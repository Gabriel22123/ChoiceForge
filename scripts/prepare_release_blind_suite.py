#!/usr/bin/env python3
"""Pin, then seal a three-task answer-blind BIG-bench release suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.prompting import render_decoder_admission, render_decoder_candidate


REPOSITORY = "https://github.com/google/BIG-bench"
REVISION = "092b196c1f8f14a54bbc62f24759d43bde46dd3b"
RAW_ROOT = f"https://raw.githubusercontent.com/google/BIG-bench/{REVISION}"
LICENSE_PATH = "LICENSE"
TASKS = {
    "causal_judgment": {
        "task": "Answer the causal-attribution question as a typical person would.",
    },
    "formal_fallacies_syllogisms_negation": {
        "task": "Determine whether the argument is deductively valid given only the stated premises.",
    },
    "movie_recommendation": {
        "task": "Choose the movie most similar to the movies listed in the context.",
    },
}
ROWS_PER_TASK = 128
MAX_TOKENS = 768


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_path(cache, task, filename):
    return cache / ("LICENSE" if filename == LICENSE_PATH else task) / Path(filename).name


def source_names(task):
    root = f"bigbench/benchmark_tasks/{task}"
    return (LICENSE_PATH, f"{root}/README.md", f"{root}/task.json")


def download(cache, name, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{RAW_ROOT}/{name}", target)


def download_only(cache, record_path):
    if record_path.exists():
        raise ValueError("Use a fresh blind source record")
    records = {}
    license_target = source_path(cache, "", LICENSE_PATH)
    download(cache, LICENSE_PATH, license_target)
    for task in TASKS:
        for name in source_names(task)[1:]:
            download(cache, name, source_path(cache, task, name))
        raw = json.loads(source_path(cache, task, source_names(task)[2]).read_text())
        examples = raw.get("examples")
        if not isinstance(examples, list) or len(examples) < ROWS_PER_TASK:
            raise ValueError("Blind task has too few examples: " + task)
        records[task] = {
            "official_rows": len(examples),
            "description": raw.get("description"),
            "files": {name: sha256(source_path(cache, task, name))
                      for name in source_names(task)},
        }
    write_json(record_path, {
        "format_version": 1,
        "name": "release-candidate-v1 fresh BIG-bench task suite",
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": "Apache-2.0",
        "tasks": records,
        "rows_per_task": ROWS_PER_TASK,
        "selection": ("For each task, admit examples under the pinned decoder tokenizer; order by "
                      "SHA-256 of task, input and candidate text without target scores; take 128."),
        "release_policy": ("Benchmark canary applies. Raw/transformed rows, labels, logits and "
                           "per-case predictions are excluded from release bundles."),
    })
    print(json.dumps({"tasks": list(TASKS), "record_sha256": sha256(record_path)}, sort_keys=True))


def verify_sources(cache, record_path):
    record = json.loads(record_path.read_text())
    if (record.get("repository") != REPOSITORY or record.get("revision") != REVISION or
            record.get("license") != "Apache-2.0" or set(record.get("tasks", {})) != set(TASKS)):
        raise ValueError("Blind source record differs")
    for task, spec in record["tasks"].items():
        if set(spec["files"]) != set(source_names(task)):
            raise ValueError("Blind source file set differs: " + task)
        for name, expected in spec["files"].items():
            if sha256(source_path(cache, task, name)) != expected:
                raise ValueError("Blind source changed: " + name)
    return record


def request_for(task, raw):
    scores = raw.get("target_scores")
    if (not isinstance(raw.get("input"), str) or not raw["input"].strip() or
            not isinstance(scores, dict) or not 2 <= len(scores) <= 32 or
            sum(value == 1 for value in scores.values()) != 1 or
            any(value not in (0, 1) for value in scores.values())):
        raise ValueError("Malformed BIG-bench example: " + task)
    choices = list(scores)
    request = {
        "task": TASKS[task]["task"],
        "context": raw["input"].strip(),
        "choices": [{"id": f"option_{index}", "description": value.strip()}
                    for index, value in enumerate(choices)],
    }
    return request, choices.index(next(choice for choice, value in scores.items() if value == 1))


def model_input_key(request):
    """Hash exactly the label-free semantics guarded as unique by load_rows."""
    return digest({
        "task": request["task"],
        "context": request["context"],
        "choice_descriptions": sorted(choice["description"] for choice in request["choices"]),
    })


def deduplicate_eligible(eligible):
    """Remove upstream duplicate inputs without consulting answers.

    The primary input-only selection key and source index establish the survivor
    before the semantic uniqueness key is checked. Candidate order therefore
    cannot make an otherwise identical model input survive twice.
    """
    ordered = sorted(eligible, key=lambda item: (item[0], item[1]))
    unique, seen = [], set()
    for item in ordered:
        key = model_input_key(item[2])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique, len(ordered) - len(unique)


def seal(cache, record_path, output, decoder_path, training_data, endpoint_manifest):
    if output.exists():
        raise ValueError("Use a fresh blind suite directory")
    source = verify_sources(cache, record_path)
    endpoints = json.loads(endpoint_manifest.read_text())
    if endpoints.get("state") != "locked_before_blind" or len(endpoints.get("endpoints", {})) != 6:
        raise ValueError("All six endpoints must be locked before blind selection")
    tokenizer = AutoTokenizer.from_pretrained(
        decoder_path, local_files_only=True, trust_remote_code=False)
    training_inputs = {digest(row["request"]) for row in load_rows(training_data)
                       if row["split"] == "train"}
    selected, reports = [], {}
    for task in TASKS:
        raw_examples = json.loads(
            source_path(cache, task, source_names(task)[2]).read_text())["examples"]
        eligible = []
        for index, raw in enumerate(raw_examples):
            request, answer = request_for(task, raw)
            admission = len(tokenizer.encode(
                render_decoder_admission(request), add_special_tokens=True))
            paths = [len(tokenizer.encode(
                render_decoder_candidate(request, choice_index), add_special_tokens=True))
                     for choice_index in range(len(request["choices"]))]
            if admission > MAX_TOKENS or max(paths) > MAX_TOKENS:
                continue
            selection_key = digest({"task": task, "input": raw["input"],
                                    "choices": list(raw["target_scores"])})
            eligible.append((selection_key, index, request, answer))
        unique_eligible, duplicates_removed = deduplicate_eligible(eligible)
        if len(unique_eligible) < ROWS_PER_TASK:
            raise ValueError("Not enough admissible blind rows: " + task)
        chosen = unique_eligible[:ROWS_PER_TASK]
        reports[task] = {"source_rows": len(raw_examples), "eligible_rows": len(eligible),
                         "unique_eligible_rows": len(unique_eligible),
                         "duplicate_model_inputs_removed": duplicates_removed,
                         "selected_rows": ROWS_PER_TASK,
                         "selected_input_sha256": digest([item[0] for item in chosen])}
        for selection_key, index, request, answer in chosen:
            if digest(request) in training_inputs:
                raise ValueError("Blind request overlaps release training data")
            selected.append({
                "id": digest({"dataset": "google/BIG-bench", "revision": REVISION,
                              "task": task, "source_index": index}),
                "group_id": f"release-blind:{task}:{selection_key}",
                "source": "bigbench-" + task,
                "family": task,
                "split": "test",
                "evaluation_regime": "fresh_release_task_family",
                "request": request,
                "label": f"option_{answer}",
                "provenance": [{
                    "dataset": "google/BIG-bench", "url": REPOSITORY,
                    "revision": REVISION, "license": "Apache-2.0",
                    "task": task, "source_index": index,
                    "label_method": "upstream_gold_target_score",
                    "transformation": "multiple-choice candidates; hash selection excludes scores",
                }],
            })
    output.mkdir(parents=True)
    cases = output / "cases.jsonl"
    cases.write_text("".join(canonical(row) + "\n" for row in sorted(selected, key=lambda row: row["id"])))
    verified = load_rows(cases)
    if len(verified) != ROWS_PER_TASK * len(TASKS):
        raise ValueError("Blind row count differs")
    write_json(output / "protocol.json", {
        "format_version": 1,
        "study": "release-candidate-v1",
        "source_record_sha256": sha256(record_path),
        "endpoint_manifest_sha256": sha256(endpoint_manifest),
        "training_data_sha256": sha256(training_data),
        "selection_is_answer_blind": True,
        "exact_training_request_overlap": 0,
        "tasks": reports,
        "rows": len(verified),
        "cases_sha256": sha256(cases),
        "role": "One-shot fresh task-family confirmation after all endpoint weights are locked.",
        "prohibited_uses": ["training", "calibration", "early stopping", "endpoint selection",
                            "hyperparameter selection"],
    })
    write_json(output / "status.json", {"state": "prepared", "official_results_read": False})
    print(json.dumps({"rows": len(verified), "cases_sha256": sha256(cases)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache/release-blind-suite-v1")
    parser.add_argument("--source-record", default="configs/release-blind-suite-v1.json")
    parser.add_argument("--output", default="runs/release-candidate-v1/blind-suite")
    parser.add_argument("--decoder-path", default="models/minicpm5-2b-base")
    parser.add_argument("--training-data", default="data/release-candidate-v1/expanded.jsonl")
    parser.add_argument("--endpoint-manifest", default="runs/release-candidate-v1/locked-endpoints.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--download-only", action="store_true")
    action.add_argument("--seal-selection", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.download_only:
        download_only(root / args.cache, root / args.source_record)
    else:
        seal(root / args.cache, root / args.source_record, root / args.output,
             root / args.decoder_path, root / args.training_data, root / args.endpoint_manifest)


if __name__ == "__main__":
    main()
