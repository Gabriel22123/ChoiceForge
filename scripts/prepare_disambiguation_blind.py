"""Pin and seal answer-blind, complete triples from BIG-bench disambiguation_qa."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from decision_model.core import digest, load_rows, write_json
from decision_model.prompting import render_decoder_admission, render_decoder_candidate


REPOSITORY = "https://github.com/google/BIG-bench"
REVISION = "092b196c1f8f14a54bbc62f24759d43bde46dd3b"
RAW_ROOT = f"https://raw.githubusercontent.com/google/BIG-bench/{REVISION}"
TASK_ROOT = "bigbench/benchmark_tasks/disambiguation_qa"
SOURCE_FILES = ("LICENSE", f"{TASK_ROOT}/README.md", f"{TASK_ROOT}/task.json")
GROUPS_TO_SELECT = 64
VARIANTS = ("m", "f", "t")
MAX_TOKENS = 768


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def local_path(cache, name):
    return cache / ("LICENSE" if name == "LICENSE" else Path(name).name)


def download_only(cache, source_record):
    for name in SOURCE_FILES:
        target = local_path(cache, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{RAW_ROOT}/{name}", target)
    task = json.loads((cache / "task.json").read_text())
    write_json(source_record, {
        "name": "BIG-bench disambiguation_qa",
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": "Apache-2.0",
        "files": {name: sha256(local_path(cache, name)) for name in SOURCE_FILES},
        "official_rows": len(task["examples"]),
        "selection": ("Form complete m/f/t triples from public example IDs; admit the entire "
                      "triple under the pinned decoder tokenizer; order groups by SHA-256 of "
                      "their IDs, inputs and choice text without target values; take 64 groups."),
        "release_policy": ("Raw and transformed task rows, labels, logits and per-case "
                           "predictions are excluded from release bundles."),
    })
    print({"downloaded": len(SOURCE_FILES), "revision": REVISION})


def verify_source(cache, source_record):
    source = json.loads(source_record.read_text())
    if (source.get("repository") != REPOSITORY or source.get("revision") != REVISION or
            source.get("license") != "Apache-2.0" or
            set(source.get("files", {})) != set(SOURCE_FILES)):
        raise ValueError("Disambiguation source record differs")
    for name, expected in source["files"].items():
        if sha256(local_path(cache, name)) != expected:
            raise ValueError("Disambiguation source changed: " + name)
    return source


def group_key(example_id):
    match = re.fullmatch(r"(.+)_([mft])_([^_]+)", example_id)
    if not match:
        raise ValueError("Unexpected disambiguation example ID: " + example_id)
    return f"{match.group(1)}:{match.group(3)}", match.group(2)


def make_request(raw):
    choices = list(raw["target_scores"])
    return {
        "task": ("Resolve what the potentially ambiguous expression refers to using only "
                 "the sentence. Choose Ambiguous when the sentence does not determine it."),
        "context": raw["input"].strip(),
        "choices": [{"id": f"option_{index}", "description": text.strip()}
                    for index, text in enumerate(choices)],
    }, list(raw["target_scores"].values())


def seal(cache, source_record, output, decoder_path, training_data):
    from transformers import AutoTokenizer

    if output.exists():
        raise ValueError("Use a fresh output directory")
    source = verify_source(cache, source_record)
    tokenizer = AutoTokenizer.from_pretrained(
        decoder_path, local_files_only=True, trust_remote_code=False)
    training_inputs = {digest(row["request"]) for row in load_rows(training_data)
                       if row["split"] == "train"}
    raw_examples = json.loads((cache / "task.json").read_text())["examples"]
    groups = {}
    for index, raw in enumerate(raw_examples):
        key, variant = group_key(raw["_id"])
        if variant in groups.setdefault(key, {}):
            raise ValueError("Duplicate disambiguation group variant: " + raw["_id"])
        groups[key][variant] = (index, raw)
    if any(set(values) != set(VARIANTS) for values in groups.values()):
        raise ValueError("Disambiguation groups are not complete m/f/t triples")

    eligible = []
    for key, variants in groups.items():
        group_rows, answer_blind = [], []
        for variant in VARIANTS:
            index, raw = variants[variant]
            request, scores = make_request(raw)
            if scores.count(1) != 1 or any(score not in (0, 1) for score in scores):
                raise ValueError("Malformed disambiguation target: " + raw["_id"])
            admission = len(tokenizer.encode(
                render_decoder_admission(request), add_special_tokens=True))
            candidate_lengths = [len(tokenizer.encode(
                render_decoder_candidate(request, choice_index), add_special_tokens=True))
                for choice_index in range(len(request["choices"]))]
            if admission > MAX_TOKENS or max(candidate_lengths) > MAX_TOKENS:
                group_rows = []
                break
            answer_index = scores.index(1)
            group_rows.append((variant, index, raw, request, answer_index))
            answer_blind.append({"id": raw["_id"], "input": raw["input"],
                                 "choices": list(raw["target_scores"])})
        if group_rows:
            eligible.append((digest({"group": key, "rows": answer_blind}), key, group_rows))
    eligible.sort(key=lambda item: item[0])
    if len(eligible) < GROUPS_TO_SELECT:
        raise ValueError("Not enough admissible complete disambiguation groups")
    chosen = eligible[:GROUPS_TO_SELECT]
    selected = []
    for _selection_key, key, rows in chosen:
        for variant, index, raw, request, answer_index in rows:
            if digest(request) in training_inputs:
                raise ValueError("Disambiguation request overlaps training data")
            selected.append({
                "id": digest({"dataset": "bigbench-disambiguation-qa",
                              "revision": REVISION, "source_id": raw["_id"]}),
                "group_id": "disambiguation-qa:" + key,
                "source": "disambiguation-qa",
                "split": "test", "request": request,
                "label": f"option_{answer_index}",
                "provenance": {"dataset": "google/BIG-bench", "revision": REVISION,
                               "task": "disambiguation_qa", "source_index": index,
                               "source_id": raw["_id"], "variant": variant},
            })
    output.mkdir(parents=True)
    cases = output / "cases.jsonl"
    with cases.open("w") as stream:
        for row in sorted(selected, key=lambda item: item["id"]):
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    loaded = load_rows(cases)
    if len(loaded) != GROUPS_TO_SELECT * len(VARIANTS):
        raise ValueError("Disambiguation blind row count differs")
    write_json(output / "protocol.json", {
        "question": ("Does train-time context advantage improve unseen pronoun resolution "
                     "and ambiguity recognition over matched raw decoder controls?"),
        "dataset": source,
        "dataset_source_record_sha256": sha256(source_record),
        "preparation_script_sha256": sha256(Path(__file__).resolve()),
        "training_data_sha256": sha256(training_data),
        "exact_training_request_overlap": 0,
        "selection_is_answer_blind": True,
        "complete_group_selection": True,
        "source_groups": len(groups), "eligible_groups": len(eligible),
        "selected_groups": GROUPS_TO_SELECT,
        "selected_group_sha256": digest([item[0] for item in chosen]),
        "selected_source_ids": [row[2]["_id"] for item in chosen for row in item[2]],
        "cases_sha256": sha256(cases), "rows": len(loaded),
        "role": ("Fresh task-level blind confirmation only; never training, calibration, "
                 "hyperparameter selection or endpoint selection."),
        "evaluation_rule": ("After all three fixed treatment endpoints and their audits are "
                            "complete, evaluate matched raw controls and treatments once."),
    })
    write_json(output / "status.json", {"state": "prepared"})
    print({"prepared": str(output), "rows": len(loaded),
           "groups": GROUPS_TO_SELECT, "cases_sha256": sha256(cases)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache/bigbench-disambiguation")
    parser.add_argument("--source-record", default="configs/disambiguation-blind-source.json")
    parser.add_argument("--output", default="runs/context-advantage-v1/disambiguation-blind")
    parser.add_argument("--decoder-path", default="models/minicpm5-2b-base")
    parser.add_argument("--training-data", default="runs/architecture-comparison-v1/cases.jsonl")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--download-only", action="store_true")
    action.add_argument("--seal-selection", action="store_true")
    args = parser.parse_args()
    if args.download_only:
        download_only(Path(args.cache), Path(args.source_record))
    else:
        seal(Path(args.cache), Path(args.source_record), Path(args.output),
             Path(args.decoder_path), Path(args.training_data))


if __name__ == "__main__":
    main()
