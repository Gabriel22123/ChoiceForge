"""Pin and seal an answer-blind BIG-bench logical-deduction subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from decision_model.core import digest, load_rows, write_json
from decision_model.prompting import render_decoder_admission, render_decoder_candidate


REPOSITORY = "https://github.com/google/BIG-bench"
REVISION = "092b196c1f8f14a54bbc62f24759d43bde46dd3b"
RAW_ROOT = f"https://raw.githubusercontent.com/google/BIG-bench/{REVISION}"
TASK_ROOT = "bigbench/benchmark_tasks/logical_deduction"
SOURCE_FILES = (
    "LICENSE",
    f"{TASK_ROOT}/README.md",
    f"{TASK_ROOT}/three_objects/task.json",
    f"{TASK_ROOT}/five_objects/task.json",
    f"{TASK_ROOT}/seven_objects/task.json",
)
SUBTASKS = ("three_objects", "five_objects", "seven_objects")
ROWS_PER_SUBTASK = 96
MAX_TOKENS = 768


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def local_path(cache, source_name):
    if source_name == "LICENSE":
        return cache / "LICENSE"
    return cache / Path(source_name).relative_to(TASK_ROOT)


def download_only(cache, source_record):
    for source_name in SOURCE_FILES:
        target = local_path(cache, source_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{RAW_ROOT}/{source_name}", target)
    write_json(source_record, {
        "name": "BIG-bench logical_deduction",
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": "Apache-2.0",
        "files": {source_name: sha256(local_path(cache, source_name))
                  for source_name in SOURCE_FILES},
        "official_rows": {subtask: len(json.loads(
            (cache / subtask / "task.json").read_text())["examples"])
            for subtask in SUBTASKS},
        "selection": ("For each subtask, admit rows under the pinned decoder tokenizer, "
                      "order by SHA-256 of subtask/input/choice text without target scores, "
                      "then take 96."),
        "release_policy": ("Raw and transformed task rows, labels, collected logits and "
                           "per-case predictions are excluded from release bundles."),
    })
    print({"downloaded": len(SOURCE_FILES), "revision": REVISION})


def verify_source(cache, source_record):
    source = json.loads(source_record.read_text())
    if (source.get("repository") != REPOSITORY or source.get("revision") != REVISION or
            source.get("license") != "Apache-2.0" or set(source.get("files", {})) != set(SOURCE_FILES)):
        raise ValueError("Logical-deduction source record differs")
    for source_name, expected in source["files"].items():
        if sha256(local_path(cache, source_name)) != expected:
            raise ValueError("Logical-deduction source changed: " + source_name)
    return source


def make_request(raw):
    choices = list(raw["target_scores"])
    return {
        "task": "Use all ordering facts to choose the statement that must be true.",
        "context": raw["input"].strip(),
        "choices": [{"id": f"option_{index}", "description": text.strip()}
                    for index, text in enumerate(choices)],
    }, choices


def seal(cache, source_record, output, decoder_path, training_data):
    from transformers import AutoTokenizer

    if output.exists():
        raise ValueError("Use a fresh output directory")
    source = verify_source(cache, source_record)
    tokenizer = AutoTokenizer.from_pretrained(
        decoder_path, local_files_only=True, trust_remote_code=False)
    training_inputs = {digest(row["request"]) for row in load_rows(training_data)
                       if row["split"] == "train"}
    selected_rows, selection = [], {}
    for subtask in SUBTASKS:
        task_file = cache / subtask / "task.json"
        task = json.loads(task_file.read_text())
        eligible = []
        for index, raw in enumerate(task["examples"]):
            request, choice_texts = make_request(raw)
            scores = list(raw["target_scores"].values())
            if scores.count(1) != 1 or any(score not in (0, 1) for score in scores):
                raise ValueError(f"Malformed logical-deduction target: {subtask}:{index}")
            admission = len(tokenizer.encode(render_decoder_admission(request), add_special_tokens=True))
            candidate_lengths = [len(tokenizer.encode(
                render_decoder_candidate(request, choice_index), add_special_tokens=True))
                for choice_index in range(len(choice_texts))]
            if admission > MAX_TOKENS or max(candidate_lengths) > MAX_TOKENS:
                continue
            selection_key = digest({"subtask": subtask, "input": raw["input"],
                                    "choices": choice_texts})
            eligible.append((selection_key, index, raw, request, scores.index(1)))
        eligible.sort(key=lambda item: item[0])
        if len(eligible) < ROWS_PER_SUBTASK:
            raise ValueError("Not enough admissible logical-deduction rows: " + subtask)
        chosen = eligible[:ROWS_PER_SUBTASK]
        selection[subtask] = {
            "source_rows": len(task["examples"]), "eligible_rows": len(eligible),
            "selected_rows": ROWS_PER_SUBTASK,
            "selected_input_sha256": digest([item[0] for item in chosen]),
            "selected_source_indexes": [item[1] for item in chosen],
        }
        for _, index, _raw, request, answer_index in chosen:
            if digest(request) in training_inputs:
                raise ValueError("Logical-deduction request overlaps training data")
            selected_rows.append({
                "id": digest({"dataset": "bigbench-logical-deduction",
                              "revision": REVISION, "subtask": subtask, "index": index}),
                "group_id": f"logical-deduction:{subtask}:{index}",
                "source": "logical-deduction-" + subtask.replace("_", "-"),
                "split": "test", "request": request, "label": f"option_{answer_index}",
                "provenance": {"dataset": "google/BIG-bench", "revision": REVISION,
                               "task": "logical_deduction", "subtask": subtask,
                               "source_index": index},
            })

    output.mkdir(parents=True)
    cases = output / "cases.jsonl"
    with cases.open("w") as stream:
        for row in sorted(selected_rows, key=lambda item: item["id"]):
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    loaded = load_rows(cases)
    if len(loaded) != len(SUBTASKS) * ROWS_PER_SUBTASK:
        raise ValueError("Logical-deduction blind row count differs")
    write_json(output / "protocol.json", {
        "question": ("Does a frozen candidate-prior correction improve unseen logical "
                     "deduction without changing the selected model endpoint?"),
        "dataset": source,
        "dataset_source_record_sha256": sha256(source_record),
        "preparation_script_sha256": sha256(Path(__file__).resolve()),
        "training_data_sha256": sha256(training_data),
        "exact_training_request_overlap": 0,
        "selection_is_answer_blind": True,
        "selection": selection,
        "cases_sha256": sha256(cases), "rows": len(loaded),
        "role": ("Fresh task-level blind confirmation only; never strength selection, "
                 "temperature fitting, training or endpoint selection."),
        "evaluation_rule": ("After one shared strength is selected from seed42/43 validation "
                            "only, evaluate raw and corrected predictions once for all three "
                            "fixed decoder endpoints."),
    })
    write_json(output / "status.json", {"state": "prepared"})
    print({"prepared": str(output), "rows": len(loaded), "cases_sha256": sha256(cases)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache/bigbench-logical-deduction")
    parser.add_argument("--source-record", default="configs/logical-deduction-blind-source.json")
    parser.add_argument("--output", default="runs/candidate-prior-v1/logical-deduction-blind")
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
