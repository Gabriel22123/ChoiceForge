#!/usr/bin/env python3
"""Prepare deterministic GSM8K answer-verification specialist rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, validate_request, write_json


TASK = "Solve the math word problem and choose its correct final numeric answer."
SOURCE_FILE = "main/train-00000-of-00001.parquet"
FINAL_PATTERN = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)\s*$")
CALC_PATTERN = re.compile(r"<<[^<>]*?=(-?\d[\d,]*(?:\.\d+)?)>>")


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def decimal_value(value: str) -> Decimal:
    try:
        result = Decimal(value.replace(",", ""))
    except (InvalidOperation, AttributeError) as error:
        raise ValueError("Malformed GSM8K numeric answer") from error
    if not result.is_finite():
        raise ValueError("Non-finite GSM8K numeric answer")
    return result


def format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def validate_raw(raw: dict) -> None:
    if set(raw) != {"question", "answer"} or any(
            not isinstance(raw[name], str) or not raw[name].strip()
            for name in ("question", "answer")):
        raise ValueError("Malformed GSM8K row")
    if FINAL_PATTERN.search(raw["answer"]) is None:
        raise ValueError("GSM8K final answer marker changed")


def selection_key(raw: dict) -> str:
    """The question alone controls admission and split assignment."""
    validate_raw(raw)
    return digest({"question": raw["question"].strip()})


def numeric_candidates(raw: dict) -> tuple[str, list[str]]:
    validate_raw(raw)
    final_match = FINAL_PATTERN.search(raw["answer"])
    correct_value = decimal_value(final_match.group(1))
    correct = format_decimal(correct_value)
    pool = []
    for token in CALC_PATTERN.findall(raw["answer"]):
        value = format_decimal(decimal_value(token))
        if value != correct and value not in pool:
            pool.append(value)
    offsets = (Decimal(1), Decimal(-1), Decimal(2), Decimal(-2))
    derived = [correct_value + value for value in offsets]
    if correct_value != 0:
        derived.extend((correct_value * 2, correct_value / 2,
                        correct_value * 10, correct_value / 10))
    for value in derived:
        text = format_decimal(value)
        if text != correct and text not in pool:
            pool.append(text)
    question = raw["question"].strip()
    pool.sort(key=lambda value: digest({"question": question, "distractor": value}))
    if len(pool) < 3:
        raise ValueError("Could not construct three distinct GSM8K distractors")
    return correct, pool[:3]


def request_and_label(raw: dict) -> tuple[dict, str]:
    correct, distractors = numeric_candidates(raw)
    question = raw["question"].strip()
    values = sorted([correct, *distractors],
                    key=lambda value: digest({"question": question, "candidate": value}))
    choices = [{"id": f"option_{index}", "description": value}
               for index, value in enumerate(values)]
    request = {"task": TASK, "context": question, "choices": choices}
    validate_request(request)
    return request, next(choice["id"] for choice in choices
                         if choice["description"] == correct)


def make_case(raw: dict, split: str, revision: str, source_hash: str) -> dict:
    request, label = request_and_label(raw)
    key = selection_key(raw)
    return {
        "id": digest({"source": "gsm8k", "selection_key": key,
                      "revision": revision, "split": split}),
        "group_id": "gsm8k:" + key,
        "source": "gsm8k",
        "family": "multi_step_numeric_reasoning",
        "split": split,
        "evaluation_regime": ("seen_task_training" if split == "train"
                              else "seen_task_new_examples"),
        "decision_type": "choice",
        "request": request,
        "label": label,
        "provenance": [{
            "origin": "GSM8K official public dataset via pinned Hugging Face conversion",
            "source_url": "https://github.com/openai/grade-school-math",
            "revision": revision,
            "official_split": "train",
            "source_file_sha256": source_hash,
            "selection_key": key,
            "license": "MIT",
            "transformation": ("selected rows by question-only hash; extracted the official final "
                               "numeric answer; built deterministic numeric distractors; excluded "
                               "solution rationale from model input; applied answer-blind candidate "
                               "permutation"),
        }],
    }


def split_rows(rows: list[dict], train_rows: int, validation_rows: int):
    keyed, seen = [], set()
    for raw in rows:
        key = selection_key(raw)
        if key in seen:
            raise ValueError("Duplicate GSM8K question")
        seen.add(key); keyed.append((key, raw))
    keyed.sort(key=lambda value: value[0])
    if train_rows < 1 or validation_rows < 1 or train_rows + validation_rows > len(keyed):
        raise ValueError("Invalid GSM8K split sizes")
    return (keyed[:train_rows], keyed[train_rows:train_rows + validation_rows],
            keyed[train_rows + validation_rows:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache/gsm8k-source")
    parser.add_argument("--source-config", default="configs/gsm8k-specialist-source.json")
    parser.add_argument("--output", default="runs/gsm8k-specialist-v1")
    parser.add_argument("--train-rows", type=int, default=512)
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache, output, config_path = root / args.cache, root / args.output, root / args.source_config
    config = json.loads(config_path.read_text())
    if config.get("license") != "MIT" or len(config.get("revision", "")) != 40:
        raise ValueError("Unexpected GSM8K source identity")
    if args.download_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=config["dataset_id"], repo_type="dataset",
                          revision=config["revision"], local_dir=cache,
                          allow_patterns=list(config["files"]))
        for relative, expected in config["files"].items():
            if file_sha256(cache / relative) != expected["sha256"]:
                raise ValueError("Downloaded GSM8K source differs: " + relative)
        print(canonical({"event": "downloaded", "revision": config["revision"],
                         "files": len(config["files"])}))
        return
    if output.exists():
        raise ValueError("Use a fresh GSM8K specialist data directory")
    import pyarrow.parquet as parquet
    path, expected = cache / SOURCE_FILE, config["files"][SOURCE_FILE]
    if file_sha256(path) != expected["sha256"]:
        raise ValueError("GSM8K source file changed")
    table = parquet.read_table(path)
    if table.num_rows != expected["rows"]:
        raise ValueError("GSM8K source row count changed")
    raw = table.to_pylist()
    train, validation, reserve = split_rows(raw, args.train_rows, args.validation_rows)
    cases = ([make_case(row, "train", config["revision"], expected["sha256"])
              for _, row in train] +
             [make_case(row, "validation", config["revision"], expected["sha256"])
              for _, row in validation])
    output.mkdir(parents=True)
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(canonical(row) + "\n" for row in cases))
    loaded = load_rows(cases_path)
    write_json(output / "protocol.json", {
        "study": "gsm8k-specialist-v1",
        "role": "public-family specialist train and consumed validation",
        "source_config": args.source_config,
        "source_config_sha256": digest(config_path.read_bytes()),
        "revision": config["revision"], "license": config["license"],
        "source_rows": len(raw), "train_rows": len(train),
        "validation_rows": len(validation), "unused_reserve_rows": len(reserve),
        "train_selection_sha256": digest([key for key, _ in train]),
        "validation_selection_sha256": digest([key for key, _ in validation]),
        "reserve_selection_sha256": digest([key for key, _ in reserve]),
        "label_positions": {split: dict(Counter(row["label"] for row in loaded
                                                 if row["split"] == split))
                            for split in ("train", "validation")},
        "answer_blind_split": True, "answer_blind_candidate_order": True,
        "solution_rationale_in_model_input": False,
        "cases_sha256": digest(cases_path.read_bytes()),
        "preparation_script_sha256": digest(Path(__file__).read_bytes()),
        "release_policy": config["release_policy"],
    })
    print(canonical({"event": "prepared", "rows": len(loaded),
                     "splits": dict(Counter(row["split"] for row in loaded)),
                     "cases_sha256": digest(cases_path.read_bytes())}))


if __name__ == "__main__":
    main()
