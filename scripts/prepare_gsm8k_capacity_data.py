#!/usr/bin/env python3
"""Prepare the frozen GSM8K capacity follow-up without reusing v1 validation."""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, write_json


BASE_PATH = Path(__file__).with_name("prepare_gsm8k_specialist_data.py")
SPEC = importlib.util.spec_from_file_location("gsm8k_capacity_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(BASE)
LINE_SEPARATORS = ("\x85", "\u2028", "\u2029")


def normalize_question(value: str) -> str:
    """Keep JSONL physical lines intact for Unicode line-separator inputs."""
    for separator in LINE_SEPARATORS:
        value = value.replace(separator, " ")
    return " ".join(value.split())


def make_case(raw: dict, split: str, revision: str, source_hash: str) -> dict:
    original_key = BASE.selection_key(raw)
    cleaned = dict(raw, question=normalize_question(raw["question"]))
    case = BASE.make_case(cleaned, split, revision, source_hash)
    case["id"] = digest({"source": "gsm8k", "selection_key": original_key,
                         "revision": revision, "split": split})
    case["group_id"] = "gsm8k:" + original_key
    case["provenance"][0]["selection_key"] = original_key
    case["provenance"][0]["transformation"] += (
        "; normalized Unicode line separators and whitespace in the question")
    return case


def capacity_split(rows: list[dict], initial_train_rows: int,
                   consumed_validation_rows: int, train_rows: int,
                   validation_rows: int):
    keyed, seen = [], set()
    for raw in rows:
        key = BASE.selection_key(raw)
        if key in seen:
            raise ValueError("Duplicate GSM8K question")
        seen.add(key); keyed.append((key, raw))
    keyed.sort(key=lambda value: value[0])
    if (initial_train_rows < 1 or consumed_validation_rows < 1 or
            train_rows < initial_train_rows or validation_rows < 1):
        raise ValueError("Invalid GSM8K capacity split sizes")
    old_validation_start = initial_train_rows
    old_validation_end = old_validation_start + consumed_validation_rows
    additional = train_rows - initial_train_rows
    new_train_end = old_validation_end + additional
    new_validation_end = new_train_end + validation_rows
    if new_validation_end > len(keyed):
        raise ValueError("GSM8K capacity split exceeds source rows")
    train = keyed[:initial_train_rows] + keyed[old_validation_end:new_train_end]
    old_validation = keyed[old_validation_start:old_validation_end]
    validation = keyed[new_train_end:new_validation_end]
    reserve = keyed[new_validation_end:]
    if ({key for key, _ in train} & {key for key, _ in old_validation} or
            {key for key, _ in train} & {key for key, _ in validation} or
            {key for key, _ in old_validation} & {key for key, _ in validation}):
        raise ValueError("GSM8K capacity split overlaps")
    return train, old_validation, validation, reserve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache/gsm8k-source")
    parser.add_argument("--source-config", default="configs/gsm8k-specialist-source.json")
    parser.add_argument("--output", default="runs/gsm8k-specialist-capacity-v2")
    parser.add_argument("--initial-train-rows", type=int, default=512)
    parser.add_argument("--consumed-validation-rows", type=int, default=256)
    parser.add_argument("--train-rows", type=int, default=2048)
    parser.add_argument("--validation-rows", type=int, default=256)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache, output, config_path = root / args.cache, root / args.output, root / args.source_config
    if output.exists():
        raise ValueError("Use a fresh GSM8K capacity data directory")
    config = json.loads(config_path.read_text())
    if config.get("license") != "MIT" or len(config.get("revision", "")) != 40:
        raise ValueError("Unexpected GSM8K source identity")
    import pyarrow.parquet as parquet
    path, expected = cache / BASE.SOURCE_FILE, config["files"][BASE.SOURCE_FILE]
    if BASE.file_sha256(path) != expected["sha256"]:
        raise ValueError("GSM8K source file changed")
    table = parquet.read_table(path)
    if table.num_rows != expected["rows"]:
        raise ValueError("GSM8K source row count changed")
    source = table.to_pylist()
    train, old_validation, validation, reserve = capacity_split(
        source, args.initial_train_rows, args.consumed_validation_rows,
        args.train_rows, args.validation_rows)
    cases = ([make_case(row, "train", config["revision"], expected["sha256"])
              for _, row in train] +
             [make_case(row, "validation", config["revision"], expected["sha256"])
              for _, row in validation])
    output.mkdir(parents=True)
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(canonical(row) + "\n" for row in cases))
    loaded = load_rows(cases_path)
    write_json(output / "protocol.json", {
        "study": "gsm8k-specialist-capacity-v2",
        "role": "consumed capacity follow-up with new same-family validation examples",
        "source_config": args.source_config,
        "source_config_sha256": digest(config_path.read_bytes()),
        "revision": config["revision"], "license": config["license"],
        "source_rows": len(source), "train_rows": len(train),
        "excluded_v1_validation_rows": len(old_validation),
        "validation_rows": len(validation), "unused_reserve_rows": len(reserve),
        "train_selection_sha256": digest([key for key, _ in train]),
        "excluded_v1_validation_sha256": digest([key for key, _ in old_validation]),
        "validation_selection_sha256": digest([key for key, _ in validation]),
        "reserve_selection_sha256": digest([key for key, _ in reserve]),
        "label_positions": {split: dict(Counter(row["label"] for row in loaded
                                                 if row["split"] == split))
                            for split in ("train", "validation")},
        "answer_blind_split": True, "answer_blind_candidate_order": True,
        "solution_rationale_in_model_input": False,
        "v1_validation_reused": False,
        "cases_sha256": digest(cases_path.read_bytes()),
        "preparation_script_sha256": digest(Path(__file__).read_bytes()),
        "base_preparation_script_sha256": digest(BASE_PATH.read_bytes()),
        "release_policy": config["release_policy"],
    })
    print(canonical({"event": "prepared", "rows": len(loaded),
                     "splits": dict(Counter(row["split"] for row in loaded)),
                     "cases_sha256": digest(cases_path.read_bytes())}))


if __name__ == "__main__":
    main()
