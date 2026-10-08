#!/usr/bin/env python3
"""Build matched control/expanded public data for the frozen release study."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet
from transformers import AutoTokenizer

from decision_model.core import canonical, digest, load_rows, summarize_rows, write_json
from decision_model.prompting import render_decoder_admission, render_decoder_candidate
from prepare_typed_decisions import convert_case, json_cell


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_inputs(root, config):
    for spec in config["data"]["inputs"]:
        path = root / spec["path"]
        if digest(path.read_bytes()) != spec["sha256"]:
            raise ValueError("Frozen release input changed: " + spec["path"])


def typed_input_key(case, rows, seed):
    return hashlib.sha256(canonical({
        "seed": seed,
        "workflow": case["workflow"],
        "case_id": case["id"],
        "requests": [row["request"] for row in rows],
    }).encode()).hexdigest()


def typed_case_rows(case, split, upstream_split, source_config, tokenizer, max_tokens):
    regime = {
        "train": "seen_task_training",
        "validation": "seen_task_validation",
        "test": "seen_task_new_examples",
    }[split]
    rows = convert_case(case, split, upstream_split, source_config["revision"],
                        source_config["license"], regime)
    for row in rows:
        request = row["request"]
        admission = len(tokenizer.encode(
            render_decoder_admission(request), add_special_tokens=True))
        paths = [len(tokenizer.encode(
            render_decoder_candidate(request, index), add_special_tokens=True))
                 for index in range(len(request["choices"]))]
        if admission > max_tokens or max(paths) > max_tokens:
            return None
    return rows


def select_typed(cases, split_counts, source_config, tokenizer, config):
    workflows = source_config["expected"]["workflows"]
    selected, report = [], {}
    for workflow in workflows:
        eligible = []
        for raw in cases:
            if raw["workflow"] != workflow:
                continue
            # Conversion validates labels, but neither labels nor probabilities enter the key.
            probe = typed_case_rows(raw, "train", "train", source_config, tokenizer,
                                    config["data"]["max_tokens"])
            if probe is not None:
                eligible.append((typed_input_key(raw, probe, config["data"]["selection_seed"]), raw))
        eligible.sort(key=lambda item: item[0])
        required = sum(count for _, count in split_counts)
        if len(eligible) < required:
            raise ValueError("Not enough token-admissible Typed Decisions cases: " + workflow)
        offset = 0
        report[workflow] = {"eligible_cases": len(eligible)}
        for split, count in split_counts:
            chosen = eligible[offset:offset + count]; offset += count
            report[workflow][split + "_cases"] = count
            report[workflow][split + "_input_sha256"] = digest([key for key, _ in chosen])
            for _, raw in chosen:
                rows = typed_case_rows(raw, split, "train", source_config, tokenizer,
                                       config["data"]["max_tokens"])
                selected.extend(rows)
    return selected, report


def select_typed_test(cases, count, source_config, tokenizer, config):
    workflows = source_config["expected"]["workflows"]
    selected, report = [], {}
    for workflow in workflows:
        eligible = []
        for raw in cases:
            if raw["workflow"] != workflow:
                continue
            probe = typed_case_rows(raw, "test", "test", source_config, tokenizer,
                                    config["data"]["max_tokens"])
            if probe is not None:
                eligible.append((typed_input_key(raw, probe, config["data"]["selection_seed"]), raw))
        eligible.sort(key=lambda item: item[0])
        if len(eligible) < count:
            raise ValueError("Not enough token-admissible Typed Decisions test cases: " + workflow)
        chosen = eligible[:count]
        report[workflow] = {"eligible_cases": len(eligible), "test_cases": count,
                            "test_input_sha256": digest([key for key, _ in chosen])}
        for _, raw in chosen:
            selected.extend(typed_case_rows(raw, "test", "test", source_config, tokenizer,
                                            config["data"]["max_tokens"]))
    return selected, report


def merge_unique(*groups):
    rows = {}
    for group in groups:
        for row in group:
            previous = rows.get(row["id"])
            if previous is not None and canonical(previous) != canonical(row):
                raise ValueError("Same row ID has different content: " + row["id"])
            rows[row["id"]] = row
    return list(rows.values())


def write_dataset(path, rows):
    path.write_text("".join(canonical(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))
    return load_rows(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/release-candidate-v1.json")
    parser.add_argument("--output", default="data/release-candidate-v1")
    parser.add_argument("--decoder-path", default="models/minicpm5-2b-base")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh release data directory")
    config = json.loads(config_path.read_text())
    verify_inputs(root, config)
    tokenizer = AutoTokenizer.from_pretrained(
        root / args.decoder_path, local_files_only=True, trust_remote_code=False)

    architecture = read_rows(root / "runs/architecture-comparison-v1/cases.jsonl")
    relation = read_rows(root / "data/public-relation-groups-v1/cases.jsonl")
    expansion = read_rows(root / "runs/task-expansion-v1/cases.jsonl")
    control_train = [row for row in architecture if row["split"] == "train"]
    relation_train = [row for row in relation if row["split"] == "train"]
    paws_train = [row for row in expansion if row["split"] == "train" and row["source"] == "paws-wiki"]
    public_expanded_train = merge_unique(relation_train, paws_train)
    common_validation = [row for row in expansion if row["split"] == "validation"]
    common_test = [row for row in expansion if row["split"] == "test"]

    source_config = json.loads((root / "configs/typed-decisions-source.json").read_text())
    train_cases = parquet.read_table(
        root / "cache/typed-decisions-ea930645/all/train-00000-of-00001.parquet").to_pylist()
    test_cases = parquet.read_table(
        root / "cache/typed-decisions-ea930645/all/test-00000-of-00001.parquet").to_pylist()
    typed_train_validation, typed_train_report = select_typed(
        train_cases,
        [("train", config["data"]["typed_train_cases_per_workflow"]),
         ("validation", config["data"]["typed_validation_cases_per_workflow"])],
        source_config, tokenizer, config)
    typed_test, typed_test_report = select_typed_test(
        test_cases, config["data"]["typed_test_cases_per_workflow"],
        source_config, tokenizer, config)
    typed_train = [row for row in typed_train_validation if row["split"] == "train"]
    typed_validation = [row for row in typed_train_validation if row["split"] == "validation"]
    expanded_train = merge_unique(public_expanded_train, typed_train)
    validation = merge_unique(common_validation, typed_validation)
    test = merge_unique(common_test, typed_test)

    expected = config["data"]
    actual = {"control_train_rows": len(control_train), "expanded_train_rows": len(expanded_train),
              "common_validation_rows": len(validation), "common_test_rows": len(test)}
    for key, value in actual.items():
        if value != expected[key]:
            raise ValueError(f"{key} differs: expected {expected[key]}, got {value}")
    if len(control_train) * 3 != len(expanded_train):
        raise ValueError("Matched example exposure count differs")

    output.mkdir(parents=True)
    control_path, expanded_path = output / "control.jsonl", output / "expanded.jsonl"
    control_rows = write_dataset(control_path, control_train + validation + test)
    expanded_rows = write_dataset(expanded_path, expanded_train + validation + test)
    if ([row["id"] for row in control_rows if row["split"] != "train"] !=
            [row["id"] for row in expanded_rows if row["split"] != "train"]):
        raise ValueError("Control and expanded evaluation rows differ")
    train_sources = Counter(row["source"] for row in expanded_train)
    if train_sources != Counter({"snli": 768, "paws-wiki": 768, "typed-decisions": 1280,
                                 "clinc": 128, "json-schema": 96, "bandit": 32}):
        raise ValueError("Expanded training source composition differs")
    manifest = {
        "format_version": 1,
        "study": config["study"],
        "protocol_sha256": digest(config_path.read_bytes()),
        "selection_is_target_blind": True,
        "typed_selection_key_fields": ["selection_seed", "workflow", "case_id", "model_requests"],
        "typed_selection_key_excludes": ["gold", "probabilities", "label", "rationale"],
        "control": {"path": "control.jsonl", "sha256": digest(control_path.read_bytes()),
                    "rows": summarize_rows(control_rows)},
        "expanded": {"path": "expanded.jsonl", "sha256": digest(expanded_path.read_bytes()),
                     "rows": summarize_rows(expanded_rows)},
        "expanded_train_sources": dict(sorted(train_sources.items())),
        "typed_train_validation_selection": typed_train_report,
        "typed_test_selection": typed_test_report,
        "evaluation_rows_identical_between_arms": True,
        "example_exposures_per_arm": len(expanded_train),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "licenses": ("Every row retains source provenance and license. The combined transformed "
                     "dataset is not relicensed as Apache-2.0; see THIRD_PARTY.md."),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"control": len(control_rows), "expanded": len(expanded_rows),
                      "train_sources": manifest["expanded_train_sources"],
                      "manifest_sha256": digest((output / "manifest.json").read_bytes())},
                     sort_keys=True))


if __name__ == "__main__":
    main()
