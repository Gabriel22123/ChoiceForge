#!/usr/bin/env python3
"""Freeze and run a public BoolQ context-repair development screen."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base
from decision_model.prompting import render_decoder_admission, render_decoder_candidate
from task_expansion_data import BOOLQ_HASH, boolq_case, boolq_request, boolq_selection_key


ARM = "boolq-functional"


def read(path):
    return json.loads(Path(path).read_text())


def verify_inputs(root, config):
    parent = config["parent"]
    exact = {
        parent["checkpoint"] + "/decision.safetensors": parent["weights_sha256"],
        parent["checkpoint"] + "/model.json": parent["model_sha256"],
        parent["checkpoint"] + "/train-predictions.json": parent["train_predictions_sha256"],
        parent["checkpoint"] + "/calibration.json": parent["calibration_sha256"],
        parent["cases"]: parent["cases_sha256"],
        config["boolq"]["cache"]: config["boolq"]["sha256"],
        config["boolq"]["consumed_cases"]: config["boolq"]["consumed_cases_sha256"],
        config["development_suite"]["cases"]: config["development_suite"]["cases_sha256"],
    }
    for name, expected in exact.items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Frozen input changed: " + name)
    suite_protocol = (root / config["development_suite"]["cases"]).parent / "protocol.json"
    if digest(suite_protocol.read_bytes()) != config["development_suite"]["protocol_sha256"]:
        raise ValueError("Consumed development protocol changed")
    verify_local_decoder_base(root / "models/minicpm5-2b-base")


def source_hashes(root):
    paths = [
        root / "src/decision_model/train.py",
        root / "src/decision_model/training_objective.py",
        root / "scripts/task_expansion_data.py",
        root / "scripts/evaluate_development_suite.py",
        root / "scripts/boolq_context_repair_report.py",
        Path(__file__).resolve(),
        root / "configs/boolq-context-repair-v1.json",
    ]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def raw_distribution(probabilities, temperature):
    if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or
            not math.isfinite(temperature) or temperature <= 0 or
            any(not isinstance(value, (int, float)) or value <= 0 or
                not math.isfinite(value) for value in probabilities.values())):
        raise ValueError("Cannot reconstruct raw parent probability distribution")
    weights = {key: float(value) ** float(temperature)
               for key, value in probabilities.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def token_admissible(tokenizer, request, max_tokens):
    prompts = [render_decoder_admission(request)] + [
        render_decoder_candidate(request, index) for index in range(len(request["choices"]))]
    return all(len(tokenizer.encode(prompt, add_special_tokens=True)) <= max_tokens
               for prompt in prompts)


def balanced_binary_weights(rows):
    counts = {label: sum(row["label"] == label for row in rows)
              for label in ("no", "yes")}
    if any(count <= 0 for count in counts.values()) or sum(counts.values()) != len(rows):
        raise ValueError("Expected nonempty binary BoolQ labels")
    return {row["id"]: len(rows) / (2 * counts[row["label"]]) for row in rows}


def prepare(root, output):
    if output.exists():
        raise ValueError("Use a fresh BoolQ context-repair study directory")
    config_path = root / "configs/boolq-context-repair-v1.json"
    frozen = read(config_path)
    if frozen["status"] != "frozen_before_reference_or_training" or frozen["seed"] != 1701:
        raise ValueError("Frozen screen identity differs")
    verify_inputs(root, frozen)
    parent = root / frozen["parent"]["checkpoint"]
    parent_cases = load_rows(root / frozen["parent"]["cases"])
    replay = [row for row in parent_cases if row["split"] == "train"]
    predictions = read(parent / "train-predictions.json")
    prediction_by_id = {row["id"]: row for row in predictions}
    if len(replay) != 512 or set(prediction_by_id) != {row["id"] for row in replay}:
        raise ValueError("Parent replay prediction coverage differs")
    temperature = read(parent / "calibration.json")["temperature"]
    teacher_field = frozen["training"]["distillation_target_field"]
    weight_field = frozen["training"]["training_weight_field"]
    replay_with_teacher = []
    for row in replay:
        item = dict(row)
        values = raw_distribution(prediction_by_id[row["id"]]["probabilities"], temperature)
        if set(values) != {choice["id"] for choice in row["request"]["choices"]}:
            raise ValueError("Replay teacher support differs")
        item[teacher_field] = values
        item[weight_field] = 1.0
        replay_with_teacher.append(item)

    if frozen["boolq"]["sha256"] != BOOLQ_HASH:
        raise ValueError("BoolQ helper and protocol hash differ")
    source_rows = pq.read_table(root / frozen["boolq"]["cache"]).to_pylist()
    consumed = load_rows(root / frozen["boolq"]["consumed_cases"])
    consumed_groups = {row["group_id"] for row in consumed if row["source"] == "boolq"}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(root / "models/minicpm5-2b-base"), local_files_only=True,
        trust_remote_code=False)
    max_tokens = read(parent / "model.json")["max_tokens"]
    eligible = []
    for row in source_rows:
        group = "boolq:" + boolq_selection_key(row)
        if group in consumed_groups:
            continue
        request = boolq_request(row)
        if token_admissible(tokenizer, request, max_tokens):
            eligible.append(row)
    selected = sorted(eligible, key=boolq_selection_key)[:
        frozen["boolq"]["train_rows"] + frozen["boolq"]["validation_rows"]]
    if len(selected) != 640:
        raise ValueError("Not enough unused token-admissible BoolQ rows")
    keys = [boolq_selection_key(row) for row in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate BoolQ selection key")
    boolq_rows = []
    for index, row in enumerate(selected):
        item = boolq_case(row)
        item["split"] = "train" if index < frozen["boolq"]["train_rows"] else "validation"
        item["evaluation_regime"] = "seen_task_training" if item["split"] == "train" else "seen_task_new_examples"
        boolq_rows.append(item)
    boolq_train = [row for row in boolq_rows if row["split"] == "train"]
    weights = balanced_binary_weights(boolq_train)
    for row in boolq_train:
        row[weight_field] = weights[row["id"]]

    cases = replay_with_teacher + boolq_rows
    cases += [row for row in parent_cases if row["split"] != "train"]
    cases.sort(key=lambda row: row["id"])
    output.mkdir(parents=True)
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(canonical(row) + "\n" for row in cases))
    checked = load_rows(cases_path)
    counts = {split: sum(row["split"] == split for row in checked)
              for split in ("train", "validation", "test")}
    if counts != {"train": 1024, "validation": 736, "test": 1376}:
        raise ValueError("BoolQ context-repair data shape differs")

    model = read(parent / "model.json")
    training = frozen["training"]
    model.update({
        "epochs": training["epochs"], "batch_size": training["batch_size"],
        "accumulation": training["accumulation"], "encoder_lr": training["encoder_lr"],
        "head_lr": training["head_lr"],
        "context_advantage_weight": training["context_advantage_weight"],
        "context_advantage_gap": training["context_advantage_gap"],
        "distillation_weight": training["distillation_weight"],
        "distillation_target_field": teacher_field,
        "distillation_allow_missing_targets": True,
        "training_weight_field": weight_field,
        "max_train_evaluation_rows": 512, "require_all_rows": True,
    })
    model_path = output / f"{ARM}.json"
    write_json(model_path, model)
    protocol = {
        **frozen,
        "config_sha256": digest(config_path.read_bytes()),
        "cases_sha256": digest(cases_path.read_bytes()),
        "model_config_sha256": digest(model_path.read_bytes()),
        "selected_boolq_keys_sha256": digest(keys),
        "selected_boolq_label_counts": {
            split: {label: sum(row["source"] == "boolq" and row["split"] == split and
                              row["label"] == label for row in boolq_rows)
                    for label in ("no", "yes")}
            for split in ("train", "validation")
        },
        "training_weighting": {
            "field": weight_field,
            "mean": sum(row[weight_field] for row in replay_with_teacher + boolq_train) /
                    (len(replay_with_teacher) + len(boolq_train)),
            "minimum": min(row[weight_field] for row in replay_with_teacher + boolq_train),
            "maximum": max(row[weight_field] for row in replay_with_teacher + boolq_train),
            "boolq_total_by_label": {
                label: sum(row[weight_field] for row in boolq_train if row["label"] == label)
                for label in ("no", "yes")
            },
        },
        "consumed_boolq_groups": len(consumed_groups),
        "parent_temperature": temperature,
        "source_files": source_hashes(root),
        "rows": counts,
    }
    write_json(output / "protocol.json", protocol)
    for name, expected in protocol["source_files"].items():
        source = root / name
        if digest(source.read_bytes()) != expected:
            raise ValueError("Screen source changed during preparation: " + name)
        target = output / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(output / "protocol-checksums.json", {
        path.name: digest(path.read_bytes()) for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    })
    write_json(output / "status.json", {"state": "prepared"})
    print(json.dumps({"state": "prepared", "rows": counts,
                      "boolq_labels": protocol["selected_boolq_label_counts"]}, sort_keys=True))


def verify(root, output, protocol):
    verify_inputs(root, protocol)
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen screen input changed: " + name)
    for name, expected in protocol["source_files"].items():
        if (digest((root / name).read_bytes()) != expected or
                digest((output / "frozen-source" / name).read_bytes()) != expected):
            raise ValueError("Frozen screen source changed: " + name)


def command(root, output, protocol, stage, args):
    verify(root, output, protocol)
    write_json(output / "status.json", {"state": stage})
    print(json.dumps({"stage": stage}), flush=True)
    with (output / f"{stage}.log").open("w") as stream:
        try:
            subprocess.run(args, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            write_json(output / "status.json", {"state": "failed", "stage": stage})
            raise


def run(root, output, device):
    protocol = read(output / "protocol.json")
    if read(output / "status.json").get("state") not in (
            "prepared", "parent-reference", "training", "auditing",
            "development-evaluation", "failed"):
        raise ValueError("Screen cannot resume from current state")
    parent = root / protocol["parent"]["checkpoint"]
    reference = output / "parent-validation-reference.json"
    if not reference.exists():
        command(root, output, protocol, "parent-reference", [
            sys.executable, "-m", "decision_model.cli", "evaluate",
            "--checkpoint", str(parent), "--data", str(output / "cases.jsonl"),
            "--output", str(reference), "--split", "validation", "--max-cases", "0",
            "--base-path", str(root / "models/minicpm5-2b-base"), "--device", device,
        ])
    target = output / ARM
    if not target.exists():
        command(root, output, protocol, "training", [
            sys.executable, "-m", "decision_model.cli", "train",
            "--data", str(output / "cases.jsonl"),
            "--config", str(output / f"{ARM}.json"), "--output", str(target),
            "--init-from", str(parent),
            "--base-path", str(root / "models/minicpm5-2b-base"), "--device", device,
        ])
    record = read(target / "run.json")
    actual_weighting = record["training_weighting"]
    expected_weighting = protocol["training_weighting"]
    weighting_matches = (
        actual_weighting["field"] == expected_weighting["field"] and
        abs(actual_weighting["mean"] - expected_weighting["mean"]) <= 1e-12 and
        abs(actual_weighting["minimum"] - expected_weighting["minimum"]) <= 1e-12 and
        abs(actual_weighting["maximum"] - expected_weighting["maximum"]) <= 1e-12)
    if (record["dataset_sha256"] != protocol["cases_sha256"] or
            record["config_sha256"] != protocol["model_config_sha256"] or
            record["warm_start_weights_sha256"] != protocol["parent"]["weights_sha256"] or
            record["updates"] != protocol["training"]["updates"] or
            record["functional_distillation"]["rows"] != 512 or
            not record["functional_distillation"]["allow_missing_targets"] or
            not weighting_matches):
        raise ValueError("BoolQ context-repair training provenance differs")
    for filename, expected in read(target / "checksums.json").items():
        if digest((target / filename).read_bytes()) != expected:
            raise ValueError("Screen checkpoint changed")
    audit = output / "audit.json"
    if not audit.exists():
        command(root, output, protocol, "auditing", [
            sys.executable, "-m", "decision_model.cli", "audit",
            "--checkpoint", str(target), "--data", str(output / "cases.jsonl"),
            "--output", str(audit), "--max-cases", "24",
            "--base-path", str(root / "models/minicpm5-2b-base"), "--device", device,
        ])
    audit_record = read(audit)
    if (audit_record["cases"] != 24 or audit_record["stable_all_orders"] != 24 or
            audit_record["max_reload_probability_difference"] > 2e-6 or
            audit_record["max_id_rename_probability_difference"] > 2e-6):
        raise ValueError("Screen audit failed")
    development = output / "development-result"
    if not development.exists():
        command(root, output, protocol, "development-evaluation", [
            sys.executable, "scripts/evaluate_development_suite.py",
            "--checkpoint", str(target),
            "--data", str(root / protocol["development_suite"]["cases"]),
            "--output", str(development),
            "--base-path", str(root / "models/minicpm5-2b-base"), "--device", device,
        ])
    endpoint = {
        "path": str(target.relative_to(root)),
        "weights_sha256": digest((target / "decision.safetensors").read_bytes()),
        "evaluation_sha256": digest((target / "evaluation.json").read_bytes()),
        "audit_sha256": digest(audit.read_bytes()),
        "development_result_sha256": digest((development / "result.json").read_bytes()),
        "parent_reference_sha256": digest(reference.read_bytes()),
    }
    write_json(output / "trained-checkpoint.json", endpoint)
    write_json(output / "status.json", {"state": "trained"})
    command(root, output, protocol, "reporting", [
        sys.executable, "scripts/boolq_context_repair_report.py",
    ])
    write_json(output / "status.json", {"state": "complete"})
    print(json.dumps({"state": "complete", "arm": ARM}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/boolq-context-repair-v1")
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if args.prepare_only:
        prepare(root, output)
    else:
        run(root, output, args.device)


if __name__ == "__main__":
    main()
