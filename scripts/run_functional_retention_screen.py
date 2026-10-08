#!/usr/bin/env python3
"""Freeze and run a warm-start functional-retention development screen."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base


ARMS = ("warm-context", "warm-functional-context")


def read(path):
    return json.loads(Path(path).read_text())


def verify_parent(root, config):
    parent = config["parent"]
    exact = {
        parent["expanded_data"]: parent["expanded_data_sha256"],
        parent["checkpoint"] + "/decision.safetensors": parent["weights_sha256"],
        parent["checkpoint"] + "/model.json": parent["model_sha256"],
        parent["checkpoint"] + "/train-predictions.json": parent["train_predictions_sha256"],
        parent["checkpoint"] + "/calibration.json": parent["calibration_sha256"],
        config["development_suite"]["cases"]: config["development_suite"]["cases_sha256"],
    }
    for name, expected in exact.items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Frozen parent input changed: " + name)
    suite_protocol = (root / config["development_suite"]["cases"]).parent / "protocol.json"
    if digest(suite_protocol.read_bytes()) != config["development_suite"]["protocol_sha256"]:
        raise ValueError("Consumed development protocol changed")
    verify_local_decoder_base(root / "models/minicpm5-2b-base")


def source_hashes(root):
    paths = [
        root / "src/decision_model/train.py",
        root / "src/decision_model/training_objective.py",
        root / "scripts/evaluate_development_suite.py",
        root / "scripts/functional_retention_screen_report.py",
        Path(__file__).resolve(),
        root / "configs/functional-retention-screen-v1.json",
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


def same_unique_ids(actual, expected):
    """Compare a selected subset without treating serialization order as identity."""
    return (len(actual) == len(expected) == len(set(actual)) == len(set(expected)) and
            set(actual) == set(expected))


def prepare(root, output):
    if output.exists():
        raise ValueError("Use a fresh functional-retention study directory")
    config_path = root / "configs/functional-retention-screen-v1.json"
    frozen = read(config_path)
    if (frozen["status"] != "frozen_before_training" or frozen["seed"] != 1701 or
            tuple(frozen["arms"]) != ARMS):
        raise ValueError("Frozen screen identity differs")
    verify_parent(root, frozen)
    parent = frozen["parent"]
    checkpoint = root / parent["checkpoint"]
    parent_run = read(checkpoint / "run.json")
    selected_ids = parent_run["evaluation_ids"]["train"]
    if (len(selected_ids) != parent["selected_train_ids"] or
            digest(selected_ids) != parent["selected_train_ids_sha256"]):
        raise ValueError("Frozen parent training subset differs")
    prediction_rows = read(checkpoint / "train-predictions.json")
    if ([row["id"] for row in prediction_rows] != selected_ids or
            len({row["id"] for row in prediction_rows}) != len(prediction_rows)):
        raise ValueError("Parent train predictions do not match frozen selection")
    temperature = read(checkpoint / "calibration.json")["temperature"]
    teacher = {row["id"]: raw_distribution(row["probabilities"], temperature)
               for row in prediction_rows}
    expanded = load_rows(root / parent["expanded_data"])
    by_id = {row["id"]: row for row in expanded}
    if any(key not in by_id or by_id[key]["split"] != "train" for key in selected_ids):
        raise ValueError("Frozen training subset is absent from expanded data")
    cases = []
    for key in selected_ids:
        item = dict(by_id[key])
        expected = {choice["id"] for choice in item["request"]["choices"]}
        if set(teacher[key]) != expected:
            raise ValueError("Teacher candidate support differs: " + key)
        item[frozen["training"]["teacher_field"]] = teacher[key]
        cases.append(item)
    cases.extend(row for row in expanded if row["split"] != "train")

    output.mkdir(parents=True)
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(canonical(row) + "\n" for row in cases))
    checked = load_rows(cases_path)
    if (len(checked) != len(cases) or
            sum(row["split"] == "train" for row in checked) != len(selected_ids)):
        raise ValueError("Functional-retention data shape differs")

    parent_model = read(checkpoint / "model.json")
    config_hashes = {}
    training = frozen["training"]
    for arm in ARMS:
        arm_config = dict(parent_model)
        arm_config.update({
            "epochs": training["epochs"],
            "batch_size": training["batch_size"],
            "accumulation": training["accumulation"],
            "encoder_lr": training["encoder_lr"],
            "head_lr": training["head_lr"],
            "context_advantage_weight": training["context_advantage_weight"],
            "context_advantage_gap": training["context_advantage_gap"],
            "distillation_target_field": training["teacher_field"],
            "distillation_weight": frozen["arms"][arm]["distillation_weight"],
            "max_train_evaluation_rows": len(selected_ids),
            "require_all_rows": True,
        })
        target = output / f"{arm}.json"
        write_json(target, arm_config)
        config_hashes[arm] = digest(target.read_bytes())

    protocol = {
        **frozen,
        "config_sha256": digest(config_path.read_bytes()),
        "cases_sha256": digest(cases_path.read_bytes()),
        "rows": len(cases),
        "train_rows": len(selected_ids),
        "teacher_distributions_sha256": digest(teacher),
        "parent_temperature": temperature,
        "config_hashes": config_hashes,
        "source_files": source_hashes(root),
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
    write_json(output / "status.json", {"state": "prepared", "completed_arms": []})
    print(json.dumps({"state": "prepared", "rows": len(cases),
                      "train_rows": len(selected_ids)}, sort_keys=True))


def verify(root, output, protocol):
    verify_parent(root, protocol)
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen screen input changed: " + name)
    for name, expected in protocol["source_files"].items():
        if (digest((root / name).read_bytes()) != expected or
                digest((output / "frozen-source" / name).read_bytes()) != expected):
            raise ValueError("Frozen screen source changed: " + name)


def command(root, output, protocol, stage, arm, args):
    verify(root, output, protocol)
    write_json(output / "status.json", {"state": stage, "arm": arm})
    print(json.dumps({"stage": stage, "arm": arm}), flush=True)
    with (output / f"{arm}-{stage}.log").open("w") as stream:
        try:
            subprocess.run(args, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            write_json(output / "status.json", {"state": "failed", "stage": stage,
                                                 "arm": arm})
            raise


def run(root, output, device):
    protocol = read(output / "protocol.json")
    if read(output / "status.json").get("state") not in (
            "prepared", "training", "auditing", "development-evaluation", "failed"):
        raise ValueError("Screen cannot resume from current state")
    checkpoint = root / protocol["parent"]["checkpoint"]
    endpoints = {}
    plan_hash = None
    for arm in ARMS:
        target = output / arm
        if not target.exists():
            command(root, output, protocol, "training", arm, [
                sys.executable, "-m", "decision_model.cli", "train",
                "--data", str(output / "cases.jsonl"),
                "--config", str(output / f"{arm}.json"),
                "--output", str(target), "--init-from", str(checkpoint),
                "--base-path", str(root / "models/minicpm5-2b-base"), "--device", device,
            ])
        run_record = read(target / "run.json")
        expected_ids = read(checkpoint / "run.json")["evaluation_ids"]["train"]
        if (run_record["dataset_sha256"] != protocol["cases_sha256"] or
                run_record["config_sha256"] != protocol["config_hashes"][arm] or
                run_record["warm_start_weights_sha256"] != protocol["parent"]["weights_sha256"] or
                run_record["updates"] != protocol["training"]["updates"] or
                not same_unique_ids(run_record["selected_ids"]["train"], expected_ids) or
                run_record["functional_distillation"]["weight"] !=
                protocol["arms"][arm]["distillation_weight"]):
            raise ValueError("Screen training provenance differs: " + arm)
        if plan_hash is None:
            plan_hash = run_record["training_plan_sha256"]
        elif run_record["training_plan_sha256"] != plan_hash:
            raise ValueError("Matched arms used different training plans")
        for filename, expected in read(target / "checksums.json").items():
            if digest((target / filename).read_bytes()) != expected:
                raise ValueError("Screen checkpoint changed: " + arm)
        audit = output / "audits" / f"{arm}.json"
        if not audit.exists():
            audit.parent.mkdir(exist_ok=True)
            command(root, output, protocol, "auditing", arm, [
                sys.executable, "-m", "decision_model.cli", "audit",
                "--checkpoint", str(target), "--data", str(output / "cases.jsonl"),
                "--output", str(audit), "--max-cases", "24",
                "--base-path", str(root / "models/minicpm5-2b-base"), "--device", device,
            ])
        audit_record = read(audit)
        if (audit_record["cases"] != 24 or audit_record["stable_all_orders"] != 24 or
                audit_record["max_reload_probability_difference"] > 2e-6 or
                audit_record["max_id_rename_probability_difference"] > 2e-6):
            raise ValueError("Screen audit failed: " + arm)
        development = output / "development-results" / arm
        if not development.exists():
            development.parent.mkdir(exist_ok=True)
            command(root, output, protocol, "development-evaluation", arm, [
                sys.executable, "scripts/evaluate_development_suite.py",
                "--checkpoint", str(target),
                "--data", str(root / protocol["development_suite"]["cases"]),
                "--output", str(development),
                "--base-path", str(root / "models/minicpm5-2b-base"), "--device", device,
            ])
        endpoints[arm] = {
            "path": str(target.relative_to(root)),
            "weights_sha256": digest((target / "decision.safetensors").read_bytes()),
            "evaluation_sha256": digest((target / "evaluation.json").read_bytes()),
            "audit_sha256": digest(audit.read_bytes()),
            "development_result_sha256": digest((development / "result.json").read_bytes()),
        }
    write_json(output / "trained-checkpoints.json", endpoints)
    write_json(output / "status.json", {"state": "trained", "arms": list(ARMS)})
    command(root, output, protocol, "reporting", "all", [
        sys.executable, "scripts/functional_retention_screen_report.py",
    ])
    write_json(output / "status.json", {"state": "complete", "arms": list(ARMS)})
    print(json.dumps({"state": "complete", "arms": list(ARMS)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/functional-retention-screen-v1")
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
