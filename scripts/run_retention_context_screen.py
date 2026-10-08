#!/usr/bin/env python3
"""Freeze and run a public-data screen for retention weighting and context gain."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base


ARMS = ("retention", "retention-context")


def read(path):
    return json.loads(Path(path).read_text())


def verify_parent(root, config):
    parent = config["parent"]
    for key in ("control", "expanded"):
        path = root / parent[f"{key}_data"]
        if digest(path.read_bytes()) != parent[f"{key}_data_sha256"]:
            raise ValueError("Parent data changed: " + key)
    checkpoint = root / parent["baseline_checkpoint"]
    if (digest((checkpoint / "decision.safetensors").read_bytes()) !=
            parent["baseline_weights_sha256"] or
            digest((checkpoint / "model.json").read_bytes()) !=
            parent["baseline_model_sha256"]):
        raise ValueError("Parent expanded checkpoint changed")
    suite = config["development_suite"]
    if (digest((root / suite["cases"]).read_bytes()) != suite["cases_sha256"] or
            digest((root / suite["cases"]).parent.joinpath("protocol.json").read_bytes()) !=
            suite["protocol_sha256"]):
        raise ValueError("Consumed development suite changed")
    verify_local_decoder_base(root / "models/minicpm5-2b-base")


def source_hashes(root):
    paths = [
        root / "src/decision_model/train.py",
        root / "src/decision_model/training_objective.py",
        root / "scripts/evaluate_development_suite.py",
        root / "scripts/retention_context_screen_report.py",
        Path(__file__).resolve(),
        root / "configs/retention-context-screen-v1.json",
    ]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def prepare(root, output):
    if output.exists():
        raise ValueError("Use a fresh retention-context study directory")
    config_path = root / "configs/retention-context-screen-v1.json"
    frozen = read(config_path)
    if (frozen["status"] != "frozen_before_training" or frozen["seed"] != 1701 or
            tuple(frozen["arms"]) != ARMS):
        raise ValueError("Frozen screen identity differs")
    verify_parent(root, frozen)
    expanded = load_rows(root / frozen["parent"]["expanded_data"])
    control = load_rows(root / frozen["parent"]["control_data"])
    retained_ids = {row["id"] for row in control if row["split"] == "train"}
    train_ids = {row["id"] for row in expanded if row["split"] == "train"}
    if not retained_ids <= train_ids:
        raise ValueError("Retained rows are absent from expanded data")
    weights = frozen["weighting"]
    if (len(retained_ids) != weights["retained_rows"] or
            len(train_ids - retained_ids) != weights["new_rows"]):
        raise ValueError("Frozen retained/new row counts differ")
    cases = []
    for row in expanded:
        item = dict(row)
        if row["split"] == "train":
            item[weights["field"]] = (weights["retained_normalized_weight"]
                                       if row["id"] in retained_ids else
                                       weights["new_normalized_weight"])
        cases.append(item)
    train = [row for row in cases if row["split"] == "train"]
    mean_weight = sum(row[weights["field"]] for row in train) / len(train)
    if abs(mean_weight - weights["mean_normalized_weight"]) > 1e-12:
        raise ValueError("Normalized training weights do not have frozen mean")

    output.mkdir(parents=True)
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(canonical(row) + "\n" for row in cases))
    checked = load_rows(cases_path)
    if len(checked) != len(cases):
        raise ValueError("Weighted dataset row count differs")

    parent_model = read(root / frozen["parent"]["baseline_checkpoint"] / "model.json")
    config_hashes = {}
    for arm in ARMS:
        arm_config = dict(parent_model)
        arm_config["training_weight_field"] = weights["field"]
        arm_config.update(frozen["arms"][arm])
        target = output / f"{arm}.json"
        write_json(target, arm_config)
        config_hashes[arm] = digest(target.read_bytes())

    protocol = {
        **frozen,
        "config_sha256": digest(config_path.read_bytes()),
        "cases_sha256": digest(cases_path.read_bytes()),
        "rows": len(cases),
        "train_rows": len(train),
        "config_hashes": config_hashes,
        "source_files": source_hashes(root),
        "retained_ids_sha256": digest(sorted(retained_ids)),
        "new_ids_sha256": digest(sorted(train_ids - retained_ids)),
        "parent_training_plan_sha256": digest(
            (root / frozen["parent"]["baseline_checkpoint"] / "training-plan.jsonl").read_bytes()),
        "parent_initial_trainable_sha256": read(
            root / frozen["parent"]["baseline_checkpoint"] / "run.json")[
                "initial_trainable_sha256"],
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
                      "train_rows": len(train), "mean_weight": mean_weight}, sort_keys=True))


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
    baseline_run = read(root / protocol["parent"]["baseline_checkpoint"] / "run.json")
    endpoints = {}
    for arm in ARMS:
        target = output / arm
        if not target.exists():
            command(root, output, protocol, "training", arm, [
                sys.executable, "-m", "decision_model.cli", "train",
                "--data", str(output / "cases.jsonl"),
                "--config", str(output / f"{arm}.json"),
                "--output", str(target),
                "--base-path", str(root / "models/minicpm5-2b-base"),
                "--device", device,
            ])
        run_record = read(target / "run.json")
        if (run_record["dataset_sha256"] != protocol["cases_sha256"] or
                run_record["config_sha256"] != protocol["config_hashes"][arm] or
                run_record["initial_trainable_sha256"] !=
                protocol["parent_initial_trainable_sha256"] or
                run_record["training_plan_sha256"] !=
                protocol["parent_training_plan_sha256"] or
                run_record["updates"] != 384 or
                run_record["selected_ids"] != baseline_run["selected_ids"] or
                run_record["training_weighting"] != {
                    "field": protocol["weighting"]["field"], "mean": 1.0,
                    "minimum": 0.6, "maximum": 1.8}):
            raise ValueError("Screen training provenance differs: " + arm)
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
        sys.executable, "scripts/retention_context_screen_report.py",
    ])
    write_json(output / "status.json", {"state": "complete", "arms": list(ARMS)})
    print(json.dumps({"state": "complete", "arms": list(ARMS)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/retention-context-screen-v1")
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
