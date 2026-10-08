#!/usr/bin/env python3
"""Run a matched evidence-only versus dual-function QASC retention screen."""
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
from decision_model.judge_factory import make_judge
from decision_model.prior_correction import prior_only_request


ARMS = ("half-qasc-evidence-only", "half-qasc-dual-function")


def read(path):
    return json.loads(Path(path).read_text())


def attach_teacher_and_weights(rows, null_teacher, replay_weight, qasc_weight):
    result = []
    replay_ids = {row["id"] for row in rows if row["split"] == "train" and row["source"] != "qasc"}
    if set(null_teacher) != replay_ids:
        raise ValueError("Null-teacher coverage differs from replay rows")
    for row in rows:
        item = dict(row)
        if row["split"] == "train":
            if row["source"] == "qasc":
                item["training_weight"] = qasc_weight
            else:
                expected = {choice["id"] for choice in row["request"]["choices"]}
                values = null_teacher[row["id"]]
                if (set(values) != expected or any(not math.isfinite(value) or value <= 0
                                                    for value in values.values()) or
                        not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6)):
                    raise ValueError("Invalid null-teacher distribution: " + row["id"])
                item["teacher_null_probabilities"] = values
                item["training_weight"] = replay_weight
        result.append(item)
    return result


def collect_null_teacher(root, parent, rows, device):
    config = read(parent / "model.json")
    judge = make_judge(config, base_path=root / "models/minicpm5-2b-base",
                       checkpoint=parent, device=device)
    judge.temperature = 1.0
    teacher = {}
    import torch
    judge.train(False)
    with torch.no_grad():
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            views = [prior_only_request(row["request"]) for row in batch]
            logits = judge.logits([judge.encode(view) for view in views]).float()
            for index, row in enumerate(batch):
                choices = row["request"]["choices"]
                probabilities = torch.softmax(logits[index, :len(choices)], -1).cpu().tolist()
                teacher[row["id"]] = {
                    choice["id"]: float(value) for choice, value in zip(choices, probabilities)
                }
    return teacher


def verify_inputs(root, config):
    exact = {
        config["parent"]["checkpoint"] + "/decision.safetensors": config["parent"]["weights_sha256"],
        config["parent"]["checkpoint"] + "/model.json": config["parent"]["model_sha256"],
        config["data"]["qasc_cases"]: config["data"]["qasc_cases_sha256"],
        config["data"]["qasc_development"]: config["data"]["qasc_development_sha256"],
        config["data"]["consumed_development"]: config["data"]["consumed_development_sha256"],
        config["baselines"]["qasc_screen_evidence"]: config["baselines"]["qasc_screen_evidence_sha256"],
        config["baselines"]["functional_retention_evidence"]: config["baselines"]["functional_retention_evidence_sha256"],
    }
    for name, expected in exact.items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Frozen dual-function input changed: " + name)
    verify_local_decoder_base(root / "models/minicpm5-2b-base")


def source_hashes(root):
    paths = [root / "src/decision_model/train.py",
             root / "src/decision_model/training_objective.py",
             root / "scripts/evaluate_development_suite.py",
             root / "scripts/qasc_dual_function_screen_report.py",
             root / "scripts/run_qasc_dual_function_screen.py",
             root / "configs/qasc-dual-function-screen-v1.json"]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def prepare(root, output, device):
    if output.exists():
        raise ValueError("Use a fresh dual-function study directory")
    config_path = root / "configs/qasc-dual-function-screen-v1.json"
    frozen = read(config_path)
    if (frozen["status"] != "frozen_before_teacher_collection_or_training" or
            tuple(frozen["arms"]) != ARMS):
        raise ValueError("Dual-function protocol was not frozen")
    verify_inputs(root, frozen)
    rows = load_rows(root / frozen["data"]["qasc_cases"])
    replay = [row for row in rows if row["split"] == "train" and row["source"] != "qasc"]
    qasc_train = [row for row in rows if row["split"] == "train" and row["source"] == "qasc"]
    if len(replay) != 512 or len(qasc_train) != 512:
        raise ValueError("Matched screen needs 512 replay and 512 QASC rows")
    output.mkdir(parents=True)
    write_json(output / "status.json", {"state": "teacher-collection"})
    parent = root / frozen["parent"]["checkpoint"]
    teacher = collect_null_teacher(root, parent, replay, device)
    write_json(output / "null-teacher.json", teacher)
    training = frozen["training"]
    cases = attach_teacher_and_weights(rows, teacher, training["replay_weight"],
                                       training["qasc_weight"])
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(canonical(row) + "\n" for row in cases))
    checked = load_rows(cases_path)
    if len(checked) != len(rows):
        raise ValueError("Prepared dual-function row count differs")
    parent_model = read(parent / "model.json")
    config_hashes = {}
    for arm in ARMS:
        model = dict(parent_model)
        model.update({
            "epochs": training["epochs"], "batch_size": training["batch_size"],
            "accumulation": training["accumulation"],
            "encoder_lr": training["encoder_lr"], "head_lr": training["head_lr"],
            "context_advantage_weight": training["context_advantage_weight"],
            "context_advantage_gap": training["context_advantage_gap"],
            "distillation_weight": training["distillation_weight"],
            "distillation_target_field": training["distillation_target_field"],
            "distillation_allow_missing_targets": True,
            "null_distillation_weight": frozen["arms"][arm]["null_distillation_weight"],
            "null_distillation_target_field": training["null_distillation_target_field"],
            "null_distillation_allow_missing_targets": True,
            "training_weight_field": training["training_weight_field"],
            "max_train_evaluation_rows": 512, "require_all_rows": True,
        })
        model_path = output / f"{arm}.json"
        write_json(model_path, model)
        config_hashes[arm] = digest(model_path.read_bytes())
    protocol = {
        **frozen, "config_sha256": digest(config_path.read_bytes()),
        "cases_sha256": digest(cases_path.read_bytes()),
        "null_teacher_sha256": digest((output / "null-teacher.json").read_bytes()),
        "null_teacher_rows": len(teacher), "config_hashes": config_hashes,
        "source_files": source_hashes(root),
        "training_weighting": {"mean": .75, "minimum": .5, "maximum": 1.0},
    }
    write_json(output / "protocol.json", protocol)
    for name, expected in protocol["source_files"].items():
        source = root / name
        if digest(source.read_bytes()) != expected:
            raise ValueError("Dual-function source changed during preparation")
        target = output / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(output / "protocol-checksums.json", {
        path.name: digest(path.read_bytes()) for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    })
    write_json(output / "status.json", {"state": "prepared"})
    print(json.dumps({"state": "prepared", "teacher_rows": len(teacher),
                      "cases_sha256": protocol["cases_sha256"]}, sort_keys=True))


def verify(root, output, protocol):
    verify_inputs(root, protocol)
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen dual-function protocol changed: " + name)
    for name, expected in protocol["source_files"].items():
        if (digest((root / name).read_bytes()) != expected or
                digest((output / "frozen-source" / name).read_bytes()) != expected):
            raise ValueError("Frozen dual-function source changed: " + name)


def command(root, output, protocol, arm, stage, args):
    verify(root, output, protocol)
    write_json(output / "status.json", {"state": stage, "arm": arm})
    print(json.dumps({"stage": stage, "arm": arm}), flush=True)
    with (output / f"{arm}-{stage}.log").open("w") as stream:
        try:
            subprocess.run(args, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            write_json(output / "status.json", {"state": "failed", "stage": stage, "arm": arm})
            raise


def run(root, output, device):
    protocol = read(output / "protocol.json")
    if read(output / "status.json").get("state") not in {
            "prepared", "training", "audit", "development", "qasc-context", "failed"}:
        raise ValueError("Dual-function screen cannot resume from this state")
    parent = root / protocol["parent"]["checkpoint"]
    base = root / "models/minicpm5-2b-base"
    plan_hash = None
    for arm in ARMS:
        target = output / arm
        if not target.exists():
            command(root, output, protocol, arm, "training", [
                sys.executable, "-m", "decision_model.cli", "train",
                "--data", str(output / "cases.jsonl"),
                "--config", str(output / f"{arm}.json"), "--output", str(target),
                "--init-from", str(parent), "--base-path", str(base), "--device", device])
        record = read(target / "run.json")
        weighting = record["training_weighting"]
        expected_null_rows = 512 if protocol["arms"][arm]["null_distillation_weight"] else 0
        if (record["dataset_sha256"] != protocol["cases_sha256"] or
                record["config_sha256"] != protocol["config_hashes"][arm] or
                record["warm_start_weights_sha256"] != protocol["parent"]["weights_sha256"] or
                record["updates"] != protocol["training"]["updates"] or
                record["functional_distillation"]["rows"] != 512 or
                record["null_functional_distillation"]["rows"] != expected_null_rows or
                abs(weighting["mean"] - .75) > 1e-12 or
                abs(weighting["minimum"] - .5) > 1e-12 or
                abs(weighting["maximum"] - 1.0) > 1e-12):
            raise ValueError("Dual-function training provenance differs: " + arm)
        if plan_hash is None:
            plan_hash = record["training_plan_sha256"]
        elif record["training_plan_sha256"] != plan_hash:
            raise ValueError("Matched dual-function arms used different plans")
        audit = output / "audits" / f"{arm}.json"
        if not audit.exists():
            audit.parent.mkdir(exist_ok=True)
            command(root, output, protocol, arm, "audit", [
                sys.executable, "-m", "decision_model.cli", "audit",
                "--checkpoint", str(target), "--data", str(output / "cases.jsonl"),
                "--output", str(audit), "--max-cases", "24",
                "--base-path", str(base), "--device", device])
        development = output / "development" / arm
        if not development.exists():
            development.parent.mkdir(exist_ok=True)
            command(root, output, protocol, arm, "development", [
                sys.executable, "scripts/evaluate_development_suite.py",
                "--checkpoint", str(target),
                "--data", str(root / protocol["data"]["consumed_development"]),
                "--output", str(development), "--base-path", str(base), "--device", device])
        qasc = output / "qasc-context" / arm
        if not qasc.exists():
            qasc.parent.mkdir(exist_ok=True)
            command(root, output, protocol, arm, "qasc-context", [
                sys.executable, "scripts/evaluate_development_suite.py",
                "--checkpoint", str(target),
                "--data", str(root / protocol["data"]["qasc_development"]),
                "--output", str(qasc), "--base-path", str(base), "--device", device])
    command(root, output, protocol, "all", "reporting", [
        sys.executable, "scripts/qasc_dual_function_screen_report.py"])
    write_json(output / "status.json", {"state": "complete"})
    print(json.dumps({"state": "complete"}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/qasc-dual-function-screen-v1")
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    prepare(root, output, args.device) if args.prepare_only else run(root, output, args.device)


if __name__ == "__main__":
    main()
