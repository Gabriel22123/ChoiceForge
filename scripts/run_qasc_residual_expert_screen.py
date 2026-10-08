#!/usr/bin/env python3
"""Run the consumed QASC screen for a frozen-parent residual expert."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base
from decision_model.judge_factory import make_judge


ARM = "frozen-parent-residual"


def read(path):
    return json.loads(Path(path).read_text())


def verify_inputs(root, config):
    exact = {
        config["parent"]["checkpoint"] + "/decision.safetensors":
            config["parent"]["weights_sha256"],
        config["parent"]["checkpoint"] + "/model.json": config["parent"]["model_sha256"],
        config["data"]["cases"]: config["data"]["cases_sha256"],
        config["data"]["qasc_development"]: config["data"]["qasc_development_sha256"],
        config["data"]["consumed_development"]:
            config["data"]["consumed_development_sha256"],
        config["baselines"]["qasc_screen_evidence"]:
            config["baselines"]["qasc_screen_evidence_sha256"],
        config["baselines"]["functional_retention_evidence"]:
            config["baselines"]["functional_retention_evidence_sha256"],
        config["baselines"]["dual_function_evidence"]:
            config["baselines"]["dual_function_evidence_sha256"],
    }
    for name, expected in exact.items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Frozen residual-expert input changed: " + name)
    verify_local_decoder_base(root / "models/minicpm5-2b-base")


def source_hashes(root):
    paths = [
        root / "src/decision_model/decoder_model.py",
        root / "src/decision_model/train.py",
        root / "scripts/evaluate_development_suite.py",
        root / "scripts/qasc_residual_expert_screen_report.py",
        root / "scripts/run_qasc_residual_expert_screen.py",
        root / "configs/qasc-residual-expert-screen-v1.json",
    ]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def prepare(root, output):
    if output.exists():
        raise ValueError("Use a fresh residual-expert study directory")
    config_path = root / "configs/qasc-residual-expert-screen-v1.json"
    frozen = read(config_path)
    if frozen["status"] != "frozen_before_training":
        raise ValueError("Residual-expert protocol was not frozen before training")
    verify_inputs(root, frozen)
    rows = load_rows(root / frozen["data"]["cases"])
    replay = [row for row in rows if row["split"] == "train" and row["source"] != "qasc"]
    qasc = [row for row in rows if row["split"] == "train" and row["source"] == "qasc"]
    if len(replay) != 512 or len(qasc) != 512:
        raise ValueError("Residual screen needs 512 replay and 512 QASC rows")
    if (any(row.get("training_weight") != frozen["training"]["replay_weight"] for row in replay) or
            any(row.get("training_weight") != frozen["training"]["qasc_weight"] for row in qasc)):
        raise ValueError("Frozen residual-expert training weights differ")
    output.mkdir(parents=True)
    shutil.copyfile(root / frozen["data"]["cases"], output / "cases.jsonl")
    parent_model = read(root / frozen["parent"]["checkpoint"] / "model.json")
    training = frozen["training"]
    model = dict(parent_model)
    model.update({
        "epochs": training["epochs"], "batch_size": training["batch_size"],
        "accumulation": training["accumulation"],
        "encoder_lr": training["encoder_lr"], "head_lr": training["head_lr"],
        "residual_head_width": training["residual_head_width"],
        "candidate_chunk_size": training["candidate_chunk_size"],
        "freeze_parent_decision_function": True,
        "track_frozen_parent_gradients": training["track_frozen_parent_gradients"],
        "context_advantage_weight": training["context_advantage_weight"],
        "context_advantage_gap": training["context_advantage_gap"],
        "distillation_weight": training["distillation_weight"],
        "distillation_target_field": training["distillation_target_field"],
        "distillation_allow_missing_targets": True,
        "null_distillation_weight": training["null_distillation_weight"],
        "null_distillation_target_field": training["null_distillation_target_field"],
        "null_distillation_allow_missing_targets": True,
        "training_weight_field": training["training_weight_field"],
        "max_train_evaluation_rows": 512, "require_all_rows": True,
    })
    write_json(output / "model.json", model)
    protocol = {
        **frozen,
        "config_sha256": digest(config_path.read_bytes()),
        "cases_sha256": digest((output / "cases.jsonl").read_bytes()),
        "model_config_sha256": digest((output / "model.json").read_bytes()),
        "source_files": source_hashes(root),
    }
    write_json(output / "protocol.json", protocol)
    for name, expected in protocol["source_files"].items():
        source = root / name
        if digest(source.read_bytes()) != expected:
            raise ValueError("Residual-expert source changed during preparation")
        target = output / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(output / "protocol-checksums.json", {
        path.name: digest(path.read_bytes()) for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    })
    write_json(output / "status.json", {"state": "prepared"})
    print(json.dumps({"state": "prepared", "cases_sha256": protocol["cases_sha256"]},
                     sort_keys=True))


def verify(root, output, protocol):
    verify_inputs(root, protocol)
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen residual-expert protocol changed: " + name)
    for name, expected in protocol["source_files"].items():
        if (digest((root / name).read_bytes()) != expected or
                digest((output / "frozen-source" / name).read_bytes()) != expected):
            raise ValueError("Residual-expert source changed: " + name)


def initial_equivalence(root, output, protocol, device):
    import torch
    from safetensors.torch import load_file

    config = read(output / "model.json")
    judge = make_judge(config, base_path=root / "models/minicpm5-2b-base",
                       checkpoint=root / protocol["parent"]["checkpoint"], device=device)
    judge.train(False)
    parent_state = load_file(
        str(root / protocol["parent"]["checkpoint"] / "decision.safetensors"), device="cpu")
    current = dict(judge.named_decision_parameters())
    parent_exact = (set(parent_state).issubset(current) and all(
        torch.equal(value, current[name].detach().cpu()) for name, value in parent_state.items()))
    rows = sorted(load_rows(output / "cases.jsonl"), key=lambda row: row["id"])[:24]
    max_difference = 0.0
    with torch.no_grad():
        for row in rows:
            encoded = judge.encode(row["request"])
            composed = judge.logits([encoded])[0].float()
            residual = judge.residual_head
            judge.residual_head = None
            parent = judge.logits([encoded])[0].float()
            judge.residual_head = residual
            max_difference = max(max_difference,
                                 (composed.softmax(-1) - parent.softmax(-1)).abs().max().item())
    result = {"rows": len(rows), "max_probability_difference": max_difference,
              "parent_parameters_exact": parent_exact}
    del judge
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def parent_parameter_difference(parent_path, endpoint_path):
    from safetensors.torch import load_file

    parent = load_file(str(parent_path), device="cpu")
    endpoint = load_file(str(endpoint_path), device="cpu")
    if not set(parent).issubset(endpoint):
        raise ValueError("Residual endpoint omits parent decision parameters")
    maximum = 0.0
    exact = True
    for name, before in parent.items():
        after = endpoint[name]
        if before.shape != after.shape:
            raise ValueError("Residual endpoint changes a parent tensor shape: " + name)
        if not before.equal(after):
            exact = False
            maximum = max(maximum, (before.float() - after.float()).abs().max().item())
    residual_names = sorted(name for name in endpoint if name.startswith("residual_head."))
    if not residual_names:
        raise ValueError("Residual endpoint has no residual expert parameters")
    return {"parent_tensors": len(parent), "parent_parameters_exact": exact,
            "parent_max_absolute_difference": maximum,
            "residual_tensors": residual_names}


def command(root, output, stage, args):
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
    verify(root, output, protocol)
    status = read(output / "status.json").get("state")
    if status not in {"prepared", "training", "audit", "development", "qasc", "failed"}:
        raise ValueError("Residual-expert screen cannot resume from this state")
    initial_path = output / "initial-equivalence.json"
    if not initial_path.exists():
        write_json(initial_path, initial_equivalence(root, output, protocol, device))
    initial = read(initial_path)
    if (not initial["parent_parameters_exact"] or
            initial["max_probability_difference"] !=
            protocol["screening_rule"]["initial_max_probability_difference"]):
        raise ValueError("Residual expert does not initialize as the exact parent function")
    parent = root / protocol["parent"]["checkpoint"]
    base = root / "models/minicpm5-2b-base"
    target = output / ARM
    if not target.exists():
        command(root, output, "training", [
            sys.executable, "-m", "decision_model.cli", "train",
            "--data", str(output / "cases.jsonl"), "--config", str(output / "model.json"),
            "--output", str(target), "--init-from", str(parent),
            "--base-path", str(base), "--device", device])
    record = read(target / "run.json")
    weighting = record["training_weighting"]
    if (record["dataset_sha256"] != protocol["cases_sha256"] or
            record["config_sha256"] != protocol["model_config_sha256"] or
            record["warm_start_weights_sha256"] != protocol["parent"]["weights_sha256"] or
            record["updates"] != protocol["training"]["updates"] or
            record["functional_distillation"]["rows"] != 512 or
            record["null_functional_distillation"]["rows"] != 512 or
            abs(weighting["mean"] - .75) > 1e-12 or
            abs(weighting["minimum"] - .5) > 1e-12 or
            abs(weighting["maximum"] - 1.0) > 1e-12 or
            not record["adapter_probe"]["parameter"].startswith("residual_head.") or
            record["adapter_probe"]["absolute_change"] <=
            protocol["screening_rule"]["residual_parameter_min_absolute_change"]):
        raise ValueError("Residual-expert training provenance differs")
    invariance_path = output / "parent-invariance.json"
    if not invariance_path.exists():
        write_json(invariance_path, parent_parameter_difference(
            parent / "decision.safetensors", target / "decision.safetensors"))
    invariance = read(invariance_path)
    if (not invariance["parent_parameters_exact"] or
            invariance["parent_max_absolute_difference"] !=
            protocol["screening_rule"]["parent_parameter_max_difference"]):
        raise ValueError("Residual training changed the frozen parent function")
    audit = output / "audit.json"
    if not audit.exists():
        command(root, output, "audit", [
            sys.executable, "-m", "decision_model.cli", "audit", "--checkpoint", str(target),
            "--data", str(output / "cases.jsonl"), "--output", str(audit), "--max-cases", "24",
            "--base-path", str(base), "--device", device])
    development = output / "development"
    if not development.exists():
        command(root, output, "development", [
            sys.executable, "scripts/evaluate_development_suite.py", "--checkpoint", str(target),
            "--data", str(root / protocol["data"]["consumed_development"]),
            "--output", str(development), "--base-path", str(base), "--device", device])
    qasc = output / "qasc"
    if not qasc.exists():
        command(root, output, "qasc", [
            sys.executable, "scripts/evaluate_development_suite.py", "--checkpoint", str(target),
            "--data", str(root / protocol["data"]["qasc_development"]),
            "--output", str(qasc), "--base-path", str(base), "--device", device])
    command(root, output, "reporting", [
        sys.executable, "scripts/qasc_residual_expert_screen_report.py"])
    write_json(output / "status.json", {"state": "complete"})
    print(json.dumps({"state": "complete"}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/qasc-residual-expert-screen-v1")
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    prepare(root, output) if args.prepare_only else run(root, output, args.device)


if __name__ == "__main__":
    main()
