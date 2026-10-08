"""Run the common-start, matched-compute task-gradient control study."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

from decision_model.core import digest, load_rows, write_json
from decision_model.evaluate import load_judge
from decision_model.train import evaluate_rows, predict_rows
from task_gradient_control import METHODS


SEEDS = (42, 43)
START = "runs/relation-group-v1/dispersed"
PARENT = "runs/task-balance-v2"
ARM_ORDER = tuple(f"seed{seed}-{method}" for seed in SEEDS for method in METHODS)


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes(root):
    paths = [*sorted((root / "src/decision_model").glob("*.py")),
             root / "scripts/task_balance_plan.py",
             root / "scripts/task_gradient_control.py",
             root / "scripts/train_task_gradient_control.py",
             root / "scripts/evaluate_reference.py",
             Path(__file__).resolve()]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def prepare(root, output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    parent = root / PARENT
    parent_protocol = read(parent / "protocol.json")
    if read(parent / "status.json")["state"] != "complete":
        raise ValueError("Parent task-balance study is incomplete")
    data = parent / "cases.jsonl"
    diagnostic = parent / "diagnostic-cases.jsonl"
    if (digest(data.read_bytes()) != parent_protocol["dataset_sha256"] or
            digest(diagnostic.read_bytes()) != parent_protocol["diagnostic_dataset_sha256"]):
        raise ValueError("Parent study datasets changed")
    rows = load_rows(data)
    diagnostic_rows = load_rows(diagnostic)
    start = root / START
    output.mkdir(parents=True)
    shutil.copyfile(data, output / "cases.jsonl")
    shutil.copyfile(diagnostic, output / "diagnostic-cases.jsonl")
    # Reuse the exact task-and-label-stratified row plans.  Only gradient
    # aggregation changes in this study.
    config_hashes = {}
    for seed in SEEDS:
        source = parent / f"seed{seed}-stratified.json"
        config = read(source)
        target = output / f"seed{seed}.json"
        write_json(target, config)
        config_hashes[str(seed)] = digest(target.read_bytes())
    initial = parent / "initial.json"
    initial_record = read(initial)
    if (initial_record["weight_sha256"] != digest((start / "decision.safetensors").read_bytes()) or
            initial_record["dataset_sha256"] != digest(data.read_bytes())):
        raise ValueError("Parent common-start reference does not match")
    shutil.copyfile(initial, output / "initial.json")

    training_inputs = {digest(row["request"]) for row in rows if row["split"] == "train"}
    overlap = sum(digest(row["request"]) in training_inputs for row in diagnostic_rows)
    if overlap:
        raise ValueError("Diagnostic input overlaps training")
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question": "At identical task-stratified rows, task-separated forwards, common starts and update budgets, can explicit gradient control reduce gradient tails and candidate-prior drift without sacrificing trained-task accuracy?",
        "seeds": list(SEEDS), "methods": list(METHODS), "arms": list(ARM_ORDER),
        "initial_checkpoint": START,
        "initial_weight_sha256": digest((start / "decision.safetensors").read_bytes()),
        "initial_model_sha256": digest((start / "model.json").read_bytes()),
        "source_files": source_hashes(root),
        "dataset_sha256": digest((output / "cases.jsonl").read_bytes()),
        "diagnostic_dataset_sha256": digest((output / "diagnostic-cases.jsonl").read_bytes()),
        "initial_reference_sha256": digest((output / "initial.json").read_bytes()),
        "config_sha256": config_hashes,
        "selected_ids": parent_protocol["selected_ids"],
        "training_rows": parent_protocol["training_rows"],
        "exact_diagnostic_training_input_overlap": overlap,
        "budget": "Per arm: the same 1024 distinct rows, one exposure each, 128 updates and three task-separated forward/backward computations per update",
        "methods_detail": {
            "raw": "Mean gradient per task, weighted 3/8 PAWS + 3/8 SNLI + 2/8 replay, then global norm clip at 1",
            "median_cap": "Within each update cap task norms above the three-task median; never amplify smaller gradients; then row-share weighting and global clip",
            "symmetric_project": "For each negative-gradient pair, apply deterministic symmetric pairwise projection computed from original gradients; then row-share weighting and global clip",
        },
        "projection_note": "PCGrad-inspired deterministic symmetric transform, not an implementation-equivalence claim for randomized PCGrad",
        "matched": [
            "same common starting adapter and head weights",
            "same exact rows, row blocks, stateless candidate permutations and task forward order within seed",
            "same task-separated forward/backward count for all methods",
            "same clean CE + 0.5 Brier risk and 3/8, 3/8, 2/8 task weights",
            "same AdamW settings, learning rates, global clipping threshold and 128 updates",
        ],
        "primary_endpoints": [
            "PAWS and SNLI raw accuracy/NLL",
            "maximum and p95 combined pre-clip gradient norm",
            "per-task raw gradient norm tails and global clipping rate",
        ],
        "diagnostic_endpoints": [
            "old-task replay sources in the fixed test set",
            "previously inspected BoolQ and WinoGrande accuracy and candidate prediction counts",
        ],
        "success_rule": "A control method must improve or preserve PAWS and SNLI across both seeds, materially reduce gradient tails, and avoid a consistently worse candidate-prior diagnostic; no single-seed winner claim",
        "checkpoint_selection": "All six final endpoints are fixed before training; no early stopping or endpoint-based method selection",
        "limitations": [
            "Two seeds do not estimate the full population distribution",
            "BoolQ and WinoGrande are previously inspected diagnostics, not fresh blind tests",
            "Task-separated batching changes padding work relative to the earlier task-balance study, so causal comparisons are within this study only",
            "Public-base pretraining contamination cannot be excluded",
        ],
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol-checksums.json", {
        path.name: digest(path.read_bytes()) for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    })
    write_json(output / "status.json", {"state": "prepared"})
    print(json.dumps({"prepared": str(output), "arms": list(ARM_ORDER)}), flush=True)


def verify(root, output, protocol):
    for name, expected in protocol["source_files"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Frozen source changed: " + name)
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen input changed: " + name)
    start = root / protocol["initial_checkpoint"]
    if (digest((start / "decision.safetensors").read_bytes()) != protocol["initial_weight_sha256"] or
            digest((start / "model.json").read_bytes()) != protocol["initial_model_sha256"]):
        raise ValueError("Common starting checkpoint changed")


def run_command(root, output, protocol, stage, arm, command):
    verify(root, output, protocol)
    write_json(output / "status.json", {"state": stage, "arm": arm})
    print(json.dumps({"stage": stage, "arm": arm}), flush=True)
    try:
        with (output / f"{arm}-{stage}.log").open("w") as stream:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        write_json(output / "status.json", {"state": "failed", "stage": stage, "arm": arm})
        raise


def diagnostic_evaluate(root, output, protocol, finals, device):
    verify(root, output, protocol)
    rows = load_rows(output / "diagnostic-cases.jsonl")
    base = root / "models/eurobert-2.1b"
    checkpoints = [(name, finals[name]) for name in protocol["arms"]]
    manifest = {
        "order": [name for name, _ in checkpoints],
        "checkpoints": dict(checkpoints),
        "dataset_sha256": protocol["diagnostic_dataset_sha256"],
        "role": "Previously inspected transfer and candidate-prior diagnostics; never training, calibration or checkpoint selection",
    }
    write_json(output / "diagnostic-manifest.json", manifest)
    results = output / "diagnostic-results"
    results.mkdir(exist_ok=True)
    for name, record in checkpoints:
        target = results / f"{name}.json"
        if target.exists():
            continue
        checkpoint = root / record["path"]
        if digest((checkpoint / "decision.safetensors").read_bytes()) != record["weights_sha256"]:
            raise ValueError("Diagnostic checkpoint changed: " + name)
        class Args:
            pass
        args = Args()
        args.checkpoint = str(checkpoint); args.base_path = str(base); args.device = device
        judge = load_judge(args)
        tokens = {row["id"]: judge.encode(row["request"]) for row in rows}
        logits = predict_rows(judge, rows, tokens)
        raw, predictions = evaluate_rows(rows, logits)
        write_json(target, {"name": name, "weights_sha256": record["weights_sha256"],
                            "model_sha256": record["model_sha256"],
                            "dataset_sha256": protocol["diagnostic_dataset_sha256"],
                            "raw": raw, "predictions": predictions})
    write_json(output / "status.json", {"state": "complete", "models": manifest["order"]})


def run(root, output, device):
    protocol = read(output / "protocol.json")
    state = read(output / "status.json")["state"]
    if state not in ("prepared", "failed", "training"):
        raise ValueError("Study cannot resume from the current state")
    base = root / "models/eurobert-2.1b"
    start = root / protocol["initial_checkpoint"]
    data = output / "cases.jsonl"
    finals = {}
    initial_hash = None
    for arm in protocol["arms"]:
        seed_text, method = arm.split("-", 1)
        seed = int(seed_text.removeprefix("seed"))
        target = output / arm
        if not target.exists():
            run_command(root, output, protocol, "training", arm, [
                sys.executable, "scripts/train_task_gradient_control.py",
                "--data", str(data), "--config", str(output / f"seed{seed}.json"),
                "--output", str(target), "--init-from", str(start),
                "--base-path", str(base), "--method", method, "--device", device,
            ])
        if not (target / "checksums.json").exists():
            raise ValueError("Incomplete arm requires explicit removal before resume: " + arm)
        record = read(target / "run.json")
        if (record["method"] != method or record["seed"] != seed or record["updates"] != 128 or
                record["selected_ids"] != protocol["selected_ids"]):
            raise ValueError("Arm protocol diverged: " + arm)
        if initial_hash is None:
            initial_hash = record["initial_trainable_sha256"]
        if record["initial_trainable_sha256"] != initial_hash:
            raise ValueError("Arms did not share identical starting trainable weights")
        for filename, expected in read(target / "checksums.json").items():
            if digest((target / filename).read_bytes()) != expected:
                raise ValueError("Final checkpoint changed: " + arm)
        finals[arm] = {
            "path": str(target.relative_to(root)),
            "weights_sha256": digest((target / "decision.safetensors").read_bytes()),
            "model_sha256": digest((target / "model.json").read_bytes()),
        }
    write_json(output / "trained-checkpoints.json", finals)
    write_json(output / "status.json", {"state": "trained", "arms": protocol["arms"]})
    diagnostic_evaluate(root, output, protocol, finals, device)
    print(json.dumps({"state": "complete", "arms": protocol["arms"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/task-gradient-control-v1")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if not args.run_prepared:
        prepare(root, output)
    if not args.prepare_only:
        run(root, output, args.device)


if __name__ == "__main__":
    main()
