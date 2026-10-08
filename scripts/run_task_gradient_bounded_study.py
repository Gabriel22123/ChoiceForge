"""Run three-seed post-projection norm-bound study."""
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
from task_balance_plan import make_orders
from task_gradient_bounded_projection import METHODS


SEEDS = (42, 43, 44)
START = "runs/relation-group-v1/dispersed"
PARENT = "runs/task-balance-v2"
RAW_PARENT = "runs/task-gradient-control-v1"
COMPOSITION_PARENT = "runs/task-gradient-composition-v1"
ARM_ORDER = tuple(f"seed{seed}-{method}" for seed in SEEDS for method in METHODS)


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes(root):
    paths = [*sorted((root / "src/decision_model").glob("*.py")),
             root / "scripts/task_balance_plan.py",
             root / "scripts/task_gradient_control.py",
             root / "scripts/task_gradient_composition.py",
             root / "scripts/task_gradient_bounded_projection.py",
             root / "scripts/train_task_gradient_control.py",
             root / "scripts/train_task_gradient_composition.py",
             root / "scripts/train_task_gradient_bounded.py",
             root / "scripts/evaluate_reference.py",
             Path(__file__).resolve()]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def prepare(root, output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    parent = root / PARENT
    raw_parent = root / RAW_PARENT
    composition_parent = root / COMPOSITION_PARENT
    parent_protocol = read(parent / "protocol.json")
    if read(parent / "status.json")["state"] != "complete":
        raise ValueError("Parent task-balance study is incomplete")
    raw_status = read(raw_parent / "status.json")
    raw_finals = read(raw_parent / "trained-checkpoints.json")
    expected_raw = [f"seed{seed}-raw" for seed in (42, 43)]
    if raw_status["state"] != "complete" or any(name not in raw_finals for name in expected_raw):
        raise ValueError("Reusable raw-control study is incomplete")
    for name in expected_raw:
        checkpoint = root / raw_finals[name]["path"]
        if digest((checkpoint / "decision.safetensors").read_bytes()) != raw_finals[name]["weights_sha256"]:
            raise ValueError("Reusable raw checkpoint changed: " + name)
    composition_status = read(composition_parent / "status.json")
    composition_finals = read(composition_parent / "trained-checkpoints.json")
    expected_composition = [f"seed{seed}-soft_cap_2x_then_project" for seed in (42, 43)]
    if (composition_status["state"] != "complete" or
            any(name not in composition_finals for name in expected_composition)):
        raise ValueError("Reusable composition study is incomplete")
    for name in expected_composition:
        checkpoint = root / composition_finals[name]["path"]
        if digest((checkpoint / "decision.safetensors").read_bytes()) != composition_finals[name]["weights_sha256"]:
            raise ValueError("Reusable composition checkpoint changed: " + name)
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
        if seed in (42, 43):
            config = read(composition_parent / f"seed{seed}.json")
        else:
            config = read(composition_parent / "seed42.json")
            config["seed"] = seed
            config["epoch_row_orders"] = make_orders(rows, "stratified", seed)
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
        "question": "Does reapplying the original two-times-median task-norm bound after conflict projection prevent projection-induced norm rebound while preserving the accuracy gains across three seeds?",
        "seeds": list(SEEDS), "methods": list(METHODS), "arms": list(ARM_ORDER),
        "initial_checkpoint": START,
        "initial_weight_sha256": digest((start / "decision.safetensors").read_bytes()),
        "initial_model_sha256": digest((start / "model.json").read_bytes()),
        "source_files": source_hashes(root),
        "dataset_sha256": digest((output / "cases.jsonl").read_bytes()),
        "diagnostic_dataset_sha256": digest((output / "diagnostic-cases.jsonl").read_bytes()),
        "initial_reference_sha256": digest((output / "initial.json").read_bytes()),
        "reused_controls": {**{name: raw_finals[name] for name in expected_raw},
                            **{name: composition_finals[name] for name in expected_composition}},
        "raw_parent_status_sha256": digest((raw_parent / "status.json").read_bytes()),
        "raw_parent_manifest_sha256": digest((raw_parent / "trained-checkpoints.json").read_bytes()),
        "composition_parent_status_sha256": digest((composition_parent / "status.json").read_bytes()),
        "composition_parent_manifest_sha256": digest((composition_parent / "trained-checkpoints.json").read_bytes()),
        "config_sha256": config_hashes,
        "selected_ids": parent_protocol["selected_ids"],
        "training_rows": parent_protocol["training_rows"],
        "exact_diagnostic_training_input_overlap": overlap,
        "budget": "Per arm: the same 1024 distinct rows, one exposure each, 128 updates and three task-separated forward/backward computations per update",
        "methods_detail": {
            "raw": "Mean gradient per task, weighted 3/8 PAWS + 3/8 SNLI + 2/8 replay, then global norm clip at 1; seeds42/43 reused, seed44 newly trained",
            "soft_cap_2x_then_project": "Apply the two-times-median cap, deterministic symmetric projection to negative-gradient pairs, row-share weighting and global clip; seeds42/43 reused, seed44 newly trained",
            "soft_cap_2x_project_recap": "Apply the same cap and projection, then cap every projected task direction again at the original pre-projection threshold without amplification before row-share weighting and global clip",
        },
        "projection_note": "Both treatments use the same PCGrad-inspired deterministic symmetric transform; the candidate adds only a non-amplifying post-projection bound and is not an implementation-equivalence claim for randomized PCGrad",
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
        "success_rule": "The bounded candidate advances to a fresh task blind evaluation only if no PAWS or SNLI accuracy is more than 1.0 point below raw in any of three seeds, its mean accuracy change across all six comparisons is nonnegative, combined-gradient p95 is below raw in all three seeds, and at seeds42/43 it does not exceed the unbounded composition p95; old diagnostics cannot rescue a primary-endpoint failure",
        "checkpoint_selection": "All nine endpoints are fixed before new training; four controls are reused bit-for-bit, five endpoints are newly trained, with no early stopping or endpoint-based method selection",
        "limitations": [
            "Three seeds still do not estimate the full population distribution",
            "BoolQ and WinoGrande are previously inspected diagnostics, not fresh blind tests",
            "Four seed42/43 controls are reused rather than retrained; validators require identical plans and first-step raw task statistics for each method within seed",
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
    raw_parent = root / RAW_PARENT
    if (digest((raw_parent / "status.json").read_bytes()) != protocol["raw_parent_status_sha256"] or
            digest((raw_parent / "trained-checkpoints.json").read_bytes()) !=
            protocol["raw_parent_manifest_sha256"]):
        raise ValueError("Reusable raw-control manifest changed")
    composition_parent = root / COMPOSITION_PARENT
    if (digest((composition_parent / "status.json").read_bytes()) != protocol["composition_parent_status_sha256"] or
            digest((composition_parent / "trained-checkpoints.json").read_bytes()) !=
            protocol["composition_parent_manifest_sha256"]):
        raise ValueError("Reusable composition manifest changed")
    for name, record in protocol["reused_controls"].items():
        checkpoint = root / record["path"]
        if (digest((checkpoint / "decision.safetensors").read_bytes()) != record["weights_sha256"] or
                digest((checkpoint / "model.json").read_bytes()) != record["model_sha256"]):
            raise ValueError("Reusable raw control changed: " + name)


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
        target = (root / protocol["reused_controls"][arm]["path"]
                  if arm in protocol["reused_controls"] else output / arm)
        if arm not in protocol["reused_controls"] and not target.exists():
            run_command(root, output, protocol, "training", arm, [
                sys.executable, "scripts/train_task_gradient_bounded.py",
                "--data", str(data), "--config", str(output / f"seed{seed}.json"),
                "--output", str(target), "--init-from", str(start),
                "--base-path", str(base), "--method", method, "--device", device,
            ])
        if not (target / "checksums.json").exists():
            raise ValueError("Incomplete arm requires explicit removal before resume: " + arm)
        record = read(target / "run.json")
        expected_version = ({"seed42-raw": "task-gradient-control-v1",
                             "seed43-raw": "task-gradient-control-v1",
                             "seed42-soft_cap_2x_then_project": "task-gradient-composition-v1",
                             "seed43-soft_cap_2x_then_project": "task-gradient-composition-v1"}.get(
                                 arm, "task-gradient-bounded-v1"))
        if (record["method"] != method or record["method_version"] != expected_version or
                record["seed"] != seed or record["updates"] != 128 or
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
    parser.add_argument("--output", default="runs/task-gradient-bounded-v1")
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
