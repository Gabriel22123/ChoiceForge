#!/usr/bin/env python3
"""Freeze, train and lock the matched release-candidate endpoints."""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base


ARMS = ("control", "expanded")


def read(path):
    return json.loads(Path(path).read_text())


def model_config(protocol, seed, arm):
    matched, model = protocol["matched"], protocol["model"]
    return {
        "model_id": model["model_id"], "revision": model["revision"],
        "architecture": model["architecture"], "seed": seed,
        "lora_layers": model["lora_layers"], "lora_rank": model["lora_rank"],
        "head_width": model["head_width"], "dropout": matched["dropout"],
        "max_tokens": protocol["data"]["max_tokens"],
        "encoder_lr": matched["encoder_lr"], "head_lr": matched["head_lr"],
        "epochs": 3 if arm == "control" else 1,
        "batch_size": matched["batch_size"], "accumulation": matched["accumulation"],
        "brier_weight": matched["brier_weight"], "consistency_weight": 0.0,
        "candidate_encoding": model["candidate_encoding"], "candidate_chunk_size": 8,
        "training_views": 1, "gradient_estimator": "clean",
        "training_target_mode": matched["training_target_mode"],
        "require_all_rows": True, "max_train_evaluation_rows": 512,
    }


def source_hashes(root):
    paths = [*sorted((root / "src/decision_model").glob("*.py")),
             root / "scripts/prepare_release_candidate_data.py",
             root / "scripts/prepare_release_blind_suite.py",
             Path(__file__).resolve(),
             root / "configs/release-candidate-v1.json",
             root / "configs/release-blind-suite-v1.json",
             root / "data/release-candidate-v1/manifest.json"]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def prepare(root, output, device):
    if output.exists():
        raise ValueError("Use a fresh release study directory")
    frozen = read(root / "configs/release-candidate-v1.json")
    if frozen["status"] != "frozen_before_training" or frozen["seeds"] != [1701, 1702, 1703]:
        raise ValueError("Release protocol identity differs")
    data_manifest = read(root / "data/release-candidate-v1/manifest.json")
    if data_manifest["protocol_sha256"] != digest(
            (root / "configs/release-candidate-v1.json").read_bytes()):
        raise ValueError("Release data was built from another protocol")
    verify_local_decoder_base(root / "models/minicpm5-2b-base")
    control_data = root / "data/release-candidate-v1/control.jsonl"
    expanded_data = root / "data/release-candidate-v1/expanded.jsonl"
    if (digest(control_data.read_bytes()) != data_manifest["control"]["sha256"] or
            digest(expanded_data.read_bytes()) != data_manifest["expanded"]["sha256"]):
        raise ValueError("Release data changed")
    output.mkdir(parents=True)
    configs = {}
    for seed in frozen["seeds"]:
        for arm in ARMS:
            name = f"seed{seed}-{arm}"
            target = output / f"{name}.json"
            write_json(target, model_config(frozen, seed, arm))
            configs[name] = digest(target.read_bytes())
    protocol = {
        **frozen,
        "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "release_config_sha256": digest((root / "configs/release-candidate-v1.json").read_bytes()),
        "blind_source_sha256": digest((root / "configs/release-blind-suite-v1.json").read_bytes()),
        "data_manifest_sha256": digest((root / "data/release-candidate-v1/manifest.json").read_bytes()),
        "data_sha256": {"control": data_manifest["control"]["sha256"],
                        "expanded": data_manifest["expanded"]["sha256"]},
        "configs_sha256": configs,
        "source_files": source_hashes(root),
        "blind_boundary": ("No blind cases may be materialized until six endpoint weights and "
                           "six order/ID audits are checksummed in locked-endpoints.json."),
    }
    write_json(output / "protocol.json", protocol)
    for name, expected in protocol["source_files"].items():
        source = root / name
        if digest(source.read_bytes()) != expected:
            raise ValueError("Release source changed during preparation: " + name)
        target = output / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    write_json(output / "frozen-source-manifest.json", {"source_files": protocol["source_files"]})
    write_json(output / "protocol-checksums.json", {
        path.name: digest(path.read_bytes()) for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    })
    write_json(output / "status.json", {"state": "prepared", "blind_materialized": False})
    print(json.dumps({"state": "prepared", "endpoints": len(configs),
                      "protocol_sha256": digest((output / "protocol.json").read_bytes())}, sort_keys=True))


def verify(root, output, protocol):
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen release input changed: " + name)
    for name, expected in protocol["source_files"].items():
        if (digest((output / "frozen-source" / name).read_bytes()) != expected or
                digest((root / name).read_bytes()) != expected):
            raise ValueError("Frozen/live release source changed: " + name)
    for arm in ARMS:
        path = root / f"data/release-candidate-v1/{arm}.jsonl"
        if digest(path.read_bytes()) != protocol["data_sha256"][arm]:
            raise ValueError("Release dataset changed: " + arm)
    verify_local_decoder_base(root / "models/minicpm5-2b-base")


def run_command(root, output, stage, endpoint, command):
    write_json(output / "status.json", {"state": stage, "endpoint": endpoint,
                                         "blind_materialized": False})
    print(json.dumps({"stage": stage, "endpoint": endpoint}), flush=True)
    log = output / f"{endpoint}-{stage}.log"
    with log.open("w") as stream:
        try:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            write_json(output / "status.json", {"state": "failed", "stage": stage,
                                                 "endpoint": endpoint,
                                                 "blind_materialized": False})
            raise


def train_and_lock(root, output, device):
    protocol = read(output / "protocol.json")
    if device != protocol["device"]:
        raise ValueError("Release device differs from frozen protocol")
    verify(root, output, protocol)
    allowed = {"prepared", "training", "auditing", "failed"}
    if read(output / "status.json").get("state") not in allowed:
        raise ValueError("Release study cannot train from current state")
    endpoints = {}
    common_eval_ids = None
    for seed in protocol["seeds"]:
        seed_initial = set()
        for arm in ARMS:
            name = f"seed{seed}-{arm}"
            target = output / name
            data = root / f"data/release-candidate-v1/{arm}.jsonl"
            config = output / f"{name}.json"
            if not target.exists():
                run_command(root, output, "training", name, [
                    sys.executable, "-m", "decision_model.cli", "train",
                    "--data", str(data), "--config", str(config), "--output", str(target),
                    "--base-path", str(root / "models/minicpm5-2b-base"), "--device", device,
                ])
            if not (target / "checksums.json").is_file():
                raise ValueError("Incomplete endpoint requires explicit cleanup: " + name)
            run = read(target / "run.json")
            if (run["dataset_sha256"] != protocol["data_sha256"][arm] or
                    run["config_sha256"] != protocol["configs_sha256"][name] or
                    run["initialization"] != "public_pretrained_base_fresh_adapter_and_head" or
                    run["seed"] != seed or run["updates"] != protocol["matched"]["optimizer_updates"] or
                    run["over_token_budget_ids"]):
                raise ValueError("Release endpoint provenance differs: " + name)
            expected_train = protocol["data"][f"{arm}_train_rows"]
            if (len(run["selected_ids"]["train"]) != expected_train or
                    len(run["selected_ids"]["validation"]) != protocol["data"]["common_validation_rows"] or
                    len(run["selected_ids"]["test"]) != protocol["data"]["common_test_rows"]):
                raise ValueError("Release endpoint row count differs: " + name)
            evaluation_ids = (run["selected_ids"]["validation"], run["selected_ids"]["test"])
            common_eval_ids = evaluation_ids if common_eval_ids is None else common_eval_ids
            if evaluation_ids != common_eval_ids:
                raise ValueError("Release arms do not share evaluation rows")
            seed_initial.add(run["initial_trainable_sha256"])
            for filename, expected in read(target / "checksums.json").items():
                if digest((target / filename).read_bytes()) != expected:
                    raise ValueError("Release endpoint changed: " + name)
            endpoints[name] = {
                "seed": seed, "arm": arm,
                "path": str(target.relative_to(root)),
                "weights_sha256": digest((target / "decision.safetensors").read_bytes()),
                "model_sha256": digest((target / "model.json").read_bytes()),
                "run_sha256": digest((target / "run.json").read_bytes()),
                "evaluation_sha256": digest((target / "evaluation.json").read_bytes()),
                "calibration_sha256": digest((target / "calibration.json").read_bytes()),
                "checksums_sha256": digest((target / "checksums.json").read_bytes()),
                "training_plan_sha256": digest((target / "training-plan.jsonl").read_bytes()),
                "initial_trainable_sha256": run["initial_trainable_sha256"],
            }
        if len(seed_initial) != 1:
            raise ValueError("Matched release arms started from different trainable tensors")

    audits = output / "audits"; audits.mkdir(exist_ok=True)
    for name, record in endpoints.items():
        result = audits / f"{name}.json"
        arm = record["arm"]
        if not result.exists():
            run_command(root, output, "auditing", name, [
                sys.executable, "-m", "decision_model.cli", "audit",
                "--checkpoint", str(root / record["path"]),
                "--data", str(root / f"data/release-candidate-v1/{arm}.jsonl"),
                "--output", str(result), "--max-cases", "24",
                "--base-path", str(root / "models/minicpm5-2b-base"), "--device", device,
            ])
        audit = read(result)
        if (audit["cases"] != 24 or audit["stable_all_orders"] != 24 or
                audit["max_reload_probability_difference"] > 2e-6 or
                audit["max_id_rename_probability_difference"] > 2e-6):
            raise ValueError("Release endpoint output audit failed: " + name)
        record["audit_sha256"] = digest(result.read_bytes())
    manifest = {"state": "locked_before_blind", "protocol_sha256": digest(
                    (output / "protocol.json").read_bytes()),
                "canonical_release_endpoint": f"seed{protocol['canonical_release_seed']}-expanded",
                "endpoints": endpoints, "blind_materialized": False}
    write_json(output / "locked-endpoints.json", manifest)
    write_json(output / "status.json", {"state": "locked_before_blind",
                                         "endpoints": len(endpoints),
                                         "blind_materialized": False,
                                         "locked_endpoints_sha256": digest(
                                             (output / "locked-endpoints.json").read_bytes())})
    print(json.dumps({"state": "locked_before_blind", "endpoints": len(endpoints)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/release-candidate-v1")
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--train-and-lock", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if args.prepare_only:
        prepare(root, output, args.device)
    else:
        train_and_lock(root, output, args.device)


if __name__ == "__main__":
    main()
