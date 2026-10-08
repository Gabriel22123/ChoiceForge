#!/usr/bin/env python3
"""Evaluate fixed interpolations between the retained parent and QASC endpoint."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base


def read(path):
    return json.loads(Path(path).read_text())


def blend_states(parent, endpoint, alpha):
    if not 0 < alpha < 1 or set(parent) != set(endpoint):
        raise ValueError("Interpolation needs matching states and 0 < alpha < 1")
    result = {}
    for name in parent:
        if parent[name].shape != endpoint[name].shape or parent[name].dtype != endpoint[name].dtype:
            raise ValueError("Interpolation tensor mismatch: " + name)
        result[name] = parent[name].mul(1.0 - alpha).add(endpoint[name], alpha=alpha)
    return result


def source_hashes(root):
    paths = [Path(__file__).resolve(),
             root / "scripts/qasc_retention_path_report.py",
             root / "configs/qasc-retention-path-v1.json"]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def verify_inputs(root, config):
    exact = {
        config["parent"]["checkpoint"] + "/decision.safetensors": config["parent"]["weights_sha256"],
        config["parent"]["checkpoint"] + "/model.json": config["parent"]["model_sha256"],
        config["qasc_endpoint"]["checkpoint"] + "/decision.safetensors": config["qasc_endpoint"]["weights_sha256"],
        config["qasc_endpoint"]["checkpoint"] + "/model.json": config["qasc_endpoint"]["model_sha256"],
        config["data"]["cases"]: config["data"]["cases_sha256"],
        config["data"]["qasc_development"]: config["data"]["qasc_development_sha256"],
        config["data"]["consumed_development"]: config["data"]["consumed_development_sha256"],
        config["baselines"]["qasc_screen_evidence"]: config["baselines"]["qasc_screen_evidence_sha256"],
        config["baselines"]["functional_retention_evidence"]: config["baselines"]["functional_retention_evidence_sha256"],
    }
    for name, expected in exact.items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Frozen interpolation input changed: " + name)
    verify_local_decoder_base(root / "models/minicpm5-2b-base")


def prepare(root, output):
    if output.exists():
        raise ValueError("Use a fresh retention-path directory")
    config_path = root / "configs/qasc-retention-path-v1.json"
    config = read(config_path)
    if config["status"] != "frozen_before_interpolation_or_evaluation":
        raise ValueError("Retention-path protocol was not frozen")
    if list(config["arms"].values()) != [0.25, 0.5, 0.75]:
        raise ValueError("Frozen interpolation grid differs")
    verify_inputs(root, config)
    from safetensors.torch import load_file, save_file
    parent_path = root / config["parent"]["checkpoint"]
    endpoint_path = root / config["qasc_endpoint"]["checkpoint"]
    parent = load_file(str(parent_path / "decision.safetensors"))
    endpoint = load_file(str(endpoint_path / "decision.safetensors"))
    rows = load_rows(root / config["data"]["cases"])
    test_ids = [row["id"] for row in rows if row["split"] == "test"]
    output.mkdir(parents=True)
    weight_hashes = {}
    for arm, alpha in config["arms"].items():
        target = output / arm
        target.mkdir()
        save_file(blend_states(parent, endpoint, alpha), str(target / "decision.safetensors"))
        shutil.copyfile(parent_path / "model.json", target / "model.json")
        write_json(target / "blend.json", {
            "alpha": alpha,
            "formula": "(1-alpha)*parent + alpha*qasc_endpoint",
            "parent_weights_sha256": config["parent"]["weights_sha256"],
            "qasc_endpoint_weights_sha256": config["qasc_endpoint"]["weights_sha256"],
        })
        write_json(target / "run.json", {
            "dataset_sha256": config["data"]["cases_sha256"],
            "selected_ids": {"test": test_ids},
            "checkpoint_selection": "fixed pre-evaluation interpolation grid",
        })
        weight_hashes[arm] = digest((target / "decision.safetensors").read_bytes())
    protocol = {**config, "config_sha256": digest(config_path.read_bytes()),
                "source_files": source_hashes(root), "weight_hashes": weight_hashes,
                "test_rows": len(test_ids)}
    write_json(output / "protocol.json", protocol)
    for name, expected in protocol["source_files"].items():
        source = root / name
        if digest(source.read_bytes()) != expected:
            raise ValueError("Retention-path source changed during preparation")
        frozen = output / "frozen-source" / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, frozen)
    write_json(output / "protocol-checksums.json", {
        path.name: digest(path.read_bytes()) for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    })
    write_json(output / "status.json", {"state": "prepared"})
    print(json.dumps({"state": "prepared", "weight_hashes": weight_hashes}, sort_keys=True))


def verify(root, output, protocol):
    verify_inputs(root, protocol)
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen retention-path protocol changed: " + name)
    for name, expected in protocol["source_files"].items():
        if (digest((root / name).read_bytes()) != expected or
                digest((output / "frozen-source" / name).read_bytes()) != expected):
            raise ValueError("Frozen retention-path source changed: " + name)
    for arm, expected in protocol["weight_hashes"].items():
        if digest((output / arm / "decision.safetensors").read_bytes()) != expected:
            raise ValueError("Interpolated checkpoint changed: " + arm)


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
            "prepared", "seen", "validation", "audit", "development", "qasc-context", "failed"}:
        raise ValueError("Retention path cannot resume from this state")
    base = root / "models/minicpm5-2b-base"
    cases = root / protocol["data"]["cases"]
    for arm in protocol["arms"]:
        checkpoint = output / arm
        seen = output / "seen" / f"{arm}.json"
        if not seen.exists():
            seen.parent.mkdir(exist_ok=True)
            command(root, output, protocol, arm, "seen", [
                sys.executable, "-m", "decision_model.cli", "evaluate",
                "--checkpoint", str(checkpoint), "--data", str(cases),
                "--output", str(seen), "--split", "test", "--max-cases", "0",
                "--base-path", str(base), "--device", device])
            shutil.copyfile(seen, checkpoint / "test-predictions-source.json")
            write_json(checkpoint / "test-predictions.json", read(seen)["predictions"])
        validation = output / "validation" / f"{arm}.json"
        if not validation.exists():
            validation.parent.mkdir(exist_ok=True)
            command(root, output, protocol, arm, "validation", [
                sys.executable, "-m", "decision_model.cli", "evaluate",
                "--checkpoint", str(checkpoint), "--data", str(cases),
                "--output", str(validation), "--split", "validation", "--max-cases", "0",
                "--base-path", str(base), "--device", device])
        audit = output / "audits" / f"{arm}.json"
        if not audit.exists():
            audit.parent.mkdir(exist_ok=True)
            command(root, output, protocol, arm, "audit", [
                sys.executable, "-m", "decision_model.cli", "audit",
                "--checkpoint", str(checkpoint), "--data", str(cases),
                "--output", str(audit), "--max-cases", "24",
                "--base-path", str(base), "--device", device])
        development = output / "development" / arm
        if not development.exists():
            development.parent.mkdir(exist_ok=True)
            command(root, output, protocol, arm, "development", [
                sys.executable, "scripts/evaluate_development_suite.py",
                "--checkpoint", str(checkpoint),
                "--data", str(root / protocol["data"]["consumed_development"]),
                "--output", str(development), "--base-path", str(base), "--device", device])
        qasc = output / "qasc-context" / arm
        if not qasc.exists():
            qasc.parent.mkdir(exist_ok=True)
            command(root, output, protocol, arm, "qasc-context", [
                sys.executable, "scripts/evaluate_development_suite.py",
                "--checkpoint", str(checkpoint),
                "--data", str(root / protocol["data"]["qasc_development"]),
                "--output", str(qasc), "--base-path", str(base), "--device", device])
    write_json(output / "status.json", {"state": "evaluated"})
    command(root, output, protocol, "all", "reporting", [
        sys.executable, "scripts/qasc_retention_path_report.py"])
    write_json(output / "status.json", {"state": "complete"})
    print(json.dumps({"state": "complete"}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/qasc-retention-path-v1")
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
