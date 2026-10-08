#!/usr/bin/env python3
"""Assemble the verified parent and merged adapter into a portable local bundle."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from decision_model.core import canonical, digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local-model-bundle-v1.json")
    parser.add_argument("--output", default="runs/choiceforge-local-bundle-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh local model bundle directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_assembly":
        raise ValueError("Local model bundle protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen local-bundle source changed: " + relative)
    sources = {
        "adapter.json": root / config["adapter"]["metadata"],
        "adapter.safetensors": root / config["adapter"]["weights"],
        "parent/model.json": root / config["parent"]["model"],
        "parent/decision.safetensors": root / config["parent"]["weights"],
        "parent/calibration.json": root / config["parent"]["calibration"],
    }
    expected = {**config["adapter"]["sha256"], **config["parent"]["sha256"]}
    if set(sources) != set(expected):
        raise ValueError("Local model bundle file plan differs")
    for relative, source in sources.items():
        if digest(source.read_bytes()) != expected[relative]:
            raise ValueError("Local model bundle source changed: " + relative)
    output.mkdir(parents=True)
    for relative, source in sources.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    files = {relative: digest((output / relative).read_bytes()) for relative in sources}
    manifest = {
        "format_version": 1, "format": "choiceforge-local-bundle-v1",
        "project": "ChoiceForge", "files": files,
        "public_base": config["public_base"],
        "contains_training_data": False, "requires_human_review": True,
        "protocol_sha256": digest(config_path.read_bytes()),
    }
    write_json(output / "bundle.json", manifest)
    if any(digest((output / relative).read_bytes()) != expected_sha
           for relative, expected_sha in files.items()):
        raise ValueError("Local model bundle copy verification failed")
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "portable local model bundle assembly",
        "bundle_manifest_sha256": digest((output / "bundle.json").read_bytes()),
        "files": files, "bytes": {
            relative: (output / relative).stat().st_size for relative in files},
        "contains_training_data": False, "passed": True,
    }
    write_json(root / config["evidence_path"], evidence)
    print(canonical({"event": "complete", "passed": True,
                     "bundle": str(output.relative_to(root)),
                     "bytes": sum(evidence["bytes"].values()),
                     "manifest_sha256": evidence["bundle_manifest_sha256"]}))


if __name__ == "__main__":
    main()
