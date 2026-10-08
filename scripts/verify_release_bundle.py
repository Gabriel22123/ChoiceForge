#!/usr/bin/env python3
"""Verify a Decision Model release bundle without loading model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORBIDDEN_PARTS = ("cases.jsonl", ".private.json", "blind-results", "cache", "models")
REQUIRED = {
    "checkpoint/decision.safetensors",
    "checkpoint/model.json",
    "checkpoint/calibration.json",
    "src/decision_model/schemas/prediction.schema.json",
    "src/decision_model/schemas/typed-prediction.schema.json",
    "data/release-candidate-v1/expanded.jsonl",
    "data/release-candidate-v1/manifest.json",
    "docs/RELEASE_QUICKSTART.md",
    "docs/evidence/release-candidate-v1.json",
}
COMPLETION_REQUIRED = {
    "docs/PROJECT_COMPLETION_AUDIT.zh-CN.md",
    "docs/evidence/project-completion-v1.json",
}


def validate_release_path(name):
    path = Path(name)
    if (path.is_absolute() or ".." in path.parts or
            any(part in FORBIDDEN_PARTS or part.endswith(".private.json")
                for part in path.parts)):
        raise ValueError("Forbidden path in release bundle: " + name)
    return name


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_bundle(bundle, require_completion=True):
    bundle = Path(bundle).resolve()
    manifest_path = bundle / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format_version") != 1 or manifest.get("release_endpoint") !=
            "seed1701-expanded" or manifest.get("base_model_included") is not False or
            manifest.get("canary_cases_included") is not False or
            manifest.get("github_published") is not False):
        raise ValueError("Release manifest identity differs")
    declared = manifest.get("files")
    required = REQUIRED | (COMPLETION_REQUIRED if require_completion else set())
    if not isinstance(declared, dict) or not required <= set(declared):
        raise ValueError("Release manifest misses required files")
    actual = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError("Release bundle contains a symlink")
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(bundle).as_posix())
    if actual != set(declared):
        raise ValueError("Release manifest file set differs from disk")
    for name, record in declared.items():
        validate_release_path(name)
        path = bundle / name
        if (set(record) != {"sha256", "bytes"} or record["bytes"] != path.stat().st_size or
                record["sha256"] != sha256(path)):
            raise ValueError("Release file integrity differs: " + name)
    weights = [name for name in actual if name.endswith(".safetensors")]
    if weights != ["checkpoint/decision.safetensors"]:
        raise ValueError("Release bundle must contain only adapter/head weights")
    evidence = json.loads((bundle / "docs/evidence/release-candidate-v1.json").read_text())
    if evidence.get("eligible") is not True or evidence.get("release_endpoint") != "seed1701-expanded":
        raise ValueError("Release evidence does not authorize the checkpoint")
    if require_completion:
        completion = json.loads((bundle / "docs/evidence/project-completion-v1.json").read_text())
        if completion.get("complete") is not True or not all(
                completion.get("requirements", {}).values()):
            raise ValueError("Project completion audit is absent or incomplete")
    return {"verified": True, "files": len(actual),
            "weights_sha256": declared["checkpoint/decision.safetensors"]["sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle, require_completion=True), sort_keys=True))


if __name__ == "__main__":
    main()
