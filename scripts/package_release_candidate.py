#!/usr/bin/env python3
"""Build the local adapter bundle only after every frozen release gate passes."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from decision_model.core import digest, write_json


CANONICAL = "seed1701-expanded"
CHECKPOINT_FILES = (
    "decision.safetensors", "model.json", "calibration.json", "checksums.json",
    "run.json", "evaluation.json",
)
ROOT_FILES = (
    ".gitignore", "LICENSE", "README.md", "README.zh-CN.md", "MODEL_CARD.md", "THIRD_PARTY.md",
    "pyproject.toml", "MANIFEST.in", "sources.lock.json",
    "requirements-tested.txt", "requirements-local-lock.txt",
)
RELEASE_SCRIPT_FILES = (
    "scripts/prepare_release_candidate_data.py",
    "scripts/run_release_candidate_study.py",
    "scripts/prepare_release_blind_suite.py",
    "scripts/evaluate_release_endpoint.py",
    "scripts/release_evaluation.py",
    "scripts/release_candidate_report.py",
    "scripts/run_release_postlock.py",
    "scripts/package_release_candidate.py",
)
RELEASE_DATA_FILES = (
    "data/release-candidate-v1/control.jsonl",
    "data/release-candidate-v1/expanded.jsonl",
    "data/release-candidate-v1/manifest.json",
)


def read(path):
    return json.loads(Path(path).read_text())


def release_bundle_files(root, study):
    """Return an explicit allowlist; canary cases and private outputs cannot enter."""
    checkpoint = study / CANONICAL
    paths = [root / name for name in ROOT_FILES + RELEASE_DATA_FILES]
    paths += [checkpoint / name for name in CHECKPOINT_FILES]
    paths += sorted((root / "src/decision_model").glob("*.py"))
    paths += sorted((root / "src/decision_model").glob("*.json"))
    paths += sorted((root / "src/decision_model/schemas").glob("*.json"))
    paths += sorted((root / "scripts").glob("*.py"))
    paths += sorted((root / "configs").glob("*.json"))
    paths += [path for path in sorted((root / "docs").glob("**/*.md"))
              if path.name != "PROJECT_COMPLETION_AUDIT.zh-CN.md"]
    paths += [path for path in sorted((root / "docs/evidence").glob("*.json"))
              if path.name != "project-completion-v1.json"]
    paths += sorted((root / "examples").glob("*.json"))
    paths += sorted((root / "tests").glob("test_*.py"))
    paths += sorted((root / "third_party").glob("*"))
    paths += [root / ".github/workflows/tests.yml"]
    paths += [study / "locked-endpoints.json", study / "gate-translation-lock.json",
              study / "blind-suite/protocol.json"]
    resolved = [path.resolve() for path in paths]
    forbidden = ("cases.jsonl", ".private.json", "blind-results", "cache/")
    if any(any(token in path.as_posix() for token in forbidden) for path in resolved):
        raise ValueError("Release allowlist contains canary material")
    if len(set(resolved)) != len(resolved) or any(not path.is_file() for path in resolved):
        raise ValueError("Release allowlist contains duplicate or missing files")
    return resolved


def main():
    from verify_release_bundle import verify_bundle

    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/release-candidate-v1")
    parser.add_argument("--output", default="dist/decision-model-minicpm5-2b-release-candidate-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study, output = root / args.study, root / args.output
    evidence = read(root / "docs/evidence/release-candidate-v1.json")
    if (evidence.get("eligible") is not True or evidence.get("release_endpoint") != CANONICAL or
            evidence.get("replication_seed_substitution_allowed") is not False):
        raise ValueError("Frozen release gates did not authorize a bundle")
    locked = read(study / "locked-endpoints.json")
    if locked.get("canonical_release_endpoint") != CANONICAL:
        raise ValueError("Canonical release endpoint differs")
    endpoint = study / CANONICAL
    if digest((endpoint / "decision.safetensors").read_bytes()) != evidence.get(
            "release_weights_sha256", locked["endpoints"][CANONICAL]["weights_sha256"]):
        raise ValueError("Canonical release weights differ")
    if output.exists():
        raise ValueError("Use a fresh release bundle directory")
    files = release_bundle_files(root, study)
    output.mkdir(parents=True)
    manifest = {}
    for source in files:
        if source.is_relative_to(endpoint.resolve()):
            relative = Path("checkpoint") / source.relative_to(endpoint.resolve())
        else:
            relative = source.relative_to(root.resolve())
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[relative.as_posix()] = {"sha256": digest(target.read_bytes()),
                                         "bytes": target.stat().st_size}
    write_json(output / "release-manifest.json", {
        "format_version": 1, "name": output.name, "release_endpoint": CANONICAL,
        "base_model_included": False, "github_published": False,
        "canary_cases_included": False, "files": manifest,
    })
    verification = verify_bundle(output, require_completion=False)
    print(json.dumps({"output": str(output), "files": len(manifest),
                      "weights_sha256": locked["endpoints"][CANONICAL]["weights_sha256"],
                      "verified": verification["verified"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
