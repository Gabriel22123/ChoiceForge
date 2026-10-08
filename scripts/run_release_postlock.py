#!/usr/bin/env python3
"""Seal the blind suite, evaluate every locked endpoint, and issue one verdict."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from decision_model.core import digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def run(root, study, stage, endpoint, command):
    write_json(study / "postlock-status.json", {"state": stage, "endpoint": endpoint})
    print(json.dumps({"stage": stage, "endpoint": endpoint}), flush=True)
    log = study / f"postlock-{stage}-{endpoint}.log"
    with log.open("w") as stream:
        subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/release-candidate-v1")
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    locked_path = study / "locked-endpoints.json"
    locked = read(locked_path)
    if locked.get("state") != "locked_before_blind" or len(locked.get("endpoints", {})) != 6:
        raise ValueError("Post-lock evaluation requires six locked endpoints")
    gate_lock = read(study / "gate-translation-lock.json")
    if (gate_lock.get("state") != "locked_before_any_endpoint_result" or
            gate_lock.get("first_endpoint_complete") is not False or
            digest((study / "gate-translation.json").read_bytes()) !=
            gate_lock.get("gate_translation_sha256")):
        raise ValueError("Gate translation was not locked before endpoint results")
    blind = study / "blind-suite"
    if not blind.exists():
        run(root, study, "seal-blind", "suite", [
            sys.executable, "scripts/prepare_release_blind_suite.py", "--seal-selection",
        ])
    protocol = read(blind / "protocol.json")
    if (protocol["endpoint_manifest_sha256"] != digest(locked_path.read_bytes()) or
            protocol["prohibited_uses"] != ["training", "calibration", "early stopping",
                                             "endpoint selection", "hyperparameter selection"]):
        raise ValueError("Sealed blind protocol differs")
    results = study / "blind-results"
    results.mkdir(exist_ok=True)
    for seed in (1701, 1702, 1703):
        for arm in ("control", "expanded"):
            name = f"seed{seed}-{arm}"
            target = results / name
            if not target.exists():
                run(root, study, "blind-evaluate", name, [
                    sys.executable, "scripts/evaluate_release_endpoint.py",
                    "--checkpoint", str(study / name), "--output", str(target),
                    "--device", args.device,
                ])
            if not (target / "checksums.json").is_file():
                raise ValueError("Incomplete blind result requires explicit cleanup: " + name)
    run(root, study, "report", "all", [
        sys.executable, "scripts/release_candidate_report.py",
    ])
    evidence = read(root / "docs/evidence/release-candidate-v1.json")
    run(root, study, "finalize-docs", "verdict", [
        sys.executable, "scripts/finalize_release_docs.py",
    ])
    run(root, study, "final-validation", "unit-tests", [
        sys.executable, "-m", "unittest", "discover", "-s", "tests",
    ])
    run(root, study, "final-validation", "release-data", [
        sys.executable, "scripts/audit_release_data.py",
    ])
    bundle = root / "dist/decision-model-minicpm5-2b-release-candidate-v1"
    if evidence["eligible"]:
        if not bundle.exists():
            run(root, study, "package", "seed1701-expanded", [
                sys.executable, "scripts/package_release_candidate.py",
            ])
    elif bundle.exists():
        raise ValueError("Ineligible study must not leave a release bundle")
    run(root, study, "completion-audit", "project", [
        sys.executable, "scripts/audit_project_completion.py",
    ])
    completion = read(root / "docs/evidence/project-completion-v1.json")
    if evidence["eligible"]:
        if completion["complete"] is not True:
            raise ValueError("Eligible model did not satisfy the original project completion audit")
        run(root, study, "finalize-package", "seed1701-expanded", [
            sys.executable, "scripts/finalize_release_bundle.py",
        ])
        run(root, study, "verify-package", "seed1701-expanded", [
            sys.executable, "scripts/verify_release_bundle.py", str(bundle),
        ])
    write_json(blind / "status.json", {
        "state": "evaluated_once", "official_results_read": True,
        "endpoints": 6, "release_eligible": evidence["eligible"],
        "evidence_sha256": digest((root / "docs/evidence/release-candidate-v1.json").read_bytes()),
    })
    write_json(study / "postlock-status.json", {
        "state": "complete", "endpoints": 6, "release_eligible": evidence["eligible"],
        "release_bundle_created": evidence["eligible"],
        "project_completion_audit": completion["complete"]})
    print(json.dumps({"state": "complete", "release_eligible": evidence["eligible"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
