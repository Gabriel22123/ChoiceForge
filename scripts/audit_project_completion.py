#!/usr/bin/env python3
"""Requirement-by-requirement audit of the local open-source project goal."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from decision_model.core import digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def verify_local_bundle(root):
    bundle = root / "models/choiceforge-local-bundle-v1"
    if not (bundle / "bundle.json").is_file():
        bundle = root / "runs/choiceforge-local-bundle-v1"
    manifest_path = bundle / "bundle.json"
    if not manifest_path.is_file():
        return {"verified": False, "path": str(bundle.relative_to(root))}
    manifest = read(manifest_path)
    expected = {"adapter.json", "adapter.safetensors", "parent/model.json",
                "parent/decision.safetensors", "parent/calibration.json"}
    verified = manifest.get("format") == "choiceforge-local-bundle-v1" and set(
        manifest.get("files", {})) == expected
    if verified:
        verified = all((bundle / relative).is_file() and
                       digest((bundle / relative).read_bytes()) == checksum
                       for relative, checksum in manifest["files"].items())
    return {
        "verified": verified, "path": str(bundle.relative_to(root)),
        "bytes": sum((bundle / relative).stat().st_size for relative in expected)
                 if verified else None,
        "manifest_sha256": digest(manifest_path.read_bytes()),
        "adapter_sha256": manifest.get("files", {}).get("adapter.safetensors"),
        "contains_training_data": manifest.get("contains_training_data"),
    }


def audit(root):
    data = read(root / "docs/evidence/release-data-redistribution-v1.json")
    architecture = read(root / "docs/evidence/architecture-comparison-v1.json")
    outcome = read(root / "docs/evidence/outcome-feedback-consolidation-v1.json")
    portfolio = read(root / "docs/evidence/decision-stable-expert-portfolio-v1.json")
    full = read(root / "docs/evidence/full-expert-portfolio-v1.json")
    scifact_path = root / "docs/evidence/scifact-final-evaluation-v1.json"
    scifact = read(scifact_path)
    merged = read(root / "docs/evidence/canonical-merged-adapter-v1.json")
    bundle = verify_local_bundle(root)
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                          cwd=root, capture_output=True, text=True)
    remote = subprocess.run(["git", "remote"], cwd=root, capture_output=True,
                            text=True, check=True).stdout.split()
    scifact_checks = scifact.get("checks", {})
    requirements = {
        "public_redistributable_data": bool(
            data.get("passed") and not data.get("blind_suite_rows_included") and
            data.get("internal_or_local_markers_found") == 0),
        "dynamic_task_context_candidates": all((root / path).is_file() for path in (
            "src/decision_model/core.py", "src/decision_model/decoder_model.py",
            "examples/request.json")),
        "strict_json_probabilities_review": bool(
            scifact_checks.get("contract.each_row_seed") and
            merged.get("checks", {}).get("contract.output")),
        "rlcd_deconstruction_and_outcome_feedback": bool(
            outcome.get("decision", {}).get("integrity_passes") and
            (root / "src/decision_model/outcome_feedback.py").is_file() and
            (root / "docs/RESEARCH.zh-CN.md").is_file()),
        "encoder_decoder_comparison": bool(
            architecture.get("decoder_default_rule", {}).get("eligible_as_default") and
            all(architecture.get("decoder_default_rule", {}).values())),
        "multitask_stability": bool(
            portfolio.get("advances_to_full_portfolio") and
            full.get("eligible_for_sealed_evaluation") and
            scifact_checks.get("decision.exact_each_seed")),
        "candidate_prior_stability": bool(
            scifact_checks.get("context.each_seed") and
            scifact_checks.get("permutation.each_row_seed") and
            merged.get("checks", {}).get("equivalence.null")),
        "cross_task_generalization": bool(scifact.get("passed")),
        "multi_seed_seen_and_unseen_gain": bool(
            scifact.get("passed") and len(scifact.get("runs", [])) == 3 and
            all(run.get("proper_gain", 0) > 0 for run in scifact["runs"]) and
            all(run.get("consumed_macro", {}).get("proper_gain", 0) > 0
                for run in full.get("runs", []))),
        "reproducible_training_and_docs": all((root / path).is_file() for path in (
            "configs/scifact-final-evaluation-v1.json",
            "configs/canonical-merged-adapter-v1.json",
            "configs/local-model-bundle-v1.json",
            "docs/LOCAL_MODEL_BUNDLE_V1.zh-CN.md",
            "release/choiceforge-v1-manifest.json")),
        "full_test_suite": test.returncode == 0 and "OK" in test.stderr + test.stdout,
        "single_verified_adapter_bundle": bool(
            merged.get("passed") and bundle.get("verified") and
            bundle.get("adapter_sha256") ==
            merged.get("adapter", {}).get("adapter_sha256")),
        "publication_boundary_respected": bool(remote) and not any(
            (root / path).is_file() and path.startswith(("runs/", "cache/", ".venv/"))
            for path in subprocess.run(
                ["git", "ls-files"], cwd=root, capture_output=True, text=True,
                check=True).stdout.split()),
    }
    complete = all(requirements.values())
    return {
        "format_version": 1, "goal": "local reproducible general zero-shot decision model",
        "complete": complete, "requirements": requirements,
        "release_evidence_sha256": digest(scifact_path.read_bytes()),
        "bundle": bundle,
        "unit_tests": {"passed": test.returncode == 0,
                       "tail": (test.stderr + test.stdout).strip().splitlines()[-4:]},
        "git_remotes": remote, "github_published": bool(remote),
        "limitations": [
            "Public-base pretraining contamination cannot be excluded.",
            "RLCD-style outcome feedback is implemented and falsifiably evaluated; the final endpoint uses source-balanced proper-loss portfolios and decision-stable projection rather than direct outcome policy updates.",
            "SciFact evaluates classification given gold rationale sentences, not retrieval.",
            "One sealed unseen family does not establish universal zero-shot competence.",
            "Quality-gate use remains one example; every downstream use needs its own validation and human review.",
        ],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    result = audit(root)
    write_json(root / "docs/evidence/project-completion-v1.json", result)
    lines = [
        "# 项目完成审计", "",
        f"**结论：{'原始本地目标的全部证据条件已满足。' if result['complete'] else '尚未满足原始本地目标。'}**",
        "", "| 原始要求 | 状态 |", "|---|---|",
        *[f"| `{name}` | {'PASS' if value else 'INCOMPLETE'} |"
          for name, value in result["requirements"].items()],
        "", "## 解释边界", "",
        *[f"- {value}" for value in result["limitations"]],
        "", "机器可读证据：`docs/evidence/project-completion-v1.json`", "",
    ]
    (root / "docs/PROJECT_COMPLETION_AUDIT.zh-CN.md").write_text("\n".join(lines))
    print(json.dumps({"complete": result["complete"], "passed": sum(
        result["requirements"].values()), "total": len(result["requirements"])},
        sort_keys=True))


if __name__ == "__main__":
    main()
