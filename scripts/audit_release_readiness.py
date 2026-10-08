"""Audit whether the selected decoder route already contains a release model.

The architecture study selected a route, not one seed-specific checkpoint.  This
script verifies the three decoder endpoints and records that distinction without
using any endpoint metric to choose a model after the fact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_model.core import digest, write_json


DECODER_ARMS = ("seed42-decoder", "seed43-decoder", "seed44-decoder")
CHECKPOINT_FILES = ("decision.safetensors", "model.json", "calibration.json")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(study: Path) -> None:
    status = read(study / "status.json")
    manifest_path = study / "result-manifest.json"
    if status.get("state") != "complete":
        raise ValueError("Architecture comparison is incomplete")
    if status.get("result_manifest_sha256") != digest(manifest_path.read_bytes()):
        raise ValueError("Architecture result manifest checksum differs")
    for relative, expected in read(manifest_path).items():
        relative_path = Path(relative)
        path = study / relative_path
        if relative_path.is_absolute() or ".." in relative_path.parts or digest(path.read_bytes()) != expected:
            raise ValueError("Architecture result changed: " + relative)


def verify_checkpoint(root: Path, arm: str, record: dict) -> dict:
    directory = root / record["path"]
    expected = {
        "decision.safetensors": record["weights_sha256"],
        "model.json": record["model_sha256"],
        "calibration.json": record["calibration_sha256"],
        "checksums.json": record["checksums_sha256"],
        "run.json": record["run_sha256"],
        "evaluation.json": record["evaluation_sha256"],
    }
    for name, checksum in expected.items():
        if digest((directory / name).read_bytes()) != checksum:
            raise ValueError(f"{arm} changed: {name}")
    local_checksums = read(directory / "checksums.json")
    for name in CHECKPOINT_FILES:
        if local_checksums.get(name) != digest((directory / name).read_bytes()):
            raise ValueError(f"{arm} loadable artifact checksum differs: {name}")

    config = read(directory / "model.json")
    run = read(directory / "run.json")
    evaluation = read(directory / "evaluation.json")
    calibration = read(directory / "calibration.json")
    if run.get("checkpoint_selection") != "fixed final epoch; test never used for selection":
        raise ValueError(f"{arm} used an unexpected checkpoint-selection rule")
    if run.get("seed") != config.get("seed") or arm != f"seed{config['seed']}-decoder":
        raise ValueError(f"{arm} seed identity differs")
    if config.get("architecture") != "causal_decoder" or config.get("candidate_encoding") != "independent":
        raise ValueError(f"{arm} is not the selected decoder route")

    raw = evaluation["test"]["raw"]
    return {
        "path": record["path"],
        "seed": config["seed"],
        "weights_sha256": record["weights_sha256"],
        "weights_bytes": (directory / "decision.safetensors").stat().st_size,
        "temperature": calibration["temperature"],
        "training_rows": len(run["selected_ids"]["train"]),
        "updates": run["updates"],
        "checkpoint_selection": run["checkpoint_selection"],
        "test_accuracy": raw["accuracy"],
        "test_nll": raw["nll"],
        "test_brier": raw["brier"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/architecture-comparison-v1")
    parser.add_argument("--output", default="docs/evidence/release-readiness-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    verify_manifest(study)
    protocol = read(study / "protocol.json")
    checkpoints = read(study / "trained-checkpoints.json")
    if protocol.get("checkpoint_selection") != "All six fixed final endpoints; no early stopping and no test-based model selection":
        raise ValueError("Architecture protocol selection rule differs")

    endpoints = {arm: verify_checkpoint(root, arm, checkpoints[arm]) for arm in DECODER_ARMS}
    shared = {
        key: {read(root / checkpoints[arm]["path"] / "model.json")[key] for arm in DECODER_ARMS}
        for key in ("model_id", "revision", "lora_layers", "lora_rank", "head_width", "candidate_encoding")
    }
    if any(len(values) != 1 for values in shared.values()):
        raise ValueError("Decoder endpoints do not share one architecture/configuration")

    required_repository_files = (
        "LICENSE", "THIRD_PARTY.md", "MODEL_CARD.md", "sources.lock.json",
        "src/decision_model/schemas/prediction.schema.json",
        "src/decision_model/schemas/typed-prediction.schema.json",
    )
    repository_checks = {name: (root / name).is_file() for name in required_repository_files}
    if not all(repository_checks.values()):
        raise ValueError("A required release/provenance file is missing")

    report = {
        "kind": "release_readiness_audit",
        "source_study": args.study,
        "source_protocol_sha256": digest((study / "protocol.json").read_bytes()),
        "route_decision": {
            "decoder_selected": True,
            "base": {key: next(iter(values)) for key, values in shared.items() if key in ("model_id", "revision")},
            "architecture": {key: next(iter(values)) for key, values in shared.items() if key not in ("model_id", "revision")},
        },
        "verified_endpoints": endpoints,
        "repository_checks": repository_checks,
        "decision": {
            "research_bundle_eligible": True,
            "single_release_checkpoint_selected": False,
            "final_model_release_ready": False,
            "reason": (
                "The frozen study selected the decoder route and retained all three fixed seed endpoints. "
                "It did not preregister a rule for selecting one endpoint as the published model."
            ),
        },
        "blockers": [
            "Freeze a final public-data training protocol and seed before training; do not select seed42/43/44 from inspected endpoint metrics.",
            "Train the final adapter on the declared public training corpus with no sealed-test checkpoint selection.",
            "Evaluate the immutable final adapter once on a fresh task-family blind suite; ARC, logical deduction, disambiguation, BoolQ and WinoGrande are already consumed diagnostics.",
            "Export one environment-independent adapter bundle with calibration, checksums, source/data provenance, install metadata and an exact reproduction command.",
        ],
        "packaging_lesson": (
            "Retain the useful weights-plus-temperature-plus-client packaging pattern, while requiring a reproducible public starting point, "
            "pinned dependencies and no missing predecessor checkpoint."
        ),
    }
    write_json(root / args.output, report)
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
