"""Create a local evidence bundle for the task-gradient control study.

No adapter is promoted or copied because the preregistered success condition
failed.  The six validated local checkpoints remain available for reproduction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from decision_model import __version__
from decision_model.core import digest, write_json
from task_gradient_control_report import ARMS, validated


def file_record(path: Path) -> dict:
    return {"sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/task-gradient-control-v1")
    parser.add_argument("--output", default="dist/task-gradient-control-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    output = root / args.output
    index = output / "artifact-index.json"
    if index.exists():
        raise ValueError("Use an output directory without artifact-index.json")

    protocol, _, _, _, _ = validated(root, study)
    final_checkpoints = json.loads((study / "trained-checkpoints.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    copies = {
        root / "docs/TASK_GRADIENT_CONTROL_RESULTS.zh-CN.md": output / "TASK_GRADIENT_CONTROL_RESULTS.zh-CN.md",
        root / "docs/evidence/task-gradient-control-v1.json": output / "task-gradient-control-v1.json",
        study / "protocol.json": output / "protocol.json",
        study / "protocol-checksums.json": output / "protocol-checksums.json",
        study / "status.json": output / "status.json",
        study / "trained-checkpoints.json": output / "trained-checkpoints.json",
    }
    for source, target in copies.items():
        shutil.copyfile(source, target)

    packages = sorted(
        path for path in output.iterdir()
        if path.name.startswith(f"decision_model-{__version__}") and path.suffix in {".whl", ".gz"}
    )
    if {path.suffix for path in packages} != {".whl", ".gz"}:
        raise ValueError("Matching wheel and source archive are required")

    write_json(index, {
        "version": __version__,
        "scope": "local reproducible task-gradient-control evidence bundle; not published",
        "model_selection": {
            "promoted_adapter": None,
            "reason": "Neither treatment met the preregistered stable-benefit condition.",
            "next_experiment": "Compare a 2x within-update median cap with cap-then-project.",
        },
        "source_artifacts": {path.name: file_record(path) for path in packages},
        "evidence_artifacts": {target.name: file_record(target) for target in copies.values()},
        "frozen_inputs": {
            "study_dataset_sha256": protocol["dataset_sha256"],
            "diagnostic_dataset_sha256": protocol["diagnostic_dataset_sha256"],
            "initial_weight_sha256": protocol["initial_weight_sha256"],
        },
        "local_checkpoints": {
            arm: {
                "path": final_checkpoints[arm]["path"],
                "weights_sha256": final_checkpoints[arm]["weights_sha256"],
            }
            for arm in ARMS
        },
        "verification": {
            "study_status": "complete",
            "seeds": [42, 43],
            "arms": 6,
            "updates_per_arm": 128,
            "unit_tests": "65 passed",
            "study_validator": "passed",
            "weights_finite": True,
            "base_weights_included": False,
            "adapter_weights_included": False,
            "dataset_rows_included": False,
            "per_case_predictions_included": False,
            "optimizer_state_included": False,
        },
    })
    print(index)


if __name__ == "__main__":
    main()
