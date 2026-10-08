"""Matched warm-start model training: clean, pathwise and score-function updates."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from decision_model.core import digest, write_json

ARMS = {"A-clean": "clean", "B-pathwise": "pathwise", "C-score-function": "score_function"}
CONTROL_KEYS = ("initial_trainable_sha256", "warm_start_weights_sha256", "selected_ids_sha256",
                "training_plan_sha256", "updates", "forward_sequences", "forward_batches",
                "training_views", "encoder_work")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/gradient-training-v1")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if output.exists():
        raise ValueError("Use a new study directory")
    data = root / "data/public-multitask-v2/cases.jsonl"
    base = root / "models/eurobert-2.1b"
    initial = root / "runs/architecture-study-v1/D-independent"
    initial_weight = digest((initial / "decision.safetensors").read_bytes())
    config = json.loads((root / "configs/eurobert-independent-pilot.json").read_text())
    config.update(epochs=1, max_train_rows=192, max_validation_rows=96, max_test_rows=192,
                  training_views=1, brier_weight=.5, consistency_weight=0.,
                  noise_samples=16, noise_sigma=.4, noise_seed=20260921, center_score_logits=True)
    paths = list((root / "src/decision_model").glob("*.py")) + [Path(__file__).resolve(), root / "scripts/evaluate_reference.py"]

    def hashes():
        return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in sorted(paths)}

    frozen = hashes()
    output.mkdir(parents=True)
    protocol = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "dataset_sha256": digest(data.read_bytes()), "source_files": frozen,
                "initial_checkpoint": "runs/architecture-study-v1/D-independent", "initial_weight_sha256": initial_weight,
                "base_config": config, "variants": ARMS,
                "comparison": {"A_vs_B": "Effect of Gaussian smoothing at fixed proper scoring rule",
                               "B_vs_C": "Different gradient estimators of the same smoothed risk"},
                "matched": ["starting adapter/head weights", "new optimizer", "data IDs and candidate order plan",
                            "192 examples / 24 updates / one view", "encoder work", "hyperparameters",
                            "B/C independent CPU Gaussian noise plan"],
                "noise": "16 Gaussian logit actions per request; sigma .4; no repeated encoder sampling; project out common-logit direction",
                "score_function": "Loss-minimizing REINFORCE surrogate, independent leave-one-out cost baseline; no group standard-deviation normalization, PPO clipping or auxiliary clean loss",
                "inference": "Clean mean logits; optional validation-only temperature; no Monte Carlo inference",
                "limitations": ["single seed and short warm-start study, not convergence or fresh training",
                                "previously inspected sources remain development diagnostics; no universal zero-shot claim",
                                "logit-gradient estimator is unbiased before optimizer/clipping; AdamW and norm clipping alter actual updates",
                                "sampled risk differs from the score-function surrogate value",
                                "public RLCD-inspired experiment, not a reconstruction of any private training"],
                "checkpoint_selection": "fixed final update, no test-based selection"}
    write_json(output / "protocol.json", protocol)
    for arm, mode in ARMS.items():
        write_json(output / (arm + ".json"), dict(config, gradient_estimator=mode))
    write_json(output / "protocol-checksums.json", {p.name: digest(p.read_bytes()) for p in output.glob("*.json")})

    def execute(stage, arm, command):
        if hashes() != frozen or digest((initial / "decision.safetensors").read_bytes()) != initial_weight:
            raise ValueError("Frozen source or starting checkpoint changed")
        write_json(output / "status.json", {"state": stage, "arm": arm})
        print(json.dumps({"event": stage, "arm": arm}), flush=True)
        try:
            with (output / (arm + "-" + stage + ".log")).open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            write_json(output / "status.json", {"state": "failed", "arm": arm, "stage": stage})
            raise

    execute("evaluate", "initial", [sys.executable, "scripts/evaluate_reference.py", "--checkpoint", str(initial),
            "--config", str(output / "A-clean.json"), "--data", str(data), "--output", str(output / "initial.json"),
            "--base-path", str(base), "--device", args.device])
    starting = json.loads((output / "initial.json").read_text())
    reference = noise_reference = None
    for arm in ARMS:
        execute("train", arm, [sys.executable, "-m", "decision_model.cli", "train", "--data", str(data),
                "--config", str(output / (arm + ".json")), "--output", str(output / arm),
                "--base-path", str(base), "--init-from", str(initial), "--device", args.device])
        run = json.loads((output / arm / "run.json").read_text())
        controls = {key: run[key] for key in CONTROL_KEYS}
        if reference is None:
            reference = controls
        if controls != reference:
            raise ValueError("Matched-training controls differ")
        if any(run["selected_ids"][split] != starting["selected_ids"][split] for split in ("validation", "test")):
            raise ValueError("Starting checkpoint evaluation used different requests")
        noise = run["objective"]["noise_plan_sha256"]
        if noise:
            if noise_reference is None:
                noise_reference = noise
            elif noise_reference != noise:
                raise ValueError("B/C noise plans differ")
        write_json(output / "matched-controls.json", {"all_arms": reference, "noisy_arms_plan_sha256": noise_reference})
        execute("audit", arm, [sys.executable, "-m", "decision_model.cli", "audit", "--checkpoint", str(output / arm),
                "--data", str(data), "--output", str(output / arm / "audit.json"), "--max-cases", "25",
                "--base-path", str(base), "--device", args.device])
    write_json(output / "status.json", {"state": "complete", "arms": list(ARMS)})


if __name__ == "__main__":
    main()
