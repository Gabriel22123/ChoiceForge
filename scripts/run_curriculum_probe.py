"""Training-fit-gated public warm-up, then the previously fixed full continuation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from decision_model.core import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/curriculum-probe-v1")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if output.exists():
        raise ValueError("Use a new study directory")
    old = root / "runs/semantic-scale-v1"
    warm = root / "runs/overfit-sanity-v1"
    if json.loads((warm / "status.json").read_text())["state"] != "complete":
        raise ValueError("Warm-up sanity check must finish first")
    warm_result = json.loads((warm / "result.json").read_text())
    if warm_result["updates"] != 80 or warm_result["final"]["metrics"]["correct"] < 23:
        raise ValueError("Predeclared training-only fit gate requires at least 23/24 after exactly 80 updates")
    config = json.loads((old / "seed-42.json").read_text())
    old_protocol = json.loads((old / "protocol.json").read_text())
    warm_protocol = json.loads((warm / "protocol.json").read_text())
    if not set(warm_protocol["selected_ids"]) <= set(old_protocol["selected_ids"]["train"]):
        raise ValueError("Warm-up introduced examples outside the fixed training selection")
    data = old / "cases.jsonl"
    if digest(data.read_bytes()) != old_protocol["dataset_sha256"]:
        raise ValueError("Fixed data changed")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    files = [*sorted((root / "src/decision_model").glob("*.py")), Path(__file__).resolve(), root / "scripts/evaluate_reference.py"]
    hashes = {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in files}
    protocol = {"purpose": "Does a short learnable warm-up improve the existing public training recipe?",
                "gate": "Only training-set fit, at least 23/24 after the predeclared fixed 80 updates; no validation/test gate",
                "warmup": warm_protocol, "warmup_weights_sha256": digest((warm / "decision.safetensors").read_bytes()),
                "dataset_sha256": old_protocol["dataset_sha256"], "config": config, "source_files": hashes,
                "continuation": "Exact same config/data as prior semantic-scale seed-42; dropout restored to .05",
                "budget": "80 balanced warm-up updates + 128 ordinary mixed-data updates; 480 + 1024 example exposures",
                "distinct_training_rows": 1024, "previous_control": "semantic-scale-v1/seed-42, 128 updates only",
                "evaluation": "Evaluate fixed 80-step warm-up, then unconditionally complete the fixed continuation and evaluate final weights; no test-based selection",
                "limitations": ["Single new seed", "Additional updates and curriculum change together; not matched-compute causal attribution",
                                "Previously inspected development evaluation, no fresh blind task",
                                "Training-set memorization is not generalization", "No proprietary RLCD reproduction"]}
    write_json(output / "protocol.json", protocol)

    def execute(stage, command):
        if any(digest((root / name).read_bytes()) != expected for name,expected in hashes.items()):
            raise ValueError("Frozen code changed")
        write_json(output / "status.json", {"state": stage})
        print(json.dumps({"stage": stage}), flush=True)
        try:
            with (output / (stage+".log")).open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            write_json(output / "status.json", {"state": "failed", "stage": stage})
            raise

    base = root / "models/eurobert-2.1b"
    execute("evaluate-warmup", [sys.executable, "scripts/evaluate_reference.py", "--checkpoint", str(warm),
            "--config", str(output / "config.json"), "--data", str(data), "--output", str(output / "warmup-evaluation.json"),
            "--base-path", str(base), "--device", args.device])
    execute("train", [sys.executable, "-m", "decision_model.cli", "train", "--data", str(data),
            "--config", str(output / "config.json"), "--output", str(output / "continued"),
            "--base-path", str(base), "--init-from", str(warm), "--device", args.device])
    run = json.loads((output / "continued/run.json").read_text())
    previous = json.loads((old / "seed-42/run.json").read_text())
    for key in ("selected_ids_sha256", "training_plan_sha256", "updates", "encoder_work"):
        if run[key] != previous[key]:
            raise ValueError("Continuation controls differ: " + key)
    if run["warm_start_weights_sha256"] != protocol["warmup_weights_sha256"]:
        raise ValueError("Wrong warm start")
    execute("audit", [sys.executable, "-m", "decision_model.cli", "audit", "--checkpoint", str(output / "continued"),
            "--data", str(data), "--output", str(output / "continued/audit.json"), "--max-cases", "25",
            "--base-path", str(base), "--device", args.device])
    write_json(output / "status.json", {"state": "complete"})


if __name__ == "__main__":
    main()
