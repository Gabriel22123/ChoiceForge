"""Frozen public semantic-training expansion, two seeds and identical evaluation."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import subprocess
import sys

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.train import balanced_subset

DATA_SHA = "88d716ce5c539d7d959d873bfc2960bf86c1427967deefa3875e88f94f355bf1"
INITIAL_SHA = "8d5475f12df673591ce10a189058e26ae805ce78425a0b3f7df2fc08ceadf10e"


def file_hashes(root):
    paths = list((root / "src/decision_model").glob("*.py"))
    paths += [root / "scripts/run_semantic_scale_study.py", root / "scripts/evaluate_reference.py"]
    return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in sorted(paths)}


def prepare(root, output):
    from transformers import AutoTokenizer
    from decision_model.model import Judge
    data = root / "data/public-multitask-v2/cases.jsonl"
    if digest(data.read_bytes()) != DATA_SHA:
        raise ValueError("Pinned public mixture changed")
    initial = root / "runs/architecture-study-v1/D-independent"
    if digest((initial / "decision.safetensors").read_bytes()) != INITIAL_SHA:
        raise ValueError("Pinned starting checkpoint changed")
    previous = json.loads((initial / "run.json").read_text())
    config = json.loads((root / "configs/eurobert-independent-pilot.json").read_text())
    config.update(epochs=1, training_views=1, brier_weight=.5, consistency_weight=0.,
                  gradient_estimator="clean", max_train_rows=0, max_validation_rows=0,
                  max_test_rows=0, max_train_evaluation_rows=128, require_all_rows=True)
    # Tokenizer-only admission check uses the exact production encoder method.
    judge = Judge.__new__(Judge)
    judge.config = config
    judge.tokenizer = AutoTokenizer.from_pretrained(str(root / "models/eurobert-2.1b"),
                                                    local_files_only=True, trust_remote_code=False)
    rows = load_rows(data)
    by_id = {r["id"]: r for r in rows}
    old_ids = set(previous["selected_ids"]["train"])
    selected = []
    for source, quota in {"clinc": 128, "json-schema": 96, "bandit": 32}.items():
        pool = [r for r in rows if r["id"] in old_ids and r["source"] == source]
        if len(pool) < quota:
            raise ValueError("Not enough replay rows")
        selected += balanced_subset(pool, quota)
    snli = [r for r in rows if r["source"] == "snli"]
    for split, quota in {"train": 768, "validation": 96, "test": 192}.items():
        pool = [r for r in snli if r["split"] == split]
        selected += balanced_subset(pool, quota)
    for split in ("validation", "test"):
        selected += [by_id[key] for key in previous["selected_ids"][split]]
    selected.sort(key=lambda row: row["id"])
    for row in selected:
        judge.encode(row["request"])
    output.mkdir(parents=True)
    study_data = output / "cases.jsonl"
    study_data.write_text("".join(canonical(r) + "\n" for r in selected))
    load_rows(study_data)  # Global group/input/split leakage checks.
    prior_gradient = json.loads((root / "runs/gradient-training-v1/A-clean/run.json").read_text())
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parent_dataset_sha256": DATA_SHA, "dataset_sha256": digest(study_data.read_bytes()),
        "initial_checkpoint": "runs/architecture-study-v1/D-independent",
        "initial_weight_sha256": INITIAL_SHA,
        "source_files": file_hashes(root), "seeds": [42, 43], "base_config": config,
        "selection": "Fixed source quotas; balanced deterministic ID order; no selection from predictions or errors",
        "counts": {split: dict(Counter(r["source"] for r in selected if r["split"] == split))
                   for split in ("train", "validation", "test")},
        "selected_ids": {split: [r["id"] for r in selected if r["split"] == split]
                         for split in ("train", "validation", "test")},
        "prior_gradient_test_overlap": len(set(prior_gradient["selected_ids"]["test"]) &
                                           {r["id"] for r in selected if r["split"] == "test"}),
        "question": "Does a larger public SNLI curriculum learn entailment while preserving replay tasks?",
        "method_choice": "Clean CE + .5 Brier is a simple fixed reference, not a winner selected from previous test scores",
        "budget": "1024 distinct training rows, one epoch, 128 updates, one view, per seed; new optimizer",
        "randomness": "Identical starting weights and data; seed changes sample order, candidate permutations and dropout",
        "primary_endpoint": "Raw SNLI test accuracy and NLL before/after; report both seeds without selecting the best",
        "secondary_endpoints": "Per-source old-task retention, held-out GoEmotions transfer, raw/calibrated probability quality",
        "checkpoint_selection": "Fixed final update only, no test-based early stopping or model selection",
        "limitations": ["Two training seeds are not population-level replication",
                        "Compared with the starting checkpoint, both semantic data and optimizer updates increase; not a matched-compute causal data-scaling estimate",
                        "SNLI is an in-distribution trained task, not zero-shot evidence",
                        "GoEmotions remains absent from training/calibration but was repeatedly inspected in development",
                        "No untouched external benchmark or proof of general zero-shot capability",
                        "Training metrics use a fixed 128-row probe; training itself consumes all 1024 rows"],
    }
    write_json(output / "protocol.json", protocol)
    for seed in protocol["seeds"]:
        write_json(output / f"seed-{seed}.json", dict(config, seed=seed))
    write_json(output / "protocol-checksums.json", {
        p.name: digest(p.read_bytes()) for p in output.iterdir() if p.suffix in (".json", ".jsonl")})
    write_json(output / "status.json", {"state": "prepared"})
    print(json.dumps(protocol["counts"]), flush=True)


def run(root, output, device):
    protocol = json.loads((output / "protocol.json").read_text())
    initial = root / protocol["initial_checkpoint"]
    data = output / "cases.jsonl"
    base = root / "models/eurobert-2.1b"
    if json.loads((output / "status.json").read_text())["state"] != "prepared":
        raise ValueError("Only a newly prepared study can run; partial runs require explicit recovery")

    def execute(stage, arm, command):
        if file_hashes(root) != protocol["source_files"]:
            raise ValueError("Frozen study source changed")
        if digest((initial / "decision.safetensors").read_bytes()) != INITIAL_SHA:
            raise ValueError("Starting checkpoint changed")
        for name, expected in json.loads((output / "protocol-checksums.json").read_text()).items():
            if digest((output / name).read_bytes()) != expected:
                raise ValueError("Frozen protocol/config/data changed: " + name)
        write_json(output / "status.json", {"state": stage, "arm": arm})
        print(json.dumps({"event": stage, "arm": arm}), flush=True)
        try:
            with (output / f"{arm}-{stage}.log").open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            write_json(output / "status.json", {"state": "failed", "arm": arm, "stage": stage})
            raise

    execute("evaluate", "initial", [sys.executable, "scripts/evaluate_reference.py", "--checkpoint", str(initial),
            "--config", str(output / "seed-42.json"), "--data", str(data), "--output", str(output / "initial.json"),
            "--base-path", str(base), "--device", device])
    initial_eval = json.loads((output / "initial.json").read_text())
    initial_parameters = None
    controls = {}
    for seed in protocol["seeds"]:
        arm = f"seed-{seed}"
        execute("train", arm, [sys.executable, "-m", "decision_model.cli", "train", "--data", str(data),
                "--config", str(output / (arm + ".json")), "--output", str(output / arm),
                "--base-path", str(base), "--init-from", str(initial), "--device", device])
        record = json.loads((output / arm / "run.json").read_text())
        if record["selected_ids"] != protocol["selected_ids"] or record["updates"] != 128:
            raise ValueError("Training selection/budget diverged from protocol")
        if any(record["selected_ids"][s] != initial_eval["selected_ids"][s] for s in ("validation", "test")):
            raise ValueError("Starting and final evaluation rows differ")
        if initial_parameters is None:
            initial_parameters = record["initial_trainable_sha256"]
        if record["initial_trainable_sha256"] != initial_parameters:
            raise ValueError("Seeds started from different trainable weights")
        controls[arm] = {key: record[key] for key in
                         ("initial_trainable_sha256", "selected_ids_sha256", "updates", "training_plan_sha256", "encoder_work")}
        write_json(output / "controls.json", controls)
        execute("audit", arm, [sys.executable, "-m", "decision_model.cli", "audit", "--checkpoint", str(output / arm),
                "--data", str(data), "--output", str(output / arm / "audit.json"), "--max-cases", "25",
                "--base-path", str(base), "--device", device])
    write_json(output / "status.json", {"state": "complete", "arms": [f"seed-{s}" for s in protocol["seeds"]]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/semantic-scale-v1")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.run_prepared:
        parser.error("Choose only one preparation mode")
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if not args.run_prepared:
        if output.exists():
            raise ValueError("Use a new study directory")
        prepare(root, output)
    if not args.prepare_only:
        run(root, output, args.device)


if __name__ == "__main__":
    main()
