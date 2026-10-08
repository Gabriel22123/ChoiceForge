"""Frozen PAWS task expansion at two seeds, followed by BoolQ/Wino transfer evaluation."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.evaluate import load_judge
from decision_model.train import evaluate_rows, predict_rows
from task_expansion_data import (BOOLQ_HASH, PAWS_HASHES, boolq_case, boolq_selection_key,
                                 paws_case, select_balanced_paws, select_boolq, summarize_labels)


SEEDS = (42, 43)
STARTS = {
    42: "runs/relation-group-v1/dispersed",
    43: "runs/relation-group-seed43-v1/dispersed",
}
PAWS_QUOTAS = {"train": 384, "validation": 48, "test": 96}  # per label


def read(path):
    return json.loads(path.read_text())


def source_hashes(root):
    paths = [*sorted((root / "src/decision_model").glob("*.py")),
             root / "scripts/task_expansion_data.py", Path(__file__).resolve(),
             root / "scripts/evaluate_reference.py"]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def parquet_rows(path, expected):
    if digest(path.read_bytes()) != expected:
        raise ValueError("Public parquet checksum differs: " + path.name)
    return pq.read_table(path).to_pylist()


def prepare(root, output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    from transformers import AutoTokenizer
    from decision_model.model import Judge

    cache = root / "cache/public-task-expansion"
    paws = {
        split: parquet_rows(cache / f"paws_{split}.parquet", PAWS_HASHES[split])
        for split in ("train", "validation", "test")
    }
    boolq = parquet_rows(cache / "boolq_validation.parquet", BOOLQ_HASH)
    relation_data = root / "data/public-relation-groups-v1/cases.jsonl"
    relation_rows = load_rows(relation_data)
    replay = [row for row in relation_rows if row["split"] == "train" and row["source"] != "snli"]
    if len(replay) != 256:
        raise ValueError("Expected the fixed 256 old-task replay rows")

    selected_paws = {
        split: [paws_case(row, split, PAWS_HASHES[split])
                for row in select_balanced_paws(paws[split], PAWS_QUOTAS[split])]
        for split in ("train", "validation", "test")
    }
    study_rows = selected_paws["train"] + replay
    study_rows += selected_paws["validation"] + [row for row in relation_rows if row["split"] == "validation"]
    study_rows += selected_paws["test"] + [row for row in relation_rows if row["split"] == "test"]
    study_rows.sort(key=lambda row: row["id"])

    base_config = read(root / STARTS[42] / "model.json")
    base_config.pop("epoch_row_orders", None)
    base_config.update(epochs=1, training_views=1, brier_weight=.5, consistency_weight=0.,
                       gradient_estimator="clean", max_train_rows=0, max_validation_rows=0,
                       max_test_rows=0, max_train_evaluation_rows=128, require_all_rows=True)
    judge = Judge.__new__(Judge)
    judge.config = base_config
    judge.tokenizer = AutoTokenizer.from_pretrained(str(root / "models/eurobert-2.1b"),
                                                    local_files_only=True, trust_remote_code=False)
    for row in study_rows:
        judge.encode(row["request"])

    def boolq_admissible(row):
        try:
            judge.encode(boolq_case(row)["request"])
            return True
        except ValueError as exc:
            if "Input too long" not in str(exc):
                raise
            return False

    frozen_boolq_source = select_boolq(boolq, 384, boolq_admissible)
    frozen_boolq = [boolq_case(row) for row in frozen_boolq_source]
    frozen_wino = load_rows(root / "runs/winogrande-blind-v1/cases.jsonl")
    blind_rows = sorted(frozen_boolq + frozen_wino, key=lambda row: row["id"])
    training_inputs = {digest(row["request"]) for row in study_rows if row["split"] == "train"}
    overlap = sum(digest(row["request"]) in training_inputs for row in blind_rows)
    if overlap:
        raise ValueError("Blind request overlaps training input")

    output.mkdir(parents=True)
    study_path, blind_path = output / "cases.jsonl", output / "blind-cases.jsonl"
    study_path.write_text("".join(canonical(row) + "\n" for row in study_rows))
    blind_path.write_text("".join(canonical(row) + "\n" for row in blind_rows))
    load_rows(study_path); load_rows(blind_path)
    start_records, configs = {}, {}
    for seed in SEEDS:
        start = root / STARTS[seed]
        if not (start / "decision.safetensors").exists():
            raise ValueError("Missing starting checkpoint")
        config = dict(base_config, seed=seed)
        configs[seed] = config
        write_json(output / f"seed-{seed}.json", config)
        start_records[str(seed)] = {
            "path": STARTS[seed],
            "weights_sha256": digest((start / "decision.safetensors").read_bytes()),
            "model_sha256": digest((start / "model.json").read_bytes()),
        }
    counts = {split: dict(Counter(row["source"] for row in study_rows if row["split"] == split))
              for split in ("train", "validation", "test")}
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question": "Does adding a licensed adversarial paraphrase task improve transfer beyond relation training?",
        "seeds": list(SEEDS), "starts": start_records,
        "final_paths": {str(seed): f"runs/task-expansion-v1/seed-{seed}-paws" for seed in SEEDS},
        "source_files": source_hashes(root),
        "study_dataset_sha256": digest(study_path.read_bytes()),
        "blind_dataset_sha256": digest(blind_path.read_bytes()),
        "relation_dataset_sha256": digest(relation_data.read_bytes()),
        "public_files": {"paws": PAWS_HASHES, "boolq_validation": BOOLQ_HASH},
        "selection": {
            "paws": "Per official split, first fixed input hashes within each label; 768 train, 96 validation, 192 test",
            "boolq": "First 384 token-admissible SHA-256(question, passage); answer excluded",
            "boolq_selection_key_sha256": digest([boolq_selection_key(row) for row in frozen_boolq_source]),
            "boolq_labels_recorded_after_freeze": summarize_labels(frozen_boolq),
            "winogrande": "Byte-identical previously frozen 384-case set; never training data",
        },
        "counts": counts,
        "selected_ids": {split: [row["id"] for row in study_rows if row["split"] == split]
                         for split in ("train", "validation", "test")},
        "blind_counts": dict(Counter(row["source"] for row in blind_rows)),
        "exact_blind_training_input_overlap": overlap,
        "budget": "Per seed: 768 PAWS-Wiki + 256 old replay rows, one epoch, 128 updates, one view",
        "objective": "Clean CE + 0.5 Brier; fresh optimizer; seed-specific balanced relation checkpoint start",
        "primary_endpoints": ["Raw BoolQ accuracy/NLL", "Raw WinoGrande accuracy/NLL"],
        "secondary_endpoints": ["PAWS test accuracy/NLL", "SNLI and old-task retention", "probability bias"],
        "checkpoint_selection": "All two final endpoints fixed before blind inference; no early stopping or winner selection",
        "limitations": ["Only two seeds", "No matched-compute non-PAWS continuation control",
                        "WinoGrande was inspected in the preceding study; BoolQ is the new score-blind task family",
                        "Public pretrained base contamination cannot be excluded"],
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol-checksums.json", {
        path.name: digest(path.read_bytes()) for path in output.iterdir()
        if path.is_file() and path.name != "status.json"})
    write_json(output / "status.json", {"state": "prepared"})
    print(json.dumps({"prepared": str(output), "counts": counts,
                      "blind_counts": protocol["blind_counts"]}), flush=True)


def verify_frozen(root, output, protocol):
    for name, expected in protocol["source_files"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Frozen source changed: " + name)
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen protocol input changed: " + name)
    for seed in protocol["seeds"]:
        start = root / protocol["starts"][str(seed)]["path"]
        if digest((start / "decision.safetensors").read_bytes()) != protocol["starts"][str(seed)]["weights_sha256"]:
            raise ValueError("Starting weights changed")


def execute(root, output, protocol, stage, arm, command):
    verify_frozen(root, output, protocol)
    write_json(output / "status.json", {"state": stage, "arm": arm})
    print(json.dumps({"stage": stage, "arm": arm}), flush=True)
    try:
        with (output / f"{arm}-{stage}.log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        write_json(output / "status.json", {"state": "failed", "arm": arm, "stage": stage})
        raise


def train_all(root, output, device):
    protocol = read(output / "protocol.json")
    if read(output / "status.json")["state"] != "prepared":
        raise ValueError("Training requires a newly prepared study")
    data, base = output / "cases.jsonl", root / "models/eurobert-2.1b"
    for seed in protocol["seeds"]:
        arm = f"seed-{seed}"; start = root / protocol["starts"][str(seed)]["path"]
        config = output / f"seed-{seed}.json"
        execute(root, output, protocol, "reference", arm,
                [sys.executable, "scripts/evaluate_reference.py", "--checkpoint", str(start),
                 "--config", str(config), "--data", str(data),
                 "--output", str(output / f"reference-seed-{seed}.json"),
                 "--base-path", str(base), "--device", device])
        target = root / protocol["final_paths"][str(seed)]
        execute(root, output, protocol, "train", arm,
                [sys.executable, "-m", "decision_model.cli", "train", "--data", str(data),
                 "--config", str(config), "--output", str(target), "--init-from", str(start),
                 "--base-path", str(base), "--device", device])
        run = read(target / "run.json")
        if run["updates"] != 128 or run["selected_ids"] != protocol["selected_ids"]:
            raise ValueError("Training selection or budget diverged")
    finals = {
        str(seed): {
            "path": protocol["final_paths"][str(seed)],
            "weights_sha256": digest((root / protocol["final_paths"][str(seed)] / "decision.safetensors").read_bytes()),
            "model_sha256": digest((root / protocol["final_paths"][str(seed)] / "model.json").read_bytes()),
        } for seed in protocol["seeds"]
    }
    write_json(output / "trained-checkpoints.json", finals)
    write_json(output / "status.json", {"state": "trained", "arms": [f"seed-{s}" for s in protocol["seeds"]]})


def blind_evaluate(root, output, device):
    protocol = read(output / "protocol.json")
    if read(output / "status.json")["state"] != "trained":
        raise ValueError("Blind evaluation starts only after every final endpoint is complete")
    verify_frozen(root, output, protocol)
    finals = read(output / "trained-checkpoints.json")
    checkpoints = []
    for seed in protocol["seeds"]:
        checkpoints.append((f"seed{seed}-start", protocol["starts"][str(seed)]))
        checkpoints.append((f"seed{seed}-paws", finals[str(seed)]))
    manifest = {"order": [name for name, _ in checkpoints], "checkpoints": {name: value for name, value in checkpoints},
                "blind_dataset_sha256": protocol["blind_dataset_sha256"]}
    write_json(output / "blind-evaluation-manifest.json", manifest)
    cases = load_rows(output / "blind-cases.jsonl")
    results = output / "blind-results"; results.mkdir()
    base = root / "models/eurobert-2.1b"
    for name, record in checkpoints:
        checkpoint = root / record["path"]
        if digest((checkpoint / "decision.safetensors").read_bytes()) != record["weights_sha256"]:
            raise ValueError("Blind checkpoint changed: " + name)
        class Args: pass
        args = Args(); args.checkpoint = str(checkpoint); args.base_path = str(base); args.device = device
        judge = load_judge(args)
        tokens = {row["id"]: judge.encode(row["request"]) for row in cases}
        logits = predict_rows(judge, cases, tokens)
        raw, predictions = evaluate_rows(cases, logits)
        write_json(results / f"{name}.json", {
            "name": name, "weights_sha256": record["weights_sha256"],
            "model_sha256": record["model_sha256"], "dataset_sha256": protocol["blind_dataset_sha256"],
            "raw": raw, "predictions": predictions,
        })
    write_json(output / "status.json", {"state": "complete", "models": manifest["order"]})
    print(json.dumps({"state": "complete", "models": manifest["order"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/task-expansion-v1")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; output = root / args.output
    if not args.run_prepared:
        prepare(root, output)
    if not args.prepare_only:
        train_all(root, output, args.device)
        blind_evaluate(root, output, args.device)


if __name__ == "__main__":
    main()
