"""Common-start mixed versus task-and-label-stratified continuation study."""
import argparse
from collections import Counter, defaultdict
import datetime
import json
from pathlib import Path
import subprocess
import sys

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.evaluate import load_judge
from decision_model.train import evaluate_rows, predict_rows
from task_balance_plan import make_orders, task_name, update_histogram


SEEDS = (42, 43)
MODES = ("mixed", "stratified")
START = "runs/relation-group-v1/dispersed"
PARENT = "runs/task-expansion-v1"


def read(path):
    return json.loads(path.read_text())


def source_hashes(root):
    paths = [*sorted((root / "src/decision_model").glob("*.py")),
             root / "scripts/task_balance_plan.py", root / "scripts/task_balance_gradient.py",
             root / "scripts/evaluate_reference.py", Path(__file__).resolve()]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def select_rows(root):
    parent = root / PARENT
    parent_protocol = read(parent / "protocol.json")
    parent_rows = load_rows(parent / "cases.jsonl")
    if digest((parent / "cases.jsonl").read_bytes()) != parent_protocol["study_dataset_sha256"]:
        raise ValueError("Parent task-expansion rows changed")
    relation_path = root / "data/public-relation-groups-v1/cases.jsonl"
    relation_rows = load_rows(relation_path)
    relation_manifest = read(root / "data/public-relation-groups-v1/manifest.json")
    if digest(relation_path.read_bytes()) != relation_manifest["dataset_sha256"]:
        raise ValueError("Relation rows changed")

    paws = [row for row in parent_rows if row["split"] == "train" and row["source"] == "paws-wiki"]
    selected_paws = []
    for label in ("different", "paraphrase"):
        pool = sorted((row for row in paws if row["label"] == label), key=lambda row: row["id"])
        if len(pool) != 384:
            raise ValueError("Unexpected parent PAWS label count")
        selected_paws.extend(pool[:192])

    groups = defaultdict(list)
    for row in relation_rows:
        if row["split"] == "train" and row["source"] == "snli":
            groups[row["group_id"]].append(row)
    eligible = [key for key, values in groups.items()
                if Counter(row["label"] for row in values) == Counter(
                    {"contradiction": 1, "neutral": 1, "entailment": 1})]
    selected_groups = sorted(eligible, key=digest)[:128]
    selected_snli = [row for key in selected_groups for row in groups[key]]
    replay = [row for row in relation_rows if row["split"] == "train" and row["source"] != "snli"]
    if len(selected_snli) != 384 or len(replay) != 256:
        raise ValueError("Unexpected SNLI or replay count")
    evaluation = [row for row in parent_rows if row["split"] != "train"]
    rows = sorted(selected_paws + selected_snli + replay + evaluation, key=lambda row: row["id"])
    return rows, relation_path, selected_groups


def prepare(root, output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    from transformers import AutoTokenizer
    from decision_model.model import Judge

    rows, relation_path, selected_groups = select_rows(root)
    parent = root / PARENT; parent_protocol = read(parent / "protocol.json")
    blind = parent / "blind-cases.jsonl"
    if digest(blind.read_bytes()) != parent_protocol["blind_dataset_sha256"]:
        raise ValueError("Diagnostic task rows changed")
    blind_rows = load_rows(blind)
    training_inputs = {digest(row["request"]) for row in rows if row["split"] == "train"}
    overlap = sum(digest(row["request"]) in training_inputs for row in blind_rows)
    if overlap:
        raise ValueError("Diagnostic input overlaps training")

    start = root / START
    start_weights = digest((start / "decision.safetensors").read_bytes())
    start_model = digest((start / "model.json").read_bytes())
    base_config = read(start / "model.json")
    base_config.pop("epoch_row_orders", None)
    base_config.update(epochs=1, training_views=1, brier_weight=.5, consistency_weight=0.,
                       gradient_estimator="clean", max_train_rows=0, max_validation_rows=0,
                       max_test_rows=0, max_train_evaluation_rows=128, require_all_rows=True)
    judge = Judge.__new__(Judge); judge.config = base_config
    judge.tokenizer = AutoTokenizer.from_pretrained(str(root / "models/eurobert-2.1b"),
                                                    local_files_only=True, trust_remote_code=False)
    for row in rows:
        judge.encode(row["request"])

    output.mkdir(parents=True)
    data = output / "cases.jsonl"; diagnostic = output / "diagnostic-cases.jsonl"
    data.write_text("".join(canonical(row) + "\n" for row in rows))
    diagnostic.write_bytes(blind.read_bytes())
    load_rows(data); load_rows(diagnostic)
    configs, plans = {}, {}
    for seed in SEEDS:
        for mode in MODES:
            arm = f"seed{seed}-{mode}"
            order = make_orders(rows, mode, seed)[0]
            config = dict(base_config, seed=seed, epoch_row_orders=[order])
            write_json(output / f"{arm}.json", config)
            configs[arm] = digest((output / f"{arm}.json").read_bytes())
            plans[arm] = update_histogram(rows, order)
    counts = {split: dict(Counter(row["source"] for row in rows if row["split"] == split))
              for split in ("train", "validation", "test")}
    selected_ids = {split: [row["id"] for row in rows if row["split"] == split]
                    for split in ("train", "validation", "test")}
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question": "At identical rows, common start, matched updates and position-level randomness, does task-and-label stratification reduce cross-task interference and candidate-prior drift?",
        "seeds": list(SEEDS), "modes": list(MODES),
        "arms": [f"seed{seed}-{mode}" for seed in SEEDS for mode in MODES],
        "initial_checkpoint": START, "initial_weight_sha256": start_weights,
        "initial_model_sha256": start_model, "source_files": source_hashes(root),
        "dataset_sha256": digest(data.read_bytes()),
        "diagnostic_dataset_sha256": digest(diagnostic.read_bytes()),
        "parent_task_expansion_dataset_sha256": parent_protocol["study_dataset_sha256"],
        "relation_dataset_sha256": digest(relation_path.read_bytes()),
        "selected_snli_group_ids_sha256": digest(selected_groups),
        "counts": counts, "selected_ids": selected_ids,
        "config_sha256": configs, "plan_histograms": plans,
        "exact_diagnostic_training_input_overlap": overlap,
        "training_rows": {"paws-wiki": 384, "snli": 384, "old-task replay": 256},
        "budget": "Per arm: 1024 distinct rows, one epoch, 128 optimizer updates, 1024 example exposures, one view",
        "mixed": "Same rows randomly permuted once; task and label counts fluctuate by update",
        "stratified": "Every update has 3 PAWS, 3 SNLI and 2 replay rows; SNLI is 1/1/1 by label; PAWS alternates 2/1 labels and is globally balanced",
        "matched": ["same common starting weights", "same rows and one exposure each", "same clean CE + 0.5 Brier objective",
                    "same optimizer, learning rates, seed and 128 updates"],
        "primary_endpoints": ["PAWS and SNLI raw accuracy/NLL", "mixed-versus-stratified difference within each seed"],
        "diagnostic_endpoints": ["BoolQ yes/no prediction counts and accuracy", "WinoGrande option prediction counts and accuracy",
                                 "pre-training task-gradient cosine at the common start"],
        "checkpoint_selection": "All four fixed final endpoints; no early stopping or winner selection; diagnostic inference only after all endpoints finish",
        "limitations": ["BoolQ and WinoGrande were already inspected and are diagnostics, not fresh blind tests",
                        "Two seeds do not estimate the population distribution",
                        "Task-dependent choice counts, token shapes and row order change RNG consumption, candidate permutations, padding work and dropout-mask assignment despite a shared seed",
                        "No explicit PCGrad or other gradient-surgery arm", "Public-base pretraining contamination cannot be excluded"],
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol-checksums.json", {
        path.name: digest(path.read_bytes()) for path in output.iterdir()
        if path.is_file() and path.name != "status.json"})
    write_json(output / "status.json", {"state": "prepared"})
    print(json.dumps({"prepared": str(output), "counts": counts,
                      "mixed_patterns": len(plans["seed42-mixed"]["task_count_patterns"]),
                      "stratified_patterns": len(plans["seed42-stratified"]["task_count_patterns"])}), flush=True)


def verify(root, output, protocol):
    for name, expected in protocol["source_files"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Frozen source changed: " + name)
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen input changed: " + name)
    start = root / protocol["initial_checkpoint"]
    if (digest((start / "decision.safetensors").read_bytes()) != protocol["initial_weight_sha256"] or
            digest((start / "model.json").read_bytes()) != protocol["initial_model_sha256"]):
        raise ValueError("Common starting checkpoint changed")


def execute(root, output, protocol, stage, arm, command):
    verify(root, output, protocol)
    write_json(output / "status.json", {"state": stage, "arm": arm})
    print(json.dumps({"stage": stage, "arm": arm}), flush=True)
    try:
        with (output / f"{arm}-{stage}.log").open("w") as stream:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        write_json(output / "status.json", {"state": "failed", "arm": arm, "stage": stage})
        raise


def run(root, output, device):
    protocol = read(output / "protocol.json")
    if read(output / "status.json")["state"] not in ("prepared", "failed"):
        raise ValueError("Study cannot resume from the current state")
    start = root / protocol["initial_checkpoint"]
    base = root / "models/eurobert-2.1b"; data = output / "cases.jsonl"
    probe = output / "gradient-probe.json"
    if not probe.exists():
        execute(root, output, protocol, "gradient-probe", "common-start",
                [sys.executable, "scripts/task_balance_gradient.py", "--checkpoint", str(start),
                 "--config", str(output / "seed42-stratified.json"), "--data", str(data),
                 "--output", str(probe), "--base-path", str(base), "--device", device])
    initial = output / "initial.json"
    if not initial.exists():
        execute(root, output, protocol, "reference", "common-start",
                [sys.executable, "scripts/evaluate_reference.py", "--checkpoint", str(start),
                 "--config", str(output / "seed42-stratified.json"), "--data", str(data),
                 "--output", str(initial), "--base-path", str(base), "--device", device])

    finals = {}; initial_hash = None
    for arm in protocol["arms"]:
        target = output / arm
        if not target.exists():
            execute(root, output, protocol, "train", arm,
                    [sys.executable, "-m", "decision_model.cli", "train", "--data", str(data),
                     "--config", str(output / f"{arm}.json"), "--output", str(target),
                     "--init-from", str(start), "--base-path", str(base), "--device", device])
        if not (target / "checksums.json").exists():
            raise ValueError("Incomplete training directory requires explicit removal before resume: " + arm)
        for name, expected in read(target / "checksums.json").items():
            if digest((target / name).read_bytes()) != expected:
                raise ValueError("Final checkpoint changed: " + arm)
        record = read(target / "run.json")
        if record["selected_ids"] != protocol["selected_ids"] or record["updates"] != 128:
            raise ValueError("Training selection or budget diverged: " + arm)
        if initial_hash is None:
            initial_hash = record["initial_trainable_sha256"]
        if record["initial_trainable_sha256"] != initial_hash:
            raise ValueError("Arms did not start from identical trainable weights")
        finals[arm] = {"path": str(target.relative_to(root)),
                       "weights_sha256": digest((target / "decision.safetensors").read_bytes()),
                       "model_sha256": digest((target / "model.json").read_bytes())}
    write_json(output / "trained-checkpoints.json", finals)
    write_json(output / "status.json", {"state": "trained", "arms": protocol["arms"]})
    diagnostic_evaluate(root, output, protocol, finals, device)


def diagnostic_evaluate(root, output, protocol, finals, device):
    verify(root, output, protocol)
    cases = load_rows(output / "diagnostic-cases.jsonl")
    base = root / "models/eurobert-2.1b"
    start = {"path": protocol["initial_checkpoint"], "weights_sha256": protocol["initial_weight_sha256"],
             "model_sha256": protocol["initial_model_sha256"]}
    checkpoints = [("common-start", start)] + [(name, finals[name]) for name in protocol["arms"]]
    manifest = {"order": [name for name, _ in checkpoints],
                "checkpoints": {name: value for name, value in checkpoints},
                "dataset_sha256": protocol["diagnostic_dataset_sha256"],
                "role": "Previously inspected held-out diagnostics; never training, calibration or checkpoint selection"}
    write_json(output / "diagnostic-manifest.json", manifest)
    results = output / "diagnostic-results"; results.mkdir(exist_ok=True)
    for name, record in checkpoints:
        target = results / f"{name}.json"
        if target.exists():
            continue
        checkpoint = root / record["path"]
        if digest((checkpoint / "decision.safetensors").read_bytes()) != record["weights_sha256"]:
            raise ValueError("Diagnostic checkpoint changed: " + name)
        class Args: pass
        args = Args(); args.checkpoint = str(checkpoint); args.base_path = str(base); args.device = device
        judge = load_judge(args)
        tokens = {row["id"]: judge.encode(row["request"]) for row in cases}
        logits = predict_rows(judge, cases, tokens)
        raw, predictions = evaluate_rows(cases, logits)
        write_json(target, {"name": name, "weights_sha256": record["weights_sha256"],
                            "model_sha256": record["model_sha256"],
                            "dataset_sha256": protocol["diagnostic_dataset_sha256"],
                            "raw": raw, "predictions": predictions})
    write_json(output / "status.json", {"state": "complete", "models": manifest["order"]})
    print(json.dumps({"state": "complete", "models": manifest["order"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/task-balance-v2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; output = root / args.output
    if not args.run_prepared:
        prepare(root, output)
    if not args.prepare_only:
        run(root, output, args.device)


if __name__ == "__main__":
    main()
