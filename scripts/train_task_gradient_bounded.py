"""Matched-compute trainer for bounded conflict projection.

This is a study trainer, separate from the release trainer so completed studies
whose source hashes are frozen remain independently verifiable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from decision_model.core import canonical, digest, load_rows, summarize_rows, write_json
from decision_model.model import Judge, verify_local_base
from decision_model.train import balanced_subset, calibrate, evaluate_rows, predict_rows
from decision_model.training_objective import TrainingObjective
try:
    from task_balance_plan import TASKS, task_name
    from task_gradient_bounded_projection import METHODS, combine
except ModuleNotFoundError:  # Allows unit tests to import this file as scripts.*.
    from scripts.task_balance_plan import TASKS, task_name
    from scripts.task_gradient_bounded_projection import METHODS, combine


ROWS_PER_UPDATE = {"paws": 3, "snli": 3, "replay": 2}
TASK_WEIGHTS = {name: count / 8 for name, count in ROWS_PER_UPDATE.items()}


def read(path):
    return json.loads(Path(path).read_text())


def trainable_hash(params):
    value = hashlib.sha256()
    for name, parameter in params:
        value.update(name.encode())
        value.update(parameter.detach().float().cpu().numpy().tobytes())
    return value.hexdigest()


def deterministic_view(row, seed):
    """Stateless candidate permutation shared by every method in one seed."""
    request = row["request"]
    choices = request["choices"].copy()
    # Keep the v1 key so candidate orders exactly match the reusable raw controls.
    permutation_seed = int(digest({"purpose": "gradient-control-choice-order-v1",
                                   "seed": seed, "row_id": row["id"]})[:16], 16)
    random.Random(permutation_seed).shuffle(choices)
    return dict(request, choices=choices)


def validate_protocol(config, rows, checkpoint):
    previous = read(Path(checkpoint) / "model.json")
    keys = ("model_id", "revision", "lora_layers", "lora_rank", "head_width",
            "candidate_encoding", "candidate_chunk_size")
    if any(config.get(key) != previous.get(key) for key in keys):
        raise ValueError("Warm-start architecture mismatch")
    if (config.get("epochs") != 1 or config.get("training_views") != 1 or
            config.get("gradient_estimator") != "clean" or
            config.get("consistency_weight") != 0 or config.get("brier_weight") != .5):
        raise ValueError("Study requires one-view clean CE + 0.5 Brier for one epoch")
    orders = config.get("epoch_row_orders")
    if not isinstance(orders, list) or len(orders) != 1:
        raise ValueError("Study requires one frozen row order")
    train = [row for row in rows if row["split"] == "train"]
    if Counter(orders[0]) != Counter(row["id"] for row in train) or len(orders[0]) != 1024:
        raise ValueError("Frozen order must expose each of the 1024 training rows once")
    by_id = {row["id"]: row for row in train}
    for offset in range(0, len(orders[0]), 8):
        block = [by_id[key] for key in orders[0][offset:offset + 8]]
        if Counter(task_name(row) for row in block) != Counter(ROWS_PER_UPDATE):
            raise ValueError("Every update must contain 3 PAWS, 3 SNLI and 2 replay rows")


def task_gradient(judge, objective, params, rows, encoded):
    sequences = [encoded[row["id"]] for row in rows]
    views = [deterministic_view(row, judge.config["seed"]) for row in rows]
    targets = torch.tensor([
        [choice["id"] for choice in view["choices"]].index(row["label"])
        for row, view in zip(rows, views)
    ], device=judge.device)
    logits = judge.logits(sequences)
    counts = [len(view["choices"]) for view in views]
    loss, risk = objective(logits, targets, counts)
    if not torch.isfinite(loss) or not torch.isfinite(risk):
        raise ValueError("Non-finite task loss")
    values = torch.autograd.grad(loss, [parameter for _, parameter in params], allow_unused=True)
    if any(value is None for value in values):
        missing = [name for (name, _), value in zip(params, values) if value is None]
        raise ValueError("Trainable parameter did not receive a task gradient: " + ", ".join(missing[:3]))
    return [value.detach() for value in values], float(risk.item()), float(loss.item()), views


def evaluate_and_save(judge, output, config, splits, tokens):
    validation_logits = predict_rows(judge, splits["validation"], tokens)
    calibration = calibrate(validation_logits, splits["validation"])
    write_json(output / "calibration.json", calibration)
    evaluation = {}
    cap = config.get("max_train_evaluation_rows", 0)
    evaluated = dict(splits)
    if cap:
        evaluated["train"] = balanced_subset(splits["train"], cap)
    for split in ("train", "validation", "test"):
        selected = evaluated[split]
        logits = validation_logits if split == "validation" else predict_rows(judge, selected, tokens)
        raw, _ = evaluate_rows(selected, logits)
        scaled, predictions = evaluate_rows(selected, logits, calibration["temperature"])
        evaluation[split] = {"raw": raw, "temperature_scaled": scaled}
        write_json(output / f"{split}-predictions.json", predictions)
    write_json(output / "evaluation.json", evaluation)
    return evaluation, {split: [row["id"] for row in evaluated[split]] for split in evaluated}


def train(args):
    if args.method not in METHODS:
        raise ValueError("Unknown gradient-control method")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new run directory")
    config = read(args.config)
    rows = load_rows(args.data)
    validate_protocol(config, rows, args.init_from)
    verify_local_base(args.base_path)
    judge = Judge(config, base_path=args.base_path, checkpoint=args.init_from, device=args.device)
    objective = TrainingObjective(config)

    tokens = {}
    for row in rows:
        tokens[row["id"]] = judge.encode(row["request"])
    train = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["id"])
    splits = {name: sorted((row for row in rows if row["split"] == name), key=lambda row: row["id"])
              for name in ("train", "validation", "test")}
    by_id = {row["id"]: row for row in train}
    order = config["epoch_row_orders"][0]
    views = {row["id"]: deterministic_view(row, config["seed"]) for row in train}
    encoded = {key: judge.encode(view) for key, view in views.items()}

    output.mkdir(parents=True)
    (output / "source").mkdir()
    source_paths = [*sorted((Path(__file__).resolve().parents[1] / "src/decision_model").glob("*.py")),
                    Path(__file__).resolve(), Path(__file__).with_name("task_gradient_bounded_projection.py"),
                    Path(__file__).with_name("task_gradient_composition.py"),
                    Path(__file__).with_name("task_gradient_control.py"),
                    Path(__file__).with_name("task_balance_plan.py")]
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
        target = output / "source" / relative.replace("/", "__")
        target.write_bytes(path.read_bytes())
        source_hashes[relative] = digest(path.read_bytes())

    params = list(judge.named_trainable())
    initial_hash = trainable_hash(params)
    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in params if name.startswith("encoder.")], "lr": config["encoder_lr"]},
        {"params": [p for name, p in params if name.startswith("head.")], "lr": config["head_lr"]},
    ], weight_decay=.01, eps=1e-6)
    torch.manual_seed(config["seed"] + 73001)
    max_updates = args.max_updates or 128
    if not 1 <= max_updates <= 128:
        raise ValueError("max-updates must be in 1..128")
    run = {
        "method": args.method,
        "method_version": "task-gradient-bounded-v1",
        "engineering_smoke": bool(args.max_updates),
        "dataset_sha256": digest(Path(args.data).read_bytes()),
        "config_sha256": digest(Path(args.config).read_bytes()),
        "warm_start_weights_sha256": digest((Path(args.init_from) / "decision.safetensors").read_bytes()),
        "initial_trainable_sha256": initial_hash,
        "seed": config["seed"], "device": str(judge.device),
        "selected": summarize_rows(rows),
        "selected_ids": {name: [row["id"] for row in splits[name]] for name in splits},
        "trainable_parameters": sum(parameter.numel() for _, parameter in params),
        "base_parameters": sum(parameter.numel() for parameter in judge.encoder.parameters()),
        "task_rows_per_update": ROWS_PER_UPDATE,
        "task_weights": TASK_WEIGHTS,
        "budget": {"planned_updates": 128, "executed_updates": max_updates,
                   "examples_per_update": 8, "task_forwards_per_update": 3},
        "candidate_permutation": "stateless SHA256-derived row permutation; identical across methods within seed",
        "dropout_seed": config["seed"] + 73001,
        "source_files": source_hashes,
    }
    write_json(output / "run.json", run)
    write_json(output / "model.json", config)

    plan_hash = hashlib.sha256()
    step_records = []
    start = time.monotonic()
    probe_name, probe = next((name, parameter) for name, parameter in params if "lora_B" in name)
    initial_probe = probe.detach().cpu().clone()
    print(canonical({"event": "start", "method": args.method, "seed": config["seed"],
                     "initial_trainable_sha256": initial_hash, "updates": max_updates}), flush=True)
    judge.train(True)
    for update in range(max_updates):
        block = [by_id[key] for key in order[update * 8:update * 8 + 8]]
        grouped = {name: [row for row in block if task_name(row) == name] for name in TASKS}
        optimizer.zero_grad(set_to_none=True)
        gradients, risks, losses, task_views = {}, {}, {}, {}
        for name in TASKS:
            gradients[name], risks[name], losses[name], task_views[name] = task_gradient(
                judge, objective, params, grouped[name], encoded)
        combined, diagnostic = combine(gradients, TASK_WEIGHTS, args.method)
        for (_, parameter), gradient in zip(params, combined):
            parameter.grad = gradient
        pre_clip = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in params], 1.0, error_if_nonfinite=True).item()
        optimizer.step()
        risk = sum(TASK_WEIGHTS[name] * risks[name] for name in TASKS)
        surrogate = sum(TASK_WEIGHTS[name] * losses[name] for name in TASKS)
        plan = {
            "update": update + 1,
            "tasks": {name: [row["id"] for row in grouped[name]] for name in TASKS},
            "choices": {name: [[choice["id"] for choice in view["choices"]]
                               for view in task_views[name]] for name in TASKS},
        }
        line = canonical(plan) + "\n"
        plan_hash.update(line.encode())
        with (output / "training-plan.jsonl").open("a") as stream:
            stream.write(line)
        record = {
            "update": update + 1, "method": args.method,
            "task_risk": risks, "task_surrogate": losses,
            "weighted_risk": risk, "weighted_surrogate": surrogate,
            "raw_task_norms": diagnostic["before"]["norms"],
            "raw_task_cosines": diagnostic["before"]["cosines"],
            "transformed_task_norms": diagnostic["after"]["norms"],
            "transformed_task_cosines": diagnostic["after"]["cosines"],
            "transformation": diagnostic["transformation"],
            "combined_pre_clip_norm": pre_clip,
            "combined_diagnostic_norm": diagnostic["combined_norm"],
            "clipped": pre_clip > 1.0,
        }
        step_records.append(record)
        with (output / "optimizer-steps.jsonl").open("a") as stream:
            stream.write(canonical(record) + "\n")
        if update == 0 or (update + 1) % 8 == 0:
            print(canonical({"event": "train", "method": args.method, "update": update + 1,
                             "risk": risk, "grad_norm": pre_clip,
                             "task_norms": diagnostic["before"]["norms"],
                             "seconds": time.monotonic() - start}), flush=True)
        del gradients, combined

    judge.save(output)
    run.update({
        "updates": max_updates,
        "training_seconds": time.monotonic() - start,
        "training_plan_sha256": plan_hash.hexdigest(),
        "objective": objective.record(),
        "gradient_clipping": {
            "max_norm": 1.0,
            "updates": max_updates,
            "clipped_updates": sum(record["clipped"] for record in step_records),
            "mean_pre_clip_norm": sum(record["combined_pre_clip_norm"] for record in step_records) / max_updates,
            "max_pre_clip_norm": max(record["combined_pre_clip_norm"] for record in step_records),
        },
        "raw_task_norm_max": {name: max(record["raw_task_norms"][name] for record in step_records)
                              for name in TASKS},
        "encoder_work": dict(judge.work),
        "adapter_probe": {"parameter": probe_name,
                          "absolute_change": (probe.detach().cpu() - initial_probe).abs().sum().item()},
    })
    torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state()},
               output / "training_state.pt")
    if not args.skip_evaluation:
        evaluation, evaluation_ids = evaluate_and_save(judge, output, config, splits, tokens)
        run["evaluation_ids"] = evaluation_ids
        run["checkpoint_selection"] = "fixed final update; test never used for selection"
    else:
        evaluation = None
        run["checkpoint_selection"] = "engineering smoke only; no evaluation"
    write_json(output / "run.json", run)
    checksum_names = ["model.json", "decision.safetensors"]
    if not args.skip_evaluation:
        checksum_names.append("calibration.json")
    write_json(output / "checksums.json", {name: digest((output / name).read_bytes()) for name in checksum_names})
    print(canonical({"event": "complete", "method": args.method, "updates": max_updates,
                     "test": evaluation["test"] if evaluation else None}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for option in ("data", "config", "output", "init-from", "base-path"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--skip-evaluation", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
