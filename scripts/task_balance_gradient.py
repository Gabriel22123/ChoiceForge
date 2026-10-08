"""Measure pairwise task-gradient cosine at a fixed checkpoint without updating it."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean

from decision_model.core import digest, load_rows, write_json
from decision_model.model import Judge
from decision_model.training_objective import TrainingObjective
from task_balance_plan import task_name


def dot(left, right):
    return sum((a.double() * b.double()).sum().item() for a, b in zip(left, right))


def norm(values):
    return math.sqrt(max(0.0, dot(values, values)))


def cosine(left, right):
    denominator = norm(left) * norm(right)
    return dot(left, right) / denominator if denominator else None


def task_gradient(judge, objective, rows):
    import torch
    requests = [row["request"] for row in rows]
    encoded = [judge.encode(request) for request in requests]
    targets = torch.tensor([
        [choice["id"] for choice in request["choices"]].index(row["label"])
        for row, request in zip(rows, requests)
    ], device=judge.device)
    logits = judge.logits(encoded)
    loss, _ = objective(logits, targets, [len(request["choices"]) for request in requests])
    parameters = [parameter for _, parameter in judge.named_trainable()]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [torch.zeros_like(parameter, device="cpu", dtype=torch.float32) if gradient is None
            else gradient.detach().float().cpu() for parameter, gradient in zip(parameters, gradients)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config.get("gradient_estimator") != "clean" or config.get("training_views") != 1:
        raise ValueError("Gradient probe expects the frozen clean one-view objective")
    rows = load_rows(Path(args.data)); by_id = {row["id"]: row for row in rows}
    order = config["epoch_row_orders"][0]
    judge = Judge(config, base_path=args.base_path, checkpoint=args.checkpoint, device=args.device)
    judge.train(False)
    objective = TrainingObjective(config)
    pairs = (("paws", "snli"), ("paws", "replay"), ("snli", "replay"))
    records = []
    for update in range(0, 128, 16):
        block = [by_id[key] for key in order[8 * update:8 * update + 8]]
        grouped = defaultdict(list)
        for row in block:
            grouped[task_name(row)].append(row)
        if {key: len(value) for key, value in grouped.items()} != {"paws": 3, "snli": 3, "replay": 2}:
            raise ValueError("Probe configuration is not the stratified plan")
        gradients = {task: task_gradient(judge, objective, grouped[task]) for task in grouped}
        records.append({
            "update_index": update,
            "norms": {task: norm(value) for task, value in gradients.items()},
            "cosines": {f"{left}:{right}": cosine(gradients[left], gradients[right])
                        for left, right in pairs},
        })
        del gradients
    aggregate = {}
    for left, right in pairs:
        key = f"{left}:{right}"; values = [record["cosines"][key] for record in records]
        aggregate[key] = {"mean": mean(values), "min": min(values), "max": max(values),
                          "negative": sum(value < 0 for value in values), "n": len(values)}
    write_json(Path(args.output), {
        "checkpoint_weights_sha256": digest((Path(args.checkpoint) / "decision.safetensors").read_bytes()),
        "config_sha256": digest(Path(args.config).read_bytes()),
        "dataset_sha256": digest(Path(args.data).read_bytes()),
        "setting": "Evaluation mode, no dropout, original candidate order, clean CE + 0.5 Brier; eight predeclared stratified updates; no optimizer step",
        "records": records,
        "aggregate": aggregate,
    })
    print(json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    main()
