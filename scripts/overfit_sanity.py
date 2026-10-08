"""Can the existing candidate scorer fit 24 public training examples? Not a benchmark."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import time

import torch

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.model import Judge, verify_local_base
from decision_model.train import balanced_subset, evaluate_rows, predict_rows
from decision_model.training_objective import TrainingObjective


def balanced_batches(rows, seed, updates):
    rng = random.Random(seed)
    labels = sorted({r["label"] for r in rows})
    pools = {label: [r for r in rows if r["label"] == label] for label in labels}
    if len(labels) != 3 or any(len(v) != 8 for v in pools.values()):
        raise ValueError("Expected eight examples of each of three labels")
    queues = {label: [] for label in labels}
    for _ in range(updates):
        batch = []
        for label in labels:
            if not queues[label]:
                queues[label] = pools[label].copy()
                rng.shuffle(queues[label])
            batch += [queues[label].pop(), queues[label].pop()]
        rng.shuffle(batch)
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/overfit-sanity-v1")
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.updates < 1 or args.updates % 4:
        raise ValueError("Use a positive multiple of four updates for equal exposure")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if output.exists():
        raise ValueError("Use a new output directory")
    data = root / "data/public-snli-v1/cases.jsonl"
    expected = "d2e11f0e7b83a5a89c72d79b637cde9c6f5673fc398e4c6850b3063e11e75aec"
    if digest(data.read_bytes()) != expected:
        raise ValueError("Pinned public SNLI data changed")
    # No validation/test predictions or labels are used in this diagnostic.
    rows = balanced_subset([r for r in load_rows(data) if r["split"] == "train"], 24)
    initial = root / "runs/architecture-study-v1/D-independent"
    config = json.loads((initial / "model.json").read_text())
    config.update(dropout=0., brier_weight=.5, consistency_weight=0., training_views=1,
                  gradient_estimator="clean", seed=42)
    base = root / "models/eurobert-2.1b"
    verify_local_base(base)
    protocol = {"purpose": "Training-set memorization sanity check; no generalization or deployment claim",
                "dataset_sha256": expected, "selected_ids": [r["id"] for r in rows],
                "starting_checkpoint": str(initial.relative_to(root)),
                "starting_weights_sha256": digest((initial / "decision.safetensors").read_bytes()),
                "config": config, "updates": args.updates, "examples_per_update": 6,
                "changes_from_previous_training": ["24 training rows repeated", "balanced labels in each update", "dropout disabled"],
                "fixed": ["same D-independent initialization", "shared marker scorer", "32-layer LoRA rank 8",
                          "encoder/head learning rates", "CE + .5 Brier", "AdamW weight decay .01", "norm clipping 1"],
                "interpretation": "Success only demonstrates fitting these training examples. Failure does not isolate a cause.",
                "source_files": {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in
                                 [*sorted((root / "src/decision_model").glob("*.py")), Path(__file__).resolve()]}}
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    (output / "cases.jsonl").write_text("".join(canonical(r) + "\n" for r in rows))
    write_json(output / "status.json", {"state": "running"})
    judge = Judge(config, base_path=base, checkpoint=initial, device=args.device)
    tokens = {r["id"]: judge.encode(r["request"]) for r in rows}
    params = list(judge.named_trainable())
    optimizer = torch.optim.AdamW([
        {"params": [p for n,p in params if n.startswith("encoder.")], "lr": config["encoder_lr"]},
        {"params": [p for n,p in params if n.startswith("head.")], "lr": config["head_lr"]},
    ], weight_decay=.01, eps=1e-6)
    objective = TrainingObjective(config)
    rng = random.Random(913)
    curve, norms = [], []
    start = time.monotonic()

    def evaluate(step):
        logits = predict_rows(judge, rows, tokens)
        metric, predictions = evaluate_rows(rows, logits)
        point = {"step": step, "seconds": time.monotonic()-start, "metrics": metric,
                 "predicted_labels": dict(Counter(max(p["probabilities"], key=p["probabilities"].get) for p in predictions))}
        curve.append(point)
        write_json(output / "curve.json", curve)
        write_json(output / "train-predictions.json", predictions)
        print(json.dumps(point), flush=True)
        judge.train(True)

    evaluate(0)
    for step, batch in enumerate(balanced_batches(rows, 42, args.updates), 1):
        optimizer.zero_grad(set_to_none=True)
        risk_total = 0.
        for begin in range(0, 6, 2):
            micro = batch[begin:begin+2]
            requests = []
            for row in micro:
                request = copy.deepcopy(row["request"])
                rng.shuffle(request["choices"])
                requests.append(request)
            targets = torch.tensor([[c["id"] for c in request["choices"]].index(row["label"])
                                    for row, request in zip(micro, requests)], device=judge.device)
            logits = judge.logits([judge.encode(request) for request in requests])
            loss, risk = objective(logits, targets, [3, 3])
            (loss/3).backward()
            risk_total += risk.item()/3
        norm = torch.nn.utils.clip_grad_norm_([p for _,p in params], 1., error_if_nonfinite=True).item()
        optimizer.step()
        norms.append(norm)
        with (output / "steps.jsonl").open("a") as log:
            log.write(canonical({"step": step, "rows": [r["id"] for r in batch], "risk": risk_total, "pre_clip_norm": norm})+"\n")
        if step % 20 == 0 or step == args.updates:
            evaluate(step)
    judge.save(output)
    write_json(output / "checksums.json", {name: digest((output/name).read_bytes()) for name in ("model.json", "decision.safetensors")})
    result = {"purpose": protocol["purpose"], "initial": curve[0], "final": curve[-1], "updates": args.updates,
              "repetitions_per_row": args.updates//4, "clipped_updates": sum(n>1 for n in norms),
              "elapsed_seconds_including_evaluation": time.monotonic()-start, "config": config}
    write_json(output / "result.json", result)
    write_json(output / "status.json", {"state": "complete"})


if __name__ == "__main__":
    main()
