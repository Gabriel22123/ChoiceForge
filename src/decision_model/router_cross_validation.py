"""Leakage-safe leave-one-family-out plans and binary route metrics."""
from __future__ import annotations

import math
import random
from collections import defaultdict


def make_lofo_plan(rows, heldout_source, seed, updates, *, source_balanced):
    """Build fixed-compute batches without exposing the held-out source."""
    if (not rows or not isinstance(heldout_source, str) or not heldout_source or
            not isinstance(seed, int) or isinstance(seed, bool) or
            not isinstance(updates, int) or isinstance(updates, bool) or updates < 1 or
            not isinstance(source_balanced, bool)):
        raise ValueError("Invalid leave-family-out plan arguments")
    groups = defaultdict(list)
    ids = set()
    for row in rows:
        row_id, source = row.get("id"), row.get("source")
        if (not isinstance(row_id, str) or not row_id or row_id in ids or
                not isinstance(source, str) or not source):
            raise ValueError("Invalid or duplicate routing row")
        ids.add(row_id)
        if source != heldout_source:
            groups[source].append(row_id)
    sources = sorted(groups)
    if heldout_source not in {row.get("source") for row in rows} or len(sources) < 2:
        raise ValueError("Held-out source is missing or leaves too few training sources")
    rng = random.Random(seed)
    batch_size = len(sources)
    plan = []
    if source_balanced:
        positions = {source: 0 for source in sources}
        for source in sources:
            groups[source].sort(); rng.shuffle(groups[source])
        for update in range(updates):
            batch = []
            for source in sources:
                position = positions[source]
                if position == len(groups[source]):
                    rng.shuffle(groups[source]); position = 0
                batch.append(groups[source][position]); positions[source] = position + 1
            rng.shuffle(batch)
            plan.append({"update": update + 1, "row_ids": batch})
    else:
        pool = sorted(row_id for values in groups.values() for row_id in values)
        position = len(pool)
        for update in range(updates):
            batch = []
            while len(batch) < batch_size:
                if position == len(pool):
                    rng.shuffle(pool); position = 0
                take = min(batch_size - len(batch), len(pool) - position)
                batch.extend(pool[position:position + take]); position += take
            plan.append({"update": update + 1, "row_ids": batch})
    if (len(plan) != updates or any(len(item["row_ids"]) != batch_size for item in plan) or
            any(row_id not in ids for item in plan for row_id in item["row_ids"])):
        raise ValueError("Invalid leave-family-out plan construction")
    return plan


def binary_route_metrics(targets, probabilities):
    """Return proper losses and class-aware metrics for route probabilities."""
    if len(targets) != len(probabilities) or not targets:
        raise ValueError("Binary route metrics require aligned nonempty values")
    if (any(value not in (0, 1) for value in targets) or
            any(not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in probabilities)):
        raise ValueError("Invalid binary targets or probabilities")
    epsilon = 1e-7
    predictions = [int(value >= 0.5) for value in probabilities]
    accuracy = sum(left == right for left, right in zip(targets, predictions)) / len(targets)
    brier = sum((float(probability) - target) ** 2
                for target, probability in zip(targets, probabilities)) / len(targets)
    log_loss = -sum(
        target * math.log(min(1.0 - epsilon, max(epsilon, float(probability)))) +
        (1 - target) * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - float(probability))))
        for target, probability in zip(targets, probabilities)) / len(targets)
    recalls = {}
    for label, name in ((0, "closed_recall"), (1, "open_recall")):
        selected = [prediction == label for target, prediction in zip(targets, predictions)
                    if target == label]
        recalls[name] = (sum(selected) / len(selected)) if selected else None
    present = [value for value in recalls.values() if value is not None]
    return {
        "rows": len(targets),
        "accuracy": accuracy,
        "balanced_accuracy": sum(present) / len(present),
        "brier": brier,
        "log_loss": log_loss,
        "target_open_rate": sum(targets) / len(targets),
        "predicted_open_rate": sum(predictions) / len(predictions),
        "mean_probability": sum(map(float, probabilities)) / len(probabilities),
        **recalls,
    }
