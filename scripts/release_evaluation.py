"""Pure release-study diagnostics over frozen evidence and null-context logits."""
from __future__ import annotations

import math
from collections import defaultdict


def softmax(values, temperature=1.0):
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
            not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Temperature must be finite and positive")
    if not isinstance(values, list) or len(values) < 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) for value in values):
        raise ValueError("Logits must be a finite list with at least two values")
    scaled = [float(value) / float(temperature) for value in values]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def context_advantage(rows, records, temperature=1.0):
    """Summarize gold log-probability gain from case-specific context."""
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(records) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Context-advantage record IDs differ")
    gains = defaultdict(list)
    per_row = []
    for row in rows:
        record = by_id[row["id"]]
        ids = [choice["id"] for choice in row["request"]["choices"]]
        if len(record["evidence_logits"]) != len(ids) or len(record["prior_logits"]) != len(ids):
            raise ValueError("Context-advantage candidate count differs")
        gold = ids.index(row["label"])
        evidence = softmax(record["evidence_logits"], temperature)
        prior = softmax(record["prior_logits"], temperature)
        gain = math.log(evidence[gold]) - math.log(prior[gold])
        gains[row["source"]].append(gain)
        per_row.append({"id": row["id"], "source": row["source"],
                        "gold_log_probability_advantage": gain})
    by_source = {source: {"rows": len(values), "mean": sum(values) / len(values),
                          "positive": sum(value > 0 for value in values)}
                 for source, values in sorted(gains.items())}
    values = [item["gold_log_probability_advantage"] for item in per_row]
    return {"rows": len(values), "mean": sum(values) / len(values),
            "positive": sum(value > 0 for value in values),
            "by_source": by_source}, per_row


def _entropy(probabilities):
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def cross_seed_null_js(record_sets, temperatures=None):
    """Compute generalized Jensen-Shannon divergence of null predictions."""
    if not isinstance(record_sets, dict) or len(record_sets) < 2:
        raise ValueError("Cross-seed JS requires at least two seeds")
    temperatures = temperatures or {seed: 1.0 for seed in record_sets}
    if set(temperatures) != set(record_sets):
        raise ValueError("Cross-seed temperatures differ from record sets")
    indexed = {}
    for seed, records in record_sets.items():
        by_id = {record["id"]: record for record in records}
        if len(by_id) != len(records):
            raise ValueError("Duplicate record ID in cross-seed JS")
        indexed[seed] = by_id
    identities = set(next(iter(indexed.values())))
    if any(set(values) != identities for values in indexed.values()):
        raise ValueError("Cross-seed record IDs differ")
    per_row = []
    for identity in sorted(identities):
        distributions = [softmax(indexed[seed][identity]["prior_logits"], temperatures[seed])
                         for seed in sorted(indexed, key=str)]
        width = len(distributions[0])
        if any(len(values) != width for values in distributions):
            raise ValueError("Cross-seed candidate count differs")
        mixture = [sum(values[index] for values in distributions) / len(distributions)
                   for index in range(width)]
        divergence = _entropy(mixture) - sum(map(_entropy, distributions)) / len(distributions)
        per_row.append({"id": identity, "js": max(0.0, divergence)})
    return {"rows": len(per_row), "mean_js": sum(item["js"] for item in per_row) / len(per_row),
            "maximum_js": max(item["js"] for item in per_row)}, per_row
