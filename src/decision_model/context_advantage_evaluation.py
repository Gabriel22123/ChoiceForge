"""Mechanistic evaluation for evidence-conditioned candidate decisions."""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median


def _log_softmax(values):
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Expected at least two finite logits")
    maximum = max(values)
    log_normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return [value - log_normalizer for value in values]


def summarize_context_advantage(rows, records, gap=0.2):
    """Summarize gold log-probability gain from real over null context.

    The statistic exactly matches the quantity constrained during training, before
    temperature calibration.  It is label aware and therefore belongs only in
    evaluation, never in blind selection or model input construction.
    """
    if (not isinstance(gap, (int, float)) or isinstance(gap, bool) or
            not math.isfinite(gap) or gap < 0):
        raise ValueError("Gap must be finite and nonnegative")
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(records) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Context-advantage record IDs differ")
    details = []
    for row in rows:
        record = by_id[row["id"]]
        choices = row["request"]["choices"]
        evidence = record["evidence_logits"]
        null = record["prior_logits"]
        if len(evidence) != len(choices) or len(null) != len(choices):
            raise ValueError("Context-advantage logit shape differs")
        ids = [choice["id"] for choice in choices]
        target = ids.index(row["label"])
        evidence_logp = _log_softmax(evidence)
        null_logp = _log_softmax(null)
        gain = evidence_logp[target] - null_logp[target]
        details.append({
            "id": row["id"], "source": row["source"],
            "gold_log_probability_gain": gain,
            "margin_shortfall": max(0.0, float(gap) - gain),
            "positive_gain": gain > 0,
            "gap_satisfied": gain >= float(gap),
            "evidence_changes_argmax": max(range(len(evidence)), key=evidence.__getitem__) !=
                                        max(range(len(null)), key=null.__getitem__),
        })

    def aggregate(points):
        gains = [point["gold_log_probability_gain"] for point in points]
        shortfalls = [point["margin_shortfall"] for point in points]
        return {
            "rows": len(points),
            "mean_gold_log_probability_gain": mean(gains),
            "median_gold_log_probability_gain": median(gains),
            "positive_gain_rate": mean(point["positive_gain"] for point in points),
            "gap_satisfaction_rate": mean(point["gap_satisfied"] for point in points),
            "mean_margin_shortfall": mean(shortfalls),
            "evidence_argmax_change_rate": mean(
                point["evidence_changes_argmax"] for point in points),
        }

    groups = defaultdict(list)
    for point in details:
        groups[point["source"]].append(point)
    return {
        "gap": float(gap),
        "overall": aggregate(details),
        "by_source": {source: aggregate(groups[source]) for source in sorted(groups)},
        "details": details,
    }
