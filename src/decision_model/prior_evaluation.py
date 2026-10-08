"""Reproducible logit collection and evaluation for contextual prior correction."""
from __future__ import annotations

import math
from collections import Counter

from .core import canonical
from .prior_correction import prior_only_request
from .train import calibrate, evaluate_rows


def collect_evidence_and_prior_logits(judge, rows):
    """Collect serial logits and cache identical label-free prior requests."""
    judge.train(False)
    prior_cache = {}
    records = []
    with judge.torch.no_grad():
        for row in rows:
            request = row["request"]
            evidence = judge.logits([judge.encode(request)])[0].float().cpu().tolist()
            null_request = prior_only_request(request)
            cache_key = canonical(null_request)
            if cache_key not in prior_cache:
                prior_cache[cache_key] = judge.logits(
                    [judge.encode(null_request)])[0].float().cpu().tolist()
            prior = prior_cache[cache_key]
            if len(evidence) != len(prior) or len(evidence) != len(request["choices"]):
                raise ValueError("Prior evaluation logit shape differs from candidate count")
            if any(not math.isfinite(value) for value in (*evidence, *prior)):
                raise ValueError("Prior evaluation produced non-finite logits")
            records.append({"id": row["id"], "evidence_logits": evidence, "prior_logits": prior})
    return records, {"rows": len(rows), "unique_prior_requests": len(prior_cache)}


def adjusted_logits(rows, records, strength):
    """Apply centered prior subtraction to stored logits in row order."""
    if (not isinstance(strength, (int, float)) or isinstance(strength, bool) or
            not math.isfinite(strength) or strength < 0):
        raise ValueError("Prior-correction strength must be finite and nonnegative")
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(records) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Prior evaluation record IDs differ")
    result = []
    for row in rows:
        record = by_id[row["id"]]
        evidence, prior = record["evidence_logits"], record["prior_logits"]
        if len(evidence) != len(prior) or len(evidence) != len(row["request"]["choices"]):
            raise ValueError("Prior evaluation logit shape differs from candidate count")
        center = sum(prior) / len(prior)
        values = [score - float(strength) * (bias - center)
                  for score, bias in zip(evidence, prior)]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Adjusted logits are non-finite")
        result.append(values)
    return result


def fit_and_evaluate(rows, records, strength):
    """Fit validation temperature and report raw/calibrated metrics."""
    logits = adjusted_logits(rows, records, strength)
    calibration = calibrate(logits, rows)
    raw, _ = evaluate_rows(rows, logits)
    scaled, predictions = evaluate_rows(rows, logits, calibration["temperature"])
    return {"strength": float(strength), "calibration": calibration,
            "raw": raw, "temperature_scaled": scaled, "predictions": predictions}


def evaluate_with_temperature(rows, records, strength, temperature):
    """Evaluate a frozen strength and validation-fitted temperature."""
    if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or
            not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Temperature must be finite and positive")
    logits = adjusted_logits(rows, records, strength)
    raw, _ = evaluate_rows(rows, logits)
    scaled, predictions = evaluate_rows(rows, logits, temperature)
    return {"strength": float(strength), "temperature": float(temperature),
            "raw": raw, "temperature_scaled": scaled, "predictions": predictions}


def candidate_marginal_gap(rows, predictions):
    """Report prediction-vs-truth marginal TV distance as a bias diagnostic."""
    by_id = {point["id"]: point for point in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Prior diagnostic prediction IDs differ")
    result = {}
    for source in sorted({row["source"] for row in rows}):
        truth, predicted = Counter(), Counter()
        source_rows = [row for row in rows if row["source"] == source]
        for row in source_rows:
            point = by_id[row["id"]]
            ids = [choice["id"] for choice in row["request"]["choices"]]
            if point["label"] != row["label"] or set(point["probabilities"]) != set(ids):
                raise ValueError("Prior diagnostic candidate contract differs")
            truth[row["label"]] += 1
            predicted[max(ids, key=lambda identity: point["probabilities"][identity])] += 1
        identities = sorted(set(truth) | set(predicted))
        count = len(source_rows)
        gap = 0.5 * sum(abs(predicted[identity] / count - truth[identity] / count)
                        for identity in identities)
        result[source] = {"rows": count, "truth": dict(truth), "predicted": dict(predicted),
                          "total_variation_gap": gap}
    return result
