"""Runtime validation for the program-built prediction contract."""
from __future__ import annotations

import math

from .core import validate_request


FIELDS = {"choice_id", "probabilities", "requires_review", "score_kind", "calibration"}
SCORE_KINDS = {
    "candidate_probability_not_a_guarantee",
    "contextual_prior_corrected_candidate_probability_not_a_guarantee",
}
CALIBRATION = {"uncalibrated", "validation_temperature"}


def validate_prediction(prediction, request, tolerance=1e-5):
    """Validate constraints that JSON Schema alone cannot express.

    Candidate keys must exactly match the request, probabilities must be finite
    and normalized, and the selected ID must use the documented first-maximum
    tie rule in request order. JSON object key order is not semantically relevant.
    """
    request = validate_request(request)
    if not isinstance(prediction, dict) or set(prediction) != FIELDS:
        raise ValueError("Prediction must contain exactly the public output fields")
    ids = [choice["id"] for choice in request["choices"]]
    probabilities = prediction["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(ids):
        raise ValueError("Probability keys must exactly match request choices")
    values = [probabilities[identity] for identity in ids]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Probabilities must be finite numbers in [0, 1]")
    if abs(sum(values) - 1.0) > tolerance:
        raise ValueError("Probabilities must sum to one")
    expected = ids[max(range(len(values)), key=values.__getitem__)]
    if prediction["choice_id"] != expected:
        raise ValueError("choice_id must be the first maximum-probability candidate")
    if prediction["requires_review"] is not True:
        raise ValueError("Research preview decisions must require human review")
    if prediction["score_kind"] not in SCORE_KINDS:
        raise ValueError("Unknown score kind")
    if prediction["calibration"] not in CALIBRATION:
        raise ValueError("Unknown calibration mode")
    return prediction
