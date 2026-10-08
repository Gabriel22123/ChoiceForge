"""Strict Boolean, Choice and Score views over the generic candidate API."""
from __future__ import annotations

import math

from .core import validate_request
from .output_contract import validate_prediction


COMMON_INPUT = {"type", "task", "context"}
COMMON_OUTPUT = {
    "type", "value", "probabilities", "top_probability", "margin",
    "requires_review", "score_kind", "calibration",
}


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def typed_to_request(payload):
    """Validate a typed request and lower it to task/context/choices."""
    if not isinstance(payload, dict):
        raise ValueError("Typed request must be an object")
    kind = payload.get("type")
    if kind not in ("boolean", "choice", "score"):
        raise ValueError("type must be boolean, choice or score")
    expected = COMMON_INPUT | {
        "boolean": {"criteria"}, "choice": {"choices"}, "score": {"levels"}
    }[kind]
    if set(payload) != expected:
        raise ValueError(f"{kind} request must contain exactly {sorted(expected)}")
    task = _nonempty(payload["task"], "task")
    context = _nonempty(payload["context"], "context")
    if kind == "boolean":
        criteria = payload["criteria"]
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            raise ValueError("boolean criteria must contain exactly true and false")
        choices = [{"id": key, "description": _nonempty(criteria[key], f"criteria.{key}")}
                   for key in ("true", "false")]
    elif kind == "choice":
        choices = payload["choices"]
    else:
        levels = payload["levels"]
        if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
            raise ValueError("score levels must contain 2..10 ordered levels")
        choices = []
        for level in levels:
            if not isinstance(level, dict) or set(level) != {"id", "description"}:
                raise ValueError("Each score level needs exactly id and description")
            choices.append(level)
    return validate_request({"task": task, "context": context, "choices": choices})


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_typed_prediction(prediction, payload, tolerance=1e-5):
    """Validate the exact public union and cross-field probability invariants."""
    request = typed_to_request(payload)
    kind = payload["type"]
    extra = {"boolean": {"probability_true"}, "choice": set(),
             "score": {"level_index", "expected_score"}}[kind]
    if not isinstance(prediction, dict) or set(prediction) != COMMON_OUTPUT | extra:
        raise ValueError("Typed prediction contains unexpected or missing fields")
    if prediction["type"] != kind:
        raise ValueError("Prediction type differs from request type")
    if kind == "boolean":
        if not isinstance(prediction["value"], bool):
            raise ValueError("Boolean value must be true or false")
        choice_id = "true" if prediction["value"] else "false"
    else:
        choice_id = prediction["value"]
    base = {
        "choice_id": choice_id,
        "probabilities": prediction["probabilities"],
        "requires_review": prediction["requires_review"],
        "score_kind": prediction["score_kind"],
        "calibration": prediction["calibration"],
    }
    validate_prediction(base, request, tolerance=tolerance)
    values = [prediction["probabilities"][choice["id"]] for choice in request["choices"]]
    ranked = sorted(values, reverse=True)
    if abs(_number(prediction["top_probability"], "top_probability") - ranked[0]) > tolerance:
        raise ValueError("top_probability differs from the distribution maximum")
    if abs(_number(prediction["margin"], "margin") - (ranked[0] - ranked[1])) > tolerance:
        raise ValueError("margin differs from the top-two probability gap")
    if kind == "boolean":
        if abs(_number(prediction["probability_true"], "probability_true") - values[0]) > tolerance:
            raise ValueError("probability_true differs from the true mass")
    elif kind == "score":
        index = prediction["level_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(values):
            raise ValueError("level_index is outside the ordered levels")
        if request["choices"][index]["id"] != prediction["value"]:
            raise ValueError("level_index and value disagree")
        expected = sum(index * probability for index, probability in enumerate(values))
        if abs(_number(prediction["expected_score"], "expected_score") - expected) > tolerance:
            raise ValueError("expected_score differs from the ordered distribution")
    return prediction


def format_typed_prediction(payload, base_prediction):
    """Build the bounded typed response from an already validated base result."""
    request = typed_to_request(payload)
    base = validate_prediction(base_prediction, request)
    probabilities = dict(base["probabilities"])
    values = [probabilities[choice["id"]] for choice in request["choices"]]
    ranked = sorted(values, reverse=True)
    common = {
        "type": payload["type"],
        "value": base["choice_id"],
        "probabilities": probabilities,
        "top_probability": ranked[0],
        "margin": ranked[0] - ranked[1],
        "requires_review": base["requires_review"],
        "score_kind": base["score_kind"],
        "calibration": base["calibration"],
    }
    if payload["type"] == "boolean":
        common["value"] = base["choice_id"] == "true"
        common["probability_true"] = probabilities["true"]
    elif payload["type"] == "score":
        common["level_index"] = next(index for index, choice in enumerate(request["choices"])
                                     if choice["id"] == base["choice_id"])
        common["expected_score"] = sum(index * probability
                                       for index, probability in enumerate(values))
    return validate_typed_prediction(common, payload)


def predict_typed(judge, payload):
    """Run one typed request through any judge implementing predict(request)."""
    request = typed_to_request(payload)
    return format_typed_prediction(payload, judge.predict(request))
