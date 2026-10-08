"""Contextual candidate-prior diagnostics for dynamic decision sets."""
from __future__ import annotations

import math
from statistics import mean

from .core import validate_request
from .output_contract import validate_prediction


NULL_CONTEXT = "No case-specific context is provided."
STRENGTH_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def select_shared_strength(validation_nll_by_seed):
    """Select one preregistered strength using validation NLL only."""
    if not isinstance(validation_nll_by_seed, dict) or len(validation_nll_by_seed) < 2:
        raise ValueError("Prior correction selection requires at least two development seeds")
    normalized = {}
    for seed, values in validation_nll_by_seed.items():
        if not isinstance(values, dict) or len(values) != len(STRENGTH_GRID):
            raise ValueError("Each seed must provide the complete strength grid")
        try:
            converted = {float(strength): value for strength, value in values.items()
                         if not isinstance(strength, bool)}
        except (TypeError, ValueError) as exc:
            raise ValueError("Each seed must provide the complete strength grid") from exc
        if set(converted) != set(STRENGTH_GRID) or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 for value in converted.values()):
            raise ValueError("Each seed must provide finite NLL for the complete strength grid")
        normalized[str(seed)] = converted
    means = {strength: mean(values[strength] for values in normalized.values())
             for strength in STRENGTH_GRID}
    selected = min(STRENGTH_GRID, key=lambda strength: (means[strength], strength))
    return {
        "selected_strength": selected,
        "selection_metric": "equal-seed mean validation NLL",
        "tie_break": "smaller strength",
        "mean_validation_nll": means,
        "validation_nll_by_seed": normalized,
    }


def prior_only_request(request):
    """Keep task and candidate semantics while removing case-specific evidence."""
    request = validate_request(request)
    return dict(request, context=NULL_CONTEXT,
                choices=[dict(choice) for choice in request["choices"]])


def corrected_logits(judge, request, strength=1.0):
    """Subtract centered null-context scores from evidence-conditioned scores.

    Centering changes no softmax probabilities but keeps the numerical scale
    interpretable.  This is an inference diagnostic until a frozen multi-seed
    study shows that it improves candidate-prior stability without harming
    primary accuracy or probability quality.
    """
    if not isinstance(strength, (int, float)) or isinstance(strength, bool) or not math.isfinite(strength) or strength < 0:
        raise ValueError("Prior-correction strength must be a finite nonnegative number")
    validate_request(request)
    judge.train(False)
    with judge.torch.no_grad():
        evidence = judge.logits([judge.encode(request)])[0]
        prior = judge.logits([judge.encode(prior_only_request(request))])[0]
        prior = prior - prior.mean()
        return evidence - float(strength) * prior


def predict_with_prior_correction(judge, request, strength=1.0, temperature=1.0):
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = corrected_logits(judge, request, strength)
    probabilities = (logits / float(temperature)).softmax(-1).float().cpu().tolist()
    if not all(math.isfinite(value) for value in probabilities):
        raise ValueError("Non-finite corrected probabilities")
    ids = [choice["id"] for choice in request["choices"]]
    return validate_prediction({
        "choice_id": ids[max(range(len(probabilities)), key=probabilities.__getitem__)],
        "probabilities": dict(zip(ids, probabilities)),
        "requires_review": True,
        "score_kind": "contextual_prior_corrected_candidate_probability_not_a_guarantee",
        "calibration": "validation_temperature" if temperature != 1 else "uncalibrated",
    }, request)
