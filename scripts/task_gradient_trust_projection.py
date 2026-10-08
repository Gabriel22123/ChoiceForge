"""Projection-correction trust region for task-gradient composition."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

try:
    from task_gradient_control import _validate, _weighted_sum, dot, norm, symmetric_pairwise_projection, task_statistics
    from task_gradient_composition import soft_median_cap
except ModuleNotFoundError:
    from scripts.task_gradient_control import _validate, _weighted_sum, dot, norm, symmetric_pairwise_projection, task_statistics
    from scripts.task_gradient_composition import soft_median_cap


METHODS = ("raw", "soft_cap_2x_then_project", "soft_cap_2x_project_trust")


def trust_blend(reference: Mapping[str, Sequence], projected: Mapping[str, Sequence],
                weights: Mapping[str, float]):
    """Keep the largest projection correction that does not raise combined norm.

    For raw combined gradient g and full capped/projected gradient p, use
    g + alpha * (p - g), alpha in [0, 1].  The bound is ||g||.  Unlike scaling
    the final vector, changing alpha changes direction and survives later global
    norm clipping.
    """
    _validate(reference, weights)
    _validate(projected, weights)
    base = _weighted_sum(reference, weights)
    full = _weighted_sum(projected, weights)
    delta = [right.detach() - left.detach() for left, right in zip(base, full)]
    base_squared = dot(base, base)
    full_squared = dot(full, full)
    delta_squared = dot(delta, delta)
    base_delta = dot(base, delta)
    if full_squared <= base_squared or delta_squared == 0:
        alpha = 1.0
    elif base_delta >= 0:
        alpha = 0.0
    else:
        alpha = min(1.0, max(0.0, -2.0 * base_delta / delta_squared))
        # Stay just inside the boundary rather than outside through float error.
        alpha *= 1.0 - 1e-7
    blended = {
        name: [left.detach() + (right.detach() - left.detach()) * alpha
               for left, right in zip(reference[name], projected[name])]
        for name in reference
    }
    combined = _weighted_sum(blended, weights)
    return blended, combined, {
        "alpha": alpha,
        "active": alpha < 1.0 - 1e-12,
        "reference_raw_combined_norm": math.sqrt(max(0.0, base_squared)),
        "full_projected_combined_norm": math.sqrt(max(0.0, full_squared)),
        "final_combined_norm": norm(combined),
        "bound_norm": math.sqrt(max(0.0, base_squared)),
    }


def combine(gradients: Mapping[str, Sequence], weights: Mapping[str, float], method: str):
    _validate(gradients, weights)
    before = task_statistics(gradients)
    if method == "raw":
        transformed = {name: [value.detach() for value in gradient]
                       for name, gradient in gradients.items()}
        combined = _weighted_sum(transformed, weights)
        transformation = {"kind": "identity"}
    else:
        capped, cap = soft_median_cap(gradients)
        projected, projection = symmetric_pairwise_projection(capped)
        if method == "soft_cap_2x_then_project":
            transformed = projected
            combined = _weighted_sum(transformed, weights)
            transformation = {
                "kind": "two_x_median_upper_cap_then_symmetric_projection",
                "cap": cap,
                "projection": projection,
            }
        elif method == "soft_cap_2x_project_trust":
            transformed, combined, trust = trust_blend(gradients, projected, weights)
            transformation = {
                "kind": "two_x_median_cap_projection_trust_region",
                "cap": cap,
                "projection": projection,
                "trust_region": trust,
            }
        else:
            raise ValueError("Unknown trust-projection method")
    return combined, {
        "before": before,
        "after": task_statistics(transformed),
        "transformation": transformation,
        "combined_norm": norm(combined),
        "weights": {name: float(weights[name]) for name in gradients},
    }
