"""Conflict projection with a non-amplifying post-projection norm bound."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

try:
    from task_gradient_control import _validate, _weighted_sum, norm, symmetric_pairwise_projection, task_statistics
    from task_gradient_composition import soft_median_cap
except ModuleNotFoundError:
    from scripts.task_gradient_control import _validate, _weighted_sum, norm, symmetric_pairwise_projection, task_statistics
    from scripts.task_gradient_composition import soft_median_cap


METHODS = ("raw", "soft_cap_2x_then_project", "soft_cap_2x_project_recap")


def upper_cap_at(gradients: Mapping[str, Sequence], threshold: float):
    if threshold < 0:
        raise ValueError("Post-projection threshold must be nonnegative")
    norms = {name: norm(value) for name, value in gradients.items()}
    scales = {name: min(1.0, threshold / value) if value > 0 else 1.0
              for name, value in norms.items()}
    transformed = {name: [value.detach().mul(scales[name]) for value in gradient]
                   for name, gradient in gradients.items()}
    return transformed, {
        "threshold_norm": threshold,
        "input_norms": norms,
        "scales": scales,
        "amplified_tasks": [],
    }


def combine(gradients: Mapping[str, Sequence], weights: Mapping[str, float], method: str):
    _validate(gradients, weights)
    before = task_statistics(gradients)
    if method == "raw":
        transformed = {name: [value.detach() for value in gradient]
                       for name, gradient in gradients.items()}
        transformation = {"kind": "identity"}
    else:
        capped, pre_cap = soft_median_cap(gradients)
        projected, projection = symmetric_pairwise_projection(capped)
        if method == "soft_cap_2x_then_project":
            transformed = projected
            transformation = {
                "kind": "two_x_median_upper_cap_then_symmetric_projection",
                "cap": pre_cap,
                "projection": projection,
            }
        elif method == "soft_cap_2x_project_recap":
            transformed, post_cap = upper_cap_at(projected, pre_cap["threshold_norm"])
            transformation = {
                "kind": "two_x_median_cap_project_then_recap",
                "pre_cap": pre_cap,
                "projection": projection,
                "post_cap": post_cap,
            }
        else:
            raise ValueError("Unknown bounded-projection method")
    combined = _weighted_sum(transformed, weights)
    return combined, {
        "before": before,
        "after": task_statistics(transformed),
        "transformation": transformation,
        "combined_norm": norm(combined),
        "weights": {name: float(weights[name]) for name in gradients},
    }
