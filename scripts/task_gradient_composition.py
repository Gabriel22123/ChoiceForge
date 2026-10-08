"""Soft magnitude control followed by optional conflict projection.

This follow-up keeps the completed task-gradient-control-v1 implementation
frozen.  It imports its audited tensor algebra and adds only the preregistered
composition treatments.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

try:
    from task_gradient_control import (
        _validate,
        _weighted_sum,
        norm,
        symmetric_pairwise_projection,
        task_statistics,
    )
except ModuleNotFoundError:
    from scripts.task_gradient_control import (
        _validate,
        _weighted_sum,
        norm,
        symmetric_pairwise_projection,
        task_statistics,
    )


METHODS = ("raw", "soft_cap_2x", "soft_cap_2x_then_project")


def soft_median_cap(gradients: Mapping[str, Sequence], multiplier: float = 2.0):
    if multiplier < 1:
        raise ValueError("Soft-cap multiplier must be at least one")
    norms = {name: norm(value) for name, value in gradients.items()}
    ordered = sorted(norms.values())
    median = ordered[len(ordered) // 2]
    threshold = multiplier * median
    scales = {
        name: min(1.0, threshold / value) if value > 0 else 1.0
        for name, value in norms.items()
    }
    transformed = {
        name: [value.detach().mul(scales[name]) for value in gradient]
        for name, gradient in gradients.items()
    }
    return transformed, {
        "median_norm": median,
        "multiplier": multiplier,
        "threshold_norm": threshold,
        "scales": scales,
        "amplified_tasks": [],
    }


def combine(gradients: Mapping[str, Sequence], weights: Mapping[str, float], method: str):
    _validate(gradients, weights)
    before = task_statistics(gradients)
    if method == "raw":
        transformed = {
            name: [value.detach() for value in gradient]
            for name, gradient in gradients.items()
        }
        transformation = {"kind": "identity"}
    elif method == "soft_cap_2x":
        transformed, cap = soft_median_cap(gradients)
        transformation = {"kind": "two_x_median_upper_cap", "cap": cap}
    elif method == "soft_cap_2x_then_project":
        capped, cap = soft_median_cap(gradients)
        transformed, projection = symmetric_pairwise_projection(capped)
        transformation = {
            "kind": "two_x_median_upper_cap_then_symmetric_projection",
            "cap": cap,
            "projection": projection,
        }
    else:
        raise ValueError("Unknown gradient-composition method")
    combined = _weighted_sum(transformed, weights)
    return combined, {
        "before": before,
        "after": task_statistics(transformed),
        "transformation": transformation,
        "combined_norm": norm(combined),
        "weights": {name: float(weights[name]) for name in gradients},
    }
