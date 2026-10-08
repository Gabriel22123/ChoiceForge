"""Deterministic task-gradient transformations for the matched control study.

The functions operate on lists of parameter-shaped tensors.  They deliberately
avoid flattening the full trainable state, which keeps the 2.1B encoder study
within local accelerator memory.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


METHODS = ("raw", "median_cap", "symmetric_project")


def _validate(gradients, weights):
    if not gradients or set(gradients) != set(weights):
        raise ValueError("Gradient tasks and weights must be the same nonempty set")
    names = tuple(gradients)
    length = len(gradients[names[0]])
    if not length or any(len(gradients[name]) != length for name in names):
        raise ValueError("Every task must provide one gradient per parameter")
    if any(not math.isfinite(float(weights[name])) or weights[name] < 0 for name in names):
        raise ValueError("Task weights must be finite and nonnegative")
    if not math.isclose(sum(float(weights[name]) for name in names), 1.0,
                            rel_tol=0, abs_tol=1e-9):
        raise ValueError("Task weights must sum to one")
    for index in range(length):
        shape = gradients[names[0]][index].shape
        if any(gradients[name][index].shape != shape for name in names):
            raise ValueError("Gradient tensor shapes differ across tasks")
    return names, length


def dot(left: Sequence, right: Sequence) -> float:
    """Return a float32-accumulated dot product without a giant flat tensor."""
    if len(left) != len(right):
        raise ValueError("Gradient lengths differ")
    total = 0.0
    for a, b in zip(left, right):
        if a.shape != b.shape:
            raise ValueError("Gradient tensor shapes differ")
        total += (a.float() * b.float()).sum().item()
    return float(total)


def norm(gradient: Sequence) -> float:
    return math.sqrt(max(0.0, dot(gradient, gradient)))


def cosine(left: Sequence, right: Sequence) -> float:
    denominator = norm(left) * norm(right)
    return dot(left, right) / denominator if denominator else 0.0


def task_statistics(gradients: Mapping[str, Sequence]) -> dict:
    names = tuple(gradients)
    norms = {name: norm(gradients[name]) for name in names}
    cosines = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            denominator = norms[left] * norms[right]
            cosines[f"{left}:{right}"] = (
                dot(gradients[left], gradients[right]) / denominator if denominator else 0.0
            )
    return {"norms": norms, "cosines": cosines}


def _scaled(gradient: Sequence, factor: float):
    return [value.detach().mul(factor) for value in gradient]


def _weighted_sum(gradients: Mapping[str, Sequence], weights: Mapping[str, float]):
    names, length = _validate(gradients, weights)
    return [sum((gradients[name][index].detach() * float(weights[name]) for name in names),
                start=gradients[names[0]][index].new_zeros(()))
            for index in range(length)]


def median_upper_cap(gradients: Mapping[str, Sequence]):
    """Cap above-median task norms while never amplifying a smaller gradient."""
    norms = {name: norm(value) for name, value in gradients.items()}
    ordered = sorted(norms.values())
    target = ordered[len(ordered) // 2]
    scales = {name: min(1.0, target / value) if value > 0 else 1.0
              for name, value in norms.items()}
    return {name: _scaled(gradients[name], scales[name]) for name in gradients}, {
        "target_norm": target,
        "scales": scales,
        "amplified_tasks": [],
    }


def symmetric_pairwise_projection(gradients: Mapping[str, Sequence]):
    """Project every negative pair symmetrically using the original gradients.

    This deterministic PCGrad-inspired transform has no shuffled projection
    order.  Each task receives the sum of its pairwise corrections computed
    from the unmodified gradients, so study arms can share an exact RNG path.
    """
    names = tuple(gradients)
    squared = {name: dot(gradients[name], gradients[name]) for name in names}
    projected = {name: [value.detach().clone() for value in gradients[name]] for name in names}
    conflicts = []
    coefficients = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            product = dot(gradients[left], gradients[right])
            key = f"{left}:{right}"
            if product >= 0 or squared[left] == 0 or squared[right] == 0:
                coefficients[key] = {left: 0.0, right: 0.0}
                continue
            # g_i <- g_i - <g_i,g_j>/||g_j||^2 g_j, for both directions.
            left_coefficient = -product / squared[right]
            right_coefficient = -product / squared[left]
            coefficients[key] = {left: left_coefficient, right: right_coefficient}
            conflicts.append(key)
            for index in range(len(projected[left])):
                projected[left][index].add_(gradients[right][index], alpha=left_coefficient)
                projected[right][index].add_(gradients[left][index], alpha=right_coefficient)
    return projected, {"conflicting_pairs": conflicts, "coefficients": coefficients}


def combine(gradients: Mapping[str, Sequence], weights: Mapping[str, float], method: str):
    """Transform and combine task gradients, returning auditable diagnostics."""
    _validate(gradients, weights)
    before = task_statistics(gradients)
    if method == "raw":
        transformed = {name: [value.detach() for value in gradient]
                       for name, gradient in gradients.items()}
        transformation = {"kind": "identity"}
    elif method == "median_cap":
        transformed, transformation = median_upper_cap(gradients)
        transformation["kind"] = "within_update_median_upper_cap"
    elif method == "symmetric_project":
        transformed, transformation = symmetric_pairwise_projection(gradients)
        transformation["kind"] = "deterministic_symmetric_pairwise_projection"
    else:
        raise ValueError("Unknown gradient-control method")
    combined = _weighted_sum(transformed, weights)
    return combined, {
        "before": before,
        "after": task_statistics(transformed),
        "transformation": transformation,
        "combined_norm": norm(combined),
        "weights": {name: float(weights[name]) for name in gradients},
    }
