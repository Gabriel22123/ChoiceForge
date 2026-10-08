"""Binary counterfactual routing for an exactly closed parent path."""
from __future__ import annotations

import math


def binary_route_target(costs, gates):
    """Choose parent (0) or expert (1), closing deterministic ties."""
    import torch
    if (costs.ndim != 1 or gates.ndim != 1 or costs.shape != gates.shape or
            len(costs) != 2 or not torch.isfinite(costs).all() or
            not torch.equal(gates, gates.new_tensor([0.0, 1.0]))):
        raise ValueError("Binary routing requires finite costs at gates [0, 1]")
    target = int(costs[1] < costs[0])
    return {
        "gate": target,
        "closed_cost": float(costs[0]),
        "open_cost": float(costs[1]),
        "selected_cost": float(costs[target]),
        "improvement": float(costs[0] - costs[target]),
    }


def hard_route(probability, threshold=0.5):
    """Convert a learned probability into an exact binary route."""
    import torch
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or
            not math.isfinite(threshold) or not 0.0 < threshold < 1.0 or
            not torch.is_tensor(probability) or not torch.isfinite(probability).all() or
            (probability < 0).any() or (probability > 1).any()):
        raise ValueError("Invalid binary route probability or threshold")
    return (probability >= float(threshold)).to(probability.dtype)
