"""Hard parent-choice preservation cost for counterfactual route targets."""
from __future__ import annotations


def add_parent_choice_cost(costs, parent, parent_null, correction, null_correction,
                           gates, *, active, change_cost=1.0):
    """Penalize gates that change either frozen-parent top choice."""
    import math
    import torch
    values = (parent, parent_null, correction, null_correction)
    if (costs.ndim != 1 or gates.ndim != 1 or costs.shape != gates.shape or
            any(value.ndim != 1 or value.shape != parent.shape for value in values) or
            len(parent) < 2 or not isinstance(active, bool) or
            not isinstance(change_cost, (int, float)) or isinstance(change_cost, bool) or
            not math.isfinite(change_cost) or change_cost < 0):
        raise ValueError("Invalid parent-choice route cost")
    if not active or change_cost == 0:
        return costs
    full = parent.unsqueeze(0) + gates.unsqueeze(1) * correction.unsqueeze(0)
    null = parent_null.unsqueeze(0) + gates.unsqueeze(1) * null_correction.unsqueeze(0)
    changed = ((full.argmax(-1) != parent.argmax()).to(costs.dtype) +
               (null.argmax(-1) != parent_null.argmax()).to(costs.dtype))
    return costs + float(change_cost) * changed
