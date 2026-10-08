"""Behavior-derived gate targets for a frozen additive expert."""
from __future__ import annotations

import math


def composite_gate_costs(parent, parent_null, correction, null_correction, target,
                         gates, *, brier_weight=0.5, context_weight=0.25,
                         context_gap=0.05, teacher=None, null_teacher=None,
                         distillation_weight=1.0, null_distillation_weight=1.0):
    """Return the per-gate task and preservation cost for one request."""
    import torch
    values = (parent, parent_null, correction, null_correction)
    if (any(value.ndim != 1 for value in values) or
            any(value.shape != parent.shape for value in values[1:]) or
            len(parent) < 2 or not 0 <= int(target) < len(parent) or
            gates.ndim != 1 or len(gates) < 2 or
            not torch.isfinite(gates).all() or (gates < 0).any() or (gates > 1).any()):
        raise ValueError("Invalid counterfactual routing tensors")
    if any(not math.isfinite(value) or value < 0 for value in (
            brier_weight, context_weight, context_gap,
            distillation_weight, null_distillation_weight)):
        raise ValueError("Invalid counterfactual routing weights")
    logits = parent.unsqueeze(0) + gates.unsqueeze(1) * correction.unsqueeze(0)
    null_logits = parent_null.unsqueeze(0) + gates.unsqueeze(1) * null_correction.unsqueeze(0)
    probabilities = logits.softmax(-1)
    one_hot = torch.nn.functional.one_hot(
        torch.tensor(int(target), device=logits.device), len(parent)).to(logits.dtype)
    costs = (-logits.log_softmax(-1)[:, int(target)] + float(brier_weight) *
             ((probabilities - one_hot) ** 2).sum(-1))
    for target_distribution, candidate_logits, weight in (
            (teacher, logits, distillation_weight),
            (null_teacher, null_logits, null_distillation_weight)):
        if target_distribution is None:
            continue
        if (target_distribution.ndim != 1 or target_distribution.shape != parent.shape or
                not torch.isfinite(target_distribution).all() or
                (target_distribution < 0).any() or
                not torch.isclose(target_distribution.sum(), target_distribution.new_tensor(1.0),
                                  atol=1e-5, rtol=0)):
            raise ValueError("Invalid counterfactual teacher distribution")
        log_target = torch.where(target_distribution > 0, target_distribution.log(),
                                 target_distribution.new_zeros(()))
        costs = costs + float(weight) * (
            target_distribution * (log_target - candidate_logits.log_softmax(-1))).sum(-1)
    advantage = (logits.log_softmax(-1)[:, int(target)] -
                 null_logits.log_softmax(-1)[:, int(target)])
    costs = costs + float(context_weight) * torch.relu(
        logits.new_tensor(float(context_gap)) - advantage)
    return costs


def optimal_gate(costs, gates):
    """Choose the smallest grid value attaining the minimum cost."""
    import torch
    if (costs.ndim != 1 or gates.ndim != 1 or costs.shape != gates.shape or
            not torch.isfinite(costs).all() or
            not torch.equal(gates, gates.sort().values)):
        raise ValueError("Invalid counterfactual gate grid or costs")
    index = int(torch.argmin(costs))
    return {"gate": float(gates[index]), "cost": float(costs[index]),
            "closed_cost": float(costs[0]), "improvement": float(costs[0] - costs[index]),
            "index": index}
