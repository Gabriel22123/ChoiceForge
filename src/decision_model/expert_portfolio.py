"""Source-robust expert portfolios with label-free decision stability."""
from __future__ import annotations


def validate_alpha_grid(alphas):
    values = tuple(float(value) for value in alphas)
    if (not values or any(not 0.0 < value <= 1.0 for value in values) or
            tuple(sorted(set(values))) != values):
        raise ValueError("Decision-stability alphas must be unique, sorted and in (0, 1]")
    return values


def decision_stable_projection(parent, parent_null, portfolio, portfolio_null, alphas):
    """Take the largest portfolio step that preserves the parent's top choice.

    The selected alpha is computed from the full-context logits and is shared
    with the null-context path. This keeps the output decision unchanged and
    prevents the context metric from receiving a different route decision.
    """
    import torch
    values = (parent, parent_null, portfolio, portfolio_null)
    if (any(not torch.is_tensor(value) or value.ndim != 1 for value in values) or
            any(value.shape != parent.shape for value in values[1:]) or
            len(parent) < 2 or any(not torch.isfinite(value).all() for value in values)):
        raise ValueError("Invalid expert-portfolio logits")
    grid = validate_alpha_grid(alphas)
    selected_full, selected_null = parent, parent_null
    selected_alpha = 0.0
    parent_choice = int(parent.argmax())
    for alpha in grid:
        full = (1.0 - alpha) * parent + alpha * portfolio
        if int(full.argmax()) == parent_choice:
            selected_full = full
            selected_null = (1.0 - alpha) * parent_null + alpha * portfolio_null
            selected_alpha = alpha
    return selected_full, selected_null, selected_alpha


def source_robust_objective(source_costs, parent_costs, source_context,
                            parent_context, proper_penalty, context_penalty):
    """Equal-source risk plus penalties for any training-source regression."""
    import torch
    values = (source_costs, parent_costs, source_context, parent_context)
    if (any(not torch.is_tensor(value) or value.ndim != 1 for value in values) or
            any(value.shape != source_costs.shape for value in values[1:]) or
            len(source_costs) < 2 or any(not torch.isfinite(value).all() for value in values) or
            min(float(proper_penalty), float(context_penalty)) < 0):
        raise ValueError("Invalid source-robust portfolio objective")
    proper_regression = torch.relu(source_costs - parent_costs)
    context_regression = torch.relu(parent_context - source_context)
    loss = source_costs.mean()
    loss = loss + float(proper_penalty) * (
        proper_regression.square().mean() + proper_regression.max().square())
    loss = loss + float(context_penalty) * (
        context_regression.square().mean() + context_regression.max().square())
    return loss, {
        "proper_regression_max": proper_regression.max(),
        "context_regression_max": context_regression.max(),
    }
