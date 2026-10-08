"""Signed counterfactual utility targets and route-regret metrics."""
from __future__ import annotations

import math

from .router_cross_validation import binary_route_metrics


def signed_route_utility(costs, gates):
    """Return positive utility when the expert costs less than the parent."""
    import torch
    if (not torch.is_tensor(costs) or not torch.is_tensor(gates) or
            costs.ndim != 1 or gates.ndim != 1 or costs.shape != gates.shape or
            len(costs) != 2 or not torch.isfinite(costs).all() or
            not torch.equal(gates, gates.new_tensor([0.0, 1.0]))):
        raise ValueError("Signed route utility requires finite costs at gates [0, 1]")
    utility = float(costs[0] - costs[1])
    return {
        "utility": utility,
        "transformed_utility": math.asinh(utility),
        "closed_cost": float(costs[0]),
        "open_cost": float(costs[1]),
        "gate": int(utility > 0.0),
    }


def utility_route_metrics(utilities, predicted_transformed_utilities):
    """Measure sign decisions, regret and transformed-utility accuracy."""
    if len(utilities) != len(predicted_transformed_utilities) or not utilities:
        raise ValueError("Utility route metrics require aligned nonempty values")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or
           not math.isfinite(value) for value in (*utilities, *predicted_transformed_utilities)):
        raise ValueError("Invalid route utilities")
    targets = [int(value > 0.0) for value in utilities]
    probabilities = [1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, float(value)))))
                     for value in predicted_transformed_utilities]
    binary = binary_route_metrics(targets, probabilities)
    predictions = [value > 0.0 for value in predicted_transformed_utilities]
    regret = mean([
        abs(float(utility)) if prediction != (utility > 0.0) else 0.0
        for utility, prediction in zip(utilities, predictions)
    ])
    transformed = [math.asinh(float(value)) for value in utilities]
    return {
        **binary,
        "mean_regret": regret,
        "always_parent_regret": mean([max(float(value), 0.0) for value in utilities]),
        "always_expert_regret": mean([max(-float(value), 0.0) for value in utilities]),
        "transformed_mae": mean([
            abs(actual - float(predicted))
            for actual, predicted in zip(transformed, predicted_transformed_utilities)
        ]),
        "zero_prediction_mae": mean([abs(value) for value in transformed]),
        "mean_utility": mean(list(map(float, utilities))),
        "mean_predicted_transformed_utility": mean(
            list(map(float, predicted_transformed_utilities))),
    }


def mean(values):
    return sum(values) / len(values)
