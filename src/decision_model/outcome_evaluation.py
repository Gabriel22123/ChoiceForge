"""Utility metrics for executable candidate-outcome decisions."""
from __future__ import annotations

import math

from .core import outcome_rewards


def outcome_metrics(rows, logits):
    if not rows or len(rows) != len(logits):
        raise ValueError("expected nonempty paired rows and logits")
    import torch
    regrets, expected, hits, full, selected_rewards = [], [], [], [], []
    per_row = []
    for row, values in zip(rows, logits):
        count = len(row["request"]["choices"])
        if len(values) != count or any(not isinstance(value, (int, float)) or
                                       not math.isfinite(value) for value in values):
            raise ValueError("invalid outcome logits")
        rewards = outcome_rewards(row)
        probabilities = torch.softmax(torch.tensor(values, dtype=torch.float64), -1).tolist()
        selected = max(range(count), key=values.__getitem__)
        best = max(rewards); selected_reward = rewards[selected]
        regret = best - selected_reward
        policy_reward = sum(p * reward for p, reward in zip(probabilities, rewards))
        hit = selected_reward == best
        regrets.append(regret); expected.append(policy_reward); hits.append(hit)
        full.append(selected_reward == 1.0); selected_rewards.append(selected_reward)
        per_row.append({"id": row["id"], "selected_index": selected,
                        "selected_reward": selected_reward, "best_reward": best,
                        "regret": regret, "policy_expected_reward": policy_reward,
                        "optimal_set_hit": hit})
    count = len(rows)
    return {
        "n": count,
        "mean_decision_regret": sum(regrets) / count,
        "mean_policy_expected_reward": sum(expected) / count,
        "mean_selected_reward": sum(selected_rewards) / count,
        "optimal_set_hit_rate": sum(hits) / count,
        "full_pass_selection_rate": sum(full) / count,
    }, per_row
