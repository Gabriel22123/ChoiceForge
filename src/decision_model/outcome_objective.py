"""Objectives for complete and chosen-only executable outcome feedback."""
from __future__ import annotations

import math

import torch


def _validate(logits, counts):
    if logits.ndim != 2 or len(counts) != logits.shape[0]:
        raise ValueError("expected batched logits and one candidate count per row")
    if any(not isinstance(count, int) or isinstance(count, bool) or
           not 2 <= count <= logits.shape[1] for count in counts):
        raise ValueError("invalid candidate count")


def masked_policy(logits, counts):
    _validate(logits, counts)
    mask = torch.arange(logits.shape[1], device=logits.device)[None, :] < torch.tensor(
        counts, device=logits.device)[:, None]
    return torch.softmax(logits.float().masked_fill(~mask, float("-inf")), -1), mask


def behavior_policy(logits, counts, exploration=0.2):
    """Freeze this mixture before logging outcomes so every action has support."""
    if (not isinstance(exploration, (int, float)) or isinstance(exploration, bool) or
            not math.isfinite(exploration) or not 0 < exploration <= 1):
        raise ValueError("exploration must be in (0,1]")
    probabilities, mask = masked_policy(logits, counts)
    uniform = mask / torch.tensor(counts, device=logits.device)[:, None]
    return (1 - exploration) * probabilities + exploration * uniform


def full_information_loss(logits, rewards, counts):
    """Oracle direct loss when every candidate's result is visible."""
    probabilities, mask = masked_policy(logits, counts)
    if rewards.shape != logits.shape or not torch.isfinite(rewards).all() or (
            rewards < 0).any() or (rewards > 1).any():
        raise ValueError("rewards must be a finite [0,1] tensor matching logits")
    if (rewards.masked_select(~mask) != 0).any():
        raise ValueError("padded rewards must be zero")
    return -(probabilities * rewards).sum(-1).mean()


def optimal_set_loss(logits, rewards, counts):
    """Negative log mass on every reward-maximizing candidate, including ties."""
    probabilities, mask = masked_policy(logits, counts)
    if rewards.shape != logits.shape or not torch.isfinite(rewards).all() or (
            rewards < 0).any() or (rewards > 1).any():
        raise ValueError("rewards must be a finite [0,1] tensor matching logits")
    if (rewards.masked_select(~mask) != 0).any():
        raise ValueError("padded rewards must be zero")
    maximum = rewards.masked_fill(~mask, float("-inf")).max(-1, keepdim=True).values
    optimal = mask & (rewards == maximum)
    return -(probabilities.masked_fill(~optimal, 0).sum(-1).clamp_min(1e-12).log()).mean()


def _logged(logits, actions, observed_rewards, behavior_probabilities, counts):
    probabilities, _ = masked_policy(logits, counts)
    if actions.ndim != 2 or actions.shape != observed_rewards.shape or actions.shape != behavior_probabilities.shape:
        raise ValueError("logged actions, rewards and probabilities must share [batch,samples]")
    if actions.shape[0] != logits.shape[0] or actions.shape[1] < 1:
        raise ValueError("logged batch shape differs from logits")
    count_tensor = torch.tensor(counts, device=actions.device)[:, None]
    if actions.dtype not in (torch.int32, torch.int64) or (actions < 0).any() or (actions >= count_tensor).any():
        raise ValueError("logged action is outside its candidate set")
    if (not torch.isfinite(observed_rewards).all() or (observed_rewards < 0).any() or
            (observed_rewards > 1).any()):
        raise ValueError("observed rewards must be finite values in [0,1]")
    if (not torch.isfinite(behavior_probabilities).all() or
            (behavior_probabilities <= 0).any() or (behavior_probabilities > 1).any()):
        raise ValueError("behavior probabilities must be finite values in (0,1]")
    selected = probabilities.gather(1, actions)
    return probabilities, selected


def ips_loo_loss(logits, actions, observed_rewards, behavior_probabilities, counts):
    """Chosen-only off-policy score-function loss with an independent LOO baseline.

    The importance weight is detached because this is a policy-gradient
    surrogate.  Each sample's baseline excludes that sample's action/reward.
    """
    _, selected = _logged(logits, actions, observed_rewards, behavior_probabilities, counts)
    samples = actions.shape[1]
    baseline = ((observed_rewards.sum(1, keepdim=True) - observed_rewards) / (samples - 1)
                if samples > 1 else torch.zeros_like(observed_rewards))
    weights = (selected / behavior_probabilities).detach()
    return -(weights * (observed_rewards - baseline).detach() * selected.clamp_min(1e-12).log()).mean()


def doubly_robust_loss(logits, reward_estimates, actions, observed_rewards,
                       behavior_probabilities, counts, reward_model_weight=1.0):
    """Chosen-only doubly robust policy loss plus observed-action reward fitting.

    The policy term remains unbiased for complete-information expected reward
    for any fixed reward estimate.  The reward head learns only from the logged
    actions; unchosen rewards are absent from this function's interface.
    """
    probabilities, selected = _logged(
        logits, actions, observed_rewards, behavior_probabilities, counts)
    if reward_estimates.shape != logits.shape or not torch.isfinite(reward_estimates).all() or (
            reward_estimates < 0).any() or (reward_estimates > 1).any():
        raise ValueError("reward estimates must be finite [0,1] values matching logits")
    if (not isinstance(reward_model_weight, (int, float)) or isinstance(reward_model_weight, bool) or
            not math.isfinite(reward_model_weight) or reward_model_weight < 0):
        raise ValueError("reward_model_weight must be finite and nonnegative")
    estimate = reward_estimates.gather(1, actions)
    direct = (probabilities * reward_estimates.detach()).sum(-1).mean()
    correction = ((selected / behavior_probabilities) *
                  (observed_rewards - estimate.detach())).mean()
    reward_loss = torch.nn.functional.mse_loss(estimate, observed_rewards)
    policy_loss = -direct - correction
    return policy_loss + reward_model_weight * reward_loss, {
        "policy_loss": policy_loss.detach(),
        "reward_model_loss": reward_loss.detach(),
        "mean_importance_weight": (selected / behavior_probabilities).detach().mean(),
    }
