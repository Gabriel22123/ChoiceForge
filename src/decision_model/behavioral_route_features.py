"""Candidate-order-invariant behavior features for expert-use routing."""
from __future__ import annotations


FEATURE_NAMES = (
    "candidate_count_log",
    "parent_entropy_norm",
    "parent_top_probability",
    "parent_margin",
    "parent_null_entropy_norm",
    "parent_null_top_probability",
    "parent_null_margin",
    "expert_entropy_norm",
    "expert_top_probability",
    "expert_margin",
    "expert_null_entropy_norm",
    "expert_null_top_probability",
    "expert_null_margin",
    "parent_expert_js",
    "parent_null_expert_null_js",
    "parent_full_null_js",
    "expert_full_null_js",
    "full_argmax_flip",
    "null_argmax_flip",
    "residual_centered_rms",
    "residual_range",
    "null_residual_centered_rms",
    "null_residual_range",
    "residual_path_difference_rms",
)


def _distribution_stats(logits):
    import math
    probabilities = logits.softmax(-1)
    ordered = probabilities.sort(descending=True).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return probabilities, (
        entropy / math.log(len(logits)),
        ordered[0],
        ordered[0] - ordered[1],
    )


def _js(left, right):
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum() +
        (right * (right.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum())


def _top_set_flip(left, right):
    """Return one only when the two maximizer sets are disjoint."""
    return float(not bool(((left == left.max()) & (right == right.max())).any()))


def behavioral_route_features(parent, parent_null, correction, null_correction):
    """Summarize parent/expert behavior without source, IDs, text or labels."""
    import math
    import torch
    values = (parent, parent_null, correction, null_correction)
    if (any(not torch.is_tensor(value) or value.ndim != 1 for value in values) or
            any(value.shape != parent.shape for value in values[1:]) or
            len(parent) < 2 or any(not torch.isfinite(value).all() for value in values)):
        raise ValueError("Invalid behavior-route logits")
    expert, expert_null = parent + correction, parent_null + null_correction
    parent_p, parent_stats = _distribution_stats(parent)
    parent_null_p, parent_null_stats = _distribution_stats(parent_null)
    expert_p, expert_stats = _distribution_stats(expert)
    expert_null_p, expert_null_stats = _distribution_stats(expert_null)
    centered = correction - correction.mean()
    null_centered = null_correction - null_correction.mean()
    features = (
        parent.new_tensor(math.log(len(parent))),
        *parent_stats,
        *parent_null_stats,
        *expert_stats,
        *expert_null_stats,
        _js(parent_p, expert_p),
        _js(parent_null_p, expert_null_p),
        _js(parent_p, parent_null_p),
        _js(expert_p, expert_null_p),
        parent.new_tensor(_top_set_flip(parent, expert)),
        parent.new_tensor(_top_set_flip(parent_null, expert_null)),
        centered.square().mean().sqrt(),
        correction.max() - correction.min(),
        null_centered.square().mean().sqrt(),
        null_correction.max() - null_correction.min(),
        (centered - null_centered).square().mean().sqrt(),
    )
    output = torch.stack(features).to(dtype=torch.float32)
    if len(output) != len(FEATURE_NAMES) or not torch.isfinite(output).all():
        raise ValueError("Invalid behavior-route feature vector")
    return output


def standardize_behavior_features(training, values):
    """Fit standardization on training rows and transform aligned matrices."""
    import torch
    if (not torch.is_tensor(training) or training.ndim != 2 or len(training) < 2 or
            training.shape[1] != len(FEATURE_NAMES) or not torch.isfinite(training).all() or
            any(not torch.is_tensor(value) or value.ndim != 2 or
                value.shape[1] != training.shape[1] or not torch.isfinite(value).all()
                for value in values)):
        raise ValueError("Invalid behavior-route feature matrices")
    mean = training.mean(0)
    scale = training.std(0, unbiased=False).clamp_min(1e-6)
    return [(value - mean) / scale for value in values], mean, scale
