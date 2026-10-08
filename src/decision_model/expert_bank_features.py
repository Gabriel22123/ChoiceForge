"""Candidate-order-invariant behavior features for a frozen specialist bank."""
from __future__ import annotations

from .behavioral_route_features import FEATURE_NAMES, behavioral_route_features


PARENT_FEATURE_COUNT = 7


def expert_bank_feature_names(expert_names):
    names = tuple(expert_names)
    if not names or len(names) != len(set(names)) or tuple(sorted(names)) != names:
        raise ValueError("Expert names must be unique, nonempty and sorted")
    return (FEATURE_NAMES[:PARENT_FEATURE_COUNT] + tuple(
        f"{expert}.{name}"
        for expert in names for name in FEATURE_NAMES[PARENT_FEATURE_COUNT:]))


def expert_bank_features(parent, parent_null, corrections, null_corrections):
    """Concatenate shared-parent and per-expert behavior without text or labels."""
    import torch
    names = tuple(sorted(corrections))
    if (not names or set(corrections) != set(null_corrections) or
            any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("Expert-bank corrections are incomplete")
    blocks = []
    for name in names:
        block = behavioral_route_features(
            parent, parent_null, corrections[name], null_corrections[name])
        blocks.append(block)
    output = torch.cat((blocks[0][:PARENT_FEATURE_COUNT],
                        *(block[PARENT_FEATURE_COUNT:] for block in blocks)))
    if (len(output) != len(expert_bank_feature_names(names)) or
            not torch.isfinite(output).all()):
        raise ValueError("Invalid expert-bank behavior vector")
    return output


def standardize_bank_features(training, values):
    import torch
    if (not torch.is_tensor(training) or training.ndim != 2 or len(training) < 2 or
            not torch.isfinite(training).all() or
            any(not torch.is_tensor(value) or value.ndim != 2 or
                value.shape[1] != training.shape[1] or not torch.isfinite(value).all()
                for value in values)):
        raise ValueError("Invalid expert-bank feature matrices")
    mean = training.mean(0)
    scale = training.std(0, unbiased=False).clamp_min(1e-6)
    return [(value - mean) / scale for value in values], mean, scale
