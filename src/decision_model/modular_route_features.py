"""Fixed parent-plus-one-expert features for modular route models."""
from __future__ import annotations

from .behavioral_route_features import FEATURE_NAMES
from .expert_bank_features import PARENT_FEATURE_COUNT


def independent_expert_features(combined, expert_names, expert_name, semantic_width):
    """Select the parent, one expert block and shared semantic suffix."""
    import torch
    names = tuple(expert_names)
    if (not torch.is_tensor(combined) or combined.ndim != 1 or
            tuple(sorted(names)) != names or len(names) != len(set(names)) or
            expert_name not in names or semantic_width < 0):
        raise ValueError("Invalid independent expert feature request")
    expert_width = len(FEATURE_NAMES) - PARENT_FEATURE_COUNT
    behavior_width = PARENT_FEATURE_COUNT + len(names) * expert_width
    if len(combined) != behavior_width + semantic_width:
        raise ValueError("Combined expert-bank feature width differs")
    index = names.index(expert_name)
    start = PARENT_FEATURE_COUNT + index * expert_width
    return torch.cat((combined[:PARENT_FEATURE_COUNT],
                      combined[start:start + expert_width],
                      combined[behavior_width:]))
