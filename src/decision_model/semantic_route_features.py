"""Candidate-order-invariant semantic summaries for frozen route features."""
from __future__ import annotations


def fixed_projection(input_width, output_width=64, seed=1701, *, dtype=None):
    import math
    import torch
    if min(input_width, output_width) < 1:
        raise ValueError("Projection widths must be positive")
    generator = torch.Generator(device="cpu"); generator.manual_seed(int(seed))
    return torch.randn(input_width, output_width, generator=generator,
                       dtype=dtype or torch.float32) / math.sqrt(output_width)


def semantic_route_features(full, null, projection):
    """Project candidate states, then summarize mean/std without candidate order."""
    import torch
    if (not torch.is_tensor(full) or not torch.is_tensor(null) or
            not torch.is_tensor(projection) or full.ndim != 2 or null.ndim != 2 or
            full.shape != null.shape or len(full) < 2 or full.shape[1] != projection.shape[0] or
            projection.ndim != 2 or not torch.isfinite(full).all() or
            not torch.isfinite(null).all() or not torch.isfinite(projection).all()):
        raise ValueError("Invalid semantic route tensors")
    full_projected = full @ projection
    null_projected = null @ projection
    output = torch.cat((full_projected.mean(0), full_projected.std(0, unbiased=False),
                        null_projected.mean(0), null_projected.std(0, unbiased=False)))
    if not torch.isfinite(output).all():
        raise ValueError("Semantic route features must be finite")
    return output
