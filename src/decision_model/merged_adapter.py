"""Exact fusion of convex portfolios of frozen residual expert heads."""
from __future__ import annotations


def build_merged_residual_adapter(hidden_size, total_width, eps=1e-5):
    """Build the compact architecture used by an exactly fused portfolio.

    Expert-specific affine LayerNorm parameters are folded into each first
    linear block.  The merged LayerNorm therefore keeps only the shared
    normalization operation and has no trainable affine parameters.
    """
    from torch import nn
    if (not isinstance(hidden_size, int) or not isinstance(total_width, int) or
            min(hidden_size, total_width) < 1 or not 0.0 < float(eps) < 1.0):
        raise ValueError("Invalid merged residual adapter dimensions")
    return nn.Sequential(
        nn.LayerNorm(hidden_size, eps=float(eps), elementwise_affine=False),
        nn.Linear(hidden_size, total_width), nn.GELU(), nn.Linear(total_width, 1))


def fuse_routed_residual_experts(modules, weights):
    """Fuse weighted CandidateHead residuals without using routers or labels."""
    import torch
    if (not isinstance(modules, dict) or not modules or set(modules) != set(weights) or
            any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
                for value in weights.values())):
        raise ValueError("Merged residual experts and weights differ")
    names = tuple(sorted(modules))
    layer_norms = [modules[name].residual[0] for name in names]
    first = [modules[name].residual[1] for name in names]
    final = [modules[name].residual[4] for name in names]
    hidden = first[0].in_features
    eps = float(layer_norms[0].eps)
    if (any(layer.out_features != 1 for layer in final) or
            any(layer.in_features != hidden for layer in first) or
            any(not layer.elementwise_affine or float(layer.eps) != eps or
                tuple(layer.normalized_shape) != (hidden,) for layer in layer_norms)):
        raise ValueError("Residual expert architectures cannot be fused")
    widths = [layer.out_features for layer in first]
    merged = build_merged_residual_adapter(hidden, sum(widths), eps)
    first_weights, first_biases, final_weights = [], [], []
    final_bias = torch.zeros_like(final[0].bias.detach())
    with torch.no_grad():
        for name, norm, input_layer, output_layer in zip(
                names, layer_norms, first, final):
            gamma, beta = norm.weight.detach(), norm.bias.detach()
            matrix, bias = input_layer.weight.detach(), input_layer.bias.detach()
            first_weights.append(matrix * gamma.unsqueeze(0))
            first_biases.append(bias + matrix @ beta)
            final_weights.append(output_layer.weight.detach() * float(weights[name]))
            final_bias.add_(output_layer.bias.detach() * float(weights[name]))
        merged[1].weight.copy_(torch.cat(first_weights, dim=0))
        merged[1].bias.copy_(torch.cat(first_biases, dim=0))
        merged[3].weight.copy_(torch.cat(final_weights, dim=1))
        merged[3].bias.copy_(final_bias)
    merged.eval()
    return merged, {
        "expert_order": list(names), "expert_widths": dict(zip(names, widths)),
        "hidden_size": hidden, "total_width": sum(widths), "layer_norm_eps": eps,
    }
