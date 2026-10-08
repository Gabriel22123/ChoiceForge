"""Distribution-aware feasibility and preference rules for specialist routing."""
from __future__ import annotations


def path_statistics(full_logits, null_logits, targets, brier_weight=0.5):
    import torch
    if (len(full_logits) != len(null_logits) or len(full_logits) < 2 or
            len(targets) != len(full_logits)):
        raise ValueError("Invalid safe-routing path tensors")
    full = torch.as_tensor(full_logits, dtype=torch.float64)
    null = torch.as_tensor(null_logits, dtype=torch.float64)
    target = torch.as_tensor(targets, dtype=torch.float64)
    if (not torch.isfinite(full).all() or not torch.isfinite(null).all() or
            not torch.isfinite(target).all() or (target < 0).any() or
            not torch.isclose(target.sum(), target.new_tensor(1.0), atol=1e-5, rtol=0)):
        raise ValueError("Safe-routing tensors must be finite distributions")
    probabilities = full.softmax(-1)
    full_log, null_log = full.log_softmax(-1), null.log_softmax(-1)
    cross_entropy = float(-(target * full_log).sum())
    brier = float(((probabilities - target) ** 2).sum())
    context = float((target * (full_log - null_log)).sum())
    return {
        "correct": int(full.argmax()) == int(target.argmax()),
        "cross_entropy": cross_entropy,
        "brier": brier,
        "context_advantage": context,
        "proper_cost": cross_entropy + float(brier_weight) * brier,
    }


def safe_improvement(parent, expert, tolerance=1e-12):
    eligible = (
        (not parent["correct"] or expert["correct"]) and
        expert["cross_entropy"] <= parent["cross_entropy"] + tolerance and
        expert["brier"] <= parent["brier"] + tolerance and
        expert["context_advantage"] + tolerance >= parent["context_advantage"]
    )
    improvement = parent["proper_cost"] - expert["proper_cost"]
    return {"eligible": bool(eligible and improvement > tolerance),
            "improvement": improvement}


def choose_safe_path(parent, experts):
    ranked = []
    for name, statistics in experts.items():
        result = safe_improvement(parent, statistics)
        if result["eligible"]:
            ranked.append((result["improvement"], name))
    if not ranked:
        return "parent", 0.0
    improvement, name = sorted(ranked, key=lambda value: (-value[0], value[1]))[0]
    return name, improvement
