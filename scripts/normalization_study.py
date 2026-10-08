"""Check whether within-group cost normalization preserves a proper-risk optimum.

This is an independent synthetic counterexample, not an evaluation of any
vendor's undisclosed training algorithm. Labels are integrated with known p.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from decision_model.core import write_json
from noise_optimum_study import solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/normalization-v1")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new output directory")
    torch.set_num_threads(2)
    repeats = 16384
    generator = torch.Generator().manual_seed(20260922)
    rows = []
    for target in (.5, .7, .9):
        for sigma in (.1, .4):
            optimum = solve(target, sigma)
            difference = optimum["optimum_logit_difference"]
            for group in (4, 16, 64):
                # Equivalent to independently perturbing two logits by N(0,sigma²).
                noise = torch.randn(repeats, group, generator=generator, dtype=torch.float64)
                actions = difference + math.sqrt(2) * sigma * noise
                score = noise / (math.sqrt(2) * sigma)
                probability = actions.sigmoid()
                costs = ((target, torch.nn.functional.softplus(-actions)),
                         (1 - target, torch.nn.functional.softplus(actions)))
                estimates = {name: torch.zeros(repeats, dtype=torch.float64) for name in
                             ("loo_score", "centered_score", "standardized_score")}
                for weight, cost in costs:
                    mean = cost.mean(1, keepdim=True)
                    loo = (cost.sum(1, keepdim=True) - cost) / (group - 1)
                    advantage = (cost - mean) / (cost.std(1, keepdim=True, unbiased=False) + 1e-8)
                    estimates["loo_score"] += weight * ((cost - loo) * score).mean(1)
                    estimates["centered_score"] += weight * ((cost - mean) * score).mean(1)
                    estimates["standardized_score"] += weight * (advantage * score).mean(1)
                estimates["pathwise"] = (probability - target).mean(1)
                measured = {name: {"mean_gradient": value.mean().item(),
                                   "standard_error": (value.std(unbiased=True) / repeats**.5).item()}
                            for name, value in estimates.items()}
                rows.append({"true_probability": target, "sigma_per_logit": sigma,
                             "samples": group, "optimum_logit_difference": difference,
                             "clean_probability_at_optimum": optimum["clean_inference_probability"],
                             "true_risk_gradient_at_optimum": optimum["noise_averaged_probability"] - target,
                             "estimators": measured})
    output.mkdir(parents=True)
    result = {"seed": 20260922, "independent_groups": repeats,
              "objective": "population binary CE with Gaussian logit noise, evaluated at its population optimum",
              "label_integration": "integrate y=1 and y=0 with known true p; normalize separately per observed label before averaging",
              "common_noise_across_estimators": True,
              "normalization": "within-group population std + 1e-8; no clipping or CE anchor",
              "gradient_direction": "cost minimization; negative means gradient descent increases the class-1 logit difference",
              "scope": "Synthetic local-stationarity check. Does not claim all policy gradient methods are biased or diagnose an external implementation end-to-end.",
              "results": rows}
    write_json(output / "results.json", result)
    representative = next(row for row in rows if row["true_probability"] == .9
                          and row["sigma_per_logit"] == .4 and row["samples"] == 16)
    print(json.dumps(representative, indent=2))


if __name__ == "__main__":
    main()
