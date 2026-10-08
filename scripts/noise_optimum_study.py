"""Binary population CE: noisy training optimum vs clean-logit inference.

This exactly specified toy problem does not use or evaluate trained model weights.
Gauss-Hermite quadrature is checked at two resolutions.
"""
import argparse
import math
from pathlib import Path

import numpy as np

from decision_model.core import write_json


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def solve(target, sigma, order=96):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    weights = weights / math.sqrt(math.pi)
    # Independent N(0,sigma²) noise on two logits means difference noise
    # N(0,2*sigma²), hence 2*sigma*x for Hermite's exp(-x²) quadrature.
    noise = 2 * sigma * nodes
    lo, hi = -30., 30.
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.dot(weights, sigmoid(mid + noise)) < target:
            lo = mid
        else:
            hi = mid
    optimum = (lo + hi) / 2
    clean = float(sigmoid(optimum))
    expected = float(np.dot(weights, sigmoid(optimum + noise)))
    return {"true_probability": target, "sigma_per_logit": sigma,
            "optimum_logit_difference": optimum,
            "clean_inference_probability": clean,
            "noise_averaged_probability": expected,
            "clean_probability_error": clean - target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/noise-optimum-v1")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new output directory")
    rows = []
    for target in (.5, .7, .9):
        for sigma in (0., .1, .4, 1.):
            row = solve(target, sigma)
            check = solve(target, sigma, order=192)
            row["quadrature_probability_difference"] = abs(row["clean_inference_probability"] - check["clean_inference_probability"])
            if row["quadrature_probability_difference"] > 1e-8:
                raise ValueError("Quadrature has not converged")
            rows.append(row)
    output.mkdir(parents=True)
    result = {"objective": "population binary CE with independent Gaussian noise on two logits",
              "inference_modes": ["softmax of unperturbed logits", "expectation of perturbed softmax"],
              "quadrature_orders": [96, 192], "results": rows,
              "scope": "Illustrates a train/inference mismatch for a known binary population; not an undisclosed algorithm or a model accuracy benchmark"}
    write_json(output / "results.json", result)
    print("true_p sigma clean_p averaged_p clean_error")
    for row in rows:
        print(" ".join(f"{row[key]:.6f}" for key in ("true_probability", "sigma_per_logit", "clean_inference_probability", "noise_averaged_probability", "clean_probability_error")))


if __name__ == "__main__":
    main()
