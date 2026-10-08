#!/usr/bin/env python3
"""Train the three frozen full-source expert portfolios before sealed evaluation."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from decision_model.core import canonical, digest, write_json


_PORTFOLIO_PATH = Path(__file__).with_name("run_decision_stable_expert_portfolio.py")
_SPEC = importlib.util.spec_from_file_location("full_portfolio_base", _PORTFOLIO_PATH)
_BASE = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_BASE)
read, mean, load_bank = _BASE.read, _BASE.mean, _BASE.load_bank
train_portfolio, evaluate_portfolio = _BASE.train_portfolio, _BASE.evaluate_portfolio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full-expert-portfolio-v1.json")
    parser.add_argument("--output", default="runs/full-expert-portfolio-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh full-portfolio output directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Full-portfolio protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen full-portfolio source changed: " + relative)
    rows, paths, full, null, targets = load_bank(root, config)
    sources = sorted({row["source"] for row in rows})
    output.mkdir(parents=True)
    runs = []
    for seed in config["training"]["seeds"]:
        weights, available, run = train_portfolio(
            rows, paths, full, null, targets, "__sealed_final__", seed, config["training"])
        if tuple(available) != tuple(range(len(paths))):
            raise ValueError("Full portfolio did not retain every path")
        evaluations = {
            source: evaluate_portfolio(
                rows, full, null, targets, source, weights, available, config["training"])
            for source in sources
        }
        weight_record = {
            "format_version": 1, "study": config["study"], "seed": seed,
            "paths": list(paths), "weights": run["weights"],
            "training_sources": sources, "training_protocol_sha256": digest(config_path.read_bytes()),
        }
        weight_path = output / f"weights-seed-{seed}.json"
        write_json(weight_path, weight_record)
        runs.append({
            **run, "weights_file": str(weight_path.relative_to(root)),
            "weights_file_sha256": digest(weight_path.read_bytes()),
            "consumed_validation": evaluations,
            "consumed_macro": {metric: mean([value[metric] for value in evaluations.values()])
                for metric in ("accuracy_delta", "cross_entropy_delta", "brier_delta",
                               "proper_gain", "context_advantage_delta", "mean_alpha")},
        })
    checks = {
        "coverage.seeds": [run["seed"] for run in runs] == config["training"]["seeds"],
        "coverage.sources": all(run["train_sources"] == sources for run in runs),
        "coverage.paths": all(set(run["weights"]) == set(paths) for run in runs),
        "weights.simplex": all(abs(sum(run["weights"].values()) - 1.0) <= 1e-6 and
                               all(value >= 0 for value in run["weights"].values())
                               for run in runs),
        "decision.exact": all(value["accuracy_delta"] == 0.0 for run in runs
                              for value in run["consumed_validation"].values()),
        "utility.consumed_each_seed": all(
            run["consumed_macro"]["proper_gain"] >= config["screening_rule"]["proper_gain_min"]
            for run in runs),
        "context.consumed_each_seed": all(
            run["consumed_macro"]["context_advantage_delta"] >=
            config["screening_rule"]["context_min_delta"] for run in runs),
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "full consumed-source portfolios frozen before sealed-family access",
        "protocol_sha256": digest(config_path.read_bytes()), "paths": list(paths),
        "training_sources": sources, "runs": runs, "checks": checks,
        "eligible_for_sealed_evaluation": all(checks.values()),
        "limitations": config["limitations"],
    }
    write_json(output / "evidence.json", evidence)
    write_json(root / config["evidence_path"], evidence)
    write_json(output / "status.json", {
        "state": "complete", "eligible_for_sealed_evaluation": all(checks.values())})
    print(canonical({"event": "complete", "eligible": all(checks.values()),
                     "weights": {run["seed"]: run["weights_file_sha256"] for run in runs},
                     "checks": checks}))


if __name__ == "__main__":
    main()
