#!/usr/bin/env python3
"""Verify the prepared executable-outcome dataset and emit aggregate evidence."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from decision_model.core import load_rows, outcome_rewards, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mbpp-outcomes-v1/data.jsonl")
    parser.add_argument("--manifest", default="data/mbpp-outcomes-v1/manifest.json")
    parser.add_argument("--source-config", default="configs/mbpp-outcomes-source.json")
    parser.add_argument("--output", default="docs/evidence/mbpp-outcomes-v1.json")
    args = parser.parse_args()
    rows = load_rows(args.data)
    manifest = json.loads(Path(args.manifest).read_text())
    config = json.loads(Path(args.source_config).read_text())
    if manifest["source_sha256"] != config["source"]["sha256"]:
        raise ValueError("source hash differs between manifest and lock")
    if manifest["source_config_sha256"] != sha256(args.source_config):
        raise ValueError("source config hash differs from preparation manifest")
    if len(rows) != manifest["rows"] or {row["split"] for row in rows} - {"train", "validation"}:
        raise ValueError("row count or development split boundary differs")
    rewards, regrets, worst, optimal_counts = [], [], [], []
    category_counts, mutation_counts, reward_counts = Counter(), Counter(), Counter()
    reference_not_full = 0
    for row in rows:
        aligned = outcome_rewards(row)
        best = max(aligned)
        rewards.extend(aligned)
        regrets.append(best - mean(aligned))
        worst.append(min(aligned))
        optimal_counts.append(sum(value == best for value in aligned))
        for candidate in row["request"]["choices"]:
            key = candidate["id"]
            outcome = row["candidate_outcomes"][key]
            reward_counts[f'{outcome["passed"]}/{outcome["total"]}'] += 1
            category_counts.update(outcome["categories"])
            operator = row["mutation_provenance"][key]["operator"]
            mutation_counts[operator] += 1
            if operator == "reference" and outcome["reward"] != 1:
                reference_not_full += 1
    if reference_not_full or any(len(set(outcome_rewards(row))) < 2 for row in rows):
        raise ValueError("reference or reward-information invariant failed")
    evidence = {
        "format_version": 1,
        "data_sha256": sha256(args.data),
        "manifest_sha256": sha256(args.manifest),
        "source_config_sha256": sha256(args.source_config),
        "source_sha256": manifest["source_sha256"],
        "rows": len(rows),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "candidates": len(rewards),
        "candidate_count_distribution": dict(sorted(Counter(
            len(row["request"]["choices"]) for row in rows).items())),
        "reward_fraction_distribution": dict(sorted(reward_counts.items())),
        "outcome_categories": dict(sorted(category_counts.items())),
        "mutation_operators": dict(sorted(mutation_counts.items())),
        "mean_candidate_reward": mean(rewards),
        "mean_uniform_policy_regret": mean(regrets),
        "mean_worst_candidate_reward": mean(worst),
        "optimal_candidate_count_distribution": dict(sorted(Counter(optimal_counts).items())),
        "reference_passes_every_test": True,
        "every_row_has_reward_difference": True,
        "official_test_rows_present": False,
        "candidate_selection_uses_outcomes": manifest["selection_uses_outcomes"],
        "presentation_order_uses_outcomes": manifest["presentation_order_uses_outcomes"],
    }
    write_json(args.output, evidence)
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
