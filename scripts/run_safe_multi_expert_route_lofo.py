#!/usr/bin/env python3
"""Multi-seed leave-one-family-out validation for a safe multi-expert router."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn

from decision_model.core import canonical, digest, load_rows, target_distribution, write_json
from decision_model.expert_bank_features import (
    expert_bank_feature_names, expert_bank_features, standardize_bank_features)
from decision_model.feature_screen import parameter_sha256
from decision_model.safe_routing import choose_safe_path, path_statistics
from decision_model.train import evaluate_rows


_ORACLE_PATH = Path(__file__).with_name("run_distribution_safe_specialist_oracle.py")
_ORACLE_SPEC = importlib.util.spec_from_file_location("safe_multi_route_oracle_helper", _ORACLE_PATH)
_ORACLE = importlib.util.module_from_spec(_ORACLE_SPEC); _ORACLE_SPEC.loader.exec_module(_ORACLE)
load_features, load_module, collect_paths = (
    _ORACLE.load_features, _ORACLE.load_module, _ORACLE.collect_paths)


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    return sum(values) / len(values)


def build_router(input_width, hidden_width, output_width):
    if min(input_width, hidden_width, output_width) < 1:
        raise ValueError("Invalid multi-expert router dimensions")
    return nn.Sequential(nn.Linear(input_width, hidden_width), nn.GELU(),
                         nn.Linear(hidden_width, output_width))


def class_weights(labels, classes, cap):
    counts = Counter(labels); total = len(labels)
    values = []
    for index in range(classes):
        count = counts.get(index, 0)
        values.append(0.0 if count == 0 else math.sqrt(total / (classes * count)))
    active = [value for value in values if value]
    scale = mean(active)
    return torch.tensor([min(cap, value / scale) if value else 0.0 for value in values])


def source_balanced_plan(rows, heldout_source, seed, updates, batch_size):
    groups = defaultdict(list)
    for row in rows:
        if row["source"] != heldout_source:
            groups[row["source"]].append(row["id"])
    if len(groups) < 2 or any(not values for values in groups.values()):
        raise ValueError("Source-balanced route plan needs multiple training sources")
    for values in groups.values():
        values.sort()
    rng = random.Random(seed); sources = sorted(groups)
    plan = []
    for update in range(updates):
        source_order = [sources[index % len(sources)] for index in range(batch_size)]
        rng.shuffle(source_order)
        row_ids = [rng.choice(groups[source]) for source in source_order]
        plan.append({"update": update + 1, "row_ids": row_ids})
    return plan


def prepare_records(rows, dataset, manifest, expert_paths, brier_weight):
    records = {}
    names = tuple(sorted(expert_paths))
    for index, row in enumerate(rows):
        key = f"{dataset}:{row['id']}"
        parent_full = manifest["rows"][key]["parent_logits"]
        parent_null = manifest["rows"][key]["parent_null_logits"]
        targets = target_distribution(row)
        parent_stats = path_statistics(parent_full, parent_null, targets, brier_weight)
        expert_stats, corrections, null_corrections = {}, {}, {}
        for name in names:
            full, null = expert_paths[name]
            expert_stats[name] = path_statistics(full[index], null[index], targets, brier_weight)
            corrections[name] = torch.tensor(full[index]) - torch.tensor(parent_full)
            null_corrections[name] = torch.tensor(null[index]) - torch.tensor(parent_null)
        target_path, improvement = choose_safe_path(parent_stats, expert_stats)
        records[row["id"]] = {
            "features": expert_bank_features(
                torch.tensor(parent_full), torch.tensor(parent_null),
                corrections, null_corrections),
            "target_path": target_path, "oracle_improvement": improvement,
            "parent": parent_stats, "experts": expert_stats,
            "parent_logits": parent_full,
            "expert_logits": {name: expert_paths[name][0][index] for name in names},
        }
    if len(records) != len(rows):
        raise ValueError("Multi-expert route record coverage differs")
    return records


def train_fold(rows, heldout_source, records, paths, feature_map, config, seed, output):
    training_rows = [row for row in rows if row["split"] == "train" and
                     row["source"] != heldout_source]
    labels = [paths.index(records[row["id"]]["target_path"]) for row in training_rows]
    weights = class_weights(labels, len(paths), config["class_weight_cap"])
    plan = source_balanced_plan(rows=[row for row in rows if row["split"] == "train"],
                                heldout_source=heldout_source, seed=seed,
                                updates=config["updates"], batch_size=config["batch_size"])
    plan_path = output / "plans" / f"{heldout_source}-{seed}.jsonl"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("".join(canonical(item) + "\n" for item in plan))
    by_id = {row["id"]: row for row in training_rows}
    torch.manual_seed(seed)
    module = build_router(len(next(iter(feature_map.values()))),
                          config["hidden_width"], len(paths))
    optimizer = torch.optim.AdamW(module.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], eps=1e-6)
    losses, norms, started = [], [], time.monotonic(); module.train()
    for item in plan:
        batch = [by_id[row_id] for row_id in item["row_ids"]]
        matrix = torch.stack([feature_map[row["id"]] for row in batch])
        target = torch.tensor([paths.index(records[row["id"]]["target_path"])
                               for row in batch])
        loss = torch.nn.functional.cross_entropy(module(matrix), target, weight=weights)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), config["gradient_clip"])
        optimizer.step(); losses.append(float(loss.detach())); norms.append(float(norm))
    return module, {
        "seed": seed, "heldout_source": heldout_source, "updates": len(plan),
        "examples": len(plan) * config["batch_size"],
        "seconds": time.monotonic() - started,
        "plan_sha256": digest(plan_path.read_bytes()),
        "class_counts": dict(Counter(paths[index] for index in labels)),
        "class_weights": {path: float(weights[index]) for index, path in enumerate(paths)},
        "loss_first": losses[0], "loss_last": losses[-1],
        "gradient_norm_mean": mean(norms), "gradient_norm_maximum": max(norms),
        "router_sha256": parameter_sha256(module),
    }


def evaluate_fold(module, rows, heldout_source, records, paths, standardized, threshold):
    heldout = [row for row in rows if row["split"] == "validation" and
               row["source"] == heldout_source]
    matrix = torch.stack([standardized[row["id"]] for row in heldout])
    module.eval()
    with torch.no_grad():
        probabilities = module(matrix).softmax(-1)
    predictions, logits, violations, gains, oracle, matches = [], [], Counter(), [], [], []
    for index, row in enumerate(heldout):
        confidence, selected = probabilities[index].max(0)
        path = paths[int(selected)] if float(confidence) >= threshold else "parent"
        record = records[row["id"]]
        statistics = record["parent"] if path == "parent" else record["experts"][path]
        parent = record["parent"]
        violations["parent_correct_lost"] += int(parent["correct"] and not statistics["correct"])
        violations["cross_entropy_regressed"] += int(
            statistics["cross_entropy"] > parent["cross_entropy"] + 1e-12)
        violations["brier_regressed"] += int(statistics["brier"] > parent["brier"] + 1e-12)
        violations["context_regressed"] += int(
            statistics["context_advantage"] + 1e-12 < parent["context_advantage"])
        predictions.append(path); matches.append(path == record["target_path"])
        gains.append(parent["proper_cost"] - statistics["proper_cost"])
        oracle.append(record["oracle_improvement"])
        logits.append(record["parent_logits"] if path == "parent"
                      else record["expert_logits"][path])
    parent_logits = [records[row["id"]]["parent_logits"] for row in heldout]
    parent_metrics, _ = evaluate_rows(heldout, parent_logits, 1.0)
    routed_metrics, _ = evaluate_rows(heldout, logits, 1.0)
    return {
        "rows": len(heldout), "path_match": mean(matches),
        "selection": dict(Counter(predictions)),
        "mean_proper_gain": mean(gains), "mean_oracle_improvement": mean(oracle),
        "mean_regret": mean([best - gain for best, gain in zip(oracle, gains)]),
        "accuracy_delta": routed_metrics["accuracy"] - parent_metrics["accuracy"],
        "cross_entropy_delta": routed_metrics["nll"] - parent_metrics["nll"],
        "violation_count": sum(violations.values()), "violations": dict(violations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe-multi-expert-route-lofo-v1.json")
    parser.add_argument("--output", default="runs/safe-multi-expert-route-lofo-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh safe multi-expert route directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Safe multi-expert route protocol was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen multi-expert route source changed: " + relative)
    data_path = root / config["data"]["path"]
    if digest(data_path.read_bytes()) != config["data"]["sha256"]:
        raise ValueError("Frozen multi-expert route data changed")
    rows = [row for row in load_rows(data_path) if row["split"] in ("train", "validation") and
            row["source"] in config["data"]["sources"]]
    counts = Counter((row["source"], row["split"]) for row in rows)
    if counts != {tuple(key.split("|")): value for key, value in config["data"]["counts"].items()}:
        raise ValueError("Multi-expert route data coverage differs")
    cache = root / config["feature_cache"]["path"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_cache"]["manifest_sha256"]:
        raise ValueError("Frozen multi-expert route feature manifest changed")
    dataset = config["data"]["dataset"]
    keys = [f"{dataset}:{row['id']}" for row in rows]
    manifest, features = load_features(cache, keys)
    model = {**config["model"], "hidden_size": manifest["hidden_size"]}
    expert_paths = {}
    for name, endpoint in config["experts"].items():
        weights = root / endpoint["weights"]
        if digest(weights.read_bytes()) != endpoint["weights_sha256"]:
            raise ValueError("Frozen multi-expert route weight changed: " + name)
        expert_paths[name] = collect_paths(
            load_module(weights, model), rows, dataset, features, manifest)
    records = prepare_records(rows, dataset, manifest, expert_paths,
                              config["proper_loss"]["brier_weight"])
    names = tuple(sorted(expert_paths)); paths = ("parent", *names)
    feature_names = expert_bank_feature_names(names)
    raw = {row["id"]: records[row["id"]]["features"] for row in rows}
    sources = sorted(config["data"]["sources"]); output.mkdir(parents=True)
    runs = []
    for heldout_source in sources:
        training = [row for row in rows if row["split"] == "train" and
                    row["source"] != heldout_source]
        heldout = [row for row in rows if row["split"] == "validation" and
                   row["source"] == heldout_source]
        train_matrix = torch.stack([raw[row["id"]] for row in training])
        heldout_matrix = torch.stack([raw[row["id"]] for row in heldout])
        standardized, fit_mean, fit_scale = standardize_bank_features(
            train_matrix, [train_matrix, heldout_matrix])
        feature_map = {**{row["id"]: value for row, value in zip(training, standardized[0])},
                       **{row["id"]: value for row, value in zip(heldout, standardized[1])}}
        for seed in config["router_seeds"]:
            module, details = train_fold(
                rows, heldout_source, records, paths, feature_map,
                config["training"], seed, output)
            details["evaluation"] = evaluate_fold(
                module, rows, heldout_source, records, paths, feature_map,
                config["training"]["confidence_threshold"])
            details["train_sources"] = [source for source in sources if source != heldout_source]
            details["standardization"] = {
                "fit_rows": len(training),
                "mean_sha256": digest(fit_mean.numpy().tobytes()),
                "scale_sha256": digest(fit_scale.numpy().tobytes()),
            }
            runs.append(details)
    by_source = {}
    for source in sources:
        selected = [run["evaluation"] for run in runs if run["heldout_source"] == source]
        by_source[source] = {metric: mean([value[metric] for value in selected])
                             for metric in ("path_match", "mean_proper_gain", "mean_regret",
                                            "accuracy_delta", "cross_entropy_delta",
                                            "violation_count")}
    aggregate = {metric: mean([run["evaluation"][metric] for run in runs])
                 for metric in ("path_match", "mean_proper_gain", "mean_regret",
                                "accuracy_delta", "cross_entropy_delta")}
    total_rows = sum(run["evaluation"]["rows"] for run in runs)
    aggregate["violation_rate"] = (sum(run["evaluation"]["violation_count"] for run in runs) /
                                   total_rows)
    rules = config["screening_rule"]
    checks = {
        "coverage.folds_seeds": len(runs) == len(sources) * len(config["router_seeds"]),
        "coverage.no_heldout_training": all(
            run["heldout_source"] not in run["train_sources"] for run in runs),
        "combined.accuracy": aggregate["accuracy_delta"] >= rules["accuracy_min_delta"],
        "combined.cross_entropy": aggregate["cross_entropy_delta"] <= rules["cross_entropy_max_delta"],
        "sources.accuracy": all(value["accuracy_delta"] >= rules["each_source_accuracy_min_delta"]
                                for value in by_source.values()),
        "sources.cross_entropy": all(
            value["cross_entropy_delta"] <= rules["each_source_cross_entropy_max_delta"]
            for value in by_source.values()),
        "safety.violation_rate": aggregate["violation_rate"] <= rules["violation_rate_max"],
        "utility.proper_gain": aggregate["mean_proper_gain"] >= rules["proper_gain_min"],
        "target.path_match": aggregate["path_match"] >= rules["path_match_min"],
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "public-data multi-seed leave-one-family-out safe multi-expert route validation",
        "protocol_sha256": digest(config_path.read_bytes()),
        "paths": list(paths), "feature_names": list(feature_names),
        "target_distribution": dict(Counter(records[row["id"]]["target_path"] for row in rows)),
        "runs": runs, "aggregate": aggregate, "by_source": by_source,
        "checks": checks, "advances_to_full_router_training": all(checks.values()),
        "limitations": config["limitations"],
    }
    write_json(output / "evidence.json", evidence)
    write_json(root / "docs/evidence/safe-multi-expert-route-lofo-v1.json", evidence)
    table = [
        f"| {source} | {value['path_match']:.4f} | {value['mean_proper_gain']:+.4f} | "
        f"{value['accuracy_delta']:+.4f} | {value['cross_entropy_delta']:+.4f} | "
        f"{value['violation_count']:.2f} |"
        for source, value in by_source.items()]
    lines = ["# 安全多专家路由留一任务族结果", "",
             "路由只使用父模型与四个专家的候选顺序不变行为特征。六个来源逐一完全留出，"
             "每折三个随机种子。", "",
             "| 留出来源 | 路径匹配率 | proper 收益 | 准确率差 | 交叉熵差 | 平均违规数 |",
             "|---|---:|---:|---:|---:|---:|", *table, "", "## 汇总", "",
             f"- 路径匹配率：`{aggregate['path_match']:.4f}`。",
             f"- 平均 proper-loss 收益：`{aggregate['mean_proper_gain']:+.4f}`。",
             f"- 准确率差：`{aggregate['accuracy_delta']:+.4f}`。",
             f"- 交叉熵差：`{aggregate['cross_entropy_delta']:+.4f}`。",
             f"- 安全违规率：`{aggregate['violation_rate']:.4f}`。", "", "## 判定", "",
             ("全部门槛通过，允许进入完整多专家路由训练。" if all(checks.values())
              else "至少一个门槛失败，不训练新的路由端点。"), "",
             "逐项门槛：" + ", ".join(
                 f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
             "机器证据：`docs/evidence/safe-multi-expert-route-lofo-v1.json`", ""]
    (root / "docs/SAFE_MULTI_EXPERT_ROUTE_LOFO_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(canonical({"event": "complete", "advances": all(checks.values()),
                     "aggregate": aggregate, "checks": checks}))


if __name__ == "__main__":
    main()
