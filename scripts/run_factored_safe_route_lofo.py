#!/usr/bin/env python3
"""Nested leave-one-family-out validation for feasibility-first expert routing."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.expert_bank_features import standardize_bank_features
from decision_model.feature_screen import parameter_sha256
from decision_model.safe_routing import safe_improvement
from decision_model.train import evaluate_rows


_BASE_PATH = Path(__file__).with_name("run_safe_multi_expert_route_lofo.py")
_BASE_SPEC = importlib.util.spec_from_file_location("factored_route_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_BASE_SPEC); _BASE_SPEC.loader.exec_module(_BASE)
read, mean, source_balanced_plan, prepare_records = (
    _BASE.read, _BASE.mean, _BASE.source_balanced_plan, _BASE.prepare_records)
load_features, load_module, collect_paths = _BASE.load_features, _BASE.load_module, _BASE.collect_paths


class FactoredRouter(nn.Module):
    def __init__(self, input_width, hidden_width, experts):
        super().__init__()
        if min(input_width, hidden_width, experts) < 1:
            raise ValueError("Invalid factored router dimensions")
        self.trunk = nn.Sequential(nn.Linear(input_width, hidden_width), nn.GELU())
        self.safety = nn.Linear(hidden_width, experts)
        self.gain = nn.Linear(hidden_width, experts)

    def forward(self, values):
        hidden = self.trunk(values)
        return self.safety(hidden), self.gain(hidden)


def safety_and_gain_targets(rows, records, names):
    labels, gains = [], []
    for row in rows:
        record = records[row["id"]]
        decisions = [safe_improvement(record["parent"], record["experts"][name])
                     for name in names]
        labels.append([float(value["eligible"]) for value in decisions])
        gains.append([value["improvement"] for value in decisions])
    return torch.tensor(labels), torch.tensor(gains)


def positive_weights(labels, cap):
    if labels.ndim != 2 or len(labels) < 1:
        raise ValueError("Invalid factored safety labels")
    positives = labels.sum(0)
    negatives = len(labels) - positives
    weights = []
    for positive, negative in zip(positives, negatives):
        if float(positive) == 0:
            weights.append(0.0)
        else:
            weights.append(min(float(cap), math.sqrt(float(negative / positive))))
    return torch.tensor(weights)


def train_model(rows, records, names, features, config, seed, updates, plan_path):
    labels, gains = safety_and_gain_targets(rows, records, names)
    gain_mean = gains.mean(0)
    gain_scale = gains.std(0, unbiased=False).clamp_min(config["gain_scale_min"])
    weights = positive_weights(labels, config["positive_weight_cap"])
    plan = source_balanced_plan(rows, "__none__", seed, updates, config["batch_size"])
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("".join(canonical(item) + "\n" for item in plan))
    row_index = {row["id"]: index for index, row in enumerate(rows)}
    torch.manual_seed(seed)
    module = FactoredRouter(len(next(iter(features.values()))), config["hidden_width"], len(names))
    optimizer = torch.optim.AdamW(module.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], eps=1e-6)
    losses, safety_losses, gain_losses, norms = [], [], [], []
    started = time.monotonic(); module.train()
    for item in plan:
        indices = [row_index[row_id] for row_id in item["row_ids"]]
        matrix = torch.stack([features[item_id] for item_id in item["row_ids"]])
        safety_logits, normalized_gain = module(matrix)
        safety_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            safety_logits, labels[indices], pos_weight=weights)
        gain_target = (gains[indices] - gain_mean) / gain_scale
        gain_loss = torch.nn.functional.smooth_l1_loss(
            normalized_gain, gain_target, beta=config["gain_huber_beta"])
        loss = safety_loss + config["gain_loss_weight"] * gain_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), config["gradient_clip"])
        optimizer.step()
        losses.append(float(loss.detach())); safety_losses.append(float(safety_loss.detach()))
        gain_losses.append(float(gain_loss.detach())); norms.append(float(norm))
    return module, gain_mean, gain_scale, {
        "updates": len(plan), "examples": len(plan) * config["batch_size"],
        "seconds": time.monotonic() - started, "plan_sha256": digest(plan_path.read_bytes()),
        "positive_counts": {name: int(labels[:, index].sum()) for index, name in enumerate(names)},
        "positive_weights": {name: float(weights[index]) for index, name in enumerate(names)},
        "gain_mean": {name: float(gain_mean[index]) for index, name in enumerate(names)},
        "gain_scale": {name: float(gain_scale[index]) for index, name in enumerate(names)},
        "loss_first": losses[0], "loss_last": losses[-1],
        "safety_loss_last": safety_losses[-1], "gain_loss_last": gain_losses[-1],
        "gradient_norm_mean": mean(norms), "gradient_norm_maximum": max(norms),
        "router_sha256": parameter_sha256(module),
    }


def predict(module, rows, features, gain_mean, gain_scale):
    matrix = torch.stack([features[row["id"]] for row in rows])
    module.eval()
    with torch.no_grad():
        logits, normalized_gain = module(matrix)
    return logits.sigmoid(), normalized_gain * gain_scale + gain_mean


def calibrate_thresholds(rows, outer_source, records, names, raw, config, seed, output):
    oof = {name: [] for name in names}
    inner_sources = sorted({row["source"] for row in rows if row["source"] != outer_source})
    inner_details = []
    for inner_source in inner_sources:
        training = [row for row in rows if row["source"] not in (outer_source, inner_source)]
        validation = [row for row in rows if row["source"] == inner_source]
        train_matrix = torch.stack([raw[row["id"]] for row in training])
        validation_matrix = torch.stack([raw[row["id"]] for row in validation])
        standardized, fit_mean, fit_scale = standardize_bank_features(
            train_matrix, [train_matrix, validation_matrix])
        features = {**{row["id"]: value for row, value in zip(training, standardized[0])},
                    **{row["id"]: value for row, value in zip(validation, standardized[1])}}
        module, gain_mean, gain_scale, details = train_model(
            training, records, names, features, config, seed,
            config["calibration_updates"],
            output / "plans" / f"outer-{outer_source}-inner-{inner_source}-{seed}.jsonl")
        probabilities, _ = predict(module, validation, features, gain_mean, gain_scale)
        labels, _ = safety_and_gain_targets(validation, records, names)
        for expert_index, name in enumerate(names):
            oof[name].extend((float(probability), bool(label)) for probability, label in
                             zip(probabilities[:, expert_index], labels[:, expert_index]))
        inner_details.append({
            "heldout_source": inner_source, "train_sources": sorted(
                {row["source"] for row in training}), "rows": len(validation),
            "standardization_mean_sha256": digest(fit_mean.numpy().tobytes()),
            "standardization_scale_sha256": digest(fit_scale.numpy().tobytes()),
            **details,
        })
    thresholds, calibration = {}, {}
    for name in names:
        unsafe = [probability for probability, label in oof[name] if not label]
        threshold = max(config["minimum_safety_threshold"],
                        (max(unsafe) + config["threshold_margin"]) if unsafe else 0.5)
        thresholds[name] = min(1.0, threshold)
        calibration[name] = {
            "rows": len(oof[name]), "safe_rows": sum(label for _, label in oof[name]),
            "unsafe_rows": len(unsafe), "maximum_unsafe_probability": max(unsafe) if unsafe else None,
            "threshold": thresholds[name],
            "empirical_false_safe": sum((not label) and probability >= thresholds[name]
                                        for probability, label in oof[name]),
        }
    return thresholds, calibration, inner_details


def evaluate(module, rows, records, names, features, gain_mean, gain_scale, thresholds):
    probabilities, predicted_gains = predict(module, rows, features, gain_mean, gain_scale)
    selected_paths, logits, gains, regrets, matches = [], [], [], [], []
    violations, unsafe_rows, selected_experts = Counter(), 0, 0
    for row_index, row in enumerate(rows):
        record = records[row["id"]]
        eligible = [(float(predicted_gains[row_index, index]), name)
                    for index, name in enumerate(names)
                    if float(probabilities[row_index, index]) >= thresholds[name] and
                    float(predicted_gains[row_index, index]) > 0]
        path = sorted(eligible, key=lambda value: (-value[0], value[1]))[0][1] if eligible else "parent"
        statistics = record["parent"] if path == "parent" else record["experts"][path]
        parent = record["parent"]
        selected_experts += int(path != "parent")
        if path != "parent":
            safe = safe_improvement(parent, statistics)["eligible"]
            unsafe_rows += int(not safe)
        violations["parent_correct_lost"] += int(parent["correct"] and not statistics["correct"])
        violations["cross_entropy_regressed"] += int(
            statistics["cross_entropy"] > parent["cross_entropy"] + 1e-12)
        violations["brier_regressed"] += int(statistics["brier"] > parent["brier"] + 1e-12)
        violations["context_regressed"] += int(
            statistics["context_advantage"] + 1e-12 < parent["context_advantage"])
        selected_paths.append(path); matches.append(path == record["target_path"])
        gain = parent["proper_cost"] - statistics["proper_cost"]
        gains.append(gain); regrets.append(record["oracle_improvement"] - gain)
        logits.append(record["parent_logits"] if path == "parent" else record["expert_logits"][path])
    parent_logits = [records[row["id"]]["parent_logits"] for row in rows]
    parent_metrics, _ = evaluate_rows(rows, parent_logits, 1.0)
    routed_metrics, _ = evaluate_rows(rows, logits, 1.0)
    return {
        "rows": len(rows), "path_match": mean(matches), "selection": dict(Counter(selected_paths)),
        "selected_experts": selected_experts, "unsafe_selected_rows": unsafe_rows,
        "expert_selection_rate": selected_experts / len(rows),
        "safe_selection_precision": (1 - unsafe_rows / selected_experts) if selected_experts else 1.0,
        "mean_proper_gain": mean(gains), "mean_regret": mean(regrets),
        "accuracy_delta": routed_metrics["accuracy"] - parent_metrics["accuracy"],
        "cross_entropy_delta": routed_metrics["nll"] - parent_metrics["nll"],
        "violation_count": sum(violations.values()), "violations": dict(violations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/factored-safe-route-lofo-v1.json")
    parser.add_argument("--output", default="runs/factored-safe-route-lofo-v1")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh factored route directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Factored route protocol was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen factored route source changed: " + relative)
    data_path = root / config["data"]["path"]
    if digest(data_path.read_bytes()) != config["data"]["sha256"]:
        raise ValueError("Frozen factored route data changed")
    rows = [row for row in load_rows(data_path) if row["split"] in ("train", "validation") and
            row["source"] in config["data"]["sources"]]
    expected_counts = {tuple(key.split("|")): value for key, value in config["data"]["counts"].items()}
    if Counter((row["source"], row["split"]) for row in rows) != expected_counts:
        raise ValueError("Factored route data coverage differs")
    cache = root / config["feature_cache"]["path"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_cache"]["manifest_sha256"]:
        raise ValueError("Frozen factored route feature manifest changed")
    dataset = config["data"]["dataset"]; keys = [f"{dataset}:{row['id']}" for row in rows]
    manifest, features = load_features(cache, keys)
    model = {**config["model"], "hidden_size": manifest["hidden_size"]}
    expert_paths = {}
    for name, endpoint in config["experts"].items():
        weights = root / endpoint["weights"]
        if digest(weights.read_bytes()) != endpoint["weights_sha256"]:
            raise ValueError("Frozen expert weight changed: " + name)
        expert_paths[name] = collect_paths(load_module(weights, model), rows, dataset, features, manifest)
    records = prepare_records(rows, dataset, manifest, expert_paths,
                              config["proper_loss"]["brier_weight"])
    names = tuple(sorted(expert_paths)); raw = {
        row["id"]: records[row["id"]]["features"] for row in rows}
    sources = sorted(config["data"]["sources"]); output.mkdir(parents=True)
    runs = []
    train_rows = [row for row in rows if row["split"] == "train"]
    for outer_source in sources:
        outer_training = [row for row in train_rows if row["source"] != outer_source]
        outer_validation = [row for row in rows if row["split"] == "validation" and
                            row["source"] == outer_source]
        train_matrix = torch.stack([raw[row["id"]] for row in outer_training])
        validation_matrix = torch.stack([raw[row["id"]] for row in outer_validation])
        standardized, fit_mean, fit_scale = standardize_bank_features(
            train_matrix, [train_matrix, validation_matrix])
        feature_map = {**{row["id"]: value for row, value in zip(outer_training, standardized[0])},
                       **{row["id"]: value for row, value in zip(outer_validation, standardized[1])}}
        for seed in config["router_seeds"]:
            thresholds, calibration, inner = calibrate_thresholds(
                train_rows, outer_source, records, names, raw, config["training"], seed, output)
            module, gain_mean, gain_scale, details = train_model(
                outer_training, records, names, feature_map, config["training"], seed,
                config["training"]["final_updates"],
                output / "plans" / f"outer-{outer_source}-final-{seed}.jsonl")
            details.update({
                "seed": seed, "heldout_source": outer_source,
                "train_sources": sorted({row["source"] for row in outer_training}),
                "thresholds": thresholds, "calibration": calibration,
                "inner_folds": inner,
                "standardization": {"fit_rows": len(outer_training),
                    "mean_sha256": digest(fit_mean.numpy().tobytes()),
                    "scale_sha256": digest(fit_scale.numpy().tobytes())},
                "evaluation": evaluate(module, outer_validation, records, names, feature_map,
                                       gain_mean, gain_scale, thresholds),
            })
            runs.append(details)
    metrics = ("path_match", "mean_proper_gain", "mean_regret", "accuracy_delta",
               "cross_entropy_delta", "expert_selection_rate", "safe_selection_precision",
               "violation_count")
    by_source = {source: {metric: mean([run["evaluation"][metric] for run in runs
                                        if run["heldout_source"] == source]) for metric in metrics}
                 for source in sources}
    aggregate = {metric: mean([run["evaluation"][metric] for run in runs]) for metric in metrics[:-1]}
    total_rows = sum(run["evaluation"]["rows"] for run in runs)
    total_selected = sum(run["evaluation"]["selected_experts"] for run in runs)
    total_unsafe = sum(run["evaluation"]["unsafe_selected_rows"] for run in runs)
    aggregate.update({
        "violation_rate": sum(run["evaluation"]["violation_count"] for run in runs) / total_rows,
        "expert_selection_rate": total_selected / total_rows,
        "safe_selection_precision": 1 - total_unsafe / total_selected if total_selected else 1.0,
    })
    rules = config["screening_rule"]
    checks = {
        "coverage.folds_seeds": len(runs) == len(sources) * len(config["router_seeds"]),
        "coverage.no_outer_leakage": all(run["heldout_source"] not in run["train_sources"] and
            all(run["heldout_source"] not in fold["train_sources"] for fold in run["inner_folds"])
            for run in runs),
        "combined.accuracy": aggregate["accuracy_delta"] >= rules["accuracy_min_delta"],
        "combined.cross_entropy": aggregate["cross_entropy_delta"] <= rules["cross_entropy_max_delta"],
        "sources.accuracy": all(value["accuracy_delta"] >= rules["each_source_accuracy_min_delta"]
                                for value in by_source.values()),
        "sources.cross_entropy": all(value["cross_entropy_delta"] <= rules["each_source_cross_entropy_max_delta"]
                                     for value in by_source.values()),
        "safety.violation_rate": aggregate["violation_rate"] <= rules["violation_rate_max"],
        "safety.selection_precision": aggregate["safe_selection_precision"] >= rules["safe_selection_precision_min"],
        "coverage.expert_selection": aggregate["expert_selection_rate"] >= rules["expert_selection_rate_min"],
        "utility.proper_gain": aggregate["mean_proper_gain"] >= rules["proper_gain_min"],
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "nested public-data leave-one-family-out factored safe route validation",
        "protocol_sha256": digest(config_path.read_bytes()), "experts": list(names),
        "runs": runs, "aggregate": aggregate, "by_source": by_source, "checks": checks,
        "advances_to_full_router_training": all(checks.values()), "limitations": config["limitations"],
    }
    write_json(output / "evidence.json", evidence)
    write_json(root / "docs/evidence/factored-safe-route-lofo-v1.json", evidence)
    table = [f"| {source} | {value['expert_selection_rate']:.4f} | "
             f"{value['safe_selection_precision']:.4f} | {value['mean_proper_gain']:+.4f} | "
             f"{value['accuracy_delta']:+.4f} | {value['cross_entropy_delta']:+.4f} |"
             for source, value in by_source.items()]
    lines = ["# 分解式安全路由留一任务族结果", "",
             "每个外层任务族完全留出；安全阈值只由训练来源的内层留出预测确定。", "",
             "| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |",
             "|---|---:|---:|---:|---:|---:|", *table, "", "## 汇总", "",
             f"- 专家选择率：`{aggregate['expert_selection_rate']:.4f}`。",
             f"- 已选专家安全精度：`{aggregate['safe_selection_precision']:.4f}`。",
             f"- 逐项安全违规率：`{aggregate['violation_rate']:.4f}`。",
             f"- 平均 proper-loss 收益：`{aggregate['mean_proper_gain']:+.4f}`。",
             f"- 准确率差：`{aggregate['accuracy_delta']:+.4f}`。",
             f"- 交叉熵差：`{aggregate['cross_entropy_delta']:+.4f}`。", "", "## 判定", "",
             ("全部门槛通过，允许进入已消费数据上的完整路由训练。" if all(checks.values())
              else "至少一个冻结门槛失败，不训练完整路由端点。"), "",
             "逐项门槛：" + ", ".join(f"`{key}`={'PASS' if value else 'FAIL'}"
                                         for key, value in checks.items()), "",
             "机器证据：`docs/evidence/factored-safe-route-lofo-v1.json`", ""]
    (root / "docs/FACTORED_SAFE_ROUTE_LOFO_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(canonical({"event": "complete", "advances": all(checks.values()),
                     "aggregate": aggregate, "checks": checks}))


if __name__ == "__main__":
    main()
