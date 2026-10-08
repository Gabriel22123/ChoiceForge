#!/usr/bin/env python3
"""Nested LOFO routing with separately calibrated safety-constraint margins."""
from __future__ import annotations

import argparse
import importlib.util
import math
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


_V1_PATH = Path(__file__).with_name("run_factored_safe_route_lofo.py")
_V1_SPEC = importlib.util.spec_from_file_location("constraint_route_v1", _V1_PATH)
_V1 = importlib.util.module_from_spec(_V1_SPEC); _V1_SPEC.loader.exec_module(_V1)
read, mean, source_balanced_plan, prepare_records = (
    _V1.read, _V1.mean, _V1.source_balanced_plan, _V1.prepare_records)
load_features, load_module, collect_paths = _V1.load_features, _V1.load_module, _V1.collect_paths


class ConstraintMarginRouter(nn.Module):
    def __init__(self, input_width, hidden_width, experts):
        super().__init__()
        self.experts = experts
        self.trunk = nn.Sequential(nn.Linear(input_width, hidden_width), nn.GELU())
        self.correctness_risk = nn.Linear(hidden_width, experts)
        self.margins = nn.Linear(hidden_width, experts * 4)

    def forward(self, values):
        hidden = self.trunk(values)
        return self.correctness_risk(hidden), self.margins(hidden).reshape(-1, self.experts, 4)


def component_targets(rows, records, names):
    risks, margins = [], []
    for row in rows:
        record = records[row["id"]]; parent = record["parent"]
        row_risks, row_margins = [], []
        for name in names:
            expert = record["experts"][name]
            row_risks.append(float(parent["correct"] and not expert["correct"]))
            row_margins.append([
                expert["cross_entropy"] - parent["cross_entropy"],
                expert["brier"] - parent["brier"],
                parent["context_advantage"] - expert["context_advantage"],
                parent["proper_cost"] - expert["proper_cost"],
            ])
        risks.append(row_risks); margins.append(row_margins)
    return torch.tensor(risks), torch.tensor(margins)


def risk_weights(labels, cap):
    positives = labels.sum(0); negatives = len(labels) - positives
    return torch.tensor([1.0 if float(positive) == 0 else
                         min(float(cap), math.sqrt(float(negative / positive)))
                         for positive, negative in zip(positives, negatives)])


def upper_quantile(values, probability):
    if not values or not 0 < probability <= 1:
        raise ValueError("Invalid conservative quantile")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def train_model(rows, records, names, features, config, seed, updates, plan_path):
    risks, margins = component_targets(rows, records, names)
    margin_mean = margins.mean(0)
    margin_scale = margins.std(0, unbiased=False).clamp_min(config["margin_scale_min"])
    weights = risk_weights(risks, config["risk_positive_weight_cap"])
    plan = source_balanced_plan(rows, "__none__", seed, updates, config["batch_size"])
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("".join(canonical(item) + "\n" for item in plan))
    row_index = {row["id"]: index for index, row in enumerate(rows)}
    torch.manual_seed(seed)
    module = ConstraintMarginRouter(len(next(iter(features.values()))),
                                    config["hidden_width"], len(names))
    optimizer = torch.optim.AdamW(module.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], eps=1e-6)
    losses, risk_losses, margin_losses, norms = [], [], [], []
    started = time.monotonic(); module.train()
    for item in plan:
        indices = [row_index[row_id] for row_id in item["row_ids"]]
        matrix = torch.stack([features[row_id] for row_id in item["row_ids"]])
        risk_logits, normalized_margins = module(matrix)
        risk_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            risk_logits, risks[indices], pos_weight=weights)
        margin_target = (margins[indices] - margin_mean) / margin_scale
        margin_loss = torch.nn.functional.smooth_l1_loss(
            normalized_margins, margin_target, beta=config["margin_huber_beta"])
        loss = risk_loss + config["margin_loss_weight"] * margin_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), config["gradient_clip"])
        optimizer.step()
        losses.append(float(loss.detach())); risk_losses.append(float(risk_loss.detach()))
        margin_losses.append(float(margin_loss.detach())); norms.append(float(norm))
    return module, margin_mean, margin_scale, {
        "updates": len(plan), "examples": len(plan) * config["batch_size"],
        "seconds": time.monotonic() - started, "plan_sha256": digest(plan_path.read_bytes()),
        "risk_positive_counts": {name: int(risks[:, index].sum())
                                 for index, name in enumerate(names)},
        "risk_positive_weights": {name: float(weights[index])
                                  for index, name in enumerate(names)},
        "loss_first": losses[0], "loss_last": losses[-1],
        "risk_loss_last": risk_losses[-1], "margin_loss_last": margin_losses[-1],
        "gradient_norm_mean": mean(norms), "gradient_norm_maximum": max(norms),
        "router_sha256": parameter_sha256(module),
    }


def predict(module, rows, features, margin_mean, margin_scale):
    matrix = torch.stack([features[row["id"]] for row in rows])
    module.eval()
    with torch.no_grad():
        risk, normalized = module(matrix)
    return risk.sigmoid(), normalized * margin_scale + margin_mean


def calibrate(rows, outer_source, records, names, raw, config, seed, output):
    samples = {name: {"risk": [], "residuals": [[], [], [], []]} for name in names}
    support, inner_details = [], []
    inner_sources = sorted({row["source"] for row in rows if row["source"] != outer_source})
    for inner_source in inner_sources:
        training = [row for row in rows if row["source"] not in (outer_source, inner_source)]
        validation = [row for row in rows if row["source"] == inner_source]
        train_matrix = torch.stack([raw[row["id"]] for row in training])
        validation_matrix = torch.stack([raw[row["id"]] for row in validation])
        standardized, fit_mean, fit_scale = standardize_bank_features(
            train_matrix, [train_matrix, validation_matrix])
        features = {**{row["id"]: value for row, value in zip(training, standardized[0])},
                    **{row["id"]: value for row, value in zip(validation, standardized[1])}}
        module, margin_mean, margin_scale, details = train_model(
            training, records, names, features, config, seed, config["calibration_updates"],
            output / "plans" / f"outer-{outer_source}-inner-{inner_source}-{seed}.jsonl")
        predicted_risk, predicted_margins = predict(
            module, validation, features, margin_mean, margin_scale)
        actual_risk, actual_margins = component_targets(validation, records, names)
        support.extend(float(value.abs().max()) for value in standardized[1])
        for expert_index, name in enumerate(names):
            samples[name]["risk"].extend((float(probability), bool(label))
                for probability, label in zip(predicted_risk[:, expert_index],
                                              actual_risk[:, expert_index]))
            for component in range(4):
                if component == 3:
                    residual = predicted_margins[:, expert_index, component] - actual_margins[:, expert_index, component]
                else:
                    residual = actual_margins[:, expert_index, component] - predicted_margins[:, expert_index, component]
                samples[name]["residuals"][component].extend(float(value) for value in residual)
        inner_details.append({
            "heldout_source": inner_source,
            "train_sources": sorted({row["source"] for row in training}),
            "rows": len(validation), "standardization_mean_sha256": digest(fit_mean.numpy().tobytes()),
            "standardization_scale_sha256": digest(fit_scale.numpy().tobytes()), **details,
        })
    calibration = {"support_cutoff": upper_quantile(support, config["calibration_quantile"]),
                   "experts": {}}
    for name in names:
        violating = [probability for probability, label in samples[name]["risk"] if label]
        risk_threshold = ((min(violating) - config["risk_threshold_margin"])
                          if violating else config["default_risk_threshold"])
        calibration["experts"][name] = {
            "risk_threshold": max(0.0, min(config["default_risk_threshold"], risk_threshold)),
            "risk_violations": len(violating),
            "residual_bounds": [upper_quantile(values, config["calibration_quantile"])
                                for values in samples[name]["residuals"]],
        }
    return calibration, inner_details


def evaluate(module, rows, records, names, features, margin_mean, margin_scale, calibration):
    risks, margins = predict(module, rows, features, margin_mean, margin_scale)
    paths, logits, gains, regrets = [], [], [], []
    violations, unsafe_rows, selected = Counter(), 0, 0
    for row_index, row in enumerate(rows):
        record = records[row["id"]]; candidates = []
        in_support = float(features[row["id"]].abs().max()) <= calibration["support_cutoff"]
        if in_support:
            for expert_index, name in enumerate(names):
                item = calibration["experts"][name]; bounds = item["residual_bounds"]
                conservative = [float(margins[row_index, expert_index, component]) + bounds[component]
                                for component in range(3)]
                gain_lower = float(margins[row_index, expert_index, 3]) - bounds[3]
                if (float(risks[row_index, expert_index]) < item["risk_threshold"] and
                        all(value <= 0 for value in conservative) and gain_lower > 0):
                    candidates.append((gain_lower, name))
        path = sorted(candidates, key=lambda value: (-value[0], value[1]))[0][1] if candidates else "parent"
        statistics = record["parent"] if path == "parent" else record["experts"][path]
        parent = record["parent"]; selected += int(path != "parent")
        unsafe_rows += int(path != "parent" and not safe_improvement(parent, statistics)["eligible"])
        violations["parent_correct_lost"] += int(parent["correct"] and not statistics["correct"])
        violations["cross_entropy_regressed"] += int(statistics["cross_entropy"] > parent["cross_entropy"] + 1e-12)
        violations["brier_regressed"] += int(statistics["brier"] > parent["brier"] + 1e-12)
        violations["context_regressed"] += int(statistics["context_advantage"] + 1e-12 < parent["context_advantage"])
        gain = parent["proper_cost"] - statistics["proper_cost"]
        gains.append(gain); regrets.append(record["oracle_improvement"] - gain); paths.append(path)
        logits.append(record["parent_logits"] if path == "parent" else record["expert_logits"][path])
    parent_logits = [records[row["id"]]["parent_logits"] for row in rows]
    parent_metrics, _ = evaluate_rows(rows, parent_logits, 1.0)
    routed_metrics, _ = evaluate_rows(rows, logits, 1.0)
    return {
        "rows": len(rows), "selection": dict(Counter(paths)), "selected_experts": selected,
        "unsafe_selected_rows": unsafe_rows, "expert_selection_rate": selected / len(rows),
        "safe_selection_precision": 1 - unsafe_rows / selected if selected else 1.0,
        "mean_proper_gain": mean(gains), "mean_regret": mean(regrets),
        "accuracy_delta": routed_metrics["accuracy"] - parent_metrics["accuracy"],
        "cross_entropy_delta": routed_metrics["nll"] - parent_metrics["nll"],
        "violation_count": sum(violations.values()), "violations": dict(violations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/constraint-margin-route-lofo-v1.json")
    parser.add_argument("--output", default="runs/constraint-margin-route-lofo-v1")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists(): raise ValueError("Use a fresh constraint-margin route directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training": raise ValueError("Protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen constraint route source changed: " + relative)
    data_path = root / config["data"]["path"]
    if digest(data_path.read_bytes()) != config["data"]["sha256"]: raise ValueError("Data changed")
    rows = [row for row in load_rows(data_path) if row["split"] in ("train", "validation") and
            row["source"] in config["data"]["sources"]]
    expected = {tuple(key.split("|")): value for key, value in config["data"]["counts"].items()}
    if Counter((row["source"], row["split"]) for row in rows) != expected: raise ValueError("Coverage changed")
    cache = root / config["feature_cache"]["path"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_cache"]["manifest_sha256"]:
        raise ValueError("Feature cache changed")
    dataset = config["data"]["dataset"]; keys = [f"{dataset}:{row['id']}" for row in rows]
    manifest, features = load_features(cache, keys); model = {**config["model"], "hidden_size": manifest["hidden_size"]}
    expert_paths = {}
    for name, endpoint in config["experts"].items():
        weights = root / endpoint["weights"]
        if digest(weights.read_bytes()) != endpoint["weights_sha256"]: raise ValueError("Expert changed: " + name)
        expert_paths[name] = collect_paths(load_module(weights, model), rows, dataset, features, manifest)
    records = prepare_records(rows, dataset, manifest, expert_paths, config["proper_loss"]["brier_weight"])
    names = tuple(sorted(expert_paths)); raw = {row["id"]: records[row["id"]]["features"] for row in rows}
    sources = sorted(config["data"]["sources"]); output.mkdir(parents=True); runs = []
    train_rows = [row for row in rows if row["split"] == "train"]
    for outer_source in sources:
        training = [row for row in train_rows if row["source"] != outer_source]
        validation = [row for row in rows if row["split"] == "validation" and row["source"] == outer_source]
        train_matrix = torch.stack([raw[row["id"]] for row in training])
        validation_matrix = torch.stack([raw[row["id"]] for row in validation])
        standardized, fit_mean, fit_scale = standardize_bank_features(train_matrix, [train_matrix, validation_matrix])
        feature_map = {**{row["id"]: value for row, value in zip(training, standardized[0])},
                       **{row["id"]: value for row, value in zip(validation, standardized[1])}}
        for seed in config["router_seeds"]:
            calibration, inner = calibrate(train_rows, outer_source, records, names, raw,
                                           config["training"], seed, output)
            module, margin_mean, margin_scale, details = train_model(
                training, records, names, feature_map, config["training"], seed,
                config["training"]["final_updates"], output / "plans" / f"outer-{outer_source}-final-{seed}.jsonl")
            details.update({"seed": seed, "heldout_source": outer_source,
                "train_sources": sorted({row["source"] for row in training}),
                "calibration": calibration, "inner_folds": inner,
                "standardization": {"fit_rows": len(training),
                    "mean_sha256": digest(fit_mean.numpy().tobytes()),
                    "scale_sha256": digest(fit_scale.numpy().tobytes())},
                "evaluation": evaluate(module, validation, records, names, feature_map,
                                       margin_mean, margin_scale, calibration)})
            runs.append(details)
    metrics = ("mean_proper_gain", "mean_regret", "accuracy_delta", "cross_entropy_delta",
               "expert_selection_rate", "safe_selection_precision", "violation_count")
    by_source = {source: {metric: mean([run["evaluation"][metric] for run in runs
                                        if run["heldout_source"] == source]) for metric in metrics}
                 for source in sources}
    aggregate = {metric: mean([run["evaluation"][metric] for run in runs]) for metric in metrics[:-1]}
    total_rows = sum(run["evaluation"]["rows"] for run in runs)
    total_selected = sum(run["evaluation"]["selected_experts"] for run in runs)
    total_unsafe = sum(run["evaluation"]["unsafe_selected_rows"] for run in runs)
    aggregate.update({"violation_rate": sum(run["evaluation"]["violation_count"] for run in runs) / total_rows,
                      "expert_selection_rate": total_selected / total_rows,
                      "safe_selection_precision": 1 - total_unsafe / total_selected if total_selected else 1.0})
    rule = config["screening_rule"]
    checks = {"coverage.folds_seeds": len(runs) == len(sources) * len(config["router_seeds"]),
      "coverage.no_outer_leakage": all(run["heldout_source"] not in run["train_sources"] and
        all(run["heldout_source"] not in fold["train_sources"] for fold in run["inner_folds"]) for run in runs),
      "combined.accuracy": aggregate["accuracy_delta"] >= rule["accuracy_min_delta"],
      "combined.cross_entropy": aggregate["cross_entropy_delta"] <= rule["cross_entropy_max_delta"],
      "sources.accuracy": all(v["accuracy_delta"] >= rule["each_source_accuracy_min_delta"] for v in by_source.values()),
      "sources.cross_entropy": all(v["cross_entropy_delta"] <= rule["each_source_cross_entropy_max_delta"] for v in by_source.values()),
      "safety.violation_rate": aggregate["violation_rate"] <= rule["violation_rate_max"],
      "safety.selection_precision": aggregate["safe_selection_precision"] >= rule["safe_selection_precision_min"],
      "coverage.expert_selection": aggregate["expert_selection_rate"] >= rule["expert_selection_rate_min"],
      "utility.proper_gain": aggregate["mean_proper_gain"] >= rule["proper_gain_min"]}
    evidence = {"format_version": 1, "study": config["study"],
      "role": "nested public-data constraint-margin route validation", "protocol_sha256": digest(config_path.read_bytes()),
      "experts": list(names), "runs": runs, "aggregate": aggregate, "by_source": by_source,
      "checks": checks, "advances_to_full_router_training": all(checks.values()), "limitations": config["limitations"]}
    write_json(output / "evidence.json", evidence); write_json(root / "docs/evidence/constraint-margin-route-lofo-v1.json", evidence)
    table = [f"| {source} | {v['expert_selection_rate']:.4f} | {v['safe_selection_precision']:.4f} | "
             f"{v['mean_proper_gain']:+.4f} | {v['accuracy_delta']:+.4f} | {v['cross_entropy_delta']:+.4f} |"
             for source, v in by_source.items()]
    lines = ["# 约束 margin 路由留一任务族结果", "", "外层任务族完全留出；各约束保守界仅由训练来源内层留出预测确定。", "",
      "| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |", "|---|---:|---:|---:|---:|---:|", *table, "", "## 汇总", "",
      f"- 专家选择率：`{aggregate['expert_selection_rate']:.4f}`。", f"- 已选专家安全精度：`{aggregate['safe_selection_precision']:.4f}`。",
      f"- 逐项安全违规率：`{aggregate['violation_rate']:.4f}`。", f"- 平均 proper-loss 收益：`{aggregate['mean_proper_gain']:+.4f}`。",
      f"- 准确率差：`{aggregate['accuracy_delta']:+.4f}`。", f"- 交叉熵差：`{aggregate['cross_entropy_delta']:+.4f}`。", "", "## 判定", "",
      ("全部门槛通过，允许进入已消费数据上的完整路由训练。" if all(checks.values()) else "至少一个冻结门槛失败，不训练完整路由端点。"), "",
      "逐项门槛：" + ", ".join(f"`{k}`={'PASS' if v else 'FAIL'}" for k, v in checks.items()), "",
      "机器证据：`docs/evidence/constraint-margin-route-lofo-v1.json`", ""]
    (root / "docs/CONSTRAINT_MARGIN_ROUTE_LOFO_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(canonical({"event": "complete", "advances": all(checks.values()), "aggregate": aggregate, "checks": checks}))


if __name__ == "__main__": main()
