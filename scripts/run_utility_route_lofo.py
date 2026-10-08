#!/usr/bin/env python3
"""Multi-seed leave-one-family-out validation for signed-utility routing."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from decision_model.behavioral_route_features import (
    FEATURE_NAMES, behavioral_route_features, standardize_behavior_features)
from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.feature_screen import RoutedResidualHead, parameter_sha256
from decision_model.router_cross_validation import make_lofo_plan
from decision_model.route_utility import signed_route_utility, utility_route_metrics


_BASE_PATH = Path(__file__).with_name("run_binary_route_lofo.py")
_BASE_SPEC = importlib.util.spec_from_file_location("binary_route_lofo_helper", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_BASE_SPEC); _BASE_SPEC.loader.exec_module(_BASE)
component_hash = _BASE.component_hash
load_features, mean, pad_batch, read, score, train_stage = (
    _BASE.load_features, _BASE.mean, _BASE.pad_batch, _BASE.read, _BASE.score, _BASE.train_stage)
composite_gate_costs = _BASE._BINARY.composite_gate_costs
add_parent_choice_cost = _BASE._BINARY.add_parent_choice_cost


def derive_utility_targets(module, rows, features, manifest, config, output):
    gates = torch.tensor(config["counterfactual_gate_grid"], dtype=torch.float32)
    records, by_source = [], defaultdict(list)
    module.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            keys = [f"cases:{row['id']}" for row in batch]
            full, null, parent, parent_null, mask, labels = pad_batch(
                batch, keys, features, manifest)
            correction = module.residual(full).squeeze(-1).masked_fill(~mask, 0.0)
            null_correction = module.residual(null).squeeze(-1).masked_fill(~mask, 0.0)
            for index, row in enumerate(batch):
                count = len(row["request"]["choices"])
                choice_ids = [choice["id"] for choice in row["request"]["choices"]]
                teacher = (torch.tensor([row["teacher_probabilities"][key]
                                         for key in choice_ids])
                           if "teacher_probabilities" in row else None)
                null_teacher = (torch.tensor([row["teacher_null_probabilities"][key]
                                              for key in choice_ids])
                                if "teacher_null_probabilities" in row else None)
                costs = composite_gate_costs(
                    parent[index, :count], parent_null[index, :count],
                    correction[index, :count], null_correction[index, :count],
                    int(labels[index]), gates, brier_weight=config["brier_weight"],
                    context_weight=config["context_advantage_weight"],
                    context_gap=config["context_advantage_gap"], teacher=teacher,
                    null_teacher=null_teacher,
                    distillation_weight=config["distillation_weight"],
                    null_distillation_weight=config["null_distillation_weight"])
                costs = add_parent_choice_cost(
                    costs, parent[index, :count], parent_null[index, :count],
                    correction[index, :count], null_correction[index, :count], gates,
                    active=teacher is not None,
                    change_cost=config["parent_choice_change_cost"])
                record = {"row_id": row["id"], **signed_route_utility(costs, gates)}
                records.append(record); by_source[row["source"]].append(record)
    path = output / "counterfactual-utility-targets.jsonl"
    path.write_text("".join(canonical(record) + "\n" for record in records))
    aggregate = lambda values: {
        "rows": len(values),
        "mean_utility": mean([row["utility"] for row in values]),
        "mean_absolute_utility": mean([abs(row["utility"]) for row in values]),
        "open_rate": mean([row["gate"] for row in values]),
    }
    summary = {
        "all": aggregate(records),
        "by_source": {source: aggregate(values) for source, values in sorted(by_source.items())},
        "targets_sha256": digest(path.read_bytes()),
        "target": "asinh(parent counterfactual cost minus expert counterfactual cost)",
    }
    return {record["row_id"]: record for record in records}, summary


def extract_behavior_features(module, rows, features, manifest):
    result = {}
    module.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            keys = [f"cases:{row['id']}" for row in batch]
            full, null, parent, parent_null, mask, _ = pad_batch(
                batch, keys, features, manifest)
            correction = module.residual(full).squeeze(-1).masked_fill(~mask, 0.0)
            null_correction = module.residual(null).squeeze(-1).masked_fill(~mask, 0.0)
            for index, row in enumerate(batch):
                count = len(row["request"]["choices"])
                result[row["id"]] = behavioral_route_features(
                    parent[index, :count], parent_null[index, :count],
                    correction[index, :count], null_correction[index, :count])
    if len(result) != len(rows):
        raise ValueError("Behavior-route feature coverage differs")
    return result


def build_router(width):
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("Invalid behavior router width")
    return nn.Sequential(nn.Linear(len(FEATURE_NAMES), width), nn.GELU(), nn.Linear(width, 1))


def train_fold(module, rows, heldout_source, targets, feature_map, config,
               output, arm, seed):
    source_balanced = arm == "utility_source_balanced"
    if arm not in ("utility_row_weighted", "utility_source_balanced"):
        raise ValueError("Unknown utility-route arm")
    plan = make_lofo_plan(
        rows, heldout_source, seed, config["updates"], source_balanced=source_balanced)
    plan_path = output / "plans" / f"{arm}-{heldout_source}-{seed}.jsonl"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("".join(canonical(item) + "\n" for item in plan))
    by_id = {row["id"]: row for row in rows}
    parameters = list(module.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"], eps=1e-6)
    losses, norms = [], []
    started = time.monotonic(); module.train()
    for item in plan:
        batch = [by_id[row_id] for row_id in item["row_ids"]]
        if any(row["source"] == heldout_source for row in batch):
            raise ValueError("Held-out source leaked into utility router training")
        matrix = torch.stack([feature_map[row["id"]] for row in batch])
        target = torch.tensor(
            [targets[row["id"]]["transformed_utility"] for row in batch],
            dtype=torch.float32)
        loss = torch.nn.functional.smooth_l1_loss(module(matrix).squeeze(-1), target)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
        optimizer.step(); losses.append(float(loss.detach())); norms.append(float(norm))
    return {
        "arm": arm,
        "seed": seed,
        "heldout_source": heldout_source,
        "updates": len(plan),
        "batch_size": len(plan[0]["row_ids"]),
        "examples": sum(len(item["row_ids"]) for item in plan),
        "seconds": time.monotonic() - started,
        "plan_sha256": digest(plan_path.read_bytes()),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "gradient_norm_mean": mean(norms),
        "gradient_norm_maximum": max(norms),
    }


def aggregate_utility_runs(runs, sources, arms):
    metric_names = (
        "accuracy", "balanced_accuracy", "brier", "log_loss", "target_open_rate",
        "predicted_open_rate", "mean_regret", "always_parent_regret",
        "always_expert_regret", "transformed_mae", "zero_prediction_mae",
        "mean_utility", "mean_predicted_transformed_utility")
    result = {}
    for arm in arms:
        selected = [run for run in runs if run["arm"] == arm]
        by_source = {}
        for source in sources:
            source_runs = [run for run in selected if run["heldout_source"] == source]
            by_source[source] = {
                key: mean([run["metrics"][key] for run in source_runs])
                for key in metric_names
            }
        result[arm] = {
            "runs": len(selected),
            **{key: mean([run["metrics"][key] for run in selected])
               for key in metric_names},
            "by_source": by_source,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/utility-route-lofo-v1.json")
    parser.add_argument("--output", default="runs/utility-route-lofo-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh utility-route LOFO study directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Utility-route LOFO protocol was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen utility-route LOFO source changed: " + relative)
    cache = root / config["feature_cache"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_manifest_sha256"]:
        raise ValueError("Frozen routed feature manifest changed")
    cases_path = root / config["data"]["cases"]
    if digest(cases_path.read_bytes()) != config["data"]["cases_sha256"]:
        raise ValueError("Frozen utility-route LOFO cases changed")
    cases = load_rows(cases_path)
    train_rows = [row for row in cases if row["split"] == "train"]
    manifest, features = load_features(
        cache, [f"cases:{row['id']}" for row in train_rows])
    training = config["training"]
    output.mkdir(parents=True)

    torch.manual_seed(training["expert_seed"])
    expert_module = RoutedResidualHead.build(
        manifest["hidden_size"], training["residual_width"], training["semantic_router_width"],
        training["dropout"], training["initial_gate_probability"])
    initial = {"residual": component_hash(expert_module, "residual"),
               "router": component_hash(expert_module, "router")}
    initial_logits, _, _ = score(expert_module, train_rows[:24], "cases", features, manifest)
    parent_logits = [manifest["rows"][f"cases:{row['id']}"]["parent_logits"]
                     for row in train_rows[:24]]
    initial_max = max(abs(left - right) for actual, parent in zip(initial_logits, parent_logits)
                      for left, right in zip(actual, parent))
    expert = train_stage(
        expert_module, "expert", train_rows, features, manifest, training, output)
    after_expert = {"residual": component_hash(expert_module, "residual"),
                    "router": component_hash(expert_module, "router")}
    if (initial_max != 0.0 or after_expert["residual"] == initial["residual"] or
            after_expert["router"] != initial["router"]):
        raise ValueError("Utility-route LOFO expert isolation failed")
    targets, target_summary = derive_utility_targets(
        expert_module, train_rows, features, manifest, training, output)
    raw_features = extract_behavior_features(
        expert_module, train_rows, features, manifest)
    sources = sorted({row["source"] for row in train_rows})
    if len(sources) != config["expected_sources"] or len(targets) != len(train_rows):
        raise ValueError("Utility-route LOFO source or target coverage differs")

    runs = []
    for arm in config["arms"]:
        for heldout_source in sources:
            training_rows = [row for row in train_rows if row["source"] != heldout_source]
            heldout = [row for row in train_rows if row["source"] == heldout_source]
            train_matrix = torch.stack([raw_features[row["id"]] for row in training_rows])
            heldout_matrix = torch.stack([raw_features[row["id"]] for row in heldout])
            (standardized, fit_mean, fit_scale) = standardize_behavior_features(
                train_matrix, [train_matrix, heldout_matrix])
            fold_features = {
                **{row["id"]: value for row, value in zip(training_rows, standardized[0])},
                **{row["id"]: value for row, value in zip(heldout, standardized[1])},
            }
            standardization = {
                "fit_rows": len(training_rows),
                "mean_sha256": digest(fit_mean.numpy().tobytes()),
                "scale_sha256": digest(fit_scale.numpy().tobytes()),
            }
            for seed in config["router_seeds"]:
                torch.manual_seed(seed)
                module = build_router(training["behavior_router_width"])
                details = train_fold(
                    module, train_rows, heldout_source, targets, fold_features,
                    training, output, arm, seed)
                module.eval()
                with torch.no_grad():
                    predicted = module(standardized[1]).squeeze(-1).sigmoid().tolist()
                utilities = [targets[row["id"]]["utility"] for row in heldout]
                details["metrics"] = utility_route_metrics(utilities, predicted)
                details["heldout_rows"] = len(heldout)
                details["train_sources"] = [source for source in sources
                                             if source != heldout_source]
                details["standardization"] = standardization
                details["router_sha256"] = parameter_sha256(module)
                runs.append(details)

    aggregate = aggregate_utility_runs(runs, sources, config["arms"])
    selected_arm = min(config["arms"], key=lambda arm: aggregate[arm]["mean_regret"])
    selected = aggregate[selected_arm]
    rules = config["screening_rule"]
    checks = {
        "mechanism.initial_exact": initial_max == 0.0,
        "mechanism.expert_stage_isolated": (
            after_expert["residual"] != initial["residual"] and
            after_expert["router"] == initial["router"]),
        "coverage.sources_seeds_arms": len(runs) == len(sources) * len(config["router_seeds"]) * 2,
        "coverage.no_heldout_training": all(
            run["heldout_source"] not in run["train_sources"] for run in runs),
        "selected.regret_vs_parent": (
            selected["mean_regret"] <= selected["always_parent_regret"] -
            rules["regret_vs_parent_min_improvement"]),
        "selected.mae_vs_zero": (
            selected["transformed_mae"] <= selected["zero_prediction_mae"] -
            rules["mae_vs_zero_min_improvement"]),
        "selected.balanced_accuracy": (
            selected["balanced_accuracy"] >= rules["balanced_accuracy_min"]),
        "selected.each_source_noninferior": all(
            selected["by_source"][source]["mean_regret"] <=
            selected["by_source"][source]["always_parent_regret"] +
            rules["each_source_regret_max_regression"]
            for source in sources),
    }
    evidence = {
        "format_version": 1,
        "study": config["study"],
        "role": "public-data multi-seed leave-one-family-out signed-utility route validation",
        "protocol_sha256": digest(config_path.read_bytes()),
        "feature_names": list(FEATURE_NAMES),
        "initial": initial,
        "initial_max_logit_difference": initial_max,
        "after_expert": after_expert,
        "expert": expert,
        "target_summary": target_summary,
        "sources": sources,
        "runs": runs,
        "aggregate": aggregate,
        "selected_arm": selected_arm,
        "checks": checks,
        "advances_to_full_router_training": all(checks.values()),
        "limitations": [
            "The frozen expert is shared across folds; only router supervision is held out.",
            "All families are public training sources already used elsewhere in the project.",
            "This validates signed-utility route transfer, not a release endpoint.",
        ],
    }
    write_json(output / "run.json", evidence)
    write_json(root / "docs/evidence/utility-route-lofo-v1.json", evidence)
    table = []
    for source in sources:
        table.append(
            f"| {source} | {selected['by_source'][source]['mean_regret']:.4f} | "
            f"{selected['by_source'][source]['always_parent_regret']:.4f} | "
            f"{selected['by_source'][source]['balanced_accuracy']:.4f} | "
            f"{selected['by_source'][source]['transformed_mae']:.4f} | "
            f"{selected['by_source'][source]['zero_prediction_mae']:.4f} |")
    lines = [
        "# 反事实效用路由留一任务族验证", "",
        "路由回归父模型与专家的带符号反事实效用；7 个公开来源逐一留出，每折每臂 3 个种子。", "",
        f"按宏平均路由遗憾预注册选择：`{selected_arm}`。", "",
        "| 留出来源 | 路由遗憾 | 始终父模型遗憾 | 符号平衡准确率 | 变换 MAE | 恒零 MAE |",
        "|---|---:|---:|---:|---:|---:|", *table, "", "## 汇总", "",
        f"- 选中效用路由宏平均遗憾：`{selected['mean_regret']:.6f}`。",
        f"- 始终父模型宏平均遗憾：`{selected['always_parent_regret']:.6f}`。",
        f"- 选中效用路由变换后 MAE：`{selected['transformed_mae']:.6f}`。",
        f"- 恒预测零的变换后 MAE：`{selected['zero_prediction_mae']:.6f}`。",
        f"- 选中效用路由宏平均平衡准确率：`{selected['balanced_accuracy']:.6f}`。", "",
        "## 判定", "",
        ("全部门槛通过；允许进入多种子完整效用路由训练。" if all(checks.values())
         else "至少一个门槛失败；不训练新的候选端点。"), "",
        "逐项门槛：" + ", ".join(
            f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
        "机器证据：`docs/evidence/utility-route-lofo-v1.json`", "",
    ]
    (root / "docs/UTILITY_ROUTE_LOFO_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {
        "state": "complete", "advances": all(checks.values())})
    print(json.dumps({"state": "complete", "advances": all(checks.values()),
                      "selected_arm": selected_arm, "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
