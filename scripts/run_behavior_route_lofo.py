#!/usr/bin/env python3
"""Multi-seed leave-one-family-out validation for behavior-derived routing."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
from torch import nn

from decision_model.behavioral_route_features import (
    FEATURE_NAMES, behavioral_route_features, standardize_behavior_features)
from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.feature_screen import RoutedResidualHead, parameter_sha256
from decision_model.router_cross_validation import binary_route_metrics, make_lofo_plan


_BASE_PATH = Path(__file__).with_name("run_binary_route_lofo.py")
_BASE_SPEC = importlib.util.spec_from_file_location("binary_route_lofo_helper", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_BASE_SPEC); _BASE_SPEC.loader.exec_module(_BASE)
aggregate_runs, component_hash, derive_binary_targets = (
    _BASE.aggregate_runs, _BASE.component_hash, _BASE.derive_binary_targets)
load_features, mean, pad_batch, read, score, train_stage = (
    _BASE.load_features, _BASE.mean, _BASE.pad_batch, _BASE.read, _BASE.score, _BASE.train_stage)


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
    source_balanced = arm == "behavior_source_balanced"
    if arm not in ("behavior_row_weighted", "behavior_source_balanced"):
        raise ValueError("Unknown behavior-route arm")
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
            raise ValueError("Held-out source leaked into behavior router training")
        matrix = torch.stack([feature_map[row["id"]] for row in batch])
        target = torch.tensor([targets[row["id"]] for row in batch], dtype=torch.float32)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            module(matrix).squeeze(-1), target)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/behavior-route-lofo-v1.json")
    parser.add_argument("--output", default="runs/behavior-route-lofo-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh behavior-route LOFO study directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Behavior-route LOFO protocol was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen behavior-route LOFO source changed: " + relative)
    baseline_path = root / config["semantic_baseline"]["path"]
    if digest(baseline_path.read_bytes()) != config["semantic_baseline"]["sha256"]:
        raise ValueError("Frozen semantic-route baseline changed")
    semantic = read(baseline_path)["aggregate"]["row_weighted"]
    cache = root / config["feature_cache"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_manifest_sha256"]:
        raise ValueError("Frozen routed feature manifest changed")
    cases_path = root / config["data"]["cases"]
    if digest(cases_path.read_bytes()) != config["data"]["cases_sha256"]:
        raise ValueError("Frozen behavior-route LOFO cases changed")
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
        raise ValueError("Behavior-route LOFO expert isolation failed")
    targets, target_summary = derive_binary_targets(
        expert_module, train_rows, features, manifest, training, output)
    raw_features = extract_behavior_features(
        expert_module, train_rows, features, manifest)
    sources = sorted({row["source"] for row in train_rows})
    if len(sources) != config["expected_sources"] or len(targets) != len(train_rows):
        raise ValueError("Behavior-route LOFO source or target coverage differs")

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
            constant = mean([targets[row["id"]] for row in training_rows])
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
                labels = [targets[row["id"]] for row in heldout]
                details["metrics"] = binary_route_metrics(labels, predicted)
                details["constant_metrics"] = binary_route_metrics(
                    labels, [constant] * len(labels))
                details["heldout_rows"] = len(heldout)
                details["train_sources"] = [source for source in sources
                                             if source != heldout_source]
                details["standardization"] = standardization
                details["router_sha256"] = parameter_sha256(module)
                runs.append(details)

    aggregate = aggregate_runs(runs, sources, config["arms"])
    selected_arm = min(config["arms"], key=lambda arm: aggregate[arm]["brier"])
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
        "selected.brier_vs_semantic": (
            selected["brier"] <= semantic["brier"] - rules["brier_vs_semantic_min_improvement"]),
        "selected.brier_vs_constant": (
            selected["brier"] <= selected["constant_brier"] -
            rules["brier_vs_constant_min_improvement"]),
        "selected.balanced_accuracy": (
            selected["balanced_accuracy"] >= rules["balanced_accuracy_min"]),
        "selected.each_source_noninferior": all(
            selected["by_source"][source]["brier"] <=
            semantic["by_source"][source]["brier"] + rules["each_source_brier_max_regression"]
            for source in sources),
    }
    evidence = {
        "format_version": 1,
        "study": config["study"],
        "role": "public-data multi-seed leave-one-family-out behavior-route validation",
        "protocol_sha256": digest(config_path.read_bytes()),
        "feature_names": list(FEATURE_NAMES),
        "initial": initial,
        "initial_max_logit_difference": initial_max,
        "after_expert": after_expert,
        "expert": expert,
        "target_summary": target_summary,
        "semantic_baseline": semantic,
        "sources": sources,
        "runs": runs,
        "aggregate": aggregate,
        "selected_arm": selected_arm,
        "checks": checks,
        "advances_to_full_router_training": all(checks.values()),
        "limitations": [
            "The frozen expert is shared across folds; only router supervision is held out.",
            "All families are public training sources already used elsewhere in the project.",
            "This validates behavior-route transfer, not a release endpoint.",
        ],
    }
    write_json(output / "run.json", evidence)
    write_json(root / "docs/evidence/behavior-route-lofo-v1.json", evidence)
    table = []
    for source in sources:
        table.append(
            f"| {source} | {semantic['by_source'][source]['brier']:.4f} | "
            f"{selected['by_source'][source]['brier']:.4f} | "
            f"{selected['by_source'][source]['balanced_accuracy']:.4f} | "
            f"{selected['by_source'][source]['target_open_rate']:.4f} | "
            f"{selected['by_source'][source]['predicted_open_rate']:.4f} |")
    lines = [
        "# 行为路由留一任务族验证", "",
        "路由只读取父模型与冻结专家的推理行为特征；7 个公开来源逐一留出，每折每臂 3 个种子。", "",
        f"按宏平均 Brier 预注册选择：`{selected_arm}`。", "",
        "| 留出来源 | 语义路由 Brier | 行为路由 Brier | 行为平衡准确率 | 目标打开率 | 预测打开率 |",
        "|---|---:|---:|---:|---:|---:|", *table, "", "## 汇总", "",
        f"- 语义路由宏平均 Brier：`{semantic['brier']:.6f}`。",
        f"- 选中行为路由宏平均 Brier：`{selected['brier']:.6f}`。",
        f"- 选中行为路由常数基线 Brier：`{selected['constant_brier']:.6f}`。",
        f"- 选中行为路由宏平均平衡准确率：`{selected['balanced_accuracy']:.6f}`。", "",
        "## 判定", "",
        ("全部门槛通过；允许进入多种子完整行为路由训练。" if all(checks.values())
         else "至少一个门槛失败；不训练新的候选端点。"), "",
        "逐项门槛：" + ", ".join(
            f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
        "机器证据：`docs/evidence/behavior-route-lofo-v1.json`", "",
    ]
    (root / "docs/BEHAVIOR_ROUTE_LOFO_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {
        "state": "complete", "advances": all(checks.values())})
    print(json.dumps({"state": "complete", "advances": all(checks.values()),
                      "selected_arm": selected_arm, "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
