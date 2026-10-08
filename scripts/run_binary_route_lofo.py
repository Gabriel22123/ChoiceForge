#!/usr/bin/env python3
"""Multi-seed leave-one-family-out validation for the binary semantic router."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.feature_screen import RoutedResidualHead, parameter_sha256
from decision_model.router_cross_validation import binary_route_metrics, make_lofo_plan


_BINARY_PATH = Path(__file__).with_name("run_qasc_binary_route_screen.py")
_BINARY_SPEC = importlib.util.spec_from_file_location("binary_route_screen_helper", _BINARY_PATH)
_BINARY = importlib.util.module_from_spec(_BINARY_SPEC); _BINARY_SPEC.loader.exec_module(_BINARY)
component_hash = _BINARY.component_hash
derive_binary_targets = _BINARY.derive_binary_targets
load_features, pad_batch, score = _BINARY.load_features, _BINARY.pad_batch, _BINARY.score
train_stage = _BINARY.train_stage


def read(path):
    return json.loads(Path(path).read_text())


def request_logits(module, null, mask):
    weights = mask.to(dtype=null.dtype).unsqueeze(-1)
    request = (null * weights).sum(dim=1) / weights.sum(dim=1)
    return module.router(request).squeeze(-1)


def train_fold(module, rows, heldout_source, targets, features, manifest, config,
               output, arm, seed):
    source_balanced = arm == "source_balanced"
    if arm not in ("row_weighted", "source_balanced"):
        raise ValueError("Unknown leave-family-out arm")
    plan = make_lofo_plan(
        rows, heldout_source, seed, config["updates"], source_balanced=source_balanced)
    plan_path = output / "plans" / f"{arm}-{heldout_source}-{seed}.jsonl"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("".join(canonical(item) + "\n" for item in plan))
    by_id = {row["id"]: row for row in rows}
    module.residual.requires_grad_(False); module.router.requires_grad_(True)
    parameters = list(module.router.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"], eps=1e-6)
    losses, norms = [], []
    started = time.monotonic(); module.train()
    for item in plan:
        batch = [by_id[row_id] for row_id in item["row_ids"]]
        if any(row["source"] == heldout_source for row in batch):
            raise ValueError("Held-out source leaked into router training")
        keys = [f"cases:{row['id']}" for row in batch]
        _, null, _, _, mask, _ = pad_batch(batch, keys, features, manifest)
        logits = request_logits(module, null, mask)
        target = torch.tensor([targets[row["id"]] for row in batch], dtype=torch.float32)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
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
        "gradient_norm_mean": sum(norms) / len(norms),
        "gradient_norm_maximum": max(norms),
        "router_sha256": component_hash(module, "router"),
    }


def probabilities(module, rows, features, manifest):
    values = []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            keys = [f"cases:{row['id']}" for row in batch]
            _, null, _, _, mask, _ = pad_batch(batch, keys, features, manifest)
            values.extend(request_logits(module, null, mask).sigmoid().tolist())
    return values


def mean(values):
    return sum(values) / len(values)


def aggregate_runs(runs, sources, arms):
    metric_names = ("accuracy", "balanced_accuracy", "brier", "log_loss",
                    "target_open_rate", "predicted_open_rate", "mean_probability")
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
            by_source[source]["constant_brier"] = mean(
                [run["constant_metrics"]["brier"] for run in source_runs])
        result[arm] = {
            "runs": len(selected),
            **{key: mean([run["metrics"][key] for run in selected])
               for key in metric_names},
            "constant_brier": mean([run["constant_metrics"]["brier"] for run in selected]),
            "by_source": by_source,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/binary-route-lofo-v1.json")
    parser.add_argument("--output", default="runs/binary-route-lofo-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh binary-route LOFO study directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Binary-route LOFO protocol was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen binary-route LOFO source changed: " + relative)
    cache = root / config["feature_cache"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_manifest_sha256"]:
        raise ValueError("Frozen routed feature manifest changed")
    cases_path = root / config["data"]["cases"]
    if digest(cases_path.read_bytes()) != config["data"]["cases_sha256"]:
        raise ValueError("Frozen binary-route LOFO cases changed")
    cases = load_rows(cases_path)
    train_rows = [row for row in cases if row["split"] == "train"]
    keys = [f"cases:{row['id']}" for row in train_rows]
    manifest, features = load_features(cache, keys)
    training = config["training"]
    output.mkdir(parents=True)

    torch.manual_seed(training["expert_seed"])
    expert_module = RoutedResidualHead.build(
        manifest["hidden_size"], training["residual_width"], training["router_width"],
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
        raise ValueError("Binary-route LOFO expert isolation failed")
    targets, target_summary = derive_binary_targets(
        expert_module, train_rows, features, manifest, training, output)
    sources = sorted({row["source"] for row in train_rows})
    if len(sources) != config["expected_sources"] or len(targets) != len(train_rows):
        raise ValueError("Binary-route LOFO source or target coverage differs")
    residual_state = {name: value.detach().clone()
                      for name, value in expert_module.residual.state_dict().items()}

    runs = []
    for arm in config["arms"]:
        for heldout_source in sources:
            heldout = [row for row in train_rows if row["source"] == heldout_source]
            training_rows = [row for row in train_rows if row["source"] != heldout_source]
            constant = mean([targets[row["id"]] for row in training_rows])
            for seed in config["router_seeds"]:
                torch.manual_seed(seed)
                module = RoutedResidualHead.build(
                    manifest["hidden_size"], training["residual_width"],
                    training["router_width"], training["dropout"],
                    training["initial_gate_probability"])
                module.residual.load_state_dict(residual_state)
                details = train_fold(
                    module, train_rows, heldout_source, targets, features, manifest,
                    training, output, arm, seed)
                predicted = probabilities(module, heldout, features, manifest)
                labels = [targets[row["id"]] for row in heldout]
                details["metrics"] = binary_route_metrics(labels, predicted)
                details["constant_metrics"] = binary_route_metrics(
                    labels, [constant] * len(labels))
                details["heldout_rows"] = len(heldout)
                details["train_sources"] = [source for source in sources
                                             if source != heldout_source]
                runs.append(details)

    aggregate = aggregate_runs(runs, sources, config["arms"])
    raw = aggregate["row_weighted"]
    balanced = aggregate["source_balanced"]
    rules = config["screening_rule"]
    checks = {
        "mechanism.initial_exact": initial_max == 0.0,
        "mechanism.expert_stage_isolated": (
            after_expert["residual"] != initial["residual"] and
            after_expert["router"] == initial["router"]),
        "coverage.sources_seeds_arms": len(runs) == len(sources) * len(config["router_seeds"]) * 2,
        "coverage.no_heldout_training": all(
            run["heldout_source"] not in run["train_sources"] for run in runs),
        "balanced.brier_vs_row": (
            balanced["brier"] <= raw["brier"] - rules["balanced_brier_vs_row_min_improvement"]),
        "balanced.brier_vs_constant": (
            balanced["brier"] <= balanced["constant_brier"] -
            rules["balanced_brier_vs_constant_min_improvement"]),
        "balanced.balanced_accuracy": (
            balanced["balanced_accuracy"] >= rules["balanced_accuracy_min"]),
        "balanced.each_source_noninferior": all(
            balanced["by_source"][source]["brier"] <=
            raw["by_source"][source]["brier"] + rules["each_source_brier_max_regression"]
            for source in sources),
    }
    evidence = {
        "format_version": 1,
        "study": config["study"],
        "role": "public-data multi-seed leave-one-family-out binary route validation",
        "protocol_sha256": digest(config_path.read_bytes()),
        "initial": initial,
        "initial_max_logit_difference": initial_max,
        "after_expert": after_expert,
        "expert": expert,
        "target_summary": target_summary,
        "sources": sources,
        "runs": runs,
        "aggregate": aggregate,
        "checks": checks,
        "advances_to_full_router_training": all(checks.values()),
        "limitations": [
            "The frozen expert is shared across folds; only router supervision is held out.",
            "All families are public training sources already used elsewhere in the project.",
            "This validates route-target transfer, not a release endpoint.",
        ],
    }
    write_json(output / "run.json", evidence)
    write_json(root / "docs/evidence/binary-route-lofo-v1.json", evidence)
    rows = []
    for source in sources:
        rows.append(
            f"| {source} | {raw['by_source'][source]['brier']:.4f} | "
            f"{balanced['by_source'][source]['brier']:.4f} | "
            f"{balanced['by_source'][source]['balanced_accuracy']:.4f} | "
            f"{balanced['by_source'][source]['target_open_rate']:.4f} | "
            f"{balanced['by_source'][source]['predicted_open_rate']:.4f} |")
    lines = [
        "# 二元路由留一任务族验证", "",
        "7 个公开来源逐一留出，每折、每训练臂运行 3 个随机种子。", "",
        "| 留出来源 | 普通 Brier | 族均衡 Brier | 族均衡平衡准确率 | 目标打开率 | 预测打开率 |",
        "|---|---:|---:|---:|---:|---:|", *rows, "", "## 汇总", "",
        f"- 普通训练宏平均 Brier：`{raw['brier']:.6f}`。",
        f"- 任务族均衡宏平均 Brier：`{balanced['brier']:.6f}`。",
        f"- 任务族均衡常数基线 Brier：`{balanced['constant_brier']:.6f}`。",
        f"- 任务族均衡宏平均平衡准确率：`{balanced['balanced_accuracy']:.6f}`。", "",
        "## 判定", "",
        ("全部门槛通过；允许进入多种子完整路由训练。" if all(checks.values())
         else "至少一个门槛失败；不训练新的候选端点。"), "",
        "逐项门槛：" + ", ".join(
            f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
        "机器证据：`docs/evidence/binary-route-lofo-v1.json`", "",
    ]
    (root / "docs/BINARY_ROUTE_LOFO_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {
        "state": "complete", "advances": all(checks.values())})
    print(json.dumps({"state": "complete", "advances": all(checks.values()),
                      "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
