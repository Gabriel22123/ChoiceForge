#!/usr/bin/env python3
"""LOFO validation for a source-robust, decision-stable expert portfolio."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import torch

from decision_model.core import canonical, digest, load_rows, target_distribution, write_json
from decision_model.expert_portfolio import (
    decision_stable_projection, source_robust_objective)


_BANK_PATH = Path(__file__).with_name("run_heterogeneous_expert_bank.py")
_SPEC = importlib.util.spec_from_file_location("portfolio_bank", _BANK_PATH)
_BANK = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_BANK)
load_features, load_module, collect_paths = (
    _BANK.load_features, _BANK.load_module, _BANK.collect_paths)
endpoint_model = _BANK.endpoint_model


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    return sum(values) / len(values)


def load_bank(root, config):
    names = tuple(sorted(config["experts"]))
    rows, full_rows, null_rows, seen_ids = [], [], [], set()
    for group in config["groups"]:
        data_path = root / group["data_path"]
        if digest(data_path.read_bytes()) != group["data_sha256"]:
            raise ValueError("Portfolio data changed: " + group["name"])
        group_rows = [row for row in load_rows(data_path)
                      if row["split"] in group["splits"] and row["source"] in group["sources"]]
        expected = {tuple(key.split("|")): value for key, value in group["counts"].items()}
        if Counter((row["source"], row["split"]) for row in group_rows) != expected:
            raise ValueError("Portfolio coverage changed: " + group["name"])
        if seen_ids.intersection(row["id"] for row in group_rows):
            raise ValueError("Portfolio row IDs collide across groups")
        seen_ids.update(row["id"] for row in group_rows)
        cache = root / group["feature_cache"]
        if digest((cache / "manifest.json").read_bytes()) != group["manifest_sha256"]:
            raise ValueError("Portfolio feature cache changed: " + group["name"])
        keys = [f"{group['dataset']}:{row['id']}" for row in group_rows]
        manifest, features = load_features(cache, keys)
        expert_paths = {}
        for name, endpoint in config["experts"].items():
            weight = root / endpoint["weights"]
            if digest(weight.read_bytes()) != endpoint["weights_sha256"]:
                raise ValueError("Portfolio expert changed: " + name)
            model = endpoint_model(config["default_model"], endpoint, manifest["hidden_size"])
            expert_paths[name] = collect_paths(
                load_module(weight, model), group_rows, group["dataset"], features, manifest)
        for index, key in enumerate(keys):
            entry = manifest["rows"][key]
            full_rows.append([entry["parent_logits"]] +
                             [expert_paths[name][0][index] for name in names])
            null_rows.append([entry["parent_null_logits"]] +
                             [expert_paths[name][1][index] for name in names])
        rows.extend(group_rows)
    if not rows or len(rows) != len(full_rows) or len(rows) != len(null_rows):
        raise ValueError("Portfolio bank is incomplete")
    width = max(len(row["request"]["choices"]) for row in rows)
    paths = ("parent", *names)
    full = torch.full((len(rows), len(paths), width), -1e4)
    null = torch.full_like(full, -1e4)
    targets = torch.zeros(len(rows), width)
    for row_index, (row, row_full, row_null) in enumerate(zip(rows, full_rows, null_rows)):
        count = len(row["request"]["choices"])
        if len(row_full) != len(paths) or len(row_null) != len(paths):
            raise ValueError("Portfolio path coverage differs")
        for path_index, (full_logits, null_logits) in enumerate(zip(row_full, row_null)):
            if len(full_logits) != count or len(null_logits) != count:
                raise ValueError("Portfolio candidate coverage differs")
            full[row_index, path_index, :count] = torch.tensor(full_logits)
            null[row_index, path_index, :count] = torch.tensor(null_logits)
        targets[row_index, :count] = torch.tensor(target_distribution(row))
    return rows, paths, full, null, targets


def proper_costs(logits, targets, brier_weight):
    log_probabilities = logits.log_softmax(-1)
    probabilities = log_probabilities.exp()
    cross_entropy = -(targets * log_probabilities).sum(-1)
    brier = ((probabilities - targets) ** 2).sum(-1)
    return cross_entropy + float(brier_weight) * brier, cross_entropy, brier


def context_advantage(full, null, targets):
    return (targets * (full.log_softmax(-1) - null.log_softmax(-1))).sum(-1)


def source_means(values, source_indices):
    return torch.stack([values[indices].mean() for indices in source_indices])


def train_portfolio(rows, paths, full, null, targets, outer_source, seed, config):
    available = [index for index, name in enumerate(paths) if name != outer_source]
    training_ids = [index for index, row in enumerate(rows)
                    if row["split"] == "train" and row["source"] != outer_source]
    train_sources = sorted({rows[index]["source"] for index in training_ids})
    local_source_indices = [torch.tensor([
        position for position, row_index in enumerate(training_ids)
        if rows[row_index]["source"] == source]) for source in train_sources]
    ids = torch.tensor(training_ids)
    selected_full, selected_null, selected_targets = full[ids], null[ids], targets[ids]
    parent_cost, _, _ = proper_costs(
        selected_full[:, 0], selected_targets, config["brier_weight"])
    parent_context = context_advantage(
        selected_full[:, 0], selected_null[:, 0], selected_targets)
    parent_source_cost = source_means(parent_cost, local_source_indices).detach()
    parent_source_context = source_means(parent_context, local_source_indices).detach()
    torch.manual_seed(seed)
    initial = torch.tensor([config["parent_initial_logit"]] +
                           [config["expert_initial_logit"]] * (len(available) - 1))
    parameter = torch.nn.Parameter(initial + torch.randn(len(available)) *
                                   config["initial_noise"])
    optimizer = torch.optim.Adam([parameter], lr=config["learning_rate"])
    curve, started = [], time.monotonic()
    for update in range(1, config["updates"] + 1):
        weights = parameter.softmax(0)
        portfolio_full = (selected_full[:, available] * weights.view(1, -1, 1)).sum(1)
        portfolio_null = (selected_null[:, available] * weights.view(1, -1, 1)).sum(1)
        costs, _, _ = proper_costs(portfolio_full, selected_targets, config["brier_weight"])
        contexts = context_advantage(portfolio_full, portfolio_null, selected_targets)
        source_cost = source_means(costs, local_source_indices)
        source_context = source_means(contexts, local_source_indices)
        loss, diagnostics = source_robust_objective(
            source_cost, parent_source_cost, source_context, parent_source_context,
            config["proper_regression_penalty"], config["context_regression_penalty"])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if update in (1, config["updates"]):
            curve.append({
                "update": update, "loss": float(loss.detach()),
                "proper_regression_max": float(diagnostics["proper_regression_max"]),
                "context_regression_max": float(diagnostics["context_regression_max"]),
            })
    weights = parameter.softmax(0).detach()
    weight_map = {paths[index]: float(weight)
                  for index, weight in zip(available, weights)}
    return weights, available, {
        "seed": seed, "heldout_source": outer_source,
        "train_sources": train_sources,
        "available_paths": [paths[index] for index in available],
        "heldout_expert_excluded": outer_source not in paths or outer_source not in weight_map,
        "updates": config["updates"], "seconds": time.monotonic() - started,
        "weights": weight_map, "weights_sha256": digest(canonical(weight_map)),
        "curve": curve,
    }


def evaluate_portfolio(rows, full, null, targets, source, weights, available, config):
    ids = [index for index, row in enumerate(rows)
           if row["split"] == "validation" and row["source"] == source]
    output_full, output_null, alphas = [], [], []
    portfolio_full = (full[ids][:, available] * weights.view(1, -1, 1)).sum(1)
    portfolio_null = (null[ids][:, available] * weights.view(1, -1, 1)).sum(1)
    for local_index, row_index in enumerate(ids):
        count = len(rows[row_index]["request"]["choices"])
        projected_full, projected_null, alpha = decision_stable_projection(
            full[row_index, 0, :count], null[row_index, 0, :count],
            portfolio_full[local_index, :count], portfolio_null[local_index, :count],
            config["projection_alphas"])
        padded_full = full[row_index, 0].clone(); padded_null = null[row_index, 0].clone()
        padded_full[:count] = projected_full; padded_null[:count] = projected_null
        output_full.append(padded_full); output_null.append(padded_null); alphas.append(alpha)
    output_full, output_null = torch.stack(output_full), torch.stack(output_null)
    parent_full, parent_null, selected_targets = full[ids, 0], null[ids, 0], targets[ids]
    parent_cost, parent_ce, parent_brier = proper_costs(
        parent_full, selected_targets, config["brier_weight"])
    output_cost, output_ce, output_brier = proper_costs(
        output_full, selected_targets, config["brier_weight"])
    parent_context = context_advantage(parent_full, parent_null, selected_targets)
    output_context = context_advantage(output_full, output_null, selected_targets)
    parent_correct = parent_full.argmax(-1) == selected_targets.argmax(-1)
    output_correct = output_full.argmax(-1) == selected_targets.argmax(-1)
    gains = parent_cost - output_cost
    return {
        "rows": len(ids), "parent_accuracy": float(parent_correct.float().mean()),
        "accuracy": float(output_correct.float().mean()),
        "accuracy_delta": float(output_correct.float().mean() - parent_correct.float().mean()),
        "cross_entropy_delta": float((output_ce - parent_ce).mean()),
        "brier_delta": float((output_brier - parent_brier).mean()),
        "proper_gain": float(gains.mean()),
        "positive_proper_rate": float((gains > 0).float().mean()),
        "context_advantage_delta": float((output_context - parent_context).mean()),
        "context_regression_rate": float((output_context < parent_context).float().mean()),
        "mean_alpha": mean(alphas), "minimum_alpha": min(alphas),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision-stable-expert-portfolio-v1.json")
    parser.add_argument("--output", default="runs/decision-stable-expert-portfolio-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh decision-stable portfolio output directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Portfolio protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen portfolio source changed: " + relative)
    rows, paths, full, null, targets = load_bank(root, config)
    sources = sorted({row["source"] for row in rows})
    output.mkdir(parents=True)
    runs = []
    for source in sources:
        for seed in config["training"]["seeds"]:
            weights, available, run = train_portfolio(
                rows, paths, full, null, targets, source, seed, config["training"])
            run["evaluation"] = evaluate_portfolio(
                rows, full, null, targets, source, weights, available, config["training"])
            runs.append(run)
    metrics = ("accuracy_delta", "cross_entropy_delta", "brier_delta", "proper_gain",
               "positive_proper_rate", "context_advantage_delta", "context_regression_rate",
               "mean_alpha")
    by_source = {source: {metric: mean([
        run["evaluation"][metric] for run in runs if run["heldout_source"] == source])
        for metric in metrics} for source in sources}
    macro = {metric: mean([value[metric] for value in by_source.values()])
             for metric in metrics}
    rule = config["screening_rule"]
    checks = {
        "coverage.folds_seeds": len(runs) == len(sources) * len(config["training"]["seeds"]),
        "coverage.no_outer_data": all(
            run["heldout_source"] not in run["train_sources"] for run in runs),
        "coverage.no_matching_expert": all(run["heldout_expert_excluded"] for run in runs),
        "decision.exact_each_run": all(
            run["evaluation"]["accuracy_delta"] == 0.0 for run in runs),
        "calibration.cross_entropy": macro["cross_entropy_delta"] <= 0.0,
        "calibration.brier": macro["brier_delta"] <= 0.0,
        "utility.proper_gain": macro["proper_gain"] >= rule["proper_gain_min"],
        "context.aggregate": macro["context_advantage_delta"] >= 0.0,
        "sources.cross_entropy": all(
            value["cross_entropy_delta"] <= rule["each_source_cross_entropy_max_delta"]
            for value in by_source.values()),
        "sources.proper_gain": all(
            value["proper_gain"] >= rule["each_source_proper_gain_min"]
            for value in by_source.values()),
        "sources.context": all(
            value["context_advantage_delta"] >= rule["each_source_context_min_delta"]
            for value in by_source.values()),
        "coverage.material_projection": macro["mean_alpha"] >= rule["mean_alpha_min"],
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "source-robust decision-stable heterogeneous expert portfolio LOFO",
        "protocol_sha256": digest(config_path.read_bytes()),
        "paths": list(paths), "runs": runs, "by_source": by_source, "macro": macro,
        "checks": checks, "advances_to_full_portfolio": all(checks.values()),
        "limitations": config["limitations"],
    }
    write_json(output / "evidence.json", evidence)
    write_json(root / config["evidence_path"], evidence)
    table = [
        f"| {source} | {value['accuracy_delta']:+.4f} | "
        f"{value['cross_entropy_delta']:+.4f} | {value['brier_delta']:+.4f} | "
        f"{value['proper_gain']:+.4f} | {value['context_advantage_delta']:+.4f} | "
        f"{value['mean_alpha']:.4f} |" for source, value in by_source.items()
    ]
    lines = [
        "# 决策稳定专家组合 V1 结果", "",
        "外层来源及其同名专家同时留出；推理投影保证父模型最终选择不翻转。", "",
        "| 留出来源 | 准确率差 | CE 差 | Brier 差 | proper 收益 | 上下文差 | 平均步长 |",
        "|---|---:|---:|---:|---:|---:|---:|", *table, "", "## 宏平均", "",
        f"- 准确率差：`{macro['accuracy_delta']:+.4f}`。",
        f"- 交叉熵差：`{macro['cross_entropy_delta']:+.4f}`。",
        f"- Brier 差：`{macro['brier_delta']:+.4f}`。",
        f"- proper-loss 收益：`{macro['proper_gain']:+.4f}`。",
        f"- 上下文优势差：`{macro['context_advantage_delta']:+.4f}`。",
        f"- 平均投影步长：`{macro['mean_alpha']:.4f}`。", "", "## 判定", "",
        ("全部冻结门槛通过；允许训练全来源组合并冻结最终评测协议。"
         if all(checks.values()) else "至少一个冻结门槛失败；不进入最终评测。"), "",
        "逐项门槛：" + ", ".join(
            f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
        f"机器证据：`{config['evidence_path']}`", "",
    ]
    (root / config["report_path"]).write_text("\n".join(lines))
    write_json(output / "status.json", {
        "state": "complete", "advances": all(checks.values())})
    print(canonical({"event": "complete", "advances": all(checks.values()),
                     "macro": macro, "checks": checks}))


if __name__ == "__main__":
    main()
