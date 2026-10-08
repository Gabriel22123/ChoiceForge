#!/usr/bin/env python3
"""LOFO semantic routing requiring unanimous safety across three seeds."""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path

import torch

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.expert_bank_features import standardize_bank_features
from decision_model.safe_routing import safe_improvement
from decision_model.train import evaluate_rows


_SEMANTIC_PATH = Path(__file__).with_name("run_semantic_factored_route_lofo.py")
_SPEC = importlib.util.spec_from_file_location("semantic_consensus_base", _SEMANTIC_PATH)
_SEMANTIC = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_SEMANTIC)
read, mean, load_features, load_module, collect_paths = (
    _SEMANTIC.read, _SEMANTIC.mean, _SEMANTIC.load_features,
    _SEMANTIC.load_module, _SEMANTIC.collect_paths)
prepare_records, combined_route_features = _SEMANTIC.prepare_records, _SEMANTIC.combined_route_features
calibrate_thresholds, train_model, predict = (
    _SEMANTIC.calibrate_thresholds, _SEMANTIC.train_model, _SEMANTIC._V1.predict)


def choose_consensus_path(row_index, names, predictions):
    eligible = []
    for expert_index, name in enumerate(names):
        safety = [float(probability[row_index, expert_index]) >= thresholds[name]
                  for probability, _, thresholds in predictions]
        predicted = [float(values[row_index, expert_index]) for _, values, _ in predictions]
        if all(safety) and all(value > 0 for value in predicted):
            eligible.append((min(predicted), name))
    return sorted(eligible, key=lambda value: (-value[0], value[1]))[0][1] if eligible else "parent"


def evaluate_consensus(members, rows, records, names, features):
    predictions = []
    for member in members:
        probabilities, gains = predict(member["module"], rows, features,
                                       member["gain_mean"], member["gain_scale"])
        predictions.append((probabilities, gains, member["thresholds"]))
    selected_paths, logits, gains, regrets, matches = [], [], [], [], []
    violations, unsafe_rows, selected_experts = Counter(), 0, 0
    for row_index, row in enumerate(rows):
        record = records[row["id"]]
        path = choose_consensus_path(row_index, names, predictions)
        statistics = record["parent"] if path == "parent" else record["experts"][path]
        parent = record["parent"]; selected_experts += int(path != "parent")
        unsafe_rows += int(path != "parent" and not safe_improvement(parent, statistics)["eligible"])
        violations["parent_correct_lost"] += int(parent["correct"] and not statistics["correct"])
        violations["cross_entropy_regressed"] += int(statistics["cross_entropy"] > parent["cross_entropy"] + 1e-12)
        violations["brier_regressed"] += int(statistics["brier"] > parent["brier"] + 1e-12)
        violations["context_regressed"] += int(statistics["context_advantage"] + 1e-12 < parent["context_advantage"])
        gain = parent["proper_cost"] - statistics["proper_cost"]
        gains.append(gain); regrets.append(record["oracle_improvement"] - gain)
        selected_paths.append(path); matches.append(path == record["target_path"])
        logits.append(record["parent_logits"] if path == "parent" else record["expert_logits"][path])
    parent_logits = [records[row["id"]]["parent_logits"] for row in rows]
    parent_metrics, _ = evaluate_rows(rows, parent_logits, 1.0)
    routed_metrics, _ = evaluate_rows(rows, logits, 1.0)
    return {"rows": len(rows), "path_match": mean(matches), "selection": dict(Counter(selected_paths)),
      "selected_experts": selected_experts, "unsafe_selected_rows": unsafe_rows,
      "expert_selection_rate": selected_experts / len(rows),
      "safe_selection_precision": 1 - unsafe_rows / selected_experts if selected_experts else 1.0,
      "mean_proper_gain": mean(gains), "mean_regret": mean(regrets),
      "accuracy_delta": routed_metrics["accuracy"] - parent_metrics["accuracy"],
      "cross_entropy_delta": routed_metrics["nll"] - parent_metrics["nll"],
      "violation_count": sum(violations.values()), "violations": dict(violations)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic-consensus-route-lofo-v1.json")
    parser.add_argument("--output", default="runs/semantic-consensus-route-lofo-v1")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists(): raise ValueError("Use a fresh semantic consensus route directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training": raise ValueError("Protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen consensus source changed: " + relative)
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
    manifest, frozen = load_features(cache, keys); model = {**config["model"], "hidden_size": manifest["hidden_size"]}
    expert_paths = {}
    for name, endpoint in config["experts"].items():
        weights = root / endpoint["weights"]
        if digest(weights.read_bytes()) != endpoint["weights_sha256"]: raise ValueError("Expert changed: " + name)
        expert_paths[name] = collect_paths(load_module(weights, model), rows, dataset, frozen, manifest)
    records = prepare_records(rows, dataset, manifest, expert_paths, config["proper_loss"]["brier_weight"])
    raw, projection = combined_route_features(rows, dataset, records, frozen, manifest["hidden_size"],
        config["semantic_features"]["projection_width"], config["semantic_features"]["projection_seed"])
    names = tuple(sorted(expert_paths)); sources = sorted(config["data"]["sources"])
    train_rows = [row for row in rows if row["split"] == "train"]; output.mkdir(parents=True); ensembles = []
    for outer_source in sources:
        training = [row for row in train_rows if row["source"] != outer_source]
        validation = [row for row in rows if row["split"] == "validation" and row["source"] == outer_source]
        train_matrix = torch.stack([raw[row["id"]] for row in training])
        validation_matrix = torch.stack([raw[row["id"]] for row in validation])
        standardized, fit_mean, fit_scale = standardize_bank_features(train_matrix, [train_matrix, validation_matrix])
        feature_map = {**{row["id"]: value for row, value in zip(training, standardized[0])},
                       **{row["id"]: value for row, value in zip(validation, standardized[1])}}
        members, member_evidence = [], []
        for seed in config["router_seeds"]:
            thresholds, calibration, inner = calibrate_thresholds(
                train_rows, outer_source, records, names, raw, config["training"], seed, output)
            module, gain_mean, gain_scale, details = train_model(
                training, records, names, feature_map, config["training"], seed,
                config["training"]["final_updates"], output / "plans" / f"outer-{outer_source}-final-{seed}.jsonl")
            members.append({"module": module, "gain_mean": gain_mean, "gain_scale": gain_scale,
                            "thresholds": thresholds})
            member_evidence.append({**details, "seed": seed, "thresholds": thresholds,
                "calibration": calibration, "inner_folds": inner})
        ensembles.append({"heldout_source": outer_source,
          "train_sources": sorted({row["source"] for row in training}), "members": member_evidence,
          "standardization": {"fit_rows": len(training), "mean_sha256": digest(fit_mean.numpy().tobytes()),
                              "scale_sha256": digest(fit_scale.numpy().tobytes())},
          "evaluation": evaluate_consensus(members, validation, records, names, feature_map)})
    metrics = ("path_match", "mean_proper_gain", "mean_regret", "accuracy_delta", "cross_entropy_delta",
               "expert_selection_rate", "safe_selection_precision", "violation_count")
    by_source = {run["heldout_source"]: {metric: run["evaluation"][metric] for metric in metrics}
                 for run in ensembles}
    aggregate = {metric: mean([run["evaluation"][metric] for run in ensembles]) for metric in metrics[:-1]}
    total_rows = sum(run["evaluation"]["rows"] for run in ensembles)
    total_selected = sum(run["evaluation"]["selected_experts"] for run in ensembles)
    total_unsafe = sum(run["evaluation"]["unsafe_selected_rows"] for run in ensembles)
    aggregate.update({"violation_rate": sum(run["evaluation"]["violation_count"] for run in ensembles) / total_rows,
                      "expert_selection_rate": total_selected / total_rows,
                      "safe_selection_precision": 1 - total_unsafe / total_selected if total_selected else 1.0})
    rule = config["screening_rule"]
    checks = {"coverage.folds_seeds": len(ensembles) == len(sources) and all(
        [member["seed"] for member in run["members"]] == config["router_seeds"] for run in ensembles),
      "coverage.no_outer_leakage": all(run["heldout_source"] not in run["train_sources"] and all(
        run["heldout_source"] not in fold["train_sources"] for member in run["members"]
        for fold in member["inner_folds"]) for run in ensembles),
      "combined.accuracy": aggregate["accuracy_delta"] >= rule["accuracy_min_delta"],
      "combined.cross_entropy": aggregate["cross_entropy_delta"] <= rule["cross_entropy_max_delta"],
      "sources.accuracy": all(v["accuracy_delta"] >= rule["each_source_accuracy_min_delta"] for v in by_source.values()),
      "sources.cross_entropy": all(v["cross_entropy_delta"] <= rule["each_source_cross_entropy_max_delta"] for v in by_source.values()),
      "safety.violation_rate": aggregate["violation_rate"] <= rule["violation_rate_max"],
      "safety.selection_precision": aggregate["safe_selection_precision"] >= rule["safe_selection_precision_min"],
      "coverage.expert_selection": aggregate["expert_selection_rate"] >= rule["expert_selection_rate_min"],
      "utility.proper_gain": aggregate["mean_proper_gain"] >= rule["proper_gain_min"]}
    evidence = {"format_version": 1, "study": config["study"],
      "role": "nested public-data unanimous semantic safe route validation",
      "protocol_sha256": digest(config_path.read_bytes()), "experts": list(names),
      "semantic_features": {**config["semantic_features"], "projection_sha256": digest(projection.numpy().tobytes()),
                            "combined_width": len(next(iter(raw.values())))},
      "ensembles": ensembles, "aggregate": aggregate, "by_source": by_source, "checks": checks,
      "advances_to_full_router_training": all(checks.values()), "limitations": config["limitations"]}
    write_json(output / "evidence.json", evidence); write_json(root / "docs/evidence/semantic-consensus-route-lofo-v1.json", evidence)
    table = [f"| {source} | {v['expert_selection_rate']:.4f} | {v['safe_selection_precision']:.4f} | "
             f"{v['mean_proper_gain']:+.4f} | {v['accuracy_delta']:+.4f} | {v['cross_entropy_delta']:+.4f} |"
             for source, v in by_source.items()]
    lines = ["# 语义安全路由三种子一致性结果", "", "只有三个种子都判定安全且收益为正的专家才被放行。", "",
      "| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |", "|---|---:|---:|---:|---:|---:|", *table, "", "## 汇总", "",
      f"- 专家选择率：`{aggregate['expert_selection_rate']:.4f}`。", f"- 已选专家安全精度：`{aggregate['safe_selection_precision']:.4f}`。",
      f"- 逐项安全违规率：`{aggregate['violation_rate']:.4f}`。", f"- 平均 proper-loss 收益：`{aggregate['mean_proper_gain']:+.4f}`。",
      f"- 准确率差：`{aggregate['accuracy_delta']:+.4f}`。", f"- 交叉熵差：`{aggregate['cross_entropy_delta']:+.4f}`。", "", "## 判定", "",
      ("全部门槛通过，允许进入已消费数据上的完整路由训练。" if all(checks.values()) else "至少一个冻结门槛失败，不训练完整路由端点。"), "",
      "逐项门槛：" + ", ".join(f"`{k}`={'PASS' if v else 'FAIL'}" for k, v in checks.items()), "",
      "机器证据：`docs/evidence/semantic-consensus-route-lofo-v1.json`", ""]
    (root / "docs/SEMANTIC_CONSENSUS_ROUTE_LOFO_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(canonical({"event": "complete", "advances": all(checks.values()), "aggregate": aggregate, "checks": checks}))


if __name__ == "__main__": main()
