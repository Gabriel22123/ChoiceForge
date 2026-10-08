#!/usr/bin/env python3
"""LOFO factored safe routing with frozen permutation-invariant semantics."""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path

import torch

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.expert_bank_features import standardize_bank_features
from decision_model.semantic_route_features import fixed_projection, semantic_route_features


_V1_PATH = Path(__file__).with_name("run_factored_safe_route_lofo.py")
_V1_SPEC = importlib.util.spec_from_file_location("semantic_factored_v1", _V1_PATH)
_V1 = importlib.util.module_from_spec(_V1_SPEC); _V1_SPEC.loader.exec_module(_V1)
read, mean, load_features, load_module, collect_paths = (
    _V1.read, _V1.mean, _V1.load_features, _V1.load_module, _V1.collect_paths)
prepare_records, calibrate_thresholds, train_model, evaluate = (
    _V1.prepare_records, _V1.calibrate_thresholds, _V1.train_model, _V1.evaluate)


def combined_route_features(rows, dataset, records, frozen_features, hidden_size,
                            projection_width, projection_seed):
    projection = fixed_projection(hidden_size, projection_width, projection_seed)
    output = {}
    for row in rows:
        full, null = frozen_features[f"{dataset}:{row['id']}"]
        semantic = semantic_route_features(full, null, projection)
        output[row["id"]] = torch.cat((records[row["id"]]["features"], semantic))
    return output, projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic-factored-route-lofo-v1.json")
    parser.add_argument("--output", default="runs/semantic-factored-route-lofo-v1")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists(): raise ValueError("Use a fresh semantic factored route directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training": raise ValueError("Protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen semantic route source changed: " + relative)
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
    manifest, features = load_features(cache, keys)
    model = {**config["model"], "hidden_size": manifest["hidden_size"]}
    expert_paths = {}
    for name, endpoint in config["experts"].items():
        weights = root / endpoint["weights"]
        if digest(weights.read_bytes()) != endpoint["weights_sha256"]: raise ValueError("Expert changed: " + name)
        expert_paths[name] = collect_paths(load_module(weights, model), rows, dataset, features, manifest)
    records = prepare_records(rows, dataset, manifest, expert_paths,
                              config["proper_loss"]["brier_weight"])
    raw, projection = combined_route_features(
        rows, dataset, records, features, manifest["hidden_size"],
        config["semantic_features"]["projection_width"],
        config["semantic_features"]["projection_seed"])
    names = tuple(sorted(expert_paths)); sources = sorted(config["data"]["sources"])
    output.mkdir(parents=True); runs = []; train_rows = [row for row in rows if row["split"] == "train"]
    for outer_source in sources:
        training = [row for row in train_rows if row["source"] != outer_source]
        validation = [row for row in rows if row["split"] == "validation" and row["source"] == outer_source]
        train_matrix = torch.stack([raw[row["id"]] for row in training])
        validation_matrix = torch.stack([raw[row["id"]] for row in validation])
        standardized, fit_mean, fit_scale = standardize_bank_features(train_matrix, [train_matrix, validation_matrix])
        feature_map = {**{row["id"]: value for row, value in zip(training, standardized[0])},
                       **{row["id"]: value for row, value in zip(validation, standardized[1])}}
        for seed in config["router_seeds"]:
            thresholds, calibration, inner = calibrate_thresholds(
                train_rows, outer_source, records, names, raw, config["training"], seed, output)
            module, gain_mean, gain_scale, details = train_model(
                training, records, names, feature_map, config["training"], seed,
                config["training"]["final_updates"], output / "plans" / f"outer-{outer_source}-final-{seed}.jsonl")
            details.update({"seed": seed, "heldout_source": outer_source,
                "train_sources": sorted({row["source"] for row in training}),
                "thresholds": thresholds, "calibration": calibration, "inner_folds": inner,
                "standardization": {"fit_rows": len(training),
                    "mean_sha256": digest(fit_mean.numpy().tobytes()),
                    "scale_sha256": digest(fit_scale.numpy().tobytes())},
                "evaluation": evaluate(module, validation, records, names, feature_map,
                                       gain_mean, gain_scale, thresholds)})
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
      "role": "nested public-data semantic factored safe route validation",
      "protocol_sha256": digest(config_path.read_bytes()), "experts": list(names),
      "semantic_features": {**config["semantic_features"], "projection_sha256": digest(projection.numpy().tobytes()),
                            "combined_width": len(next(iter(raw.values())))},
      "runs": runs, "aggregate": aggregate, "by_source": by_source, "checks": checks,
      "advances_to_full_router_training": all(checks.values()), "limitations": config["limitations"]}
    write_json(output / "evidence.json", evidence); write_json(root / "docs/evidence/semantic-factored-route-lofo-v1.json", evidence)
    table = [f"| {source} | {v['expert_selection_rate']:.4f} | {v['safe_selection_precision']:.4f} | "
             f"{v['mean_proper_gain']:+.4f} | {v['accuracy_delta']:+.4f} | {v['cross_entropy_delta']:+.4f} |"
             for source, v in by_source.items()]
    lines = ["# 语义增强分解式安全路由留一任务族结果", "",
      "冻结 2B 语义摘要与行为特征拼接；外层任务族完全留出，阈值只由训练来源内层留出预测确定。", "",
      "| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |", "|---|---:|---:|---:|---:|---:|", *table, "", "## 汇总", "",
      f"- 专家选择率：`{aggregate['expert_selection_rate']:.4f}`。", f"- 已选专家安全精度：`{aggregate['safe_selection_precision']:.4f}`。",
      f"- 逐项安全违规率：`{aggregate['violation_rate']:.4f}`。", f"- 平均 proper-loss 收益：`{aggregate['mean_proper_gain']:+.4f}`。",
      f"- 准确率差：`{aggregate['accuracy_delta']:+.4f}`。", f"- 交叉熵差：`{aggregate['cross_entropy_delta']:+.4f}`。", "", "## 判定", "",
      ("全部门槛通过，允许进入已消费数据上的完整路由训练。" if all(checks.values()) else "至少一个冻结门槛失败，不训练完整路由端点。"), "",
      "逐项门槛：" + ", ".join(f"`{k}`={'PASS' if v else 'FAIL'}" for k, v in checks.items()), "",
      "机器证据：`docs/evidence/semantic-factored-route-lofo-v1.json`", ""]
    (root / "docs/SEMANTIC_FACTORED_ROUTE_LOFO_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(canonical({"event": "complete", "advances": all(checks.values()), "aggregate": aggregate, "checks": checks}))


if __name__ == "__main__": main()
