#!/usr/bin/env python3
"""Nine-family, six-expert semantic consensus route validation."""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path

import torch

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.expert_bank_features import standardize_bank_features


_CONSENSUS_PATH = Path(__file__).with_name("run_semantic_consensus_route_lofo.py")
_SPEC = importlib.util.spec_from_file_location("heterogeneous_consensus_base", _CONSENSUS_PATH)
_BASE = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_BASE)
read, mean, load_features, load_module, collect_paths = (
    _BASE.read, _BASE.mean, _BASE.load_features, _BASE.load_module, _BASE.collect_paths)
prepare_records, combined_route_features = _BASE.prepare_records, _BASE.combined_route_features
calibrate_thresholds, train_model, evaluate_consensus = (
    _BASE.calibrate_thresholds, _BASE.train_model, _BASE.evaluate_consensus)

_BANK_PATH = Path(__file__).with_name("run_heterogeneous_expert_bank.py")
_BANK_SPEC = importlib.util.spec_from_file_location("heterogeneous_consensus_bank", _BANK_PATH)
_BANK = importlib.util.module_from_spec(_BANK_SPEC); _BANK_SPEC.loader.exec_module(_BANK)
endpoint_model = _BANK.endpoint_model


def load_route_group(root, group, expert_endpoints, default_model, brier_weight,
                     semantic_config):
    data_path = root / group["data_path"]
    if digest(data_path.read_bytes()) != group["data_sha256"]:
        raise ValueError("Data changed: " + group["name"])
    rows = [row for row in load_rows(data_path)
            if row["split"] in group["splits"] and row["source"] in group["sources"]]
    expected = {tuple(key.split("|")): value for key, value in group["counts"].items()}
    if Counter((row["source"], row["split"]) for row in rows) != expected:
        raise ValueError("Coverage changed: " + group["name"])
    cache = root / group["feature_cache"]
    if digest((cache / "manifest.json").read_bytes()) != group["manifest_sha256"]:
        raise ValueError("Cache changed: " + group["name"])
    keys = [f"{group['dataset']}:{row['id']}" for row in rows]
    manifest, frozen = load_features(cache, keys)
    paths = {}
    for name, endpoint in expert_endpoints.items():
        weight = root / endpoint["weights"]
        if digest(weight.read_bytes()) != endpoint["weights_sha256"]:
            raise ValueError("Expert changed: " + name)
        model = endpoint_model(default_model, endpoint, manifest["hidden_size"])
        paths[name] = collect_paths(
            load_module(weight, model), rows, group["dataset"], frozen, manifest)
    records = prepare_records(rows, group["dataset"], manifest, paths, brier_weight)
    raw, projection = combined_route_features(
        rows, group["dataset"], records, frozen, manifest["hidden_size"],
        semantic_config["projection_width"], semantic_config["projection_seed"])
    return rows, records, raw, projection


def aggregate_results(ensembles, sources):
    metrics = ("path_match", "mean_proper_gain", "mean_regret", "accuracy_delta",
               "cross_entropy_delta", "expert_selection_rate", "safe_selection_precision",
               "violation_count")
    by_source = {run["heldout_source"]: {
        metric: run["evaluation"][metric] for metric in metrics} for run in ensembles}
    aggregate = {metric: mean([run["evaluation"][metric] for run in ensembles])
                 for metric in metrics[:-1]}
    total_rows = sum(run["evaluation"]["rows"] for run in ensembles)
    total_selected = sum(run["evaluation"]["selected_experts"] for run in ensembles)
    total_unsafe = sum(run["evaluation"]["unsafe_selected_rows"] for run in ensembles)
    aggregate.update({
        "violation_rate": sum(run["evaluation"]["violation_count"]
                              for run in ensembles) / total_rows,
        "expert_selection_rate": total_selected / total_rows,
        "safe_selection_precision": 1 - total_unsafe / total_selected
        if total_selected else 1.0,
    })
    if set(by_source) != set(sources):
        raise ValueError("Aggregate source coverage changed")
    return aggregate, by_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/heterogeneous-semantic-consensus-route-v2.json")
    parser.add_argument("--output", default="runs/heterogeneous-semantic-consensus-route-v2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh heterogeneous consensus output directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen source changed: " + relative)

    rows, records, raw, projection = [], {}, {}, None
    for group in config["groups"]:
        group_rows, group_records, group_raw, group_projection = load_route_group(
            root, group, config["experts"], config["default_model"],
            config["proper_loss"]["brier_weight"], config["semantic_features"])
        if set(records).intersection(group_records):
            raise ValueError("Row IDs collide across groups")
        if projection is not None and not torch.equal(projection, group_projection):
            raise ValueError("Projection changed across groups")
        projection = group_projection
        rows.extend(group_rows); records.update(group_records); raw.update(group_raw)

    names = tuple(sorted(config["experts"]))
    sources = sorted({row["source"] for row in rows})
    train_rows = [row for row in rows if row["split"] == "train"]
    output.mkdir(parents=True)
    ensembles = []
    for outer_source in sources:
        training = [row for row in train_rows if row["source"] != outer_source]
        validation = [row for row in rows
                      if row["split"] == "validation" and row["source"] == outer_source]
        train_matrix = torch.stack([raw[row["id"]] for row in training])
        validation_matrix = torch.stack([raw[row["id"]] for row in validation])
        standardized, fit_mean, fit_scale = standardize_bank_features(
            train_matrix, [train_matrix, validation_matrix])
        feature_map = {
            **{row["id"]: value for row, value in zip(training, standardized[0])},
            **{row["id"]: value for row, value in zip(validation, standardized[1])},
        }
        members, evidence_members = [], []
        for seed in config["router_seeds"]:
            thresholds, calibration, inner = calibrate_thresholds(
                train_rows, outer_source, records, names, raw, config["training"], seed, output)
            module, gain_mean, gain_scale, details = train_model(
                training, records, names, feature_map, config["training"], seed,
                config["training"]["final_updates"],
                output / "plans" / f"outer-{outer_source}-final-{seed}.jsonl")
            members.append({"module": module, "gain_mean": gain_mean,
                            "gain_scale": gain_scale, "thresholds": thresholds})
            evidence_members.append({**details, "seed": seed, "thresholds": thresholds,
                                     "calibration": calibration, "inner_folds": inner})
        ensembles.append({
            "heldout_source": outer_source,
            "train_sources": sorted({row["source"] for row in training}),
            "members": evidence_members,
            "standardization": {
                "fit_rows": len(training),
                "mean_sha256": digest(fit_mean.numpy().tobytes()),
                "scale_sha256": digest(fit_scale.numpy().tobytes()),
            },
            "evaluation": evaluate_consensus(members, validation, records, names, feature_map),
        })

    aggregate, by_source = aggregate_results(ensembles, sources)
    rule = config["screening_rule"]
    checks = {
        "coverage.folds_seeds": len(ensembles) == len(sources) and all(
            [member["seed"] for member in run["members"]] == config["router_seeds"]
            for run in ensembles),
        "coverage.no_outer_leakage": all(
            run["heldout_source"] not in run["train_sources"] and all(
                run["heldout_source"] not in fold["train_sources"]
                for member in run["members"] for fold in member["inner_folds"])
            for run in ensembles),
        "combined.accuracy": aggregate["accuracy_delta"] >= rule["accuracy_min_delta"],
        "combined.cross_entropy": aggregate["cross_entropy_delta"] <= rule["cross_entropy_max_delta"],
        "sources.accuracy": all(value["accuracy_delta"] >= rule["each_source_accuracy_min_delta"]
                                for value in by_source.values()),
        "sources.cross_entropy": all(
            value["cross_entropy_delta"] <= rule["each_source_cross_entropy_max_delta"]
            for value in by_source.values()),
        "safety.violation_rate": aggregate["violation_rate"] <= rule["violation_rate_max"],
        "safety.selection_precision": (
            aggregate["safe_selection_precision"] >= rule["safe_selection_precision_min"]),
        "coverage.expert_selection": (
            aggregate["expert_selection_rate"] >= rule["expert_selection_rate_min"]),
        "utility.proper_gain": aggregate["mean_proper_gain"] >= rule["proper_gain_min"],
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "nine-family six-expert heterogeneous semantic consensus LOFO",
        "protocol_sha256": digest(config_path.read_bytes()), "experts": list(names),
        "semantic_features": {
            **config["semantic_features"],
            "projection_sha256": digest(projection.numpy().tobytes()),
            "combined_width": len(next(iter(raw.values()))),
        },
        "ensembles": ensembles, "aggregate": aggregate, "by_source": by_source,
        "checks": checks, "advances_to_full_router_training": all(checks.values()),
        "limitations": config["limitations"],
    }
    write_json(output / "evidence.json", evidence)
    write_json(root / config["evidence_path"], evidence)
    table = [
        f"| {source} | {value['expert_selection_rate']:.4f} | "
        f"{value['safe_selection_precision']:.4f} | {value['mean_proper_gain']:+.4f} | "
        f"{value['accuracy_delta']:+.4f} | {value['cross_entropy_delta']:+.4f} |"
        for source, value in by_source.items()
    ]
    lines = [
        f"# {config['display_name']}结果", "",
        "六个异构专家在九个已消费来源上执行三种子一致性路由；每个外层来源完全留出。", "",
        "| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |",
        "|---|---:|---:|---:|---:|---:|", *table, "", "## 汇总", "",
        f"- 专家选择率：`{aggregate['expert_selection_rate']:.4f}`。",
        f"- 已选专家安全精度：`{aggregate['safe_selection_precision']:.4f}`。",
        f"- 逐项安全违规率：`{aggregate['violation_rate']:.4f}`。",
        f"- 平均 proper-loss 收益：`{aggregate['mean_proper_gain']:+.4f}`。",
        f"- 准确率差：`{aggregate['accuracy_delta']:+.4f}`。",
        f"- 交叉熵差：`{aggregate['cross_entropy_delta']:+.4f}`。", "", "## 判定", "",
        ("全部门槛通过，允许进入已消费数据上的完整路由训练。"
         if all(checks.values()) else "至少一个冻结门槛失败，不训练完整路由端点。"), "",
        "逐项门槛：" + ", ".join(
            f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
        f"机器证据：`{config['evidence_path']}`", "",
    ]
    (root / config["report_path"]).write_text("\n".join(lines))
    write_json(output / "status.json", {
        "state": "complete", "advances": all(checks.values())})
    print(canonical({"event": "complete", "advances": all(checks.values()),
                     "aggregate": aggregate, "checks": checks}))


if __name__ == "__main__":
    main()
