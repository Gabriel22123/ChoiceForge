#!/usr/bin/env python3
"""Seven-family LOFO route with independently trained per-expert modules."""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path

import torch

from decision_model.core import canonical, digest, write_json
from decision_model.expert_bank_features import standardize_bank_features
from decision_model.modular_route_features import independent_expert_features
from decision_model.safe_routing import safe_improvement
from decision_model.train import evaluate_rows


_EXTENDED_PATH = Path(__file__).with_name("run_extended_semantic_consensus_route.py")
_SPEC = importlib.util.spec_from_file_location("modular_route_base", _EXTENDED_PATH)
_BASE = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_BASE)
read, mean, load_route_group = _BASE.read, _BASE.mean, _BASE.load_route_group
calibrate_thresholds, train_model, predict = (
    _BASE.calibrate_thresholds, _BASE.train_model, _BASE._BASE._SEMANTIC._V1.predict)


def expert_feature_maps(raw, names, semantic_width):
    return {name: {row_id: independent_expert_features(value, names, name, semantic_width)
                   for row_id, value in raw.items()} for name in names}


def evaluate_modules(modules, rows, records, names, features):
    predictions = {}
    for name in names:
        predictions[name] = [(predict(member["module"], rows, features[name],
                                      member["gain_mean"], member["gain_scale"]),
                              member["threshold"]) for member in modules[name]]
    paths, logits, gains, regrets, matches = [], [], [], [], []
    violations, unsafe, selected = Counter(), 0, 0
    for row_index, row in enumerate(rows):
        candidates = []; record = records[row["id"]]
        for name in names:
            seed_values = []
            for (probabilities, predicted_gains), threshold in predictions[name]:
                probability = float(probabilities[row_index, 0]); gain = float(predicted_gains[row_index, 0])
                seed_values.append((probability >= threshold, gain))
            if all(safe and gain > 0 for safe, gain in seed_values):
                candidates.append((min(gain for _, gain in seed_values), name))
        path = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1] if candidates else "parent"
        parent = record["parent"]; statistics = parent if path == "parent" else record["experts"][path]
        selected += int(path != "parent"); unsafe += int(path != "parent" and not safe_improvement(parent, statistics)["eligible"])
        violations["parent_correct_lost"] += int(parent["correct"] and not statistics["correct"])
        violations["cross_entropy_regressed"] += int(statistics["cross_entropy"] > parent["cross_entropy"] + 1e-12)
        violations["brier_regressed"] += int(statistics["brier"] > parent["brier"] + 1e-12)
        violations["context_regressed"] += int(statistics["context_advantage"] + 1e-12 < parent["context_advantage"])
        gain = parent["proper_cost"] - statistics["proper_cost"]
        gains.append(gain); regrets.append(record["oracle_improvement"] - gain); matches.append(path == record["target_path"]); paths.append(path)
        logits.append(record["parent_logits"] if path == "parent" else record["expert_logits"][path])
    parent_logits = [records[row["id"]]["parent_logits"] for row in rows]
    pm, _ = evaluate_rows(rows, parent_logits, 1.0); rm, _ = evaluate_rows(rows, logits, 1.0)
    return {"rows": len(rows), "selection": dict(Counter(paths)), "path_match": mean(matches),
      "selected_experts": selected, "unsafe_selected_rows": unsafe,
      "expert_selection_rate": selected / len(rows), "safe_selection_precision": 1 - unsafe / selected if selected else 1.0,
      "mean_proper_gain": mean(gains), "mean_regret": mean(regrets),
      "accuracy_delta": rm["accuracy"] - pm["accuracy"], "cross_entropy_delta": rm["nll"] - pm["nll"],
      "violation_count": sum(violations.values()), "violations": dict(violations)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/modular-expert-route-v1.json")
    parser.add_argument("--output", default="runs/modular-expert-route-v1"); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; config_path, output = root / args.config, root / args.output
    if output.exists(): raise ValueError("Use a fresh modular route output directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training": raise ValueError("Protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected: raise ValueError("Frozen source changed: " + relative)
    rows, records, raw, projection = [], {}, {}, None
    for group in config["groups"]:
        gr, grec, graw, gproj = load_route_group(root, group, config["experts"], config["model"],
            config["proper_loss"]["brier_weight"], config["semantic_features"])
        if set(records).intersection(grec): raise ValueError("Row IDs collide")
        if projection is not None and not torch.equal(projection, gproj): raise ValueError("Projection changed")
        projection = gproj; rows.extend(gr); records.update(grec); raw.update(graw)
    names = tuple(sorted(config["experts"])); semantic_width = 4 * config["semantic_features"]["projection_width"]
    expert_raw = expert_feature_maps(raw, names, semantic_width)
    if any(len(next(iter(values.values()))) != config["module_input_width"] for values in expert_raw.values()):
        raise ValueError("Independent module width changed")
    sources = sorted({row["source"] for row in rows}); train_rows = [row for row in rows if row["split"] == "train"]
    output.mkdir(parents=True); ensembles = []
    for outer_source in sources:
        training = [row for row in train_rows if row["source"] != outer_source]
        validation = [row for row in rows if row["split"] == "validation" and row["source"] == outer_source]
        modules, evidence_modules, standardized_maps = {}, {}, {}
        for name in names:
            train_matrix = torch.stack([expert_raw[name][row["id"]] for row in training])
            val_matrix = torch.stack([expert_raw[name][row["id"]] for row in validation])
            standardized, fit_mean, fit_scale = standardize_bank_features(train_matrix, [train_matrix, val_matrix])
            fmap = {**{row["id"]: value for row, value in zip(training, standardized[0])},
                    **{row["id"]: value for row, value in zip(validation, standardized[1])}}
            standardized_maps[name] = fmap; modules[name] = []; evidence_modules[name] = []
            for seed in config["router_seeds"]:
                thresholds, calibration, inner = calibrate_thresholds(
                    train_rows, outer_source, records, (name,), expert_raw[name], config["training"], seed, output)
                module, gain_mean, gain_scale, details = train_model(
                    training, records, (name,), fmap, config["training"], seed, config["training"]["final_updates"],
                    output / "plans" / f"outer-{outer_source}-expert-{name}-{seed}.jsonl")
                modules[name].append({"module": module, "gain_mean": gain_mean, "gain_scale": gain_scale,
                                      "threshold": thresholds[name]})
                evidence_modules[name].append({**details, "seed": seed, "threshold": thresholds[name],
                  "calibration": calibration[name], "inner_folds": inner})
            evidence_modules[name + "_standardization"] = {"fit_rows": len(training),
              "mean_sha256": digest(fit_mean.numpy().tobytes()), "scale_sha256": digest(fit_scale.numpy().tobytes())}
        ensembles.append({"heldout_source": outer_source, "train_sources": sorted({row["source"] for row in training}),
          "modules": evidence_modules, "evaluation": evaluate_modules(modules, validation, records, names, standardized_maps)})
    metrics = ("path_match", "mean_proper_gain", "mean_regret", "accuracy_delta", "cross_entropy_delta",
               "expert_selection_rate", "safe_selection_precision", "violation_count")
    by_source = {run["heldout_source"]: {m: run["evaluation"][m] for m in metrics} for run in ensembles}
    aggregate = {m: mean([run["evaluation"][m] for run in ensembles]) for m in metrics[:-1]}
    total_rows = sum(run["evaluation"]["rows"] for run in ensembles); total_selected = sum(run["evaluation"]["selected_experts"] for run in ensembles)
    total_unsafe = sum(run["evaluation"]["unsafe_selected_rows"] for run in ensembles)
    aggregate.update({"violation_rate": sum(run["evaluation"]["violation_count"] for run in ensembles) / total_rows,
      "expert_selection_rate": total_selected / total_rows, "safe_selection_precision": 1 - total_unsafe / total_selected if total_selected else 1.0})
    rule = config["screening_rule"]
    checks = {"coverage.modules": all(all(name in run["modules"] for name in names) for run in ensembles),
      "coverage.no_outer_leakage": all(run["heldout_source"] not in run["train_sources"] for run in ensembles),
      "combined.accuracy": aggregate["accuracy_delta"] >= rule["accuracy_min_delta"],
      "combined.cross_entropy": aggregate["cross_entropy_delta"] <= rule["cross_entropy_max_delta"],
      "sources.accuracy": all(v["accuracy_delta"] >= rule["each_source_accuracy_min_delta"] for v in by_source.values()),
      "sources.cross_entropy": all(v["cross_entropy_delta"] <= rule["each_source_cross_entropy_max_delta"] for v in by_source.values()),
      "safety.violation_rate": aggregate["violation_rate"] <= rule["violation_rate_max"],
      "safety.selection_precision": aggregate["safe_selection_precision"] >= rule["safe_selection_precision_min"],
      "coverage.expert_selection": aggregate["expert_selection_rate"] >= rule["expert_selection_rate_min"],
      "utility.proper_gain": aggregate["mean_proper_gain"] >= rule["proper_gain_min"]}
    evidence = {"format_version": 1, "study": config["study"], "role": "independent per-expert semantic consensus LOFO",
      "protocol_sha256": digest(config_path.read_bytes()), "experts": list(names), "module_input_width": config["module_input_width"],
      "projection_sha256": digest(projection.numpy().tobytes()), "ensembles": ensembles, "aggregate": aggregate,
      "by_source": by_source, "checks": checks, "advances_to_full_router_training": all(checks.values()),
      "limitations": config["limitations"]}
    write_json(output / "evidence.json", evidence); write_json(root / "docs/evidence/modular-expert-route-v1.json", evidence)
    table = [f"| {s} | {v['expert_selection_rate']:.4f} | {v['safe_selection_precision']:.4f} | {v['mean_proper_gain']:+.4f} | {v['accuracy_delta']:+.4f} | {v['cross_entropy_delta']:+.4f} |" for s,v in by_source.items()]
    lines = ["# 独立专家模块语义一致性路由结果", "", "每个专家使用固定 280 维输入并独立训练、校准。", "",
      "| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |", "|---|---:|---:|---:|---:|---:|", *table, "", "## 汇总", "",
      f"- 专家选择率：`{aggregate['expert_selection_rate']:.4f}`。", f"- 安全精度：`{aggregate['safe_selection_precision']:.4f}`。",
      f"- 违规率：`{aggregate['violation_rate']:.4f}`。", f"- proper 收益：`{aggregate['mean_proper_gain']:+.4f}`。",
      f"- 准确率差：`{aggregate['accuracy_delta']:+.4f}`。", f"- 交叉熵差：`{aggregate['cross_entropy_delta']:+.4f}`。", "", "## 判定", "",
      ("全部门槛通过。" if all(checks.values()) else "至少一个冻结门槛失败。"), "",
      "机器证据：`docs/evidence/modular-expert-route-v1.json`", ""]
    (root / "docs/MODULAR_EXPERT_ROUTE_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(canonical({"event": "complete", "advances": all(checks.values()), "aggregate": aggregate, "checks": checks}))


if __name__ == "__main__": main()
