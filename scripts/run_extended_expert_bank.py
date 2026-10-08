#!/usr/bin/env python3
"""Cross-evaluate five frozen experts on seven consumed public families."""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.safe_routing import safe_improvement
from decision_model.train import evaluate_rows


_BASE_PATH = Path(__file__).with_name("run_semantic_factored_route_lofo.py")
_SPEC = importlib.util.spec_from_file_location("extended_bank_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_BASE)
read, mean, load_features, load_module, collect_paths = (
    _BASE.read, _BASE.mean, _BASE.load_features, _BASE.load_module, _BASE.collect_paths)
prepare_records = _BASE.prepare_records


def load_group(root, group, experts, model_config, brier_weight):
    data_path = root / group["data_path"]
    if digest(data_path.read_bytes()) != group["data_sha256"]:
        raise ValueError("Extended-bank data changed: " + group["name"])
    rows = [row for row in load_rows(data_path) if row["split"] in group["splits"] and
            row["source"] in group["sources"]]
    expected = {tuple(key.split("|")): value for key, value in group["counts"].items()}
    if Counter((row["source"], row["split"]) for row in rows) != expected:
        raise ValueError("Extended-bank group coverage changed: " + group["name"])
    cache = root / group["feature_cache"]
    if digest((cache / "manifest.json").read_bytes()) != group["manifest_sha256"]:
        raise ValueError("Extended-bank cache changed: " + group["name"])
    keys = [f"{group['dataset']}:{row['id']}" for row in rows]
    manifest, features = load_features(cache, keys)
    model = {**model_config, "hidden_size": manifest["hidden_size"]}
    paths = {name: collect_paths(module, rows, group["dataset"], features, manifest)
             for name, module in experts(model).items()}
    records = prepare_records(rows, group["dataset"], manifest, paths, brier_weight)
    return rows, records


def summarize(rows, records, expert_names):
    validation = [row for row in rows if row["split"] == "validation"]
    matrix = defaultdict(dict)
    for source in sorted({row["source"] for row in validation}):
        selected = [row for row in validation if row["source"] == source]
        parent_logits = [records[row["id"]]["parent_logits"] for row in selected]
        parent_metrics, _ = evaluate_rows(selected, parent_logits, 1.0)
        for name in expert_names:
            expert_logits = [records[row["id"]]["expert_logits"][name] for row in selected]
            expert_metrics, _ = evaluate_rows(selected, expert_logits, 1.0)
            decisions = [safe_improvement(records[row["id"]]["parent"],
                                          records[row["id"]]["experts"][name])
                         for row in selected]
            matrix[source][name] = {
                "rows": len(selected),
                "accuracy_delta": expert_metrics["accuracy"] - parent_metrics["accuracy"],
                "cross_entropy_delta": expert_metrics["nll"] - parent_metrics["nll"],
                "mean_proper_gain": mean([value["improvement"] for value in decisions]),
                "safe_rate": mean([value["eligible"] for value in decisions]),
            }
    oracle_logits, violations, selection = [], Counter(), Counter()
    for row in validation:
        record = records[row["id"]]; path = record["target_path"]; selection[path] += 1
        statistics = record["parent"] if path == "parent" else record["experts"][path]
        parent = record["parent"]
        violations["parent_correct_lost"] += int(parent["correct"] and not statistics["correct"])
        violations["cross_entropy_regressed"] += int(statistics["cross_entropy"] > parent["cross_entropy"] + 1e-12)
        violations["brier_regressed"] += int(statistics["brier"] > parent["brier"] + 1e-12)
        violations["context_regressed"] += int(statistics["context_advantage"] + 1e-12 < parent["context_advantage"])
        oracle_logits.append(record["parent_logits"] if path == "parent" else record["expert_logits"][path])
    parent_logits = [records[row["id"]]["parent_logits"] for row in validation]
    parent_metrics, _ = evaluate_rows(validation, parent_logits, 1.0)
    oracle_metrics, _ = evaluate_rows(validation, oracle_logits, 1.0)
    by_source = {}
    for source in sorted(matrix):
        indices = [index for index, row in enumerate(validation) if row["source"] == source]
        source_rows = [validation[index] for index in indices]
        pm, _ = evaluate_rows(source_rows, [parent_logits[index] for index in indices], 1.0)
        om, _ = evaluate_rows(source_rows, [oracle_logits[index] for index in indices], 1.0)
        by_source[source] = {"rows": len(indices), "accuracy_delta": om["accuracy"] - pm["accuracy"],
                             "cross_entropy_delta": om["nll"] - pm["nll"]}
    return dict(matrix), {"rows": len(validation), "selection": dict(selection),
        "violations": dict(violations), "violation_count": sum(violations.values()),
        "parent_accuracy": parent_metrics["accuracy"], "oracle_accuracy": oracle_metrics["accuracy"],
        "accuracy_delta": oracle_metrics["accuracy"] - parent_metrics["accuracy"],
        "parent_cross_entropy": parent_metrics["nll"], "oracle_cross_entropy": oracle_metrics["nll"],
        "cross_entropy_delta": oracle_metrics["nll"] - parent_metrics["nll"],
        "by_source": by_source}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extended-expert-bank-v1.json")
    parser.add_argument("--output", default="runs/extended-expert-bank-v1")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists(): raise ValueError("Use a fresh extended-bank output directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_evaluation": raise ValueError("Protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen extended-bank source changed: " + relative)
    def experts(model):
        modules = {}
        for name, endpoint in config["experts"].items():
            weight = root / endpoint["weights"]
            if digest(weight.read_bytes()) != endpoint["weights_sha256"]:
                raise ValueError("Extended-bank expert changed: " + name)
            modules[name] = load_module(weight, model)
        return modules
    rows, records = [], {}
    for group in config["groups"]:
        group_rows, group_records = load_group(
            root, group, experts, config["model"], config["proper_loss"]["brier_weight"])
        duplicate = set(records).intersection(group_records)
        if duplicate: raise ValueError("Extended-bank row IDs collide across groups")
        rows.extend(group_rows); records.update(group_records)
    matrix, oracle = summarize(rows, records, tuple(sorted(config["experts"])))
    rule = config["screening_rule"]
    checks = {"coverage.sources": set(matrix) == set(rule["required_sources"]),
              "coverage.experts": all(set(values) == set(config["experts"]) for values in matrix.values()),
              "safety.zero_violations": oracle["violation_count"] == 0,
              "sources.accuracy": all(value["accuracy_delta"] >= 0 for value in oracle["by_source"].values()),
              "sources.cross_entropy": all(value["cross_entropy_delta"] <= 1e-12
                                            for value in oracle["by_source"].values())}
    evidence = {"format_version": 1, "study": config["study"],
      "role": "five-expert seven-family public cross evaluation", "protocol_sha256": digest(config_path.read_bytes()),
      "matrix": matrix, "oracle": oracle, "checks": checks, "passed": all(checks.values()),
      "limitations": config["limitations"]}
    output.mkdir(parents=True); write_json(output / "evidence.json", evidence)
    write_json(root / "docs/evidence/extended-expert-bank-v1.json", evidence)
    header = "| 来源 | " + " | ".join(sorted(config["experts"])) + " |"
    separator = "|---|" + "---:|" * len(config["experts"])
    table = ["| " + source + " | " + " | ".join(
        f"{matrix[source][name]['mean_proper_gain']:+.4f} / {matrix[source][name]['safe_rate']:.2%}"
        for name in sorted(config["experts"])) + " |" for source in sorted(matrix)]
    lines = ["# TruthfulQA 扩展专家库交叉验证结果", "",
      "单元格为 `平均 proper-loss 收益 / 安全可行率`。", "", header, separator, *table, "", "## 安全 oracle", "",
      f"- 验证行：`{oracle['rows']}`。", f"- 准确率：`{oracle['parent_accuracy']:.4f}` → `{oracle['oracle_accuracy']:.4f}`（`{oracle['accuracy_delta']:+.4f}`）。",
      f"- 交叉熵：`{oracle['parent_cross_entropy']:.4f}` → `{oracle['oracle_cross_entropy']:.4f}`（`{oracle['cross_entropy_delta']:+.4f}`）。",
      f"- 逐行违规：`{oracle['violation_count']}`。", f"- 路径分布：`{oracle['selection']}`。", "", "## 判定", "",
      ("全部完整性与安全门槛通过。" if all(checks.values()) else "至少一个完整性或安全门槛失败。"), "",
      "机器证据：`docs/evidence/extended-expert-bank-v1.json`", ""]
    (root / "docs/EXTENDED_EXPERT_BANK_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "passed": all(checks.values())})
    print(canonical({"event": "complete", "passed": all(checks.values()), "oracle": oracle, "checks": checks}))


if __name__ == "__main__": main()
