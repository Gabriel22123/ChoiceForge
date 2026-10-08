#!/usr/bin/env python3
"""Cross-evaluate independently shaped frozen experts on consumed families."""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, write_json


BASE_PATH = Path(__file__).with_name("run_extended_expert_bank.py")
SPEC = importlib.util.spec_from_file_location("heterogeneous_bank_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(BASE)
read, load_features, load_module, collect_paths = (
    BASE.read, BASE.load_features, BASE.load_module, BASE.collect_paths)
prepare_records, summarize = BASE.prepare_records, BASE.summarize


def endpoint_model(default: dict, endpoint: dict, hidden_size: int) -> dict:
    model = {**default, **endpoint.get("model", {}), "hidden_size": hidden_size}
    required = {"hidden_size", "residual_width", "router_width", "dropout",
                "initial_gate_probability"}
    if set(model) != required or any(model[key] <= 0 for key in
                                     ("hidden_size", "residual_width", "router_width")):
        raise ValueError("Invalid heterogeneous expert model config")
    return model


def load_group(root: Path, group: dict, config: dict):
    data_path = root / group["data_path"]
    if digest(data_path.read_bytes()) != group["data_sha256"]:
        raise ValueError("Heterogeneous-bank data changed: " + group["name"])
    rows = [row for row in load_rows(data_path) if row["split"] in group["splits"] and
            row["source"] in group["sources"]]
    expected = {tuple(key.split("|")): value for key, value in group["counts"].items()}
    if Counter((row["source"], row["split"]) for row in rows) != expected:
        raise ValueError("Heterogeneous-bank group coverage changed: " + group["name"])
    cache = root / group["feature_cache"]
    if digest((cache / "manifest.json").read_bytes()) != group["manifest_sha256"]:
        raise ValueError("Heterogeneous-bank cache changed: " + group["name"])
    keys = [f"{group['dataset']}:{row['id']}" for row in rows]
    manifest, features = load_features(cache, keys)
    modules = {}
    for name, endpoint in config["experts"].items():
        weight = root / endpoint["weights"]
        if digest(weight.read_bytes()) != endpoint["weights_sha256"]:
            raise ValueError("Heterogeneous-bank expert changed: " + name)
        model = endpoint_model(config["default_model"], endpoint, manifest["hidden_size"])
        modules[name] = load_module(weight, model)
    paths = {name: collect_paths(module, rows, group["dataset"], features, manifest)
             for name, module in modules.items()}
    records = prepare_records(rows, group["dataset"], manifest, paths,
                              config["proper_loss"]["brier_weight"])
    return rows, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/heterogeneous-expert-bank-v2.json")
    parser.add_argument("--output", default="runs/heterogeneous-expert-bank-v2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh heterogeneous-bank output directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_evaluation":
        raise ValueError("Protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen heterogeneous-bank source changed: " + relative)
    rows, records = [], {}
    for group in config["groups"]:
        group_rows, group_records = load_group(root, group, config)
        if set(records).intersection(group_records):
            raise ValueError("Heterogeneous-bank row IDs collide across groups")
        rows.extend(group_rows); records.update(group_records)
    matrix, oracle = summarize(rows, records, tuple(sorted(config["experts"])))
    rule = config["screening_rule"]
    checks = {
        "coverage.sources": set(matrix) == set(rule["required_sources"]),
        "coverage.experts": all(set(values) == set(config["experts"])
                                for values in matrix.values()),
        "safety.zero_violations": oracle["violation_count"] == 0,
        "sources.accuracy": all(value["accuracy_delta"] >= 0
                                for value in oracle["by_source"].values()),
        "sources.cross_entropy": all(value["cross_entropy_delta"] <= 1e-12
                                     for value in oracle["by_source"].values()),
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "heterogeneous expert public-family cross evaluation",
        "protocol_sha256": digest(config_path.read_bytes()),
        "expert_models": {name: endpoint_model(config["default_model"], endpoint, 2048)
                          for name, endpoint in config["experts"].items()},
        "matrix": matrix, "oracle": oracle, "checks": checks,
        "passed": all(checks.values()), "limitations": config["limitations"],
    }
    output.mkdir(parents=True)
    write_json(output / "evidence.json", evidence)
    write_json(root / config["evidence_path"], evidence)
    names = sorted(config["experts"])
    header = "| 来源 | " + " | ".join(names) + " |"
    separator = "|---|" + "---:|" * len(names)
    table = ["| " + source + " | " + " | ".join(
        f"{matrix[source][name]['mean_proper_gain']:+.4f} / "
        f"{matrix[source][name]['safe_rate']:.2%}" for name in names) + " |"
             for source in sorted(matrix)]
    lines = [f"# {config['display_name']} 交叉验证结果", "",
      "单元格为 `平均 proper-loss 收益 / 安全可行率`。", "", header, separator,
      *table, "", "## 安全 oracle", "",
      f"- 验证行：`{oracle['rows']}`。",
      f"- 准确率：`{oracle['parent_accuracy']:.4f}` → `{oracle['oracle_accuracy']:.4f}`（`{oracle['accuracy_delta']:+.4f}`）。",
      f"- 交叉熵：`{oracle['parent_cross_entropy']:.4f}` → `{oracle['oracle_cross_entropy']:.4f}`（`{oracle['cross_entropy_delta']:+.4f}`）。",
      f"- 逐行违规：`{oracle['violation_count']}`。",
      f"- 路径分布：`{oracle['selection']}`。", "", "## 判定", "",
      ("全部完整性与安全门槛通过。" if all(checks.values()) else
       "至少一个完整性或安全门槛失败。"), "",
      f"机器证据：`{config['evidence_path']}`", ""]
    (root / config["report_path"]).write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "passed": all(checks.values())})
    print(canonical({"event": "complete", "passed": all(checks.values()),
                     "oracle": oracle, "checks": checks}))


if __name__ == "__main__":
    main()
