#!/usr/bin/env python3
"""Screen frozen QASC expert utility on the consumed public ARC families."""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from decision_model.core import digest, load_rows, write_json
from decision_model.feature_screen import RoutedResidualHead, parameter_sha256
from decision_model.route_utility import signed_route_utility
from decision_model.train import evaluate_rows


_BASE_PATH = Path(__file__).with_name("run_qasc_binary_route_screen.py")
_BASE_SPEC = importlib.util.spec_from_file_location("arc_expert_utility_helper", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_BASE_SPEC); _BASE_SPEC.loader.exec_module(_BASE)
composite_gate_costs = _BASE.composite_gate_costs
load_features, pad_batch = _BASE.load_features, _BASE.pad_batch


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/arc-expert-utility-screen-v1.json")
    parser.add_argument("--output", default="runs/arc-expert-utility-screen-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh ARC expert-utility screen directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_screen":
        raise ValueError("ARC expert-utility protocol was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen ARC expert-utility source changed: " + relative)
    data_path = root / config["data"]["path"]
    if digest(data_path.read_bytes()) != config["data"]["sha256"]:
        raise ValueError("Frozen ARC rows changed")
    cache = root / config["feature_cache"]["path"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_cache"]["manifest_sha256"]:
        raise ValueError("Frozen ARC feature manifest changed")
    expert_path = root / config["expert"]["weights"]
    if digest(expert_path.read_bytes()) != config["expert"]["weights_sha256"]:
        raise ValueError("Frozen QASC expert weights changed")

    rows = load_rows(data_path)
    keys = [f"arc:{row['id']}" for row in rows]
    manifest, features = load_features(cache, keys)
    model = config["model"]
    module = RoutedResidualHead.build(
        manifest["hidden_size"], model["residual_width"], model["router_width"],
        model["dropout"], model["initial_gate_probability"])
    module.load_state_dict(load_file(str(expert_path)))
    module.eval()
    gates = torch.tensor([0.0, 1.0], dtype=torch.float32)
    utilities, parent_logits, expert_logits = [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            batch_keys = [f"arc:{row['id']}" for row in batch]
            full, null, parent, parent_null, mask, labels = pad_batch(
                batch, batch_keys, features, manifest)
            correction = module.residual(full).squeeze(-1).masked_fill(~mask, 0.0)
            null_correction = module.residual(null).squeeze(-1).masked_fill(~mask, 0.0)
            for index, row in enumerate(batch):
                count = len(row["request"]["choices"])
                costs = composite_gate_costs(
                    parent[index, :count], parent_null[index, :count],
                    correction[index, :count], null_correction[index, :count],
                    int(labels[index]), gates,
                    brier_weight=config["cost"]["brier_weight"],
                    context_weight=config["cost"]["context_advantage_weight"],
                    context_gap=config["cost"]["context_advantage_gap"],
                    teacher=None, null_teacher=None,
                    distillation_weight=0.0, null_distillation_weight=0.0)
                target = signed_route_utility(costs, gates)
                utilities.append({"row_id": row["id"], "source": row["source"], **target})
                parent_logits.append(parent[index, :count].tolist())
                expert_logits.append((parent[index, :count] + correction[index, :count]).tolist())

    parent_metrics, _ = evaluate_rows(rows, parent_logits, 1.0)
    expert_metrics, _ = evaluate_rows(rows, expert_logits, 1.0)
    grouped = defaultdict(list)
    for record in utilities:
        grouped[record["source"]].append(record)
    by_source = {}
    for source, values in sorted(grouped.items()):
        source_rows = [row for row in rows if row["source"] == source]
        source_parent = [value for row, value in zip(rows, parent_logits) if row["source"] == source]
        source_expert = [value for row, value in zip(rows, expert_logits) if row["source"] == source]
        parent, _ = evaluate_rows(source_rows, source_parent, 1.0)
        expert, _ = evaluate_rows(source_rows, source_expert, 1.0)
        by_source[source] = {
            "rows": len(values),
            "mean_utility": mean([value["utility"] for value in values]),
            "mean_absolute_utility": mean([abs(value["utility"]) for value in values]),
            "open_rate": mean([value["gate"] for value in values]),
            "parent_accuracy": parent["accuracy"],
            "expert_accuracy": expert["accuracy"],
            "parent_nll": parent["nll"],
            "expert_nll": expert["nll"],
        }
    summary = {
        "rows": len(rows),
        "mean_utility": mean([value["utility"] for value in utilities]),
        "mean_absolute_utility": mean([abs(value["utility"]) for value in utilities]),
        "open_rate": mean([value["gate"] for value in utilities]),
        "parent_accuracy": parent_metrics["accuracy"],
        "expert_accuracy": expert_metrics["accuracy"],
        "parent_nll": parent_metrics["nll"],
        "expert_nll": expert_metrics["nll"],
        "by_source": by_source,
    }
    rules = config["screening_rule"]
    checks = {
        "coverage.rows": len(rows) == config["data"]["rows"],
        "coverage.sources": set(by_source) == set(config["data"]["sources"]),
        "mechanism.weights_loaded": parameter_sha256(module) == config["expert"]["parameter_sha256"],
        "utility.combined": summary["mean_utility"] >= rules["combined_mean_utility_min"],
        "utility.each_source": all(
            value["mean_utility"] >= rules["each_source_mean_utility_min"]
            for value in by_source.values()),
        "accuracy.combined": (
            summary["expert_accuracy"] - summary["parent_accuracy"] >=
            rules["combined_accuracy_min_delta"]),
        "nll.each_source": all(
            value["expert_nll"] - value["parent_nll"] <= rules["each_source_nll_max_delta"]
            for value in by_source.values()),
    }
    evidence = {
        "format_version": 1,
        "study": config["study"],
        "role": "consumed public-family frozen-expert utility screen",
        "protocol_sha256": digest(config_path.read_bytes()),
        "summary": summary,
        "checks": checks,
        "qualifies_as_positive_utility_family": all(checks.values()),
        "limitations": [
            "ARC was previously consumed as an architecture evaluation and is not fresh evidence.",
            "This screen evaluates an existing QASC expert and does not train a router.",
            "Raw and transformed ARC rows remain outside the release source boundary.",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "run.json", evidence)
    write_json(root / "docs/evidence/arc-expert-utility-screen-v1.json", evidence)
    table = []
    for source, value in by_source.items():
        table.append(
            f"| {source} | {value['rows']} | {value['mean_utility']:+.4f} | "
            f"{value['open_rate']:.4f} | {value['parent_accuracy']:.4f} | "
            f"{value['expert_accuracy']:.4f} | {value['parent_nll']:.4f} | "
            f"{value['expert_nll']:.4f} |")
    lines = [
        "# ARC 冻结专家效用筛选", "",
        "把现有 QASC 冻结专家直接应用于已消费的 ARC-Challenge 与 ARC-Easy。", "",
        "| 来源 | 行数 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 父 NLL | 专家 NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|", *table, "", "## 汇总", "",
        f"- 合并平均效用：`{summary['mean_utility']:+.6f}`。",
        f"- 合并应打开率：`{summary['open_rate']:.6f}`。",
        f"- 合并准确率：父模型 `{summary['parent_accuracy']:.6f}`，专家 `{summary['expert_accuracy']:.6f}`。",
        f"- 合并 NLL：父模型 `{summary['parent_nll']:.6f}`，专家 `{summary['expert_nll']:.6f}`。", "",
        "## 判定", "",
        ("全部门槛通过；ARC 可作为一个已消费的正效用训练族候选。" if all(checks.values())
         else "至少一个门槛失败；ARC 不计为正效用训练族。"), "",
        "逐项门槛：" + ", ".join(
            f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
        "机器证据：`docs/evidence/arc-expert-utility-screen-v1.json`", "",
    ]
    (root / "docs/ARC_EXPERT_UTILITY_SCREEN.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {
        "state": "complete", "qualifies": all(checks.values())})
    print(json.dumps({"state": "complete", "qualifies": all(checks.values()),
                      "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
