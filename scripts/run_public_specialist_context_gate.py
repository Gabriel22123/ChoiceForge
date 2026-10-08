#!/usr/bin/env python3
"""Run the frozen public specialist trainer and enforce a hard context gate."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from decision_model.core import canonical, digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def context_deltas(evaluation: dict) -> dict[str, float]:
    expert = evaluation["context_advantage"]
    parent = evaluation["parent_context_advantage"]
    if set(expert) != set(parent) or not expert:
        raise ValueError("Context summaries have different source coverage")
    return {source: expert[source]["mean"] - parent[source]["mean"]
            for source in sorted(expert)}


def hard_context_gate(seeds: list[dict], minimum_delta: float) -> tuple[bool, list[dict]]:
    details = []
    for item in seeds:
        deltas = context_deltas(item["evaluation"])
        details.append({"seed": item["seed"], "by_source": deltas,
                        "minimum_delta": min(deltas.values())})
    return all(item["minimum_delta"] >= minimum_delta for item in details), details


def render_report(config: dict, evidence: dict) -> str:
    rows = []
    for item in evidence["seeds"]:
        value = item["evaluation"]
        minimum_context = next(
            detail["minimum_delta"] for detail in evidence["context_gate"]
            if detail["seed"] == item["seed"])
        rows.append(
            f"| {item['seed']} | {value['mean_utility']:+.4f} | {value['open_rate']:.4f} | "
            f"{value['parent_accuracy']:.4f} | {value['expert_accuracy']:.4f} | "
            f"{value['accuracy_delta']:+.4f} | {value['nll_delta']:+.4f} | "
            f"{minimum_context:+.4f} |")
    checks = evidence["checks"]
    lines = [
        f"# {config['display_name']} 多随机种子专家训练", "",
        f"在固定 2B 表示和父决策头上，为 {config['display_name']} 训练独立残差专家。"
        "验证集已消费，结果只用于机制筛选。", "",
        "| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 | 最小上下文优势差 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|", *rows, "", "## 判定", "",
        (f"全部门槛通过；{config['display_name']} 可计为一个已消费的正效用专家族。"
         if evidence["qualifies_as_positive_utility_family"] else
         f"至少一个门槛失败；{config['display_name']} 暂不计为正效用专家族。"), "",
        "逐项门槛：" + ", ".join(
            f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
        f"机器证据：`{config['evidence_path']}`", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    config = read(config_path)
    if output.exists():
        raise ValueError("Use a fresh specialist training directory")
    minimum = config.get("screening_rule", {}).get("context_advantage_min_delta")
    if not isinstance(minimum, (int, float)):
        raise ValueError("Frozen protocol must define context_advantage_min_delta")
    expected = config.get("source_files", {}).get("scripts/run_public_specialist_context_gate.py")
    if expected != digest(Path(__file__).read_bytes()):
        raise ValueError("Frozen context-gate runner changed")
    subprocess.run([
        sys.executable, str(root / "scripts/run_public_specialist.py"),
        "--config", args.config, "--output", args.output,
    ], cwd=root, check=True)
    evidence_path = output / "evidence.json"
    evidence = read(evidence_path)
    passed, details = hard_context_gate(evidence["seeds"], float(minimum))
    evidence["checks"]["context.each_source_each_seed"] = passed
    evidence["context_gate"] = details
    evidence["qualifies_as_positive_utility_family"] = all(evidence["checks"].values())
    write_json(evidence_path, evidence)
    write_json(root / config["evidence_path"], evidence)
    (root / config["report_path"]).write_text(render_report(config, evidence))
    write_json(output / "status.json", {
        "state": "complete",
        "qualifies": evidence["qualifies_as_positive_utility_family"],
    })
    print(canonical({"event": "context_gate_complete",
                     "qualifies": evidence["qualifies_as_positive_utility_family"],
                     "context_gate": passed, "checks": evidence["checks"]}))


if __name__ == "__main__":
    main()
