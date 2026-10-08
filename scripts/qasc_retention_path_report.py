#!/usr/bin/env python3
"""Report the frozen QASC parameter-path retention diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

from decision_model.core import digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path(__file__).resolve().parents[1]
    study = root / "runs/qasc-retention-path-v1"
    protocol = read(study / "protocol.json")
    baseline = read(root / protocol["baselines"]["qasc_screen_evidence"])["summary"]
    retained_parent = read(root / protocol["baselines"]["functional_retention_evidence"])[
        "arms"]["warm-functional-context"]
    rules = protocol["screening_rule"]
    arms = {}
    for arm, alpha in protocol["arms"].items():
        seen = read(study / "seen" / f"{arm}.json")["raw"]
        validation = read(study / "validation" / f"{arm}.json")["raw"]["by_source"]["qasc"]
        development = read(study / "development" / arm / "result.json")
        qasc = read(study / "qasc-context" / arm / "result.json")
        audit = read(study / "audits" / f"{arm}.json")
        context = development["context_advantage"]
        qasc_context = qasc["context_advantage"]
        context_deltas = {
            source: value["mean"] - retained_parent["context_advantage"]["by_source"][source]["mean"]
            for source, value in context["by_source"].items()
        }
        checks = {
            "seen.accuracy": seen["accuracy"] - baseline["seen"]["parent_accuracy"] >= rules["seen_accuracy_vs_parent_min_delta"],
            "seen.nll": seen["nll"] - baseline["seen"]["parent_nll"] <= rules["seen_nll_vs_parent_max_delta"],
            "development.accuracy": development["raw"]["accuracy"] - baseline["consumed_development"]["parent_accuracy"] >= rules["development_accuracy_vs_parent_min_delta"],
            "development.context_advantage_retention": min(context_deltas.values()) >= rules["development_each_context_advantage_vs_parent_min_delta"],
            "qasc.accuracy": validation["accuracy"] - baseline["qasc_validation"]["parent_accuracy"] >= rules["qasc_accuracy_vs_parent_min_delta"],
            "qasc.nll": validation["nll"] - baseline["qasc_validation"]["parent_nll"] <= rules["qasc_nll_vs_parent_max_delta"],
            "qasc.context_advantage": qasc_context["mean"] - baseline["qasc_validation"]["parent_context_advantage"]["mean"] >= rules["qasc_context_advantage_vs_parent_min_delta"],
            "audit": audit["cases"] == audit["stable_all_orders"] == 24 and audit["max_reload_probability_difference"] <= 2e-6 and audit["max_id_rename_probability_difference"] <= 2e-6,
        }
        arms[arm] = {
            "alpha": alpha,
            "seen": {"accuracy": seen["accuracy"], "nll": seen["nll"]},
            "development": {"accuracy": development["raw"]["accuracy"],
                            "context_advantage": context, "context_delta_vs_parent": context_deltas},
            "qasc": {"accuracy": validation["accuracy"], "nll": validation["nll"],
                     "context_advantage": qasc_context["mean"]},
            "audit": {key: audit[key] for key in ("cases", "stable_all_orders", "correct_all_orders", "max_reload_probability_difference", "max_id_rename_probability_difference")},
            "checks": checks, "passes": all(checks.values()),
        }
    eligible = [name for name, value in arms.items() if value["passes"]]
    selected = (sorted(eligible, key=lambda name: (-arms[name]["qasc"]["accuracy"], arms[name]["alpha"]))[0]
                if eligible else None)
    evidence = {
        "format_version": 1, "study": protocol["study"],
        "protocol_sha256": digest((study / "protocol.json").read_bytes()),
        "role": "consumed development path diagnostic; no release or multiseed claim",
        "arms": arms, "eligible_arms": eligible, "selected_training_dose_target": selected,
        "requires_gradient_training_replication": selected is not None,
        "new_sealed_tasks_required_for_release": True,
    }
    evidence_path = root / "docs/evidence/qasc-retention-path-v1.json"
    write_json(evidence_path, evidence)
    lines = ["# QASC 参数路径保留诊断", "",
             "本实验不再读取新数据，而是在函数保留父端点与 QASC 端点之间预先固定三个线性插值位置。它回答的是：现有更新方向上是否存在同时保留旧函数、又获得 QASC 收益的较小步长。所有任务均已消费，只能用于选择下一轮训练剂量。", "",
             "| 插值 | 已见准确率 | 已见 NLL | 开发准确率 | QASC 准确率 | QASC NLL | QASC 材料优势 | 通过 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for arm, value in arms.items():
        lines.append(f"| {value['alpha']:.2f} | {value['seen']['accuracy']:.4f} | {value['seen']['nll']:.4f} | {value['development']['accuracy']:.4f} | {value['qasc']['accuracy']:.4f} | {value['qasc']['nll']:.4f} | {value['qasc']['context_advantage']:.4f} | {'PASS' if value['passes'] else 'FAIL'} |")
    lines += ["", "## 冻结判断", ""]
    if selected:
        lines.append(f"`{selected}` 通过全部开发门槛，只作为下一轮从父端点重新训练时的目标剂量；插值权重本身不发布、不进入多种子。")
    else:
        lines.append("没有插值位置通过全部门槛。当前 QASC 更新方向上不存在已测得的简单保留折中，需要改变训练约束或数据组合。")
    lines += ["", "候选换序、权重重载和候选 ID 重命名审计分别独立执行。任何后续发布结论仍需未读取的新任务族。", "", "机器证据：`docs/evidence/qasc-retention-path-v1.json`", ""]
    (root / "docs/QASC_RETENTION_PATH.zh-CN.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
