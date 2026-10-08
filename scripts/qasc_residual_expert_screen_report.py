#!/usr/bin/env python3
"""Report the consumed QASC frozen-parent residual-expert screen."""
from __future__ import annotations

import json
from pathlib import Path

from decision_model.core import digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def advancement(checks):
    return all(checks.values())


def main():
    root = Path(__file__).resolve().parents[1]
    study = root / "runs/qasc-residual-expert-screen-v1"
    protocol = read(study / "protocol.json")
    baseline = read(root / protocol["baselines"]["qasc_screen_evidence"])["summary"]
    parent = read(root / protocol["baselines"]["functional_retention_evidence"])[
        "arms"]["warm-functional-context"]
    rules = protocol["screening_rule"]
    evaluation = read(study / "frozen-parent-residual" / "evaluation.json")
    seen = evaluation["test"]["raw"]
    qasc_validation = evaluation["validation"]["raw"]["by_source"]["qasc"]
    development = read(study / "development" / "result.json")
    qasc = read(study / "qasc" / "result.json")
    audit = read(study / "audit.json")
    initial = read(study / "initial-equivalence.json")
    invariance = read(study / "parent-invariance.json")
    run = read(study / "frozen-parent-residual" / "run.json")
    context_deltas = {
        source: value["mean"] - parent["context_advantage"]["by_source"][source]["mean"]
        for source, value in development["context_advantage"]["by_source"].items()
    }
    checks = {
        "mechanism.initial_exact": initial["max_probability_difference"] <=
            rules["initial_max_probability_difference"] and initial["parent_parameters_exact"],
        "mechanism.parent_parameters_exact": invariance["parent_parameters_exact"] and
            invariance["parent_max_absolute_difference"] <=
            rules["parent_parameter_max_difference"],
        "mechanism.residual_changed": run["adapter_probe"]["absolute_change"] >
            rules["residual_parameter_min_absolute_change"],
        "seen.accuracy": seen["accuracy"] - baseline["seen"]["parent_accuracy"] >=
            rules["seen_accuracy_vs_parent_min_delta"],
        "seen.nll": seen["nll"] - baseline["seen"]["parent_nll"] <=
            rules["seen_nll_vs_parent_max_delta"],
        "development.accuracy": development["raw"]["accuracy"] -
            baseline["consumed_development"]["parent_accuracy"] >=
            rules["development_accuracy_vs_parent_min_delta"],
        "development.context_retention": min(context_deltas.values()) >=
            rules["development_each_context_advantage_vs_parent_min_delta"],
        "qasc.accuracy": qasc_validation["accuracy"] -
            baseline["qasc_validation"]["parent_accuracy"] >=
            rules["qasc_accuracy_vs_parent_min_delta"],
        "qasc.nll": qasc_validation["nll"] - baseline["qasc_validation"]["parent_nll"] <=
            rules["qasc_nll_vs_parent_max_delta"],
        "qasc.context": qasc["context_advantage"]["mean"] -
            baseline["qasc_validation"]["parent_context_advantage"]["mean"] >=
            rules["qasc_context_advantage_vs_parent_min_delta"],
        "audit": audit["cases"] == audit["stable_all_orders"] == 24 and
            audit["max_reload_probability_difference"] <= 2e-6 and
            audit["max_id_rename_probability_difference"] <= 2e-6,
    }
    selected = advancement(checks)
    evidence = {
        "format_version": 1, "study": protocol["study"],
        "protocol_sha256": digest((study / "protocol.json").read_bytes()),
        "role": "consumed single-seed mechanism screen; no multiseed or release claim",
        "mechanism": {"initial": initial, "parent_invariance": invariance,
                      "residual_probe": run["adapter_probe"]},
        "seen": {"accuracy": seen["accuracy"], "nll": seen["nll"]},
        "development": {"accuracy": development["raw"]["accuracy"],
                        "context_advantage": development["context_advantage"],
                        "context_delta_vs_parent": context_deltas},
        "qasc": {"accuracy": qasc_validation["accuracy"], "nll": qasc_validation["nll"],
                 "context_advantage": qasc["context_advantage"]["mean"]},
        "audit": {key: audit[key] for key in (
            "cases", "stable_all_orders", "correct_all_orders",
            "max_reload_probability_difference", "max_id_rename_probability_difference")},
        "checks": checks, "advances_to_new_task_family_screen": selected,
        "new_sealed_tasks_required_for_any_release_claim": True,
    }
    write_json(root / "docs/evidence/qasc-residual-expert-screen-v1.json", evidence)
    causal = development["context_advantage"]["by_source"]["bigbench-causal_judgment"]["mean"]
    lines = [
        "# QASC 冻结父函数残差专家筛选", "",
        "父 LoRA 与原读出头在训练期间逐张量冻结；新增零初始化候选残差头。全部评测任务均已消费，本结果只回答机制问题。", "",
        "| 已见准确率 | 已见 NLL | 开发准确率 | 因果材料优势 | QASC 准确率 | QASC NLL | QASC 材料优势 |", 
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {seen['accuracy']:.4f} | {seen['nll']:.4f} | {development['raw']['accuracy']:.4f} | {causal:.4f} | {qasc_validation['accuracy']:.4f} | {qasc_validation['nll']:.4f} | {qasc['context_advantage']['mean']:.4f} |",
        "", "## 机制审计", "",
        f"- 训练前最大概率差：`{initial['max_probability_difference']:.10g}`。",
        f"- 训练后父参数最大绝对差：`{invariance['parent_max_absolute_difference']:.10g}`。",
        f"- 残差探针绝对变化：`{run['adapter_probe']['absolute_change']:.10g}`。",
        "", "## 结论", "",
    ]
    lines.append(
        "全部冻结门槛通过；残差专家可进入新的公开任务族筛选，但不能据此形成多种子或发布结论。"
        if selected else
        "至少一个冻结门槛未通过；残差专家不进入新的任务族训练。"
    )
    lines += ["", "逐项门槛：" + ", ".join(
        f"`{name}`={'PASS' if value else 'FAIL'}" for name, value in checks.items()),
        "", "机器证据：`docs/evidence/qasc-residual-expert-screen-v1.json`", ""]
    (root / "docs/QASC_RESIDUAL_EXPERT_SCREEN.zh-CN.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
