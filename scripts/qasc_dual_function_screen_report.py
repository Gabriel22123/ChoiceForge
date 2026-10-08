#!/usr/bin/env python3
"""Report the matched QASC dual-function distillation screen."""
from __future__ import annotations

import json
from pathlib import Path

from decision_model.core import digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path(__file__).resolve().parents[1]
    study = root / "runs/qasc-dual-function-screen-v1"
    protocol = read(study / "protocol.json")
    qasc_base = read(root / protocol["baselines"]["qasc_screen_evidence"])["summary"]
    parent_base = read(root / protocol["baselines"]["functional_retention_evidence"])[
        "arms"]["warm-functional-context"]
    rules = protocol["screening_rule"]
    arms = {}
    for arm in protocol["arms"]:
        evaluation = read(study / arm / "evaluation.json")
        seen = evaluation["test"]["raw"]
        qasc_validation = evaluation["validation"]["raw"]["by_source"]["qasc"]
        development = read(study / "development" / arm / "result.json")
        qasc = read(study / "qasc-context" / arm / "result.json")
        audit = read(study / "audits" / f"{arm}.json")
        context_deltas = {
            source: value["mean"] - parent_base["context_advantage"]["by_source"][source]["mean"]
            for source, value in development["context_advantage"]["by_source"].items()
        }
        checks = {
            "seen.accuracy": seen["accuracy"] - qasc_base["seen"]["parent_accuracy"] >= rules["seen_accuracy_vs_parent_min_delta"],
            "seen.nll": seen["nll"] - qasc_base["seen"]["parent_nll"] <= rules["seen_nll_vs_parent_max_delta"],
            "development.accuracy": development["raw"]["accuracy"] - qasc_base["consumed_development"]["parent_accuracy"] >= rules["development_accuracy_vs_parent_min_delta"],
            "development.context_retention": min(context_deltas.values()) >= rules["development_each_context_advantage_vs_parent_min_delta"],
            "qasc.accuracy": qasc_validation["accuracy"] - qasc_base["qasc_validation"]["parent_accuracy"] >= rules["qasc_accuracy_vs_parent_min_delta"],
            "qasc.nll": qasc_validation["nll"] - qasc_base["qasc_validation"]["parent_nll"] <= rules["qasc_nll_vs_parent_max_delta"],
            "qasc.context": qasc["context_advantage"]["mean"] - qasc_base["qasc_validation"]["parent_context_advantage"]["mean"] >= rules["qasc_context_advantage_vs_parent_min_delta"],
            "audit": audit["cases"] == audit["stable_all_orders"] == 24 and audit["max_reload_probability_difference"] <= 2e-6 and audit["max_id_rename_probability_difference"] <= 2e-6,
        }
        arms[arm] = {
            "null_distillation_weight": protocol["arms"][arm]["null_distillation_weight"],
            "seen": {"accuracy": seen["accuracy"], "nll": seen["nll"]},
            "development": {"accuracy": development["raw"]["accuracy"],
                            "context_advantage": development["context_advantage"],
                            "context_delta_vs_parent": context_deltas},
            "qasc": {"accuracy": qasc_validation["accuracy"], "nll": qasc_validation["nll"],
                     "context_advantage": qasc["context_advantage"]["mean"]},
            "audit": {key: audit[key] for key in ("cases", "stable_all_orders", "correct_all_orders", "max_reload_probability_difference", "max_id_rename_probability_difference")},
            "absolute_checks": checks, "passes_absolute": all(checks.values()),
        }
    control = arms["half-qasc-evidence-only"]
    dual = arms["half-qasc-dual-function"]
    causal = "bigbench-causal_judgment"
    mechanism_delta = (dual["development"]["context_advantage"]["by_source"][causal]["mean"] -
                       control["development"]["context_advantage"]["by_source"][causal]["mean"])
    mechanism_pass = mechanism_delta >= rules["dual_causal_context_advantage_vs_control_min_delta"]
    selected = dual["passes_absolute"] and mechanism_pass
    evidence = {
        "format_version": 1, "study": protocol["study"],
        "protocol_sha256": digest((study / "protocol.json").read_bytes()),
        "role": "consumed single-seed development screen; no multiseed or release claim",
        "arms": arms,
        "dual_causal_context_advantage_delta_vs_control": mechanism_delta,
        "dual_mechanism_check": mechanism_pass,
        "selects_next_data_expansion_route": selected,
        "new_sealed_tasks_required_for_release": True,
    }
    write_json(root / "docs/evidence/qasc-dual-function-screen-v1.json", evidence)
    lines = ["# QASC 双函数蒸馏筛选", "",
             "两组从同一父端点出发，使用相同顺序、半权重 QASC、旧请求概率蒸馏和材料优势损失。实验组唯一新增项是：对旧任务移除材料后的父模型候选分布做 KL 蒸馏。", "",
             "| 组别 | 已见准确率 | 已见 NLL | 开发准确率 | 因果材料优势 | QASC 准确率 | QASC NLL | QASC 材料优势 | 绝对门槛 |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm, value in arms.items():
        causal_value = value["development"]["context_advantage"]["by_source"][causal]["mean"]
        lines.append(f"| {arm} | {value['seen']['accuracy']:.4f} | {value['seen']['nll']:.4f} | {value['development']['accuracy']:.4f} | {causal_value:.4f} | {value['qasc']['accuracy']:.4f} | {value['qasc']['nll']:.4f} | {value['qasc']['context_advantage']:.4f} | {'PASS' if value['passes_absolute'] else 'FAIL'} |")
    lines += ["", "## 机制与结论", "",
              f"双函数组相对匹配对照的因果材料优势变化：{mechanism_delta:+.4f}（要求至少 +{rules['dual_causal_context_advantage_vs_control_min_delta']:.2f}）。", ""]
    lines.append("双函数蒸馏通过全部绝对门槛和机制门槛，可进入新的公开任务扩展设计；仍不进入多种子或发布验证。" if selected else "双函数蒸馏没有同时通过全部绝对门槛与机制门槛，不选择为下一训练路线。")
    lines += ["", "所有评估任务均已消费。任何发布结论必须使用未读取的新任务族。", "", "机器证据：`docs/evidence/qasc-dual-function-screen-v1.json`", ""]
    (root / "docs/QASC_DUAL_FUNCTION_SCREEN.zh-CN.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
