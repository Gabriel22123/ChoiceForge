"""Write aggregate evidence and the human report after full study validation."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from decision_model.core import write_json
from task_gradient_composition import METHODS
from task_gradient_composition_report import SEEDS, validated


def pct(value):
    return f"{100 * value:.1f}%"


def pp(value):
    return f"{100 * value:+.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/task-gradient-composition-v1")
    parser.add_argument("--report", default="docs/TASK_GRADIENT_COMPOSITION_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/task-gradient-composition-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, arms, diagnostic, paired, diagnostic_paired = validated(root, root / args.study)

    treatment = "soft_cap_2x_then_project"
    accuracy_deltas = []
    per_seed = {}
    for seed in SEEDS:
        raw = arms[f"seed{seed}-raw"]["raw"]["by_source"]
        current = arms[f"seed{seed}-{treatment}"]["raw"]["by_source"]
        deltas = {source: current[source]["accuracy"] - raw[source]["accuracy"]
                  for source in ("paws-wiki", "snli")}
        accuracy_deltas.extend(deltas.values())
        per_seed[str(seed)] = {
            "accuracy_delta": deltas,
            "raw_p95": arms[f"seed{seed}-raw"]["training"]["combined_norm"]["p95"],
            "treatment_p95": arms[f"seed{seed}-{treatment}"]["training"]["combined_norm"]["p95"],
        }

    prior_gaps = {}
    diagnostic_gap_pass = True
    for source, candidate in (("boolq", "yes"), ("winogrande-1.1", "option1")):
        raw_counts = [diagnostic[f"seed{seed}-raw"]["choice_counts"][source]["predicted"].get(candidate, 0)
                      for seed in SEEDS]
        treatment_counts = [diagnostic[f"seed{seed}-{treatment}"]["choice_counts"][source]["predicted"].get(candidate, 0)
                            for seed in SEEDS]
        raw_gap = abs(raw_counts[0] - raw_counts[1])
        treatment_gap = abs(treatment_counts[0] - treatment_counts[1])
        prior_gaps[source] = {
            "candidate": candidate, "raw_counts": raw_counts,
            "treatment_counts": treatment_counts,
            "raw_between_seed_gap": raw_gap,
            "treatment_between_seed_gap": treatment_gap,
        }
        diagnostic_gap_pass &= treatment_gap <= raw_gap

    success = {
        "no_primary_accuracy_drop_below_minus_1pp": min(accuracy_deltas) >= -0.01,
        "mean_primary_accuracy_delta_nonnegative": mean(accuracy_deltas) >= 0,
        "p95_below_raw_in_both_seeds": all(value["treatment_p95"] < value["raw_p95"]
                                             for value in per_seed.values()),
        "candidate_prior_between_seed_gaps_not_worse": diagnostic_gap_pass,
    }
    success["advances_to_fresh_blind_evaluation"] = all(success.values())
    evidence = {
        "protocol": protocol,
        "arms": {name: {"raw": value["raw"], "training": value["training"]}
                 for name, value in arms.items()},
        "paired_method_minus_raw": paired,
        "diagnostic": {name: {"raw": value["raw"], "choice_counts": value["choice_counts"]}
                       for name, value in diagnostic.items()},
        "diagnostic_paired_method_minus_raw": diagnostic_paired,
        "success_rule": {"checks": success, "per_seed": per_seed,
                         "candidate_prior_gaps": prior_gaps},
    }
    write_json(root / args.evidence, evidence)

    labels = {"raw": "原始聚合", "soft_cap_2x": "2×中位数截断",
              "soft_cap_2x_then_project": "截断后投影"}
    lines = [
        "# 软截断后再处理梯度冲突，能否同时稳住尺度与方向？", "", "## 结论", "",
        "**组合方案显著改善了准确率稳定性，但没有完整通过本轮预注册晋级条件。**", "",
        "两个种子的 PAWS/SNLI 最差变化为 -0.5 个百分点，四项平均提高 2.2 个百分点；seed42 的合并梯度 P95 从 57.53 降到 31.41，但 seed43 从 32.74 升到 37.50。BoolQ 的跨种子 `yes` 数差从 68 降到 37，WinoGrande 的 `option1` 数差却从 10 升到 15。", "",
        "因此它是当前更有希望的研究候选，但尚不能晋级为默认训练方案。下一轮应约束投影后的范数再做第三种子验证，规则冻结后才进入新的任务盲测。", "",
        "## 主任务结果", "",
        "| 种子 / 方法 | PAWS | 相对 raw | PAWS NLL | SNLI | 相对 raw | SNLI NLL | 总准确率 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        baseline = arms[f"seed{seed}-raw"]["raw"]["by_source"]
        for method in METHODS:
            value = arms[f"seed{seed}-{method}"]["raw"]
            paws, snli = value["by_source"]["paws-wiki"], value["by_source"]["snli"]
            lines.append(f"| seed{seed} {labels[method]} | {pct(paws['accuracy'])} | {pp(paws['accuracy']-baseline['paws-wiki']['accuracy'])} | {paws['nll']:.4f} | {pct(snli['accuracy'])} | {pp(snli['accuracy']-baseline['snli']['accuracy'])} | {snli['nll']:.4f} | {pct(value['accuracy'])} |")

    lines += ["", "## 梯度尺度与干预", "", "| 种子 / 方法 | 合并范数 P95 | 最大值 | 截断更新 | 冲突更新 |", "|---|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for method in METHODS:
            training = arms[f"seed{seed}-{method}"]["training"]
            norm = training["combined_norm"]
            lines.append(f"| seed{seed} {labels[method]} | {norm['p95']:.2f} | {norm['max']:.2f} | {training['cap_updates']}/128 | {training['conflict_updates']}/128 |")

    lines += ["", "## 预注册条件", ""]
    for name, passed in success.items():
        lines.append(f"- {'通过' if passed else '未通过'}：`{name}`")
    lines += ["", "## 组合方案的配对区间", "", "| 任务 / 种子 | 准确率变化 | 组重采样 95% 区间 |", "|---|---:|---:|"]
    for source, title in (("paws-wiki", "PAWS"), ("snli", "SNLI")):
        for seed in SEEDS:
            value = paired[f"seed{seed}-{treatment}_minus_raw"][source]
            low, high = value["paired_group_bootstrap_95_percentile"]
            lines.append(f"| {title} / seed{seed} | {pp(value['accuracy_change'])} | [{pp(low)}, {pp(high)}] |")
    lines += ["", "四个区间都跨 0；两个种子不能形成最终模型选择。", "", "## 已观察诊断", ""]
    for source, value in prior_gaps.items():
        lines.append(f"- {source} `{value['candidate']}` 的两种子预测数：raw {value['raw_counts']}（差 {value['raw_between_seed_gap']}），组合 {value['treatment_counts']}（差 {value['treatment_between_seed_gap']}）。")
    lines += ["", "## 边界", "", "- raw 终点逐位复用上一轮已验证权重；新方法与它们的训练计划、候选换序、任务顺序和首次更新原始梯度完全一致。", "- 每组仍只有 128 次更新、1,024 条训练行和两个随机种子。", "- 协议预先规定候选先验诊断不得一致恶化，但没有给出计数公式；报告保守地具体化为两个诊断集的跨种子候选计数差都不得扩大。即使不采用这项事后操作化，seed43 P95 失败也足以阻止晋级。", "- BoolQ/WinoGrande 是旧诊断；下一步必须先约束投影后范数并增加第三种子，再冻结全新的任务级保留集。", "- 机器可读证据：[evidence/task-gradient-composition-v1.json](evidence/task-gradient-composition-v1.json)。", ""]
    (root / args.report).write_text("\n".join(lines))
    print({"success": success, "accuracy_deltas": accuracy_deltas,
           "p95": per_seed, "candidate_prior_gaps": prior_gaps})


if __name__ == "__main__":
    main()
