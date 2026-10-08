"""Validate, summarize and report the three-seed gradient trust study."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from decision_model.core import write_json
from task_gradient_trust_report import METHODS, PRIMARY, SEEDS, validated


def pct(value):
    return f"{100 * value:.1f}%"


def pp(value):
    return f"{100 * value:+.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/task-gradient-trust-v1")
    parser.add_argument("--report", default="docs/TASK_GRADIENT_TRUST_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/task-gradient-trust-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, arms, diagnostic, paired, diagnostic_paired = validated(root, root / args.study)
    trust_method = "soft_cap_2x_project_trust"
    unbounded_method = "soft_cap_2x_then_project"

    changes, per_seed = [], {}
    for seed in SEEDS:
        raw = arms[f"seed{seed}-raw"]
        trust = arms[f"seed{seed}-{trust_method}"]
        unbounded = arms[f"seed{seed}-{unbounded_method}"]
        deltas = {source: trust["raw"]["by_source"][source]["accuracy"] - raw["raw"]["by_source"][source]["accuracy"]
                  for source in PRIMARY}
        changes.extend(deltas.values())
        per_seed[str(seed)] = {
            "accuracy_delta": deltas,
            "raw_p95": raw["training"]["combined_norm"]["p95"],
            "unbounded_p95": unbounded["training"]["combined_norm"]["p95"],
            "trust_p95": trust["training"]["combined_norm"]["p95"],
            "trust_active_updates": trust["training"]["trust_active_updates"],
            "trust_partial_updates": trust["training"]["trust_partial_updates"],
            "maximum_bound_delta": trust["training"]["maximum_bound_delta"],
        }

    checks = {
        "no_primary_accuracy_drop_below_minus_1pp": min(changes) >= -0.01,
        "mean_primary_accuracy_delta_nonnegative": mean(changes) >= 0,
        "p95_not_above_raw_in_every_seed": all(value["trust_p95"] <= value["raw_p95"] for value in per_seed.values()),
        "seed42_43_p95_not_above_unbounded": all(per_seed[str(seed)]["trust_p95"] <= per_seed[str(seed)]["unbounded_p95"] for seed in (42, 43)),
        "trust_bound_never_violated": all(value["maximum_bound_delta"] <= max(1e-5, value["raw_p95"] * 2e-6) for value in per_seed.values()),
        "direction_changed_at_least_10pct_every_seed": all(value["trust_active_updates"] >= 13 for value in per_seed.values()),
    }
    checks["advances_to_fresh_blind_evaluation"] = all(checks.values())
    evidence = {
        "protocol": protocol,
        "arms": {name: {"raw": value["raw"], "training": value["training"]} for name, value in arms.items()},
        "paired_method_minus_raw": paired,
        "diagnostic": {name: {"raw": value["raw"], "choice_counts": value["choice_counts"]} for name, value in diagnostic.items()},
        "diagnostic_paired_method_minus_raw": diagnostic_paired,
        "success_rule": {"checks": checks, "per_seed": per_seed, "mean_primary_accuracy_delta": mean(changes)},
    }
    write_json(root / args.evidence, evidence)

    labels = {"raw": "原始聚合", "soft_cap_2x_then_project": "无界截断+投影",
              "soft_cap_2x_project_trust": "投影修正信赖域"}
    outcome = "通过" if checks["advances_to_fresh_blind_evaluation"] else "未通过"
    lines = [
        "# 限制投影修正幅度，能否稳定保留多任务收益？", "", "## 结论", "",
        f"**本轮预注册晋级规则：{outcome}。**", "",
        f"投影修正信赖域在六个主任务比较中的平均准确率变化为 {pp(mean(changes))} 个百分点，最差变化为 {pp(min(changes))} 个百分点。它直接限制每次修正后的合并梯度范数不超过同一步原始聚合范数；验证器重新检查了全部 384 个 trust 更新。", "",
        "信赖域只约束单步几何性质。是否值得作为默认训练方法，仍由三个种子的准确率、梯度 P95 和干预覆盖率共同决定；旧 BoolQ/WinoGrande 诊断不能推翻主终点失败。", "",
        "## 主任务结果", "",
        "| 种子 / 方法 | PAWS | 相对 raw | PAWS NLL | SNLI | 相对 raw | SNLI NLL | 总准确率 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        baseline = arms[f"seed{seed}-raw"]["raw"]["by_source"]
        for method in METHODS:
            value = arms[f"seed{seed}-{method}"]["raw"]
            paws, snli = value["by_source"]["paws-wiki"], value["by_source"]["snli"]
            lines.append(f"| seed{seed} {labels[method]} | {pct(paws['accuracy'])} | {pp(paws['accuracy']-baseline['paws-wiki']['accuracy'])} | {paws['nll']:.4f} | {pct(snli['accuracy'])} | {pp(snli['accuracy']-baseline['snli']['accuracy'])} | {snli['nll']:.4f} | {pct(value['accuracy'])} |")

    lines += ["", "## 梯度尺度与信赖域干预", "", "| 种子 / 方法 | 合并范数 P95 | 最大值 | 信赖域激活 | 部分步长 | 最大越界量 |", "|---|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for method in METHODS:
            training = arms[f"seed{seed}-{method}"]["training"]
            norm = training["combined_norm"]
            active = training["trust_active_updates"] if method == trust_method else 0
            partial = training["trust_partial_updates"] if method == trust_method else 0
            violation = training["maximum_bound_delta"] if method == trust_method else None
            lines.append(f"| seed{seed} {labels[method]} | {norm['p95']:.2f} | {norm['max']:.2f} | {active}/128 | {partial}/128 | {'—' if violation is None else f'{violation:.2e}'} |")

    lines += ["", "## 预注册条件", ""]
    for name, passed in checks.items():
        lines.append(f"- {'通过' if passed else '未通过'}：`{name}`")
    lines += ["", "## Trust 相对 raw 的配对区间", "", "| 任务 / 种子 | 准确率变化 | 组重采样 95% 区间 |", "|---|---:|---:|"]
    for source, title in (("paws-wiki", "PAWS"), ("snli", "SNLI")):
        for seed in SEEDS:
            value = paired[f"seed{seed}-{trust_method}_minus_raw"][source]
            low, high = value["paired_group_bootstrap_95_percentile"]
            lines.append(f"| {title} / seed{seed} | {pp(value['accuracy_change'])} | [{pp(low)}, {pp(high)}] |")

    lines += ["", "## 边界", "",
              "- 三个种子都从同一公共权重起点开始；每个端点固定使用 1,024 条训练行、一次曝光和 128 次更新。",
              "- seed42/43 的 raw 与无界组合端点逐位复用已验证研究；本轮重新验证权重、配置、训练计划、第一次原始任务梯度和父清单。",
              "- seed44 的三个端点为本轮新训练；同一种子内训练计划逐字节一致。",
              "- 信赖域约束的是修正后的合并梯度范数，不保证最终准确率单调改善。",
              "- BoolQ/WinoGrande 已经在此前实验中查看过，因此只作为候选先验诊断。",
              "- 三个随机种子仍不足以估计完整分布；公共底座的预训练污染无法排除。",
              "- 机器可读证据：[evidence/task-gradient-trust-v1.json](evidence/task-gradient-trust-v1.json)。", ""]
    (root / args.report).write_text("\n".join(lines))
    print({"success": checks, "per_seed": per_seed, "mean_primary_accuracy_delta": mean(changes)})


if __name__ == "__main__":
    main()
