#!/usr/bin/env python3
"""Validate and report the frozen warm-start functional-retention screen."""
from __future__ import annotations

import json
from pathlib import Path

from decision_model.core import digest, write_json


ARMS = ("warm-context", "warm-functional-context")


def read(path):
    return json.loads(Path(path).read_text())


def compare(candidate, control, expanded, rule, prefix):
    sources = set(control["by_source"])
    if set(candidate["by_source"]) != sources or set(expanded["by_source"]) != sources:
        raise ValueError("Source sets differ")
    deltas = {source: candidate["by_source"][source]["accuracy"] -
              control["by_source"][source]["accuracy"] for source in sorted(sources)}
    checks = {
        f"{prefix}.combined_accuracy": candidate["accuracy"] - expanded["accuracy"] >=
            rule[f"{prefix}_combined_accuracy_vs_parent_expanded_min_delta"],
        f"{prefix}.each_source_accuracy": min(deltas.values()) >=
            rule[f"{prefix}_each_source_accuracy_vs_parent_control_min_delta"],
    }
    if prefix == "seen":
        checks["seen.nll"] = candidate["nll"] - expanded["nll"] <= rule[
            "seen_nll_vs_parent_expanded_max_delta"]
    return {
        "candidate_accuracy": candidate["accuracy"],
        "parent_control_accuracy": control["accuracy"],
        "parent_expanded_accuracy": expanded["accuracy"],
        "accuracy_delta_vs_parent_expanded": candidate["accuracy"] - expanded["accuracy"],
        "nll": candidate["nll"],
        "nll_delta_vs_parent_expanded": candidate["nll"] - expanded["nll"],
        "source_accuracy_delta_vs_parent_control": deltas,
        "minimum_source_accuracy_delta_vs_parent_control": min(deltas.values()),
    }, checks


def main():
    root = Path(__file__).resolve().parents[1]
    study = root / "runs/functional-retention-screen-v1"
    protocol = read(study / "protocol.json")
    if read(study / "status.json")["state"] not in ("trained", "reporting"):
        raise ValueError("Screen endpoints are not both trained")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen screen input changed: " + name)
    for name, expected in protocol["source_files"].items():
        if (digest((root / name).read_bytes()) != expected or
                digest((study / "frozen-source" / name).read_bytes()) != expected):
            raise ValueError("Frozen screen source changed: " + name)
    endpoints = read(study / "trained-checkpoints.json")
    if set(endpoints) != set(ARMS):
        raise ValueError("Screen endpoint set differs")
    for arm, record in endpoints.items():
        endpoint = root / record["path"]
        development = study / "development-results" / arm / "result.json"
        if (digest((endpoint / "decision.safetensors").read_bytes()) != record["weights_sha256"] or
                digest((endpoint / "evaluation.json").read_bytes()) != record["evaluation_sha256"] or
                digest(development.read_bytes()) != record["development_result_sha256"]):
            raise ValueError("Screen endpoint checksum differs: " + arm)

    parent = root / "runs/release-candidate-v1"
    control_seen = read(parent / "seed1701-control/evaluation.json")["test"]["raw"]
    expanded_seen = read(parent / "seed1701-expanded/evaluation.json")["test"]["raw"]
    control_dev = read(parent / "blind-results/seed1701-control/result.json")["raw"]
    expanded_dev = read(parent / "blind-results/seed1701-expanded/result.json")["raw"]
    rule = protocol["screening_rule"]
    arms = {}
    for arm in ARMS:
        seen = read(study / arm / "evaluation.json")["test"]["raw"]
        development_record = read(study / "development-results" / arm / "result.json")
        seen_summary, seen_checks = compare(seen, control_seen, expanded_seen, rule, "seen")
        development_summary, development_checks = compare(
            development_record["raw"], control_dev, expanded_dev, rule, "development")
        checks = {**seen_checks, **development_checks}
        context = development_record["context_advantage"]
        checks["development.context_advantage_each_source"] = all(
            values["mean"] > 0 for values in context["by_source"].values())
        arms[arm] = {
            "seen": seen_summary,
            "consumed_development_suite": development_summary,
            "context_advantage": context,
            "checks": checks,
            "passes_screen": all(checks.values()),
            "advances_to_multiseed": (
                arm == "warm-functional-context" and all(checks.values())),
        }
    evidence = {
        "format_version": 1,
        "study": protocol["study"],
        "protocol_sha256": digest((study / "protocol.json").read_bytes()),
        "development_suite_is_fresh": False,
        "development_suite_release_eligible": False,
        "arms": arms,
        "advancing_arms": [arm for arm in ARMS if arms[arm]["advances_to_multiseed"]],
        "new_fresh_blind_required_for_any_release_claim": True,
    }
    write_json(root / "docs/evidence/functional-retention-screen-v1.json", evidence)
    lines = [
        "# 函数级保留与上下文更新筛选", "",
        "本实验从同一个 Expanded 端点继续训练。两组使用相同公开样本、顺序、学习率和上下文约束；唯一差异是是否用父模型完整概率分布做 KL 锚定。已消费诊断集只用于开发。", "",
        "| 方案 | 已见准确率 | 相对 Expanded | 已见最差源相对 Control | 开发集准确率 | 相对 Expanded | 开发集最差源相对 Control | 通过全部条件 | 晋级 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for arm in ARMS:
        item = arms[arm]
        lines.append(
            f"| `{arm}` | {item['seen']['candidate_accuracy']:.4f} | "
            f"{item['seen']['accuracy_delta_vs_parent_expanded']:+.4f} | "
            f"{item['seen']['minimum_source_accuracy_delta_vs_parent_control']:+.4f} | "
            f"{item['consumed_development_suite']['candidate_accuracy']:.4f} | "
            f"{item['consumed_development_suite']['accuracy_delta_vs_parent_expanded']:+.4f} | "
            f"{item['consumed_development_suite']['minimum_source_accuracy_delta_vs_parent_control']:+.4f} | "
            f"{'是' if item['passes_screen'] else '否'} | "
            f"{'是' if item['advances_to_multiseed'] else '否'} |")
    lines += ["", "## 冻结检查", ""]
    for arm in ARMS:
        lines += [f"### `{arm}`", "", "| 条件 | 结果 |", "|---|---|"]
        lines += [f"| `{name}` | {'PASS' if value else 'FAIL'} |"
                  for name, value in sorted(arms[arm]["checks"].items())]
        lines.append("")
    lines += [
        "## 结论边界", "",
        "- 教师分布来自同一个公开数据父端点；公开真实标签仍保留，未被教师替换。",
        "- 单种子开发筛选只能淘汰方案，不能证明稳定性或通用零样本能力。",
        "- 只有函数级保留组有资格晋级；任何发布实验都必须另取从未读取的新任务族。",
        "", "机器证据：`docs/evidence/functional-retention-screen-v1.json`",
    ]
    (root / "docs/FUNCTIONAL_RETENTION_SCREEN.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"advancing_arms": evidence["advancing_arms"]}, sort_keys=True))


if __name__ == "__main__":
    main()
