#!/usr/bin/env python3
"""Validate and report the frozen retention/context single-seed screen."""
from __future__ import annotations

import json
from pathlib import Path

from decision_model.core import digest, write_json


ARMS = ("retention", "retention-context")


def read(path):
    return json.loads(Path(path).read_text())


def compare(candidate, control, expanded, rule, prefix):
    sources = set(control["by_source"])
    if set(candidate["by_source"]) != sources or set(expanded["by_source"]) != sources:
        raise ValueError("Seen source sets differ")
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
    study = root / "runs/retention-context-screen-v1"
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
        development = development_record["raw"]
        seen_summary, seen_checks = compare(
            seen, control_seen, expanded_seen, rule, "seen")
        development_summary, development_checks = compare(
            development, control_dev, expanded_dev, rule, "development")
        checks = {**seen_checks, **development_checks}
        context = development_record["context_advantage"]
        if arm == "retention-context":
            checks["development.context_advantage_each_source"] = all(
                values["mean"] > 0 for values in context["by_source"].values())
        arms[arm] = {
            "seen": seen_summary,
            "consumed_development_suite": development_summary,
            "context_advantage": context,
            "checks": checks,
            "advances_to_multiseed": all(checks.values()),
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
    write_json(root / "docs/evidence/retention-context-screen-v1.json", evidence)
    lines = [
        "# 保留能力加权与上下文约束筛选", "",
        "本实验只用于选择下一轮多种子方案。开发诊断集已经在 Release Candidate v1 中读取，不能再次作为新盲测或发布证据。", "",
        "| 方案 | 已见准确率 | 相对 Expanded | 已见最差源相对 Control | 开发集准确率 | 相对 Expanded | 开发集最差源相对 Control | 晋级 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
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
            f"{'是' if item['advances_to_multiseed'] else '否'} |")
    lines += ["", "## 冻结检查", ""]
    for arm in ARMS:
        lines += [f"### `{arm}`", "", "| 条件 | 结果 |", "|---|---|"]
        lines += [f"| `{name}` | {'PASS' if value else 'FAIL'} |"
                  for name, value in sorted(arms[arm]["checks"].items())]
        lines.append("")
    lines += [
        "## 结论边界", "",
        "- 训练数据全部来自本项目已审计的可再分发公开数据。",
        "- 单种子筛选只能淘汰明显失败的方案，不能证明稳定性。",
        "- 任何晋级方案都必须重新冻结三个种子，并使用从未读取的新任务族做一次性盲测。",
        "", "机器证据：`docs/evidence/retention-context-screen-v1.json`",
    ]
    (root / "docs/RETENTION_CONTEXT_SCREEN.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"advancing_arms": evidence["advancing_arms"]}, sort_keys=True))


if __name__ == "__main__":
    main()
