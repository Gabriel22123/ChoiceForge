#!/usr/bin/env python3
"""Validate and report the frozen BoolQ context-repair screen."""
from __future__ import annotations

import json
from pathlib import Path

from decision_model.core import digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def source_deltas(candidate, control):
    if set(candidate["by_source"]) != set(control["by_source"]):
        raise ValueError("Seen source sets differ")
    return {source: candidate["by_source"][source]["accuracy"] -
            control["by_source"][source]["accuracy"]
            for source in sorted(control["by_source"])}


def main():
    root = Path(__file__).resolve().parents[1]
    study = root / "runs/boolq-context-repair-v1"
    protocol = read(study / "protocol.json")
    if read(study / "status.json")["state"] not in ("trained", "reporting"):
        raise ValueError("Screen endpoint is not trained")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen screen input changed: " + name)
    for name, expected in protocol["source_files"].items():
        if (digest((root / name).read_bytes()) != expected or
                digest((study / "frozen-source" / name).read_bytes()) != expected):
            raise ValueError("Frozen screen source changed: " + name)
    endpoint = read(study / "trained-checkpoint.json")
    checkpoint = root / endpoint["path"]
    development = study / "development-result/result.json"
    reference = study / "parent-validation-reference.json"
    if (digest((checkpoint / "decision.safetensors").read_bytes()) != endpoint["weights_sha256"] or
            digest((checkpoint / "evaluation.json").read_bytes()) != endpoint["evaluation_sha256"] or
            digest(development.read_bytes()) != endpoint["development_result_sha256"] or
            digest(reference.read_bytes()) != endpoint["parent_reference_sha256"]):
        raise ValueError("Screen endpoint checksum differs")

    release = root / "runs/release-candidate-v1"
    control_seen = read(release / "seed1701-control/evaluation.json")["test"]["raw"]
    control_dev = read(release / "blind-results/seed1701-control/result.json")["raw"]
    parent = root / protocol["parent"]["checkpoint"]
    parent_seen = read(parent / "evaluation.json")["test"]["raw"]
    parent_dev = read(
        root / "runs/functional-retention-screen-v1/development-results/"
        "warm-functional-context/result.json")["raw"]
    candidate_evaluation = read(checkpoint / "evaluation.json")
    candidate_seen = candidate_evaluation["test"]["raw"]
    candidate_dev_record = read(development)
    candidate_dev = candidate_dev_record["raw"]
    candidate_boolq = candidate_evaluation["validation"]["raw"]["by_source"]["boolq"]
    parent_boolq = read(reference)["raw"]["by_source"]["boolq"]
    seen_by_source = source_deltas(candidate_seen, control_seen)
    dev_by_source = source_deltas(candidate_dev, control_dev)
    rule = protocol["screening_rule"]
    checks = {
        "seen.accuracy": candidate_seen["accuracy"] - parent_seen["accuracy"] >=
            rule["seen_accuracy_vs_parent_min_delta"],
        "seen.nll": candidate_seen["nll"] - parent_seen["nll"] <=
            rule["seen_nll_vs_parent_max_delta"],
        "seen.each_source_accuracy": min(seen_by_source.values()) >=
            rule["seen_each_source_accuracy_vs_release_control_min_delta"],
        "development.accuracy": candidate_dev["accuracy"] - parent_dev["accuracy"] >=
            rule["development_accuracy_vs_parent_min_delta"],
        "development.each_source_accuracy": min(dev_by_source.values()) >=
            rule["development_each_source_accuracy_vs_release_control_min_delta"],
        "development.context_advantage_each_source": all(
            values["mean"] > 0 for values in
            candidate_dev_record["context_advantage"]["by_source"].values()),
        "boolq.validation_accuracy": candidate_boolq["accuracy"] - parent_boolq["accuracy"] >=
            rule["boolq_validation_accuracy_vs_parent_min_delta"],
    }
    summary = {
        "seen": {
            "candidate_accuracy": candidate_seen["accuracy"],
            "parent_accuracy": parent_seen["accuracy"],
            "accuracy_delta": candidate_seen["accuracy"] - parent_seen["accuracy"],
            "candidate_nll": candidate_seen["nll"],
            "parent_nll": parent_seen["nll"],
            "nll_delta": candidate_seen["nll"] - parent_seen["nll"],
            "source_accuracy_delta_vs_release_control": seen_by_source,
            "minimum_source_delta": min(seen_by_source.values()),
        },
        "consumed_development": {
            "candidate_accuracy": candidate_dev["accuracy"],
            "parent_accuracy": parent_dev["accuracy"],
            "accuracy_delta": candidate_dev["accuracy"] - parent_dev["accuracy"],
            "source_accuracy_delta_vs_release_control": dev_by_source,
            "minimum_source_delta": min(dev_by_source.values()),
            "context_advantage": candidate_dev_record["context_advantage"],
        },
        "boolq_validation": {
            "candidate_accuracy": candidate_boolq["accuracy"],
            "parent_accuracy": parent_boolq["accuracy"],
            "accuracy_delta": candidate_boolq["accuracy"] - parent_boolq["accuracy"],
            "rows": candidate_boolq["n"],
        },
    }
    evidence = {
        "format_version": 1,
        "study": protocol["study"],
        "protocol_sha256": digest((study / "protocol.json").read_bytes()),
        "development_suite_is_fresh": False,
        "development_suite_release_eligible": False,
        "summary": summary,
        "checks": checks,
        "advances_to_multiseed": all(checks.values()),
        "new_fresh_blind_required_for_any_release_claim": True,
    }
    write_json(root / "docs/evidence/boolq-context-repair-v1.json", evidence)
    lines = [
        "# BoolQ 上下文修复筛选", "",
        "本实验只使用公开数据，从函数保留端点继续训练。BoolQ 选择顺序不读取答案；此前已消费的 384 条 BoolQ 全部排除。生成的数据受 CC-BY-SA-3.0 约束，只保存在本地，不进入 Apache-2.0 源码树。", "",
        "| 指标 | 父端点 | BoolQ 修复端点 | 变化 |",
        "|---|---:|---:|---:|",
        f"| 已见准确率 | {summary['seen']['parent_accuracy']:.4f} | {summary['seen']['candidate_accuracy']:.4f} | {summary['seen']['accuracy_delta']:+.4f} |",
        f"| 已见 NLL | {summary['seen']['parent_nll']:.4f} | {summary['seen']['candidate_nll']:.4f} | {summary['seen']['nll_delta']:+.4f} |",
        f"| 已消费开发准确率 | {summary['consumed_development']['parent_accuracy']:.4f} | {summary['consumed_development']['candidate_accuracy']:.4f} | {summary['consumed_development']['accuracy_delta']:+.4f} |",
        f"| BoolQ validation 准确率 | {summary['boolq_validation']['parent_accuracy']:.4f} | {summary['boolq_validation']['candidate_accuracy']:.4f} | {summary['boolq_validation']['accuracy_delta']:+.4f} |",
        "", "## 冻结检查", "", "| 条件 | 结果 |", "|---|---|",
    ]
    lines += [f"| `{name}` | {'PASS' if value else 'FAIL'} |"
              for name, value in sorted(checks.items())]
    lines += [
        "", "## 结论边界", "",
        "- BoolQ validation 参与开发筛选，不是盲测。",
        "- 已消费因果任务只用于验证原先失败是否修复，不能形成发布结论。",
        "- 即使全部通过，也只能晋级多种子；发布仍需从未读取的新任务族。",
        "", "机器证据：`docs/evidence/boolq-context-repair-v1.json`",
    ]
    (root / "docs/BOOLQ_CONTEXT_REPAIR.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"advances_to_multiseed": evidence["advances_to_multiseed"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
