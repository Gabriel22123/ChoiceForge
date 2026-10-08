#!/usr/bin/env python3
"""Validate and report the frozen QASC evidence-composition screen."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from decision_model.core import digest, load_rows, write_json


def read(path):
    return json.loads(Path(path).read_text())


def source_deltas(candidate, control):
    if set(candidate["by_source"]) != set(control["by_source"]):
        raise ValueError("Seen source sets differ")
    return {source: candidate["by_source"][source]["accuracy"] -
            control["by_source"][source]["accuracy"]
            for source in sorted(control["by_source"])}


def prediction_marginal(rows, predictions):
    selected = {row["id"]: row for row in rows if row["source"] == "qasc"}
    predicted = {row["id"]: row for row in predictions if row["id"] in selected}
    if set(predicted) != set(selected):
        raise ValueError("QASC prediction coverage differs")
    truth, choices = Counter(), Counter()
    for identifier, row in selected.items():
        point = predicted[identifier]
        ids = [choice["id"] for choice in row["request"]["choices"]]
        if point["label"] != row["label"] or set(point["probabilities"]) != set(ids):
            raise ValueError("QASC prediction support differs")
        truth[row["label"]] += 1
        choices[max(ids, key=lambda identity: point["probabilities"][identity])] += 1
    count = len(selected)
    identities = sorted(truth)
    gap = 0.5 * sum(abs(choices[identity] / count - truth[identity] / count)
                    for identity in identities)
    return {"rows": count, "truth": dict(truth), "predicted": dict(choices),
            "prediction_truth_total_variation": gap}


def main():
    root = Path(__file__).resolve().parents[1]
    study = root / "runs/qasc-evidence-screen-v1"
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
    qasc_parent_context = study / "qasc-parent-context-result/result.json"
    qasc_candidate_context = study / "qasc-candidate-context-result/result.json"
    exact = {
        checkpoint / "decision.safetensors": endpoint["weights_sha256"],
        checkpoint / "evaluation.json": endpoint["evaluation_sha256"],
        development: endpoint["development_result_sha256"],
        reference: endpoint["parent_reference_sha256"],
        qasc_parent_context: endpoint["qasc_parent_context_result_sha256"],
        qasc_candidate_context: endpoint["qasc_candidate_context_result_sha256"],
    }
    if any(digest(path.read_bytes()) != expected for path, expected in exact.items()):
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
    candidate_qasc = candidate_evaluation["validation"]["raw"]["by_source"]["qasc"]
    parent_reference = read(reference)
    parent_qasc = parent_reference["raw"]["by_source"]["qasc"]
    candidate_context = read(qasc_candidate_context)["context_advantage"]
    parent_context = read(qasc_parent_context)["context_advantage"]
    if (set(candidate_context["by_source"]) != {"qasc"} or
            set(parent_context["by_source"]) != {"qasc"}):
        raise ValueError("QASC context diagnostic source differs")
    candidate_context_mean = candidate_context["by_source"]["qasc"]["mean"]
    parent_context_mean = parent_context["by_source"]["qasc"]["mean"]
    cases = [row for row in load_rows(study / "cases.jsonl")
             if row["split"] == "validation"]
    candidate_marginal = prediction_marginal(
        cases, read(checkpoint / "validation-predictions.json"))
    parent_marginal = prediction_marginal(cases, parent_reference["predictions"])
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
        "qasc.validation_accuracy_delta": (
            candidate_qasc["accuracy"] - parent_qasc["accuracy"] >=
            rule["qasc_validation_accuracy_vs_parent_min_delta"]),
        "qasc.validation_accuracy_absolute": (
            candidate_qasc["accuracy"] >= rule["qasc_validation_accuracy_minimum"]),
        "qasc.validation_nll": (
            candidate_qasc["nll"] - parent_qasc["nll"] <=
            rule["qasc_validation_nll_vs_parent_max_delta"]),
        "qasc.context_advantage_absolute": (
            candidate_context_mean > rule["qasc_context_advantage_minimum"]),
        "qasc.context_advantage_delta": (
            candidate_context_mean - parent_context_mean >=
            rule["qasc_context_advantage_vs_parent_min_delta"]),
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
        "qasc_validation": {
            "candidate_accuracy": candidate_qasc["accuracy"],
            "parent_accuracy": parent_qasc["accuracy"],
            "accuracy_delta": candidate_qasc["accuracy"] - parent_qasc["accuracy"],
            "candidate_nll": candidate_qasc["nll"],
            "parent_nll": parent_qasc["nll"],
            "nll_delta": candidate_qasc["nll"] - parent_qasc["nll"],
            "candidate_context_advantage": candidate_context,
            "parent_context_advantage": parent_context,
            "context_advantage_delta": candidate_context_mean - parent_context_mean,
            "candidate_marginal": candidate_marginal,
            "parent_marginal": parent_marginal,
        },
    }
    evidence = {
        "format_version": 1,
        "study": protocol["study"],
        "protocol_sha256": digest((study / "protocol.json").read_bytes()),
        "qasc_validation_is_fresh_for_this_screen": True,
        "qasc_validation_release_eligible": False,
        "development_suite_is_fresh": False,
        "development_suite_release_eligible": False,
        "summary": summary,
        "checks": checks,
        "advances_to_multiseed": all(checks.values()),
        "new_fresh_blind_required_for_any_release_claim": True,
    }
    write_json(root / "docs/evidence/qasc-evidence-screen-v1.json", evidence)
    qasc = summary["qasc_validation"]
    lines = [
        "# QASC 八候选证据组合筛选", "",
        "本实验从函数保留端点续训，加入 512 条 QASC 训练题；每题保留两条科学事实和全部八个候选。候选按不读取答案的文本哈希重排，原始 A–H 字母不进入模型，再按重排后的正确位置做等量抽样。QASC 官方以 CC BY 4.0 发布。", "",
        "| 指标 | 父端点 | QASC 端点 | 变化 |",
        "|---|---:|---:|---:|",
        f"| 已见准确率 | {summary['seen']['parent_accuracy']:.4f} | {summary['seen']['candidate_accuracy']:.4f} | {summary['seen']['accuracy_delta']:+.4f} |",
        f"| 已见 NLL | {summary['seen']['parent_nll']:.4f} | {summary['seen']['candidate_nll']:.4f} | {summary['seen']['nll_delta']:+.4f} |",
        f"| 已消费开发准确率 | {summary['consumed_development']['parent_accuracy']:.4f} | {summary['consumed_development']['candidate_accuracy']:.4f} | {summary['consumed_development']['accuracy_delta']:+.4f} |",
        f"| QASC validation 准确率 | {qasc['parent_accuracy']:.4f} | {qasc['candidate_accuracy']:.4f} | {qasc['accuracy_delta']:+.4f} |",
        f"| QASC validation NLL | {qasc['parent_nll']:.4f} | {qasc['candidate_nll']:.4f} | {qasc['nll_delta']:+.4f} |",
        f"| QASC 上下文优势 | {qasc['parent_context_advantage']['mean']:.4f} | {qasc['candidate_context_advantage']['mean']:.4f} | {qasc['context_advantage_delta']:+.4f} |",
        "", "## 冻结检查", "", "| 条件 | 结果 |", "|---|---|",
    ]
    lines += [f"| `{name}` | {'PASS' if value else 'FAIL'} |"
              for name, value in sorted(checks.items())]
    lines += [
        "", "## 候选边际诊断", "",
        f"- 父端点预测与均衡真值的位置边际 TV：{qasc['parent_marginal']['prediction_truth_total_variation']:.4f}。",
        f"- QASC 端点预测与均衡真值的位置边际 TV：{qasc['candidate_marginal']['prediction_truth_total_variation']:.4f}。",
        "- 候选 ID 不进入模型；此统计用于发现答案文本打分坍缩，不用于宣称模型学习了位置。",
        "", "## 结论边界", "",
        "- QASC validation 在本轮报告生成后即成为已消费开发集，不是发布盲测。",
        "- 已消费的三类开发任务只用于检查旧失败，不能形成新的发布结论。",
        "- 即使全部通过，也只能晋级多种子；发布仍需从未读取的新任务族。",
        "- BIG-bench 的 canary 明确要求任务数据不要进入训练语料，因此本轮没有使用新增 BIG-bench 数据。",
        "", "机器证据：`docs/evidence/qasc-evidence-screen-v1.json`",
    ]
    (root / "docs/QASC_EVIDENCE_SCREEN.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"advances_to_multiseed": evidence["advances_to_multiseed"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
