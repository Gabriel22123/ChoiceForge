#!/usr/bin/env python3
"""Independently recompute every frozen release-candidate advancement gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from decision_model.core import digest, write_json
from release_evaluation import cross_seed_null_js


SEEDS = (1701, 1702, 1703)
ARMS = ("control", "expanded")


def read(path):
    return json.loads(Path(path).read_text())


def probability_loss(metrics):
    keys = [key for key in ("nll", "cross_entropy") if key in metrics]
    if len(keys) != 1:
        raise ValueError("Expected exactly one probability-loss metric")
    return metrics[keys[0]]


def paired_gate(metrics, retained_sources=None):
    wins = sum(metrics[(seed, "expanded")]["accuracy"] >
               metrics[(seed, "control")]["accuracy"] for seed in SEEDS)
    mean_accuracy = {arm: mean(metrics[(seed, arm)]["accuracy"] for seed in SEEDS)
                     for arm in ARMS}
    mean_loss = {arm: mean(probability_loss(metrics[(seed, arm)]) for seed in SEEDS)
                 for arm in ARMS}
    sources = set(metrics[(SEEDS[0], ARMS[0])]["by_source"])
    if any(set(metrics[(seed, arm)]["by_source"]) != sources
           for seed in SEEDS for arm in ARMS):
        raise ValueError("Paired metric task sets differ")
    regressions = {
        f"seed{seed}:{source}": (
            metrics[(seed, "expanded")]["by_source"][source]["accuracy"] -
            metrics[(seed, "control")]["by_source"][source]["accuracy"])
        for seed in SEEDS for source in sorted(sources)
    }
    result = {
        "paired_expanded_accuracy_wins": wins,
        "mean_accuracy": mean_accuracy,
        "mean_probability_loss": mean_loss,
        "task_seed_accuracy_delta": regressions,
        "minimum_task_seed_accuracy_delta": min(regressions.values()),
    }
    if retained_sources is not None:
        if not set(retained_sources) <= sources:
            raise ValueError("Retained source is absent from common test")
        endpoint_means = {
            (seed, arm): mean(metrics[(seed, arm)]["by_source"][source]["accuracy"]
                              for source in retained_sources)
            for seed in SEEDS for arm in ARMS
        }
        result["retained_task_mean_accuracy"] = {
            arm: mean(endpoint_means[(seed, arm)] for seed in SEEDS) for arm in ARMS
        }
    return result


def load_inputs(root, study):
    locked = read(study / "locked-endpoints.json")
    if locked.get("state") != "locked_before_blind" or len(locked.get("endpoints", {})) != 6:
        raise ValueError("Six endpoints are not locked")
    gate_lock = read(study / "gate-translation-lock.json")
    gate = read(study / "gate-translation.json")
    if (gate_lock["state"] != "locked_before_any_endpoint_result" or
            digest((study / "gate-translation.json").read_bytes()) !=
            gate_lock["gate_translation_sha256"]):
        raise ValueError("Gate translation lock differs")
    blind_protocol = read(study / "blind-suite/protocol.json")
    if blind_protocol["endpoint_manifest_sha256"] != digest(
            (study / "locked-endpoints.json").read_bytes()):
        raise ValueError("Blind suite was not sealed against these endpoints")
    seen, blind, records, temperatures, contracts = {}, {}, {}, {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            name = f"seed{seed}-{arm}"
            endpoint = study / name
            spec = locked["endpoints"][name]
            if (digest((endpoint / "decision.safetensors").read_bytes()) != spec["weights_sha256"] or
                    digest((endpoint / "evaluation.json").read_bytes()) != spec["evaluation_sha256"]):
                raise ValueError("Locked endpoint weights differ: " + name)
            seen[(seed, arm)] = read(endpoint / "evaluation.json")["test"]["raw"]
            result_dir = study / "blind-results" / name
            checksums = read(result_dir / "checksums.json")
            for filename, expected in checksums.items():
                if digest((result_dir / filename).read_bytes()) != expected:
                    raise ValueError("Blind result changed: " + name + "/" + filename)
            result = read(result_dir / "result.json")
            if (result["endpoint"] != name or
                    result["checkpoint_sha256"] != spec["weights_sha256"] or
                    result["endpoint_manifest_sha256"] != digest(
                        (study / "locked-endpoints.json").read_bytes()) or
                    result["blind_protocol_sha256"] != digest(
                        (study / "blind-suite/protocol.json").read_bytes()) or
                    result["cases_sha256"] != blind_protocol["cases_sha256"] or
                    result["training_or_selection_use"] is not False):
                raise ValueError("Blind result provenance differs: " + name)
            blind[(seed, arm)] = result["raw"]
            contracts[(seed, arm)] = result["strict_typed_contract"]
            records[(seed, arm)] = read(result_dir / "logits.private.json")
            temperatures[(seed, arm)] = result["temperature"]
    return gate, seen, blind, records, temperatures, contracts


def evaluate_gates(gate, seen, blind, records, temperatures, contracts):
    retained = gate["seen_tasks"]["retained_sources"]
    seen_summary = paired_gate(seen, retained)
    blind_summary = paired_gate(blind)
    prior = {}
    for arm in ARMS:
        summary, _ = cross_seed_null_js(
            {seed: records[(seed, arm)] for seed in SEEDS},
            {seed: temperatures[(seed, arm)] for seed in SEEDS})
        prior[arm] = summary
    # Context summaries live beside blind metrics and are added in main. Here
    # the record presence check keeps the prior gate total and explicit.
    for seed in SEEDS:
        endpoint_records = records[(seed, "expanded")]
        if not endpoint_records:
            raise ValueError("Missing prior records")
    checks = {
        "seen.paired_wins_at_least_two": seen_summary["paired_expanded_accuracy_wins"] >= 2,
        "seen.mean_probability_loss_not_worse":
            seen_summary["mean_probability_loss"]["expanded"] <=
            seen_summary["mean_probability_loss"]["control"],
        "seen.maximum_task_seed_regression":
            seen_summary["minimum_task_seed_accuracy_delta"] >= -0.02,
        "seen.retained_task_mean_not_worse":
            seen_summary["retained_task_mean_accuracy"]["expanded"] >=
            seen_summary["retained_task_mean_accuracy"]["control"],
        "blind.paired_wins_at_least_two": blind_summary["paired_expanded_accuracy_wins"] >= 2,
        "blind.mean_accuracy_strictly_higher":
            blind_summary["mean_accuracy"]["expanded"] > blind_summary["mean_accuracy"]["control"],
        "blind.mean_probability_loss_not_worse":
            blind_summary["mean_probability_loss"]["expanded"] <=
            blind_summary["mean_probability_loss"]["control"],
        "blind.maximum_task_seed_regression":
            blind_summary["minimum_task_seed_accuracy_delta"] >= -0.02,
        "prior.cross_seed_null_js": prior["expanded"]["mean_js"] <=
            prior["control"]["mean_js"] + 0.01,
        "contract.all_endpoints": all(
            contracts[(seed, arm)]["all_passed"] and
            contracts[(seed, arm)]["pass_rate"] == 1.0 and
            contracts[(seed, arm)]["cases"] == gate["contract"]["required_cases_per_endpoint"]
            for seed in SEEDS for arm in ARMS),
    }
    return checks, seen_summary, blind_summary, prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/release-candidate-v1")
    parser.add_argument("--evidence", default="docs/evidence/release-candidate-v1.json")
    parser.add_argument("--report", default="docs/RELEASE_CANDIDATE_RESULTS.zh-CN.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    gate, seen, blind, records, temperatures, contracts = load_inputs(root, study)
    checks, seen_summary, blind_summary, prior = evaluate_gates(
        gate, seen, blind, records, temperatures, contracts)
    context = {
        f"seed{seed}": read(study / "blind-results" / f"seed{seed}-expanded" /
                            "result.json")["context_advantage"]
        for seed in SEEDS
    }
    for seed, summary in context.items():
        for source, values in summary["by_source"].items():
            checks[f"prior.context_advantage.{seed}.{source}"] = values["mean"] > 0
    eligible = all(checks.values())
    evidence = {
        "format_version": 1, "study": "release-candidate-v1",
        "gate_translation_sha256": digest((study / "gate-translation.json").read_bytes()),
        "locked_endpoints_sha256": digest((study / "locked-endpoints.json").read_bytes()),
        "blind_protocol_sha256": digest((study / "blind-suite/protocol.json").read_bytes()),
        "seen": seen_summary, "blind": blind_summary, "candidate_prior": prior,
        "context_advantage": context, "checks": checks, "eligible": eligible,
        "release_endpoint": "seed1701-expanded" if eligible else None,
        "replication_seed_substitution_allowed": False,
    }
    evidence_path = root / args.evidence
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(evidence_path, evidence)
    lines = [
        "# Release Candidate v1 结果", "",
        f"**结论：{'通过全部冻结门槛，seed1701-expanded 是唯一候选。' if eligible else '未通过全部冻结门槛，不产生发布候选。'}**",
        "", "## 核心结果", "",
        "| 范围 | Control 平均准确率 | Expanded 平均准确率 | 配对胜出种子 | 最差任务×种子退化 |",
        "|---|---:|---:|---:|---:|",
        (f"| 已见公共任务 | {seen_summary['mean_accuracy']['control']:.4f} | "
         f"{seen_summary['mean_accuracy']['expanded']:.4f} | "
         f"{seen_summary['paired_expanded_accuracy_wins']}/3 | "
         f"{seen_summary['minimum_task_seed_accuracy_delta']:.4f} |"),
        (f"| 新任务族盲测 | {blind_summary['mean_accuracy']['control']:.4f} | "
         f"{blind_summary['mean_accuracy']['expanded']:.4f} | "
         f"{blind_summary['paired_expanded_accuracy_wins']}/3 | "
         f"{blind_summary['minimum_task_seed_accuracy_delta']:.4f} |"),
        "", "## 冻结门槛", "",
        "| 门槛 | 结果 |", "|---|---|",
        *[f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in sorted(checks.items())],
        "", "## 解释边界", "",
        "- 盲测任务只在六个端点和审计结果锁定后物化，未参与训练、校准、早停或选参。",
        "- Typed Decisions 是公开教师标签，衡量的是教师一致性，不等同于真实业务行动成功率。",
        "- 本协议只能控制本项目内的调参泄漏，无法证明基础模型预训练语料不含盲测任务。",
        "- 即使复制种子表现更好，也不能替换预先指定的 seed1701。",
        "", f"机器可复核证据：`{args.evidence}`",
    ]
    report_path = root / args.report
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"eligible": eligible, "checks_passed": sum(checks.values()),
                      "checks_total": len(checks), "evidence": str(evidence_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
