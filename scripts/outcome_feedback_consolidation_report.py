#!/usr/bin/env python3
"""Verify and report the fixed-log outcome-feedback consolidation follow-up."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_model.core import digest, write_json


ARMS = ("R-ips-loo-replay", "D-doubly-robust-replay", "A-consolidated-optimal-set")


def mean(records, key):
    return sum(record[key] for record in records) / len(records)


def compare(records, baseline, rule):
    regret_key = f"regret_delta_A_minus_{baseline}"
    reward_key = f"expected_reward_delta_A_minus_{baseline}"
    checks = {
        "mean_regret_lower": mean(records, regret_key) < 0,
        "seed_regret_wins": sum(record[regret_key] < 0 for record in records) >=
                            rule["A_seed_wins_required_against_each"],
        "maximum_single_seed_regret_regression": max(record[regret_key] for record in records) <=
                                                  rule["maximum_single_seed_regret_regression"],
        "mean_expected_reward_does_not_decline": mean(records, reward_key) >= 0,
    }
    return {"baseline": baseline, "checks": checks, "passes": all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/executable-outcome-replay-followup-v1.json")
    parser.add_argument("--run", default="runs/executable-outcome-feedback-consolidation-v1")
    parser.add_argument("--evidence", default="docs/evidence/outcome-feedback-consolidation-v1.json")
    parser.add_argument("--report", default="docs/OUTCOME_FEEDBACK_CONSOLIDATION_RESULTS.zh-CN.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path, run_root = root / args.protocol, root / args.run
    protocol = json.loads(protocol_path.read_text())
    status = json.loads((run_root / "status.json").read_text())
    if (status.get("state") != "complete" or status.get("official_test_parsed") or
            status.get("validation_is_fresh_confirmation")):
        raise ValueError("follow-up status violates the registered development boundary")

    records, endpoints = [], {}
    for endpoint in protocol["endpoints"]:
        loaded = {}
        for arm in ARMS:
            path = run_root / endpoint["id"] / arm
            checksums = json.loads((path / "checksums.json").read_text())
            if any(digest((path / name).read_bytes()) != value for name, value in checksums.items()):
                raise ValueError("endpoint checksum differs")
            run = json.loads((path / "run.json").read_text())
            if (run["new_environment_queries"] != 0 or
                    run["complete_reward_table_visible_to_training"] or
                    run["behavior_log_sha256"] != endpoint["behavior_log_sha256"]):
                raise ValueError("feedback information boundary differs")
            loaded[arm] = {
                "run": run,
                "evaluation": json.loads((path / "evaluation.json").read_text()),
                "curve": json.loads((path / "curve.json").read_text()),
                "checksums_sha256": digest((path / "checksums.json").read_bytes()),
            }
        controls = {(item["run"]["initial_policy_head_sha256"],
                     item["run"]["training_plan_sha256"]) for item in loaded.values()}
        integrity = int(len(controls) != 1)
        metrics = {arm: loaded[arm]["evaluation"]["validation"] for arm in ARMS}
        r, d, a = (metrics[arm] for arm in ARMS)
        coverage = loaded[ARMS[0]]["run"]["coverage"]
        if any(loaded[arm]["run"]["coverage"] != coverage for arm in ARMS):
            integrity += 1
        record = {
            "endpoint": endpoint["id"], "seed": endpoint["seed"],
            "complete_rows": coverage["complete_rows"],
            "incomplete_rows": coverage["incomplete_rows"],
            "regret_R": r["mean_decision_regret"],
            "regret_D": d["mean_decision_regret"],
            "regret_A": a["mean_decision_regret"],
            "regret_delta_A_minus_R": a["mean_decision_regret"] - r["mean_decision_regret"],
            "regret_delta_A_minus_D": a["mean_decision_regret"] - d["mean_decision_regret"],
            "expected_reward_R": r["mean_policy_expected_reward"],
            "expected_reward_D": d["mean_policy_expected_reward"],
            "expected_reward_A": a["mean_policy_expected_reward"],
            "expected_reward_delta_A_minus_R": a["mean_policy_expected_reward"] - r["mean_policy_expected_reward"],
            "expected_reward_delta_A_minus_D": a["mean_policy_expected_reward"] - d["mean_policy_expected_reward"],
            "optimal_hit_R": r["optimal_set_hit_rate"],
            "optimal_hit_D": d["optimal_set_hit_rate"],
            "optimal_hit_A": a["optimal_set_hit_rate"],
            "integrity_violations": integrity,
        }
        records.append(record)
        endpoints[endpoint["id"]] = loaded

    if len(records) != 3 or len({record["seed"] for record in records}) != 3:
        raise ValueError("expected exactly three seed comparisons")
    rule = protocol["development_success_rule"]
    versus_r = compare(records, "R", rule)
    versus_d = compare(records, "D", rule)
    integrity_passes = sum(record["integrity_violations"] for record in records) == \
                       rule["integrity_violations_allowed"]
    decision = {
        "versus_R": versus_r,
        "versus_D": versus_d,
        "integrity_passes": integrity_passes,
        "advances_to_confirmatory_protocol": versus_r["passes"] and versus_d["passes"] and integrity_passes,
    }
    evidence = {
        "kind": "outcome_feedback_consolidation_development_followup",
        "protocol_sha256": digest(protocol_path.read_bytes()),
        "official_test_parsed": False,
        "validation_is_fresh_confirmation": False,
        "new_environment_queries": 0,
        "comparisons": records,
        "decision": decision,
        "endpoints": endpoints,
    }
    write_json(root / args.evidence, evidence)

    lines = [
        "# 结果反馈汇总实验结果", "",
        "这是在已经消耗的验证集上进行的开发跟进，不是新鲜确认试验。"
        "官方 MBPP 测试段仍未解析，实验没有新增环境查询。", "",
        "A 臂只从已记录的选择动作反馈中汇总确定性结果；只有某行所有候选都被探索观测后，"
        "才对该行的全部并列最优候选进行学习。", "",
        "| 端点 | 完整/不完整训练行 | R regret | D regret | A regret | A-R | A-D |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f'| {record["endpoint"]} | {record["complete_rows"]}/{record["incomplete_rows"]} | '
            f'{record["regret_R"]:.4f} | {record["regret_D"]:.4f} | {record["regret_A"]:.4f} | '
            f'{record["regret_delta_A_minus_R"]:+.4f} | {record["regret_delta_A_minus_D"]:+.4f} |')
    lines += ["", "| 端点 | R 期望奖励 | D 期望奖励 | A 期望奖励 | A-R | A-D | A 最优集命中 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        lines.append(
            f'| {record["endpoint"]} | {record["expected_reward_R"]:.4f} | '
            f'{record["expected_reward_D"]:.4f} | {record["expected_reward_A"]:.4f} | '
            f'{record["expected_reward_delta_A_minus_R"]:+.4f} | '
            f'{record["expected_reward_delta_A_minus_D"]:+.4f} | {record["optimal_hit_A"]:.4f} |')
    lines += ["", "## 预注册判定", "",
              f'`advances_to_confirmatory_protocol = {str(decision["advances_to_confirmatory_protocol"]).lower()}`', ""]
    for label, comparison in (("A_vs_R", versus_r), ("A_vs_D", versus_d)):
        lines.append(f'- `{label}.passes`: {str(comparison["passes"]).lower()}')
        for key, value in comparison["checks"].items():
            lines.append(f'  - `{key}`: {str(value).lower()}')
    lines.append(f'- `integrity_passes`: {str(integrity_passes).lower()}')

    mean_values = {key: mean(records, key) for key in (
        "regret_R", "regret_D", "regret_A", "expected_reward_R", "expected_reward_D", "expected_reward_A")}
    lines += ["", "## 解释", "",
              f'A 的三种子平均 regret 为 {mean_values["regret_A"]:.4f}，R 为 '
              f'{mean_values["regret_R"]:.4f}，D 为 {mean_values["regret_D"]:.4f}。'
              "A 相对 D 是 2 胜 1 平，但相对 R 只有 1 胜 2 平，没有达到“至少两个种子严格获胜”的规则。", "",
              f'A 的平均策略期望奖励为 {mean_values["expected_reward_A"]:.4f}，R 为 '
              f'{mean_values["expected_reward_R"]:.4f}，D 为 {mean_values["expected_reward_D"]:.4f}。'
              "这说明汇总确定性反馈值得继续研究，但当前证据不足以触发官方测试或完整 LoRA 训练。", "",
              "A 在 100 个优化 epoch 时已将 seed42/43 的验证 regret 降至 0.0444，"
              "seed44 在 100 epoch 为 0.0741、200 epoch 回到 0.0815。"
              "后续应预先冻结训练时长或只依赖训练信号的停止规则，不能继续用当前验证曲线调参后冒充新鲜证据。"]
    (root / args.report).write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
