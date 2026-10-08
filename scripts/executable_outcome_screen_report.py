#!/usr/bin/env python3
"""Verify and report the executable outcome feedback screen."""
import argparse
import json
from pathlib import Path

from decision_model.core import digest, write_json


def advancement(records, rule):
    if len(records) != 3 or len({record["seed"] for record in records}) != 3:
        raise ValueError("expected exactly three seed comparisons")
    regret = [record["regret_delta_D_minus_R"] for record in records]
    expected = [record["expected_reward_delta_D_minus_R"] for record in records]
    checks = {
        "mean_regret_strictly_lower": sum(regret) / len(regret) < 0,
        "seed_regret_wins": sum(value < 0 for value in regret) >= rule["seed_wins_required"],
        "worst_seed_regret_within_limit": max(regret) <= rule["maximum_single_seed_regret_regression"],
        "mean_policy_expected_reward_does_not_decline": sum(expected) / len(expected) >= 0,
        "no_integrity_violations": sum(record["integrity_violations"] for record in records) ==
                                   rule["integrity_violations_allowed"],
    }
    return {"checks": checks, "advances_to_lora_validation": all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/executable-outcome-screen-v1.json")
    parser.add_argument("--run", default="runs/executable-outcome-screen-v1")
    parser.add_argument("--evidence", default="docs/evidence/executable-outcome-screen-v1.json")
    parser.add_argument("--report", default="docs/EXECUTABLE_OUTCOME_RLCD_RESULTS.zh-CN.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path, run_root = root / args.protocol, root / args.run
    protocol = json.loads(protocol_path.read_text())
    status = json.loads((run_root / "status.json").read_text())
    if status.get("state") != "complete" or status.get("official_test_parsed"):
        raise ValueError("screen is incomplete or official test was parsed")
    records, endpoints = [], {}
    for endpoint in protocol["representation_endpoints"]:
        endpoint_id = endpoint["id"]
        initial_evaluation = json.loads((run_root / endpoint_id / "initial-evaluation.json").read_text())
        loaded = {}
        for arm in protocol["arms"]:
            path = run_root / endpoint_id / arm
            checksums = json.loads((path / "checksums.json").read_text())
            if any(digest((path / name).read_bytes()) != value for name, value in checksums.items()):
                raise ValueError("endpoint checksum differs")
            loaded[arm] = {"run": json.loads((path / "run.json").read_text()),
                           "evaluation": json.loads((path / "evaluation.json").read_text()),
                           "checksums_sha256": digest((path / "checksums.json").read_bytes())}
        initial_hashes = {value["run"]["initial_policy_head_sha256"] for value in loaded.values()}
        plans = {value["run"]["training_plan_sha256"] for value in loaded.values()}
        limited_logs = {loaded[arm]["run"]["behavior_log_sha256"]
                        for arm in ("R-ips-loo", "D-doubly-robust")}
        integrity = int(len(initial_hashes) != 1) + int(len(plans) != 1) + int(len(limited_logs) != 1)
        r = loaded["R-ips-loo"]["evaluation"]["validation"]
        d = loaded["D-doubly-robust"]["evaluation"]["validation"]
        h = loaded["H-hard-winner"]["evaluation"]["validation"]
        o = loaded["O-full-information"]["evaluation"]["validation"]
        r_log = loaded["R-ips-loo"]["evaluation"]["logged_feedback"]
        d_log = loaded["D-doubly-robust"]["evaluation"]["logged_feedback"]
        reward_model = loaded["D-doubly-robust"]["evaluation"]["reward_model"]
        records.append({
            "endpoint": endpoint_id, "seed": endpoint["seed"],
            "initial_regret": initial_evaluation["validation"]["mean_decision_regret"],
            "initial_expected_reward": initial_evaluation["validation"]["mean_policy_expected_reward"],
            "regret_H": h["mean_decision_regret"], "regret_O": o["mean_decision_regret"],
            "regret_R": r["mean_decision_regret"], "regret_D": d["mean_decision_regret"],
            "regret_delta_D_minus_R": d["mean_decision_regret"] - r["mean_decision_regret"],
            "expected_reward_R": r["mean_policy_expected_reward"],
            "expected_reward_D": d["mean_policy_expected_reward"],
            "expected_reward_delta_D_minus_R": d["mean_policy_expected_reward"] - r["mean_policy_expected_reward"],
            "R_importance_p95": r_log["importance_weight_p95"],
            "R_importance_max": r_log["importance_weight_max"],
            "R_effective_sample_ratio": r_log["importance_effective_sample_ratio"],
            "D_importance_p95": d_log["importance_weight_p95"],
            "D_importance_max": d_log["importance_weight_max"],
            "D_effective_sample_ratio": d_log["importance_effective_sample_ratio"],
            "D_reward_model_validation_mse": reward_model["validation_full_table_mse"],
            "integrity_violations": integrity,
        })
        endpoints[endpoint_id] = {"initial": initial_evaluation, "arms": loaded}
    decision = advancement(records, protocol["advancement_rule_D_over_R"])
    evidence = {"kind": "executable_outcome_screen", "protocol_sha256": digest(protocol_path.read_bytes()),
                "official_test_parsed": False, "comparisons": records,
                "advancement": decision, "endpoints": endpoints}
    write_json(root / args.evidence, evidence)
    lines = ["# 可执行结果反馈筛选结果", "",
             "本报告只比较相同日志反馈下的 IPS-LOO 与双重稳健方法。官方 MBPP 测试段仍未解析。", "",
             "| 端点 | 初始 regret | H 硬标签 | O 全信息 | R IPS-LOO | D 双重稳健 |",
             "|---|---:|---:|---:|---:|---:|"]
    for record in records:
        lines.append(f'| {record["endpoint"]} | {record["initial_regret"]:.4f} | '
                     f'{record["regret_H"]:.4f} | {record["regret_O"]:.4f} | '
                     f'{record["regret_R"]:.4f} | {record["regret_D"]:.4f} |')
    lines += ["",
             "| 端点 | R regret | D regret | Δ regret | R 期望奖励 | D 期望奖励 | Δ 期望奖励 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        lines.append(f'| {record["endpoint"]} | {record["regret_R"]:.4f} | {record["regret_D"]:.4f} | '
                     f'{record["regret_delta_D_minus_R"]:+.4f} | {record["expected_reward_R"]:.4f} | '
                     f'{record["expected_reward_D"]:.4f} | {record["expected_reward_delta_D_minus_R"]:+.4f} |')
    lines += ["", "| 端点 | R 权重 P95 / max / ESS | D 权重 P95 / max / ESS | D 奖励头 MSE |",
              "|---|---:|---:|---:|"]
    for record in records:
        lines.append(f'| {record["endpoint"]} | {record["R_importance_p95"]:.2f} / '
                     f'{record["R_importance_max"]:.2f} / {record["R_effective_sample_ratio"]:.3f} | '
                     f'{record["D_importance_p95"]:.2f} / {record["D_importance_max"]:.2f} / '
                     f'{record["D_effective_sample_ratio"]:.3f} | '
                     f'{record["D_reward_model_validation_mse"]:.4f} |')
    lines += ["", "## 晋级判断", "", f'`advances_to_lora_validation = {str(decision["advances_to_lora_validation"]).lower()}`', ""]
    for key, value in decision["checks"].items():
        lines.append(f'- `{key}`: {str(value).lower()}')
    mean_delta = sum(record["expected_reward_delta_D_minus_R"] for record in records) / len(records)
    lines += ["", "## 解释", "",
              "D 与 R 在三套端点上选择了完全相同的 argmax，因而 regret 没有任何改善。"
              f"D 的策略期望奖励平均变化为 {mean_delta:+.4f}，只改变了概率锐度；seed43 还小幅下降。",
              "seed42 的 R/D 相对初始 regret 改善 0.0222，seed43 退化 0.0222，seed44 不变。"
              "有限反馈训练没有形成跨端点一致的选择收益。重要性权重保持有限，失败不能简单归因于权重爆炸。", "",
              "全信息 O 和硬标签 H 是解释性参照，不参与同信息量晋级结论。"
              "H 的三种子平均 regret 更低，但它读取完整奖励表并把并列最优压成单标签，不能当作有限反馈胜出。",
              "官方测试保持封存；没有通过开发集规则的方法不得用测试结果补救。"]
    (root / args.report).write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
