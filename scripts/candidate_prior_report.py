"""Independently validate and report the frozen candidate-prior study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from decision_model.core import digest, load_rows, write_json
from decision_model.prior_correction import STRENGTH_GRID, select_shared_strength
from decision_model.prior_evaluation import (candidate_marginal_gap,
                                             evaluate_with_temperature,
                                             fit_and_evaluate)


SEEDS = (42, 43, 44)
DEVELOPMENT_SEEDS = (42, 43)
PRIMARY = ("paws-wiki", "snli")
RETAINED = ("bandit", "clinc", "goemotions", "json-schema")
BLIND = ("logical-deduction-three-objects", "logical-deduction-five-objects",
         "logical-deduction-seven-objects")


def read(path):
    return json.loads(Path(path).read_text())


def assert_close(actual, expected, path="root"):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("Keys differ at " + path)
        for key in actual:
            assert_close(actual[key], expected[key], path + "." + str(key))
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("Lengths differ at " + path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_close(left, right, path + f"[{index}]")
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Values differ at {path}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Values differ at {path}: {actual!r} != {expected!r}")


def validated(root, study):
    protocol = read(study / "protocol.json")
    status = read(study / "status.json")
    if (status.get("state") != "complete" or protocol["seeds"] != list(SEEDS) or
            protocol["development_seeds"] != list(DEVELOPMENT_SEEDS) or
            protocol["confirmation_seed"] != 44 or protocol["strength_grid"] != list(STRENGTH_GRID) or
            status.get("result_manifest_sha256") != digest((study / "result-manifest.json").read_bytes())):
        raise ValueError("Candidate-prior study is incomplete")
    for name, expected in read(study / "result-manifest.json").items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or digest((study / relative).read_bytes()) != expected:
            raise ValueError("Candidate-prior result changed: " + name)
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Candidate-prior frozen input changed: " + name)
    for name, expected in protocol["source_files"].items():
        snapshot = study / "frozen-source" / name
        live = root / name
        if digest(snapshot.read_bytes()) != expected or (digest(live.read_bytes()) != expected and not snapshot.is_file()):
            raise ValueError("Candidate-prior source changed: " + name)
    architecture = root / protocol["architecture_study"]
    if (digest((architecture / "result-manifest.json").read_bytes()) !=
            protocol["architecture_result_manifest_sha256"] or
            digest((architecture / "cases.jsonl").read_bytes()) != protocol["architecture_cases_sha256"]):
        raise ValueError("Architecture evidence changed")
    for seed in SEEDS:
        checkpoint = root / protocol["checkpoints"][str(seed)]["path"]
        expected = protocol["checkpoints"][str(seed)]
        if (digest((checkpoint / "decision.safetensors").read_bytes()) != expected["weights_sha256"] or
                digest((checkpoint / "model.json").read_bytes()) != expected["model_sha256"] or
                digest((checkpoint / "checksums.json").read_bytes()) != expected["checksums_sha256"]):
            raise ValueError("Candidate-prior checkpoint changed: " + str(seed))
    blind_protocol = read(study / "logical-deduction-blind/protocol.json")
    blind_rows = load_rows(study / "logical-deduction-blind/cases.jsonl")
    if (digest((study / "logical-deduction-blind/protocol.json").read_bytes()) != protocol["blind_protocol_sha256"] or
            digest((study / "logical-deduction-blind/cases.jsonl").read_bytes()) != protocol["blind_cases_sha256"] or
            blind_protocol["rows"] != 288 or len(blind_rows) != 288 or
            not blind_protocol["selection_is_answer_blind"] or
            blind_protocol["exact_training_request_overlap"] != 0):
        raise ValueError("Logical-deduction blind evidence changed")

    rows = load_rows(architecture / "cases.jsonl")
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    validation_nll, grids = {}, {}
    for seed in SEEDS:
        records = read(study / f"logits/seed{seed}-validation.json")["records"]
        grid = {str(strength): {key: value for key, value in
                               fit_and_evaluate(validation_rows, records, strength).items()
                               if key != "predictions"}
                for strength in STRENGTH_GRID}
        stored = read(study / f"validation-grid/seed{seed}.json")
        assert_close(grid, stored, f"validation-grid.seed{seed}")
        grids[str(seed)] = grid
        validation_nll[str(seed)] = {
            strength: grid[str(strength)]["temperature_scaled"]["nll"]
            for strength in STRENGTH_GRID}
    expected_selection = select_shared_strength({str(seed): validation_nll[str(seed)]
                                                 for seed in DEVELOPMENT_SEEDS})
    expected_selection["temperatures_by_seed"] = {
        str(seed): grids[str(seed)][str(expected_selection["selected_strength"])]["calibration"]["temperature"]
        for seed in SEEDS}
    expected_selection["raw_temperatures_by_seed"] = {
        str(seed): grids[str(seed)]["0.0"]["calibration"]["temperature"] for seed in SEEDS}
    # Match the persisted JSON representation: numeric mapping keys are strings.
    expected_selection = json.loads(json.dumps(expected_selection))
    selection = read(study / "selection.json")
    assert_close(expected_selection, selection, "selection")
    selection_sha = digest((study / "selection.json").read_bytes())
    if status["selection_sha256"] != selection_sha:
        raise ValueError("Candidate-prior selection identity differs")

    results = {}
    for seed in SEEDS:
        stored = read(study / f"results/seed{seed}.json")
        if (stored["seed"] != seed or stored["selection_sha256"] != selection_sha or
                not math.isclose(stored["selected_strength"], selection["selected_strength"])):
            raise ValueError("Candidate-prior result identity differs: " + str(seed))
        splits = {}
        for split, split_rows in (("test", test_rows), ("blind", blind_rows)):
            records = read(study / f"logits/seed{seed}-{split}.json")
            if records["selection_sha256_before_collection"] != selection_sha:
                raise ValueError("Blind/test logits were collected before frozen selection identity")
            raw = evaluate_with_temperature(
                split_rows, records["records"], 0.0,
                selection["raw_temperatures_by_seed"][str(seed)])
            corrected = evaluate_with_temperature(
                split_rows, records["records"], selection["selected_strength"],
                selection["temperatures_by_seed"][str(seed)])
            recalculated = {
                "raw": raw, "corrected": corrected,
                "raw_marginal_gap": candidate_marginal_gap(split_rows, raw["predictions"]),
                "corrected_marginal_gap": candidate_marginal_gap(split_rows, corrected["predictions"]),
            }
            assert_close(recalculated, stored["splits"][split], f"result.seed{seed}.{split}")
            splits[split] = recalculated
        audit = read(study / f"audits/seed{seed}.json")
        if (audit["cases"] != 12 or audit["stable_all_orders"] > 12 or
                audit["stable_under_id_rename"] > 12 or
                not 0 <= audit["max_probability_difference"] <= 2e-6):
            raise ValueError("Candidate-prior audit differs: " + str(seed))
        results[str(seed)] = {"splits": splits, "audit": audit}
    return protocol, blind_protocol, selection, grids, results


def advancement(selection, results):
    seed44 = results["44"]["splits"]["test"]
    primary_accuracy = []
    primary_nll = []
    retained_accuracy = []
    retained_nll = []
    for source in PRIMARY:
        raw = seed44["raw"]["temperature_scaled"]["by_source"][source]
        corrected = seed44["corrected"]["temperature_scaled"]["by_source"][source]
        primary_accuracy.append(corrected["accuracy"] - raw["accuracy"])
        primary_nll.append(corrected["nll"] - raw["nll"])
    for source in RETAINED:
        raw = seed44["raw"]["temperature_scaled"]["by_source"][source]
        corrected = seed44["corrected"]["temperature_scaled"]["by_source"][source]
        retained_accuracy.append(corrected["accuracy"] - raw["accuracy"])
        retained_nll.append(corrected["nll"] - raw["nll"])
    blind_accuracy, blind_nll, blind_combined_wins = [], [], 0
    for seed in SEEDS:
        split = results[str(seed)]["splits"]["blind"]
        per_seed = []
        for source in BLIND:
            raw = split["raw"]["temperature_scaled"]["by_source"][source]
            corrected = split["corrected"]["temperature_scaled"]["by_source"][source]
            delta = corrected["accuracy"] - raw["accuracy"]
            per_seed.append(delta)
            blind_accuracy.append(delta)
            blind_nll.append(corrected["nll"] - raw["nll"])
        blind_combined_wins += mean(per_seed) > 0
    checks = {
        "nonzero_selected_strength": selection["selected_strength"] > 0,
        "seed44_primary_no_accuracy_loss_below_minus_1pp": min(primary_accuracy) >= -0.01,
        "seed44_primary_mean_accuracy_not_worse": mean(primary_accuracy) >= 0,
        "seed44_primary_mean_calibrated_nll_not_worse": mean(primary_nll) <= 0,
        "seed44_retained_mean_accuracy_not_worse": mean(retained_accuracy) >= 0,
        "seed44_retained_mean_calibrated_nll_not_worse": mean(retained_nll) <= 0,
        "blind_combined_accuracy_wins_at_least_two_seeds": blind_combined_wins >= 2,
        "blind_mean_accuracy_improves": mean(blind_accuracy) > 0,
        "blind_mean_calibrated_nll_not_worse": mean(blind_nll) <= 0,
        "all_real_model_audits_stable": all(
            results[str(seed)]["audit"]["stable_all_orders"] == 12 and
            results[str(seed)]["audit"]["stable_under_id_rename"] == 12
            for seed in SEEDS),
    }
    checks["eligible_as_default"] = all(checks.values())
    effects = {"seed44_primary_accuracy_delta": primary_accuracy,
               "seed44_primary_nll_delta": primary_nll,
               "seed44_retained_accuracy_delta": retained_accuracy,
               "seed44_retained_nll_delta": retained_nll,
               "blind_accuracy_delta": blind_accuracy, "blind_nll_delta": blind_nll}
    return checks, effects


def pct(value):
    return f"{100 * value:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/candidate-prior-v1")
    parser.add_argument("--report", default="docs/CANDIDATE_PRIOR_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/candidate-prior-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, blind_protocol, selection, grids, results = validated(root, root / args.study)
    checks, effects = advancement(selection, results)
    evidence = {"protocol": protocol, "blind_protocol": blind_protocol,
                "selection": selection, "validation_grid": grids,
                "results": results, "advancement": checks, "effects": effects}
    write_json(root / args.evidence, evidence)

    lines = ["# 候选先验修正：三种子冻结研究", "", "## 结论", "",
             ("**修正方法通过全部预注册条件，可进入默认推理路线。**" if checks["eligible_as_default"]
             else "**修正方法未通过全部预注册条件，不进入默认推理路线。**"), "",
             f"开发种子只用 validation NLL 选择出的共享强度为 `{selection['selected_strength']}`。",
             "BIG-bench logical_deduction 在选择强度并写入哈希后才首次运行。", "",
             "冻结的首版独立校验器存在一个仅影响复核的 JSON 键类型错误：内存中的浮点 λ 键与落盘后的",
             "字符串键直接比较。修复版只在比较前做 JSON 规范化，没有改动协议、数据、模型、选择、",
             "logits、预测、指标或晋级规则；修复后全部证据通过重算。哈希和错误信息见",
             "[校验器勘误](evidence/candidate-prior-validator-errata.json)。", "",
             "## 开发集强度选择", "",
             "| λ | seed42 NLL | seed43 NLL | 等权平均 |", "|---:|---:|---:|---:|"]
    for strength in STRENGTH_GRID:
        left = grids["42"][str(strength)]["temperature_scaled"]["nll"]
        right = grids["43"][str(strength)]["temperature_scaled"]["nll"]
        lines.append(f"| {strength:g} | {left:.4f} | {right:.4f} | {mean((left, right)):.4f} |")
    lines += ["", "## seed44 确认任务", "",
              "| 任务 | raw 准确率 | 修正准确率 | raw NLL | 修正 NLL |",
              "|---|---:|---:|---:|---:|"]
    test = results["44"]["splits"]["test"]
    for source in (*PRIMARY, *RETAINED):
        raw = test["raw"]["temperature_scaled"]["by_source"][source]
        corrected = test["corrected"]["temperature_scaled"]["by_source"][source]
        lines.append(f"| {source} | {pct(raw['accuracy'])} | {pct(corrected['accuracy'])} | {raw['nll']:.4f} | {corrected['nll']:.4f} |")
    lines += ["", "## 新任务盲测：BIG-bench logical_deduction", "",
              "| 种子 / 子任务 | raw 准确率 | 修正准确率 | raw NLL | 修正 NLL |",
              "|---|---:|---:|---:|---:|"]
    for seed in SEEDS:
        split = results[str(seed)]["splits"]["blind"]
        for source in BLIND:
            raw = split["raw"]["temperature_scaled"]["by_source"][source]
            corrected = split["corrected"]["temperature_scaled"]["by_source"][source]
            lines.append(f"| seed{seed} / {source.removeprefix('logical-deduction-')} | {pct(raw['accuracy'])} | {pct(corrected['accuracy'])} | {raw['nll']:.4f} | {corrected['nll']:.4f} |")
    lines += ["", "## 预注册晋级规则", ""]
    lines += [f"- {'通过' if value else '未通过'}：`{name}`" for name, value in checks.items()]
    lines += ["", "## 输出审计", "",
              "| 种子 | 换序稳定 | ID 改名稳定 | 最大概率差 |",
              "|---|---:|---:|---:|"]
    for seed in SEEDS:
        audit = results[str(seed)]["audit"]
        lines.append(f"| seed{seed} | {audit['stable_all_orders']}/12 | {audit['stable_under_id_rename']}/12 | {audit['max_probability_difference']:.3g} |")
    lines += ["", "## 边界", "",
              "- 修正增加一次空上下文候选评分，推理工作量接近翻倍；本轮未重新测延迟。",
              "- 空上下文分数可能包含有效常识，不保证都是偏差。",
              "- BIG-bench 题目可能存在于底座预训练语料；盲测只排除本项目调参泄漏。",
              "- 本报告比较固定端点上的推理变换，不证明训练阶段已经消除候选先验。",
              f"- 机器可读证据：[{Path(args.evidence).name}](evidence/{Path(args.evidence).name})。", ""]
    (root / args.report).write_text("\n".join(lines))
    print({"selected_strength": selection["selected_strength"], "advancement": checks})


if __name__ == "__main__":
    main()
