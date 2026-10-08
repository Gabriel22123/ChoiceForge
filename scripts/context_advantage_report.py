"""Independently validate and report the frozen context-advantage study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from decision_model.context_advantage_evaluation import summarize_context_advantage
from decision_model.core import digest, load_rows, write_json


SEEDS = (42, 43, 44)
PRIMARY = ("paws-wiki", "snli")
RETAINED = ("bandit", "clinc", "goemotions", "json-schema")


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
    elif (isinstance(actual, (int, float)) and not isinstance(actual, bool) and
          isinstance(expected, (int, float)) and not isinstance(expected, bool)):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Values differ at {path}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Values differ at {path}: {actual!r} != {expected!r}")


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        raise ValueError("Empty percentile input")
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def group_metrics(rows, predictions):
    by_id = {point["id"]: point for point in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Blind prediction IDs differ")
    groups = {}
    ambiguous_correct = ambiguous_rows = 0
    for row in rows:
        point = by_id[row["id"]]
        choices = row["request"]["choices"]
        ids = [choice["id"] for choice in choices]
        if point["label"] != row["label"] or set(point["probabilities"]) != set(ids):
            raise ValueError("Blind prediction contract differs")
        prediction = max(ids, key=lambda identity: point["probabilities"][identity])
        correct = prediction == row["label"]
        groups.setdefault(row["group_id"], []).append(correct)
        label_description = next(choice["description"] for choice in choices
                                 if choice["id"] == row["label"])
        if label_description == "Ambiguous":
            ambiguous_rows += 1
            ambiguous_correct += correct
    if set(map(len, groups.values())) != {3}:
        raise ValueError("Blind groups are not complete triples")
    complete = sum(all(values) for values in groups.values())
    return {
        "groups": len(groups), "complete_groups": complete,
        "complete_group_accuracy": complete / len(groups),
        "ambiguous_rows": ambiguous_rows,
        "ambiguous_correct": ambiguous_correct,
        "ambiguous_recall": ambiguous_correct / ambiguous_rows,
    }


def advancement(effects, audits):
    checks = {
        "primary_combined_accuracy_wins_at_least_two_seeds":
            sum(value > 0 for value in effects["primary_combined_accuracy_by_seed"]) >= 2,
        "primary_no_seed_task_accuracy_loss_below_minus_1pp":
            min(effects["primary_accuracy_all"]) >= -0.01,
        "primary_mean_raw_nll_not_worse": mean(effects["primary_nll_all"]) <= 0,
        "retained_mean_accuracy_not_worse": mean(effects["retained_accuracy_all"]) >= 0,
        "retained_mean_raw_nll_not_worse": mean(effects["retained_nll_all"]) <= 0,
        "mechanism_gain_wins_at_least_two_seeds":
            sum(value > 0 for value in effects["mechanism_gain_by_seed"]) >= 2,
        "mechanism_mean_gain_improves": mean(effects["mechanism_gain_by_seed"]) > 0,
        "mechanism_mean_shortfall_decreases":
            mean(effects["mechanism_shortfall_by_seed"]) < 0,
        "all_treatment_audits_stable": all(audits),
        "blind_combined_accuracy_wins_at_least_two_seeds":
            sum(value > 0 for value in effects["blind_accuracy_by_seed"]) >= 2,
        "blind_mean_accuracy_improves": mean(effects["blind_accuracy_by_seed"]) > 0,
        "blind_mean_raw_nll_not_worse": mean(effects["blind_nll_by_seed"]) <= 0,
        "blind_mean_ambiguous_recall_not_worse":
            mean(effects["blind_ambiguous_recall_by_seed"]) >= 0,
        "blind_mean_complete_group_accuracy_not_worse":
            mean(effects["blind_complete_group_accuracy_by_seed"]) >= 0,
    }
    checks["eligible_as_default"] = all(checks.values())
    return checks


def validated(root, study):
    protocol = read(study / "protocol.json")
    status = read(study / "status.json")
    if (status.get("state") != "complete" or protocol["seeds"] != list(SEEDS) or
            status.get("result_manifest_sha256") != digest(
                (study / "result-manifest.json").read_bytes())):
        raise ValueError("Context-advantage study is incomplete")
    for name, expected in read(study / "result-manifest.json").items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or digest(
                (study / relative).read_bytes()) != expected:
            raise ValueError("Context-advantage result changed: " + name)
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen protocol input changed: " + name)
    if read(study / "frozen-source-manifest.json") != {
            "source_files": protocol["source_files"]}:
        raise ValueError("Frozen source manifest differs")
    for name, expected in protocol["source_files"].items():
        snapshot = study / "frozen-source" / name
        if digest(snapshot.read_bytes()) != expected:
            raise ValueError("Frozen source snapshot changed: " + name)
        live = root / name
        if live.is_file() and digest(live.read_bytes()) != expected:
            raise ValueError("Live source changed after protocol freeze: " + name)
    control_study = root / protocol["control_study"]
    if digest((control_study / "result-manifest.json").read_bytes()) != \
            protocol["control_result_manifest_sha256"]:
        raise ValueError("Control study changed")
    blind_protocol = read(study / "disambiguation-blind/protocol.json")
    blind_rows = load_rows(study / "disambiguation-blind/cases.jsonl")
    if (digest((study / "disambiguation-blind/protocol.json").read_bytes()) !=
            protocol["blind_protocol_sha256"] or
            digest((study / "disambiguation-blind/cases.jsonl").read_bytes()) !=
            protocol["blind_cases_sha256"] or len(blind_rows) != 192 or
            blind_protocol["selected_groups"] != 64 or
            not blind_protocol["selection_is_answer_blind"] or
            not blind_protocol["complete_group_selection"]):
        raise ValueError("Blind evidence changed")
    rows = load_rows(study / "cases.jsonl")
    test_rows = [row for row in rows if row["split"] == "test"]
    treatments = read(study / "trained-checkpoints.json")
    results, audit_flags = {}, []
    for seed in SEEDS:
        control_record = protocol["controls"][str(seed)]
        control = root / control_record["path"]
        treatment_record = treatments[str(seed)]
        treatment = root / treatment_record["path"]
        for directory, record, label in ((control, control_record, "control"),
                                         (treatment, treatment_record, "treatment")):
            for filename, field in (("decision.safetensors", "weights_sha256"),
                                    ("model.json", "model_sha256"),
                                    ("checksums.json", "checksums_sha256"),
                                    ("run.json", "run_sha256"),
                                    ("evaluation.json", "evaluation_sha256"),
                                    ("training-plan.jsonl", "training_plan_sha256"),
                                    ("optimizer-steps.jsonl", "optimizer_steps_sha256")):
                if digest((directory / filename).read_bytes()) != record[field]:
                    raise ValueError(f"{label} checkpoint changed: {seed}/{filename}")
        if (control / "training-plan.jsonl").read_bytes() != \
                (treatment / "training-plan.jsonl").read_bytes():
            raise ValueError("Training plans differ: " + str(seed))
        control_run, treatment_run = read(control / "run.json"), read(treatment / "run.json")
        if control_run["initial_trainable_sha256"] != treatment_run["initial_trainable_sha256"]:
            raise ValueError("Initial trainable parameters differ: " + str(seed))
        audit = read(study / f"audit-results/seed{seed}-treatment.json")
        audit_ok = (audit["cases"] == 24 and audit["stable_all_orders"] == 24 and
                    audit["max_reload_probability_difference"] <= 2e-6 and
                    audit["max_id_rename_probability_difference"] <= 2e-6)
        audit_flags.append(audit_ok)
        arms = {}
        for arm, directory, record in (("control", control, control_record),
                                       ("treatment", treatment, treatment_record)):
            mechanism = read(study / f"mechanism/seed{seed}-{arm}.json")
            if (mechanism["checkpoint_weight_sha256"] != record["weights_sha256"] or
                    mechanism["dataset_sha256"] != protocol["dataset_sha256"] or
                    mechanism["collection"]["rows"] != len(test_rows)):
                raise ValueError("Mechanism identity differs: " + str(seed) + arm)
            recalculated = summarize_context_advantage(
                test_rows, mechanism["records"], protocol["intervention"]["gap"])
            assert_close(recalculated, mechanism["summary"], f"mechanism.{seed}.{arm}")
            blind = read(study / f"blind-results/seed{seed}-{arm}.json")
            if (blind["dataset_sha256"] != protocol["blind_cases_sha256"] or
                    blind["weights_sha256"] != record["weights_sha256"] or
                    blind["raw"]["n"] != 192):
                raise ValueError("Blind result identity differs: " + str(seed) + arm)
            arms[arm] = {
                "run": read(directory / "run.json"),
                "evaluation": read(directory / "evaluation.json"),
                "mechanism": recalculated,
                "blind": blind,
                "blind_groups": group_metrics(blind_rows, blind["predictions"]),
                "gradient_norm_p95": percentile([
                    json.loads(line)["pre_clip_grad_norm"]
                    for line in (directory / "optimizer-steps.jsonl").read_text().splitlines()
                    if line.strip()], .95),
            }
        results[str(seed)] = {"control": arms["control"],
                              "treatment": arms["treatment"], "audit": audit}
    return protocol, blind_protocol, results, audit_flags


def calculate_effects(results):
    effects = {
        "primary_combined_accuracy_by_seed": [], "primary_accuracy_all": [],
        "primary_nll_all": [], "retained_accuracy_all": [], "retained_nll_all": [],
        "mechanism_gain_by_seed": [], "mechanism_shortfall_by_seed": [],
        "blind_accuracy_by_seed": [], "blind_nll_by_seed": [],
        "blind_ambiguous_recall_by_seed": [],
        "blind_complete_group_accuracy_by_seed": [],
    }
    for seed in SEEDS:
        result = results[str(seed)]
        control = result["control"]
        treatment = result["treatment"]
        primary_accuracy = []
        for source in PRIMARY:
            left = control["evaluation"]["test"]["raw"]["by_source"][source]
            right = treatment["evaluation"]["test"]["raw"]["by_source"][source]
            accuracy_delta = right["accuracy"] - left["accuracy"]
            primary_accuracy.append(accuracy_delta)
            effects["primary_accuracy_all"].append(accuracy_delta)
            effects["primary_nll_all"].append(right["nll"] - left["nll"])
        effects["primary_combined_accuracy_by_seed"].append(mean(primary_accuracy))
        for source in RETAINED:
            left = control["evaluation"]["test"]["raw"]["by_source"][source]
            right = treatment["evaluation"]["test"]["raw"]["by_source"][source]
            effects["retained_accuracy_all"].append(right["accuracy"] - left["accuracy"])
            effects["retained_nll_all"].append(right["nll"] - left["nll"])
        left_m = control["mechanism"]["overall"]
        right_m = treatment["mechanism"]["overall"]
        effects["mechanism_gain_by_seed"].append(
            right_m["mean_gold_log_probability_gain"] -
            left_m["mean_gold_log_probability_gain"])
        effects["mechanism_shortfall_by_seed"].append(
            right_m["mean_margin_shortfall"] - left_m["mean_margin_shortfall"])
        effects["blind_accuracy_by_seed"].append(
            treatment["blind"]["raw"]["accuracy"] - control["blind"]["raw"]["accuracy"])
        effects["blind_nll_by_seed"].append(
            treatment["blind"]["raw"]["nll"] - control["blind"]["raw"]["nll"])
        effects["blind_ambiguous_recall_by_seed"].append(
            treatment["blind_groups"]["ambiguous_recall"] -
            control["blind_groups"]["ambiguous_recall"])
        effects["blind_complete_group_accuracy_by_seed"].append(
            treatment["blind_groups"]["complete_group_accuracy"] -
            control["blind_groups"]["complete_group_accuracy"])
    return effects


def pct(value):
    return f"{100 * value:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/context-advantage-v1")
    parser.add_argument("--report", default="docs/CONTEXT_ADVANTAGE_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/context-advantage-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, blind_protocol, results, audit_flags = validated(root, root / args.study)
    effects = calculate_effects(results)
    checks = advancement(effects, audit_flags)
    evidence = {"protocol": protocol, "blind_protocol": blind_protocol,
                "results": results, "effects": effects, "advancement": checks}
    write_json(root / args.evidence, evidence)

    lines = ["# 训练时上下文优势约束：三种子冻结研究", "", "## 结论", "",
             ("**通过全部预注册条件，可进入默认训练路线。**" if checks["eligible_as_default"]
              else "**未通过全部预注册条件，不进入默认训练路线。**"), "",
             "本研究只改变一件事：训练时要求真实上下文相对空上下文提高正确候选的对数概率。",
             "空上下文分支停止梯度并使用 eval 模式，因此不会改变对照组的 dropout 随机数序列。", "",
             "## 已见任务", "",
             "| 种子 / 任务 | 对照准确率 | 约束准确率 | 对照 NLL | 约束 NLL |",
             "|---|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for source in (*PRIMARY, *RETAINED):
            control = results[str(seed)]["control"]["evaluation"]["test"]["raw"]["by_source"][source]
            treatment = results[str(seed)]["treatment"]["evaluation"]["test"]["raw"]["by_source"][source]
            lines.append(f"| seed{seed} / {source} | {pct(control['accuracy'])} | {pct(treatment['accuracy'])} | {control['nll']:.4f} | {treatment['nll']:.4f} |")
    lines += ["", "## 机制指标", "",
              "`gold gain` 是真实上下文与空上下文下正确候选 log-probability 的差；越大说明模型越依赖材料。",
              "`shortfall` 是该差距未达到预注册 0.2 margin 的剩余量；越小越好。", "",
              "| 种子 | 对照 gold gain | 约束 gold gain | 对照 shortfall | 约束 shortfall |",
              "|---|---:|---:|---:|---:|"]
    for seed in SEEDS:
        left = results[str(seed)]["control"]["mechanism"]["overall"]
        right = results[str(seed)]["treatment"]["mechanism"]["overall"]
        lines.append(f"| seed{seed} | {left['mean_gold_log_probability_gain']:.4f} | {right['mean_gold_log_probability_gain']:.4f} | {left['mean_margin_shortfall']:.4f} | {right['mean_margin_shortfall']:.4f} |")
    lines += ["", "## 新任务盲测：BIG-bench disambiguation_qa", "",
              "64 个答案不可见抽取的完整 m/f/t 三元组，共 192 题；所有训练端点和审计完成后才首次推理。", "",
              "| 种子 | 对照准确率 | 约束准确率 | 对照 NLL | 约束 NLL | 对照 Ambiguous recall | 约束 Ambiguous recall | 对照完整组 | 约束完整组 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        control = results[str(seed)]["control"]
        treatment = results[str(seed)]["treatment"]
        lines.append(f"| seed{seed} | {pct(control['blind']['raw']['accuracy'])} | {pct(treatment['blind']['raw']['accuracy'])} | {control['blind']['raw']['nll']:.4f} | {treatment['blind']['raw']['nll']:.4f} | {pct(control['blind_groups']['ambiguous_recall'])} | {pct(treatment['blind_groups']['ambiguous_recall'])} | {control['blind_groups']['complete_groups']}/64 | {treatment['blind_groups']['complete_groups']}/64 |")
    lines += ["", "## 训练成本与结构审计", "",
              "| 种子 | 对照训练秒数 | 约束训练秒数 | 对照梯度 p95 | 约束梯度 p95 | 换序稳定 |",
              "|---|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        control = results[str(seed)]["control"]
        treatment = results[str(seed)]["treatment"]
        audit = results[str(seed)]["audit"]
        lines.append(f"| seed{seed} | {control['run']['training_seconds']:.1f} | {treatment['run']['training_seconds']:.1f} | {control['gradient_norm_p95']:.3f} | {treatment['gradient_norm_p95']:.3f} | {audit['stable_all_orders']}/24 |")
    lines += ["", "正式推理不运行空上下文分支，模型结构、参数量和一次请求的推理工作与 decoder 对照相同。", "",
              "## 预注册晋级规则", ""]
    lines += [f"- {'通过' if value else '未通过'}：`{name}`" for name, value in checks.items()]
    lines += ["", "## 边界", "",
              "- 本轮固定使用一个 weight 和 margin；smoke test 只验证实现可运行，没有用于选效果。",
              "- BIG-bench 可能存在于底座预训练语料；盲测只排除本项目训练和调参泄漏。",
              "- 机制指标证明这个约束是否改变材料依赖程度，不单独证明所有任务泛化。",
              f"- 机器可读证据：[{Path(args.evidence).name}](evidence/{Path(args.evidence).name})。", ""]
    (root / args.report).write_text("\n".join(lines))
    print({"advancement": checks})


if __name__ == "__main__":
    main()
