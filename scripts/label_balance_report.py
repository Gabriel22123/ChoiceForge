"""Verify/report balanced versus variable-label dispersed relation training."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from decision_model.core import digest, load_rows, write_json
from relation_group_multiseed_report import neutral_stats
from semantic_scale_report import NAMES, paired_diagnostics


def read(path):
    return json.loads(path.read_text())


def stats_from_matrix(matrix):
    wrapper = {"arms": {"value": {"paired_vs_initial": {"snli": {"confusion": {"trained": matrix}}}}}}
    return neutral_stats(wrapper, "value")


def load_experiment(root, study):
    if read(study / "status.json")["state"] != "complete":
        raise ValueError("Both variable-label arms and audits must finish")
    protocol = read(study / "protocol.json")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen input changed: " + name)
    rows = load_rows(study / "cases.jsonl"); by_id = {row["id"]: row for row in rows}
    test = [row for row in rows if row["split"] == "test"]
    results = []
    for seed in protocol["seeds"]:
        arm = f"seed-{seed}"; variable_dir = study / arm
        control = protocol["controls"][str(seed)]
        balanced_study = root / control["balanced_study"]; balanced_dir = balanced_study / "dispersed"
        variable_config = read(study / f"seed-{seed}.json"); balanced_config = read(balanced_study / "dispersed.json")
        changed = {key for key in set(variable_config) | set(balanced_config)
                   if variable_config.get(key) != balanced_config.get(key)}
        if changed != {"epoch_row_orders"}:
            raise ValueError("Configs differ beyond row orders")
        for balanced_order, variable_order in zip(balanced_config["epoch_row_orders"], variable_config["epoch_row_orders"]):
            if Counter(balanced_order) != Counter(variable_order):
                raise ValueError("Training rows differ")
            for balanced_id, variable_id in zip(balanced_order, variable_order):
                before, after = by_id[balanced_id], by_id[variable_id]
                if before["source"] != after["source"] or before["group_id"] != after["group_id"]:
                    raise ValueError("Premise/source position differs")
                if before["source"] != "snli" and balanced_id != variable_id:
                    raise ValueError("Replay position differs")
        variable_run, balanced_run = read(variable_dir / "run.json"), read(balanced_dir / "run.json")
        if variable_run["updates"] != protocol["updates"] or balanced_run["updates"] != protocol["updates"]:
            raise ValueError("Wrong update budget")
        if variable_run["selected_ids"] != balanced_run["selected_ids"] or variable_run["selected_ids"] != protocol["selected_ids"]:
            raise ValueError("Selection differs")
        if (variable_run["warm_start_weights_sha256"] != protocol["initial_weight_sha256"] or
                balanced_run["warm_start_weights_sha256"] != protocol["initial_weight_sha256"] or
                variable_run["initial_trainable_sha256"] != balanced_run["initial_trainable_sha256"]):
            raise ValueError("Initial weights differ")
        for directory in (variable_dir, balanced_dir):
            for name, expected in read(directory / "checksums.json").items():
                if digest((directory / name).read_bytes()) != expected:
                    raise ValueError("Checkpoint changed: " + str(directory / name))
        variable_plan = [json.loads(line) for line in (variable_dir / "training-plan.jsonl").read_text().splitlines()]
        balanced_plan = [json.loads(line) for line in (balanced_dir / "training-plan.jsonl").read_text().splitlines()]
        if len(variable_plan) != 1024 or len(balanced_plan) != 1024:
            raise ValueError("Wrong microbatch count")
        for variable, balanced in zip(variable_plan, balanced_plan):
            if variable["epoch"] != balanced["epoch"]:
                raise ValueError("Epoch differs")
            for variable_id, balanced_id in zip(variable["rows"], balanced["rows"]):
                before, after = by_id[balanced_id], by_id[variable_id]
                if before["source"] != after["source"] or before["group_id"] != after["group_id"]:
                    raise ValueError("Executed premise/source position differs")
                if before["source"] != "snli" and balanced_id != variable_id:
                    raise ValueError("Executed replay differs")
        variable_predictions = read(variable_dir / "test-predictions.json")
        balanced_predictions = read(balanced_dir / "test-predictions.json")
        paired = paired_diagnostics(test, variable_predictions, balanced_predictions)
        variable_eval, balanced_eval = read(variable_dir / "evaluation.json"), read(balanced_dir / "evaluation.json")
        for source, counts in paired.items():
            if variable_eval["test"]["raw"]["by_source"][source]["correct"] != counts["both_correct"] + counts["regressed"]:
                raise ValueError("Variable-label metrics disagree")
            if balanced_eval["test"]["raw"]["by_source"][source]["correct"] != counts["both_correct"] + counts["fixed"]:
                raise ValueError("Balanced metrics disagree")
        variable_matrix = paired["snli"]["confusion"]["initial"]
        balanced_matrix = paired["snli"]["confusion"]["trained"]
        results.append({"seed": seed,
                        "variable": {"evaluation": variable_eval, "audit": read(variable_dir / "audit.json"),
                                     "neutral": stats_from_matrix(variable_matrix),
                                     "weights_sha256": digest((variable_dir / "decision.safetensors").read_bytes()),
                                     "encoder_work": variable_run["encoder_work"]},
                        "balanced": {"evaluation": balanced_eval, "audit": read(balanced_dir / "audit.json"),
                                     "neutral": stats_from_matrix(balanced_matrix),
                                     "weights_sha256": digest((balanced_dir / "decision.safetensors").read_bytes()),
                                     "encoder_work": balanced_run["encoder_work"]},
                        "paired_balanced_minus_variable": paired,
                        "premise_and_replay_positions_match": True})
    return {"protocol": protocol, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/relation-label-balance-v1")
    parser.add_argument("--report", default="docs/LABEL_BALANCE_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/relation-label-balance-v1.json")
    parser.add_argument("--flip-evidence", default="docs/evidence/relation-label-balance-flips-v1.json")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    result = load_experiment(root, root / args.study)
    flip_path = root / args.flip_evidence
    if flip_path.exists():
        flip = read(flip_path)
        for record in result["results"]:
            name = f"seed{record['seed']}-variable"; model = flip["models"][name]
            if model["weight_sha256"] != record["variable"]["weights_sha256"]:
                raise ValueError("Flip probe used different variable-label weights")
            record["variable"]["evidence_flip"] = {"correct": model["metrics"]["correct"], "n": model["metrics"]["n"],
                "complete_groups": sum(group["all_three_correct"] for group in model["groups"].values()), "groups": len(model["groups"])}
    write_json(root / args.evidence, result)
    deltas = []
    for record in result["results"]:
        variable = record["variable"]["evaluation"]["test"]["raw"]["by_source"]["snli"]
        balanced = record["balanced"]["evaluation"]["test"]["raw"]["by_source"]["snli"]
        deltas.append(balanced["accuracy"] - variable["accuracy"])
    direction = "都提高" if all(delta > 0 for delta in deltas) else "没有在两个种子中都提高"
    lines = ["# 更新内标签平衡：强制 2/2/2 是否有帮助？", "", "## 结论", "",
             f"在数据、前提位置、旧任务位置和两轮预算一致时，强制每次更新三类各 2 条{direction} SNLI 总准确率。",
             f"两个种子的描述性平均变化为 {100*mean(deltas):+.1f} 个百分点。", "",
             "## 推断逐类结果", "", "| seed | 标签组织 | 矛盾 | 信息不足 | 支持 | SNLI 总计 | 信息不足精确率 | 原始 NLL |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for record in result["results"]:
        for key, title in (("variable", "自然波动"), ("balanced", "每类 2 条")):
            value = record[key]; metric = value["evaluation"]["test"]["raw"]["by_source"]["snli"]
            neutral = value["neutral"]; per = neutral["per_class_correct"]
            lines.append(f"| {record['seed']} | {title} | {per['contradiction']}/64 | {per['neutral']}/64 | {per['entailment']}/64 | "
                         f"{metric['correct']}/192（{metric['accuracy']:.1%}） | {neutral['precision']:.1%} | {metric['nll']:.4f} |")
    lines += ["", "## 平衡减自然波动：配对变化", "", "| seed | 任务 | 准确率变化 | 修正 / 引入错误 | 案例分组重采样 95% 区间 |",
              "|---:|---|---:|---:|---:|"]
    for record in result["results"]:
        for source, title in NAMES.items():
            paired = record["paired_balanced_minus_variable"][source]
            low, high = paired["paired_group_bootstrap_95_percentile"]
            lines.append(f"| {record['seed']} | {title} | {100*paired['accuracy_change']:+.1f} 个百分点 | "
                         f"{paired['fixed']} / {paired['regressed']} | [{100*low:+.1f}, {100*high:+.1f}] |")
    lines += ["", "区间只反映固定开发案例的条件不确定性，不包含种子、调参或多重比较不确定性。", "",
              "## 旧任务终点", "", "| seed | 标签组织 | 意图 | Schema | 代码策略 | 未训练情绪 |", "|---:|---|---:|---:|---:|---:|"]
    for record in result["results"]:
        for key, title in (("variable", "自然波动"), ("balanced", "每类 2 条")):
            sources = record[key]["evaluation"]["test"]["raw"]["by_source"]
            lines.append("| " + str(record["seed"]) + " | " + title + " | " + " | ".join(
                f"{sources[source]['correct']}/{sources[source]['n']}（{sources[source]['accuracy']:.1%}）"
                for source in ("clinc", "json-schema", "bandit", "goemotions")) + " |")
    if all("evidence_flip" in record["variable"] for record in result["results"]):
        lines += ["", "## 证据翻转诊断", "", "| seed | 自然波动单题 / 完整组 | 平衡单题 / 完整组 |", "|---:|---:|---:|"]
        existing = {42: read(root / "docs/evidence/relation-group-flips-v1.json"),
                    43: read(root / "docs/evidence/relation-group-seed43-flips-v1.json")}
        for record in result["results"]:
            seed = record["seed"]; variable = record["variable"]["evidence_flip"]
            model = existing[seed]["models"]["dispersed"]
            if model["weight_sha256"] != record["balanced"]["weights_sha256"]:
                raise ValueError("Existing balanced flip weights differ")
            complete = sum(group["all_three_correct"] for group in model["groups"].values())
            lines.append(f"| {seed} | {variable['correct']}/{variable['n']}、{variable['complete_groups']}/{variable['groups']} | "
                         f"{model['metrics']['correct']}/{model['metrics']['n']}、{complete}/{len(model['groups'])} |")
    lines += ["", "## 控制与边界", "",
              "- 每个位置的来源、前提 group_id 和旧任务 ID 与平衡对照一致；每个样本每轮仍恰好出现一次。",
              "- 自然波动组只把同一前提的三类样本重新分配到其三个固定位置；因此假设文本对应的微批和 dropout 掩码也随处理变化。",
              "- 两组候选序列和有效 token 总量相同，padding 工作量可以不同，不宣称硬件 FLOPs 完全相等。",
              "- SNLI 是训练任务，测试已多次观察；结果不是通用零样本或独立盲测证明。",
              "- 两个固定终点全部保留，不按测试表现选择模型。", "",
              f"机器可读记录：[evidence/{Path(args.evidence).name}](evidence/{Path(args.evidence).name})。", ""]
    (root / args.report).write_text("\n".join(lines))
    print(json.dumps({"seeds": [record["seed"] for record in result["results"]], "snli_deltas": deltas}, ensure_ascii=False))


if __name__ == "__main__":
    main()
