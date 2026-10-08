"""Verify and report the fixed one- versus two-epoch relation-training endpoints."""
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


def load_experiment(root, study):
    if read(study / "status.json")["state"] != "complete":
        raise ValueError("Both one-epoch arms and audits must finish")
    protocol = read(study / "protocol.json")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen input changed: " + name)
    rows = load_rows(study / "cases.jsonl")
    test = [row for row in rows if row["split"] == "test"]
    results = []
    for seed in protocol["seeds"]:
        arm = f"seed-{seed}"
        one_dir = study / arm
        control = protocol["controls"][str(seed)]
        two_study = root / control["two_epoch_study"]
        two_dir = two_study / "dispersed"
        one_config = read(study / f"seed-{seed}.json")
        two_config = read(two_study / "dispersed.json")
        changed = {key for key in set(one_config) | set(two_config)
                   if one_config.get(key) != two_config.get(key)}
        if changed != {"epochs", "epoch_row_orders"} or one_config["epoch_row_orders"] != two_config["epoch_row_orders"][:1]:
            raise ValueError("One- and two-epoch configs differ beyond the second epoch")
        one_run, two_run = read(one_dir / "run.json"), read(two_dir / "run.json")
        if one_run["updates"] != 128 or two_run["updates"] != 256:
            raise ValueError("Wrong update budget")
        if one_run["selected_ids"] != two_run["selected_ids"] or one_run["selected_ids"] != protocol["selected_ids"]:
            raise ValueError("Selection differs")
        if (one_run["warm_start_weights_sha256"] != protocol["initial_weight_sha256"] or
                two_run["warm_start_weights_sha256"] != protocol["initial_weight_sha256"]):
            raise ValueError("Starting checkpoint differs")
        if one_run["initial_trainable_sha256"] != two_run["initial_trainable_sha256"]:
            raise ValueError("Initial trainable tensors differ")
        for directory in (one_dir, two_dir):
            for name, expected in read(directory / "checksums.json").items():
                if digest((directory / name).read_bytes()) != expected:
                    raise ValueError("Checkpoint changed: " + str(directory / name))
        one_plan = [json.loads(line) for line in (one_dir / "training-plan.jsonl").read_text().splitlines()]
        two_plan = [json.loads(line) for line in (two_dir / "training-plan.jsonl").read_text().splitlines()]
        first_two = [point for point in two_plan if point["epoch"] == 1]
        if one_plan != first_two or len(one_plan) != 512:
            raise ValueError("Executed first-epoch plans differ")
        one_steps = [json.loads(line) for line in (one_dir / "optimizer-steps.jsonl").read_text().splitlines()]
        two_steps = [json.loads(line) for line in (two_dir / "optimizer-steps.jsonl").read_text().splitlines()]
        if one_steps != two_steps[:128]:
            raise ValueError("First 128 optimizer records differ")
        one_curve, two_curve = read(one_dir / "curve.json"), read(two_dir / "curve.json")
        if one_curve != two_curve[:1] or len(two_curve) != 2:
            raise ValueError("First-epoch training curve differs")
        one_predictions = read(one_dir / "test-predictions.json")
        two_predictions = read(two_dir / "test-predictions.json")
        paired = paired_diagnostics(test, one_predictions, two_predictions)
        one_eval, two_eval = read(one_dir / "evaluation.json"), read(two_dir / "evaluation.json")
        for source, counts in paired.items():
            correct = counts["both_correct"] + counts["regressed"]
            if one_eval["test"]["raw"]["by_source"][source]["correct"] != correct:
                raise ValueError("One-epoch metrics disagree with predictions")
            correct = counts["both_correct"] + counts["fixed"]
            if two_eval["test"]["raw"]["by_source"][source]["correct"] != correct:
                raise ValueError("Two-epoch metrics disagree with predictions")
        two_matrix = paired["snli"]["confusion"]["trained"]
        one_matrix = paired["snli"]["confusion"]["initial"]
        # neutral_stats expects the trained matrix in the standard report shape.
        one_wrapper = {"arms": {"one": {"paired_vs_initial": {"snli": {"confusion": {"trained": one_matrix}}}}}}
        two_wrapper = {"arms": {"two": {"paired_vs_initial": {"snli": {"confusion": {"trained": two_matrix}}}}}}
        one_neutral = neutral_stats(one_wrapper, "one")
        two_neutral = neutral_stats(two_wrapper, "two")
        results.append({"seed": seed,
                        "one_epoch": {"evaluation": one_eval, "audit": read(one_dir / "audit.json"),
                                      "neutral": one_neutral, "weights_sha256": digest((one_dir / "decision.safetensors").read_bytes()),
                                      "encoder_work": one_run["encoder_work"]},
                        "two_epoch": {"evaluation": two_eval, "audit": read(two_dir / "audit.json"),
                                      "neutral": two_neutral, "weights_sha256": digest((two_dir / "decision.safetensors").read_bytes()),
                                      "encoder_work": two_run["encoder_work"]},
                        "paired_two_minus_one": paired,
                        "first_epoch_execution_exact_match": True})
    return {"protocol": protocol, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/relation-second-epoch-v1")
    parser.add_argument("--report", default="docs/SECOND_EPOCH_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/relation-second-epoch-v1.json")
    parser.add_argument("--flip-evidence", default="docs/evidence/relation-second-epoch-flips-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = load_experiment(root, root / args.study)
    flip_path = root / args.flip_evidence
    if flip_path.exists():
        flip = read(flip_path)
        by_name = flip["models"]
        for record in result["results"]:
            name = f"seed{record['seed']}-one"
            if by_name[name]["weight_sha256"] != record["one_epoch"]["weights_sha256"]:
                raise ValueError("Flip probe used different one-epoch weights")
            record["one_epoch"]["evidence_flip"] = {
                "correct": by_name[name]["metrics"]["correct"], "n": by_name[name]["metrics"]["n"],
                "complete_groups": sum(group["all_three_correct"] for group in by_name[name]["groups"].values()),
                "groups": len(by_name[name]["groups"])}
    write_json(root / args.evidence, result)
    lines = ["# 第二轮训练贡献：同数据一轮与两轮对照", "", "## 结论", ""]
    deltas = []
    for record in result["results"]:
        one = record["one_epoch"]["evaluation"]["test"]["raw"]["by_source"]["snli"]
        two = record["two_epoch"]["evaluation"]["test"]["raw"]["by_source"]["snli"]
        deltas.append(two["accuracy"] - one["accuracy"])
    direction = "都提高" if all(delta > 0 for delta in deltas) else "没有在两个种子中都提高"
    lines += [f"在固定打散、标签平衡和第一轮执行记录完全一致的条件下，继续第二轮训练{direction} SNLI 总准确率。",
              f"两个种子的描述性平均变化为 {100*mean(deltas):+.1f} 个百分点。第二轮同时增加 128 次优化和一次重复曝光，当前结论不区分二者。", "",
              "## 推断逐类结果", "", "| seed | 终点 | 矛盾 | 信息不足 | 支持 | SNLI 总计 | 信息不足精确率 | 原始 NLL |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for record in result["results"]:
        for key, title in (("one_epoch", "一轮"), ("two_epoch", "两轮")):
            value = record[key]
            metric = value["evaluation"]["test"]["raw"]["by_source"]["snli"]
            neutral = value["neutral"]; per = neutral["per_class_correct"]
            lines.append(f"| {record['seed']} | {title} | {per['contradiction']}/64 | {per['neutral']}/64 | {per['entailment']}/64 | "
                         f"{metric['correct']}/192（{metric['accuracy']:.1%}） | {neutral['precision']:.1%} | {metric['nll']:.4f} |")
    lines += ["", "## 第二轮减第一轮：配对变化", "", "| seed | 任务 | 准确率变化 | 修正 / 引入错误 | 案例分组重采样 95% 区间 |",
              "|---:|---|---:|---:|---:|"]
    for record in result["results"]:
        for source, title in NAMES.items():
            paired = record["paired_two_minus_one"][source]
            low, high = paired["paired_group_bootstrap_95_percentile"]
            lines.append(f"| {record['seed']} | {title} | {100*paired['accuracy_change']:+.1f} 个百分点 | "
                         f"{paired['fixed']} / {paired['regressed']} | [{100*low:+.1f}, {100*high:+.1f}] |")
    lines += ["", "区间只反映固定开发案例的条件不确定性，不包含种子、调参或多重比较不确定性。", "",
              "## 旧任务终点", "", "| seed | 终点 | 意图 | Schema | 代码策略 | 未训练情绪 |", "|---:|---|---:|---:|---:|---:|"]
    for record in result["results"]:
        for key, title in (("one_epoch", "一轮"), ("two_epoch", "两轮")):
            sources = record[key]["evaluation"]["test"]["raw"]["by_source"]
            lines.append("| " + str(record["seed"]) + " | " + title + " | " + " | ".join(
                f"{sources[source]['correct']}/{sources[source]['n']}（{sources[source]['accuracy']:.1%}）"
                for source in ("clinc", "json-schema", "bandit", "goemotions")) + " |")
    if all("evidence_flip" in record["one_epoch"] for record in result["results"]):
        lines += ["", "## 证据翻转诊断", "", "| seed | 一轮单题 / 完整组 | 两轮单题 / 完整组 |", "|---:|---:|---:|"]
        existing = {42: read(root / "docs/evidence/relation-group-flips-v1.json"),
                    43: read(root / "docs/evidence/relation-group-seed43-flips-v1.json")}
        for record in result["results"]:
            seed = record["seed"]; one = record["one_epoch"]["evidence_flip"]
            model = existing[seed]["models"]["dispersed"]
            if model["weight_sha256"] != record["two_epoch"]["weights_sha256"]:
                raise ValueError("Existing two-epoch flip weights differ")
            complete = sum(group["all_three_correct"] for group in model["groups"].values())
            lines.append(f"| {seed} | {one['correct']}/{one['n']}、{one['complete_groups']}/{one['groups']} | "
                         f"{model['metrics']['correct']}/{model['metrics']['n']}、{complete}/{len(model['groups'])} |")
    lines += ["", "## 控制与边界", "",
              "- 每个种子的一轮配置只删除第二轮；第一轮的 512 个微批、候选顺序、128 条优化记录和曲线点逐项一致。",
              "- 两轮终点继续使用同一个优化器，不是从一轮权重重启优化器。",
              "- 第二轮既增加优化步数，也让同一批样本再次曝光，本实验只测二者合并后的净效果。",
              "- SNLI 是训练任务，测试已多次观察；结果不是通用零样本或独立盲测证明。",
              "- 一轮和两轮固定终点全部保留，不按测试表现选择模型。", "",
              f"机器可读记录：[evidence/{Path(args.evidence).name}](evidence/{Path(args.evidence).name})。", ""]
    (root / args.report).write_text("\n".join(lines))
    print(json.dumps({"seeds": [record["seed"] for record in result["results"]], "snli_deltas": deltas}, ensure_ascii=False))


if __name__ == "__main__":
    main()
