"""Combine independently executed relation-group studies without selecting a winner."""
import argparse
import json
from pathlib import Path
from statistics import mean

from decision_model.core import write_json
from relation_group_plan import LABELS
from relation_group_report import load_study
from semantic_scale_report import NAMES


def read(path):
    return json.loads(path.read_text())


def neutral_stats(result, arm):
    matrix = result["arms"][arm]["paired_vs_initial"]["snli"]["confusion"]["trained"]
    correct = matrix["neutral"].get("neutral", 0)
    predicted = sum(row.get("neutral", 0) for row in matrix.values())
    return {"correct": correct, "total": sum(matrix["neutral"].values()),
            "predicted": predicted, "recall": correct / sum(matrix["neutral"].values()),
            "precision": correct / predicted if predicted else None,
            "per_class_correct": {label: matrix[label].get(label, 0) for label in LABELS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", action="append", default=[])
    parser.add_argument("--flip-evidence", action="append", default=[])
    parser.add_argument("--report", default="docs/RELATION_GROUP_MULTISEED.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/relation-group-multiseed-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    studies = args.study or ["runs/relation-group-v1", "runs/relation-group-seed43-v1"]
    loaded = [load_study(root, root / path) for path in studies]
    seeds = [value["protocol"]["seed"] for value in loaded]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")
    reference = loaded[0]["protocol"]
    for value in loaded[1:]:
        protocol = value["protocol"]
        for key in ("dataset_sha256", "initial_weight_sha256", "selected_ids", "updates",
                    "epochs", "example_exposures"):
            if protocol[key] != reference[key]:
                raise ValueError("Studies differ on controlled field: " + key)
    records = []
    for path, seed, result in zip(studies, seeds, loaded):
        arms = {}
        for arm in ("grouped", "dispersed"):
            sources = result["arms"][arm]["evaluation"]["test"]["raw"]["by_source"]
            arms[arm] = {"sources": sources, "neutral": neutral_stats(result, arm),
                         "order_audit": {key: result["arms"][arm]["audit"][key]
                                         for key in ("cases", "stable_all_orders", "correct_all_orders")},
                         "weights_sha256": result["arms"][arm]["weights_sha256"]}
        records.append({"study": path, "seed": seed, "arms": arms,
                        "paired_grouped_minus_dispersed": result["paired_grouped_minus_dispersed"]})
    flip_paths = args.flip_evidence or ["docs/evidence/relation-group-flips-v1.json",
                                        "docs/evidence/relation-group-seed43-flips-v1.json"]
    flips = {}
    if all((root / path).exists() for path in flip_paths):
        for path, record in zip(flip_paths, records):
            value = read(root / path)
            per_arm = {}
            for arm in ("grouped", "dispersed"):
                model = value["models"][arm]
                if model["weight_sha256"] != record["arms"][arm]["weights_sha256"]:
                    raise ValueError("Flip probe weights differ for seed " + str(record["seed"]))
                per_arm[arm] = {"correct": model["metrics"]["correct"],
                                "n": model["metrics"]["n"],
                                "complete_groups": sum(group["all_three_correct"] for group in model["groups"].values()),
                                "groups": len(model["groups"])}
            flips[str(record["seed"])] = per_arm
    summary = {}
    for arm in ("grouped", "dispersed"):
        summary[arm] = {
            "mean_snli_accuracy": mean(r["arms"][arm]["sources"]["snli"]["accuracy"] for r in records),
            "mean_snli_nll": mean(r["arms"][arm]["sources"]["snli"]["nll"] for r in records),
            "mean_neutral_recall": mean(r["arms"][arm]["neutral"]["recall"] for r in records),
            "mean_neutral_precision": mean(r["arms"][arm]["neutral"]["precision"] for r in records),
        }
    deltas = [r["paired_grouped_minus_dispersed"]["snli"]["accuracy_change"] for r in records]
    evidence = {"kind": "fixed two-seed relation-group replication", "seeds": seeds,
                "controls": {key: reference[key] for key in ("dataset_sha256", "initial_weight_sha256",
                                                              "selected_ids", "updates", "epochs", "example_exposures")},
                "studies": records, "summary": summary,
                "grouped_minus_dispersed_snli_accuracy_by_seed": deltas,
                "mean_grouped_minus_dispersed_snli_accuracy": mean(deltas),
                "evidence_flips": flips,
                "limitations": ["Only two predeclared conventional seeds; no seed-level confidence interval",
                                "Evaluation was repeatedly inspected during development",
                                "SNLI is a trained task, not zero-shot",
                                "Mean across two seeds is descriptive, not a population estimate"]}
    write_json(root / args.evidence, evidence)
    lines = ["# 同前提成组训练：两个种子的复核", "", "## 结论", ""]
    signs = [delta > 0 for delta in deltas]
    if all(signs):
        lines.append(f"两个种子里成组的 SNLI 总准确率都更高，平均差值为 {100*mean(deltas):+.1f} 个百分点。")
    elif not any(signs):
        lines.append(f"两个种子里成组都没有提高 SNLI 总准确率，平均差值为 {100*mean(deltas):+.1f} 个百分点。")
    else:
        lines.append(f"成组优势没有跨两个种子复现，平均差值为 {100*mean(deltas):+.1f} 个百分点。")
    lines += ["两次结果仍不足以建立稳定算法优势；逐类关系、概率质量和证据翻转必须一起看。", "",
              "## 逐种子结果", "", "| seed | 组 | 矛盾 | 信息不足 | 支持 | SNLI 总计 | 信息不足精确率 | 原始 NLL |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        for arm in ("grouped", "dispersed"):
            n = record["arms"][arm]["neutral"]
            metric = record["arms"][arm]["sources"]["snli"]
            per = n["per_class_correct"]
            lines.append(f"| {record['seed']} | {arm} | {per['contradiction']}/64 | {per['neutral']}/64 | {per['entailment']}/64 | "
                         f"{metric['correct']}/192（{metric['accuracy']:.1%}） | {n['precision']:.1%} | {metric['nll']:.4f} |")
    lines += ["", "## 两种子的描述性平均", "", "| 组 | SNLI 准确率 | 信息不足召回 | 信息不足精确率 | 原始 NLL |",
              "|---|---:|---:|---:|---:|"]
    for arm in ("grouped", "dispersed"):
        value = summary[arm]
        lines.append(f"| {arm} | {value['mean_snli_accuracy']:.1%} | {value['mean_neutral_recall']:.1%} | "
                     f"{value['mean_neutral_precision']:.1%} | {value['mean_snli_nll']:.4f} |")
    lines += ["", "这里的平均仅概括 seed42/43，不提供跨随机种子的置信区间，也不能替代更多重复。", "",
              "## 旧任务保留", "", "| seed | 组 | 意图 | Schema | 代码策略 | 未训练情绪 |",
              "|---:|---|---:|---:|---:|---:|"]
    for record in records:
        for arm in ("grouped", "dispersed"):
            sources = record["arms"][arm]["sources"]
            lines.append("| " + str(record["seed"]) + " | " + arm + " | " + " | ".join(
                f"{sources[source]['correct']}/{sources[source]['n']}（{sources[source]['accuracy']:.1%}）"
                for source in ("clinc", "json-schema", "bandit", "goemotions")) + " |")
    if flips:
        lines += ["", "## 证据翻转诊断", "", "| seed | 组 | 单题正确 | 完整三元组 |", "|---:|---|---:|---:|"]
        for record in records:
            for arm in ("grouped", "dispersed"):
                value = flips[str(record["seed"])][arm]
                lines.append(f"| {record['seed']} | {arm} | {value['correct']}/{value['n']} | "
                             f"{value['complete_groups']}/{value['groups']} |")
    lines += ["", "## 解释边界", "",
              "- 两个 study 的数据、起点、样本 ID、轮数、更新次数和曝光量已核对一致；种子及由种子决定的训练顺序不同。",
              "- 成组与打散在每次更新里的标签数和旧任务位置一致；模型仍逐条编码，没有跨样本注意力。",
              "- 相对更早实验，数据、标签平衡和轮数同时变化；只能把同一 seed 内的两组差值主要归因于批内前提组合。",
              "- 评测集已反复观察，SNLI 也参加了训练；不能作为通用零样本或独立盲测证明。",
              "- 两个固定终点全部保留，不按测试表现选择发布模型。", "",
              f"机器可读记录：[evidence/{Path(args.evidence).name}](evidence/{Path(args.evidence).name})。", ""]
    (root / args.report).write_text("\n".join(lines))
    print(json.dumps({"seeds": seeds, "deltas": deltas, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
