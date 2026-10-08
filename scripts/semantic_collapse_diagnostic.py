"""Post-hoc diagnosis of candidate preference and actual adapter movement."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import torch
from safetensors.torch import load_file

from decision_model.core import digest, metrics, write_json
from noisy_inference_diagnostic import reconstruct
from semantic_scale_report import load_study, read


def parameter_changes(before, after):
    if before.keys() != after.keys():
        raise ValueError("Checkpoint parameter sets differ")
    groups = {}
    for name, previous in before.items():
        current = after[name]
        if previous.shape != current.shape or not torch.isfinite(current).all() or not torch.isfinite(previous).all():
            raise ValueError("Invalid checkpoint parameter")
        key = "layer-" + re.search(r"layers\.(\d+)\.", name).group(1) if name.startswith("encoder.") else "head"
        entry = groups.setdefault(key, {"parameters": 0, "changed_parameters": 0,
                                       "squared_change": 0., "squared_initial": 0., "max_absolute_change": 0.})
        delta = current.float() - previous.float()
        entry["parameters"] += delta.numel()
        entry["changed_parameters"] += int((delta != 0).sum())
        entry["squared_change"] += float(delta.square().sum())
        entry["squared_initial"] += float(previous.float().square().sum())
        entry["max_absolute_change"] = max(entry["max_absolute_change"], float(delta.abs().max()))
    for entry in groups.values():
        entry["relative_l2_change"] = math.sqrt(entry["squared_change"] / entry["squared_initial"]) if entry["squared_initial"] else None
    return {"groups": groups,
            "updated_layers": sum(entry["changed_parameters"] > 0 for name, entry in groups.items() if name != "head"),
            "total_changed_parameters": sum(entry["changed_parameters"] for entry in groups.values())}


def align_predictions(predictions):
    """Stored requests may permute choices; compare meanings, never column offsets."""
    labels = sorted(predictions[0]["probabilities"])
    if any(set(p["probabilities"]) != set(labels) for p in predictions):
        raise ValueError("Candidate sets differ")
    return labels, [dict(p, probabilities={label: p["probabilities"][label] for label in labels})
                    for p in predictions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/semantic-scale-v1")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    protocol, initial, arms = load_study(study)
    rows = {r["id"]: r for r in map(json.loads, (study / "cases.jsonl").read_text().splitlines())}
    cases = {"initial": {"predictions": initial["predictions"], "temperature": initial["calibration"]["temperature"],
                         "evaluation": initial["evaluation"]}}
    cases.update({name: {"predictions": read(study / name / "test-predictions.json"),
                        "temperature": arm["calibration"]["temperature"], "evaluation": arm["evaluation"]}
                  for name, arm in arms.items()})
    original_path = root / protocol["initial_checkpoint"] / "decision.safetensors"
    if digest(original_path.read_bytes()) != protocol["initial_weight_sha256"]:
        raise ValueError("Initial checkpoint changed")
    original = load_file(str(original_path))
    result = {"kind": "Post-hoc diagnostic added after observing seed-42 label collapse; no training or checkpoint selection changes",
              "script_sha256": digest(Path(__file__).read_bytes()), "models": {},
              "limitations": ["Across-example margin variation is descriptive, not a causal context-ablation experiment",
                              "Parameter movement does not certify gradient correctness or semantic learning",
                              "Three-class SNLI development subset; not a conclusion about all unseen tasks"]}
    for name, case in cases.items():
        predictions = [p for p in case["predictions"] if rows[p["id"]]["source"] == "snli"]
        labels, predictions = align_predictions(predictions)
        recovered = reconstruct(predictions, case["temperature"])
        raw_logits = torch.stack([v for v, _ in recovered])
        raw_probabilities = raw_logits.softmax(-1)
        measured = metrics([y for _, y in recovered], raw_probabilities.tolist())
        expected = case["evaluation"]["test"]["raw"]["by_source"]["snli"]
        if abs(measured["nll"] - expected["nll"]) > 1e-5 or measured["correct"] != expected["correct"]:
            raise ValueError("Recovered logits do not match raw evaluation")
        reference = labels.index("entailment")
        gaps = raw_logits - raw_logits[:, reference:reference+1]
        summary = {"predicted_counts": dict(Counter(labels[i] for i in raw_logits.argmax(-1).tolist())),
                   "raw_metrics": measured, "raw_probability_moments": {}, "raw_score_gap_to_entailment": {}}
        for index, label in enumerate(labels):
            values = raw_probabilities[:, index]
            summary["raw_probability_moments"][label] = {"mean": values.mean().item(), "std": values.std(correction=0).item(),
                                                        "min": values.min().item(), "max": values.max().item()}
            column = gaps[:, index]
            summary["raw_score_gap_to_entailment"][label] = {
                "mean": column.mean().item(), "std": column.std(correction=0).item(),
                "min": column.min().item(), "max": column.max().item(),
                "mean_by_gold_label": {gold: column[torch.tensor([p["label"] == gold for p in predictions])].mean().item()
                                       for gold in labels}}
        if name != "initial":
            summary["parameter_updates"] = parameter_changes(original, load_file(str(study / name / "decision.safetensors")))
        result["models"][name] = summary
    write_json(root / "docs/evidence/semantic-collapse-v1.json", result)
    lines = ["# 补充诊断：概率损失下降，为什么判断没有学会？", "",
             "这是在观察到第一组标签偏好后增加的分析。没有修改训练、提前停训、选择权重或重拟合温度；两组主实验仍按原协议完成。", "",
             "## 1. 先检查实际输出", "",
             "| 版本 | 支持 | 矛盾 | 无法判断 | 原始 NLL | 原始 Brier |", "|---|---:|---:|---:|---:|---:|"]
    for name, model in result["models"].items():
        count, metric = model["predicted_counts"], model["raw_metrics"]
        lines.append(f"| {name} | {count.get('entailment',0)} | {count.get('contradiction',0)} | {count.get('neutral',0)} | {metric['nll']:.4f} | {metric['brier']:.4f} |")
    lines += ["", "单标签退化指硬决策持续落在同一个标签，不代表概率都集中在该标签；很小的固定偏好就能让接近均匀的概率总选同一项。", "",
              "测试三类各 64 条。完全不看输入、始终输出各 1/3 的概率，就能得到 NLL = ln(3) ≈ 1.0986、Brier = 2/3；固定打破平局后的准确率是 1/3。", "",
              "**严格适当评分规则约束的是概率目标，不保证模型学会利用输入。** 当模型没有学出足够的条件关系时，接近类别总体比例也能比一个错误而自信的偏好获得更低损失。温度校准同样不能补上语义判断。", "",
              "## 2. 不同材料下，原始候选分差变化多大", "",
              "从保存概率按 T×log(p) 恢复校准前分数，先核对原始 NLL/准确率，再计算相对“支持”的分差。已去掉温度缩放的影响；公共分数偏移抵消。", "",
              "| 版本 | 候选相对支持 | 平均分差 | 跨样本标准差 | 最小值 | 最大值 |", "|---|---|---:|---:|---:|---:|"]
    for name, model in result["models"].items():
        for label in ("contradiction", "neutral"):
            entry = model["raw_score_gap_to_entailment"][label]
            lines.append(f"| {name} | {label} | " + " | ".join(f"{entry[k]:.5f}" for k in ("mean", "std", "min", "max")) + " |")
    lines += ["", "该统计可以说明选项偏好与材料间变化的大小关系，但不是因果干预，不能据此断言模型完全不读取上下文。进一步定位需固定任务和候选，改变关键证据做反事实对照。", "",
              "## 3. 是否只是 LoRA 没有更新", "",
              "| 版本 | 发生参数变化的 LoRA 层 | 数值发生变化的可训练参数 |", "|---|---:|---:|"]
    for name in arms:
        entry = result["models"][name]["parameter_updates"]
        lines.append(f"| {name} | {entry['updated_layers']}/32 | {entry['total_changed_parameters']:,} |")
    lines += ["", "原始底座未更新；上表只比较适配器和评分头。权重变化能排除“适配器完全没动”，不能证明每个梯度正确，也不能替代能力评测。", "",
              "## 下一步该验证的假设", "",
              "1. 已补充仅改变关键证据的[证据翻转测试](EVIDENCE_FLIPS.zh-CN.md)，单独报告，不改变本轮训练或选模。",
              "2. 再分别检验训练预算、任务表述/评分位置与关系表示，使用训练/验证结果选择配置。",
              "3. 若重做梯度方法比较，保持相同数据、初始化和预算；本轮不能用来宣称策略梯度优于或劣于直接求导。", "",
              "完整数值见[evidence/semantic-collapse-v1.json](evidence/semantic-collapse-v1.json)。", ""]
    (root / "docs/SEMANTIC_COLLAPSE.zh-CN.md").write_text("\n".join(lines))
    print(root / "docs/SEMANTIC_COLLAPSE.zh-CN.md")


if __name__ == "__main__":
    main()
