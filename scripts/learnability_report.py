"""Verify memorization/readout diagnostics and fixed curriculum continuation."""
import argparse
from collections import Counter
import json
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from semantic_scale_report import paired_diagnostics


def read(path):
    return json.loads(path.read_text())


def verify_checkpoint(path):
    for name, expected in read(path / "checksums.json").items():
        if Path(name).name != name or digest((path/name).read_bytes()) != expected:
            raise ValueError("Checkpoint integrity failed")


def load_experiment(root, study):
    sanity = root / "runs/overfit-sanity-v1"
    frozen = root / "runs/frozen-readout-v1"
    old = root / "runs/semantic-scale-v1"
    for directory in (sanity, frozen, study):
        if read(directory / "status.json")["state"] != "complete":
            raise ValueError("All planned stages must finish")
    memory = read(sanity / "result.json")
    memory_protocol = read(sanity / "protocol.json")
    rows = load_rows(sanity / "cases.jsonl")
    if any(r["split"] != "train" for r in rows) or [r["id"] for r in rows] != memory_protocol["selected_ids"]:
        raise ValueError("Memorization selection differs")
    steps = [json.loads(line) for line in (sanity / "steps.jsonl").read_text().splitlines()]
    counts = Counter(identity for step in steps for identity in step["rows"])
    if len(steps) != 80 or set(counts) != set(memory_protocol["selected_ids"]) or set(counts.values()) != {20}:
        raise ValueError("Memorization exposure differs")
    by_id = {r["id"]: r for r in rows}
    if any(sorted(Counter(by_id[key]["label"] for key in step["rows"]).values()) != [2,2,2] for step in steps):
        raise ValueError("Warm-up batches were not label-balanced")
    verify_checkpoint(sanity)
    head_probe = read(frozen / "result.json")
    if head_probe["protocol"]["selected_ids"] != memory_protocol["selected_ids"]:
        raise ValueError("Frozen probes used different examples")
    if abs(head_probe["arms"]["marker"]["initial"]["metrics"]["nll"] - memory["initial"]["metrics"]["nll"]) > 1e-4:
        raise ValueError("Cached marker representation does not reproduce the live initial model")
    for name in ("marker-no-context", "mean-no-context"):
        if head_probe["arms"][name]["final"]["metrics"]["correct"] != 8:
            raise ValueError("Negative control failed")
    protocol = read(study / "protocol.json")
    final = study / "continued"
    run = read(final / "run.json")
    previous_run = read(old / "seed-42/run.json")
    for key in ("selected_ids_sha256", "training_plan_sha256", "updates", "encoder_work"):
        if run[key] != previous_run[key]:
            raise ValueError("Continuation budget/order differs: " + key)
    if digest((final / "training-plan.jsonl").read_bytes()) != run["training_plan_sha256"]:
        raise ValueError("Continuation plan changed")
    if run["warm_start_weights_sha256"] != digest((sanity / "decision.safetensors").read_bytes()):
        raise ValueError("Wrong warm-up weights")
    if read(final / "model.json") != read(old / "seed-42.json"):
        raise ValueError("Continuation hyperparameters changed")
    verify_checkpoint(final)
    for name, expected in run["source_files"].items():
        if digest((final / "source" / name).read_bytes()) != expected or protocol["source_files"]["src/decision_model/"+name] != expected:
            raise ValueError("Continuation source snapshot differs")
    warm_eval = read(study / "warmup-evaluation.json")
    if any(warm_eval["selected_ids"][s] != run["selected_ids"][s] for s in ("validation", "test")):
        raise ValueError("Warm-up and continuation evaluation IDs differ")
    data = old / "cases.jsonl"
    if digest(data.read_bytes()) != run["dataset_sha256"]:
        raise ValueError("Evaluation dataset differs")
    all_rows = load_rows(data)
    test_rows = [r for r in all_rows if r["split"] == "test"]
    paired = paired_diagnostics(test_rows, read(old / "seed-42/test-predictions.json"), read(final / "test-predictions.json"))
    evaluation = read(final / "evaluation.json")
    for source, counts in paired.items():
        if evaluation["test"]["raw"]["by_source"][source]["correct"] != counts["both_correct"] + counts["fixed"]:
            raise ValueError("Paired comparison disagrees with accuracy")
    snli_test_ids = {r["id"] for r in test_rows if r["source"] == "snli"}
    snli_predictions = [p for p in read(final / "test-predictions.json") if p["id"] in snli_test_ids]
    snli_confusion = {label: dict(Counter(max(p["probabilities"], key=p["probabilities"].get)
                         for p in snli_predictions if p["label"] == label))
                      for label in ("contradiction", "neutral", "entailment")}
    warm_ids = set(memory_protocol["selected_ids"])
    snli_train_ids = {r["id"] for r in all_rows if r["source"] == "snli" and r["split"] == "train"}
    probe = [p for p in read(final / "train-predictions.json") if p["id"] in snli_train_ids]
    probe_groups = {}
    for name, selected in (("repeated_during_warmup", [p for p in probe if p["id"] in warm_ids]),
                           ("only_mixed_training", [p for p in probe if p["id"] not in warm_ids])):
        probe_groups[name] = {"n": len(selected), "correct": sum(max(p["probabilities"], key=p["probabilities"].get) == p["label"] for p in selected)}
    return {"protocol": protocol, "memorization": memory, "memorization_curve": read(sanity / "curve.json"),
            "frozen_readout": head_probe, "initial": read(old / "initial.json"),
            "previous": read(old / "seed-42/evaluation.json"), "warmup": warm_eval,
            "continued": {"run": run, "evaluation": evaluation, "audit": read(final / "audit.json"),
                          "weights_sha256": digest((final / "decision.safetensors").read_bytes())},
            "paired_vs_previous": paired, "snli_training_probe_exposure": probe_groups,
            "snli_test_confusion": snli_confusion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/curriculum-probe-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = load_experiment(root, root / args.study)
    write_json(root / "docs/evidence/learnability-v1.json", result)
    names = {"snli": "SNLI 推断", "clinc": "意图排序", "json-schema": "Schema", "bandit": "局部代码策略", "goemotions": "情绪（未参加微调）"}
    lines = ["# 从记忆检查到公开数据续训", "",
             "## 1. 当前结构能不能学会训练样本", "",
             "固定 24 条公开 SNLI 训练样本、每类 8 条；关闭 dropout，每批每类 2 条，从原 D-independent 开始训练固定 80 次。其他优化设置保持原值。", "",
             "| 更新次数 | 训练样本答对 | 原始 NLL |", "|---|---:|---:|"]
    for point in result["memorization_curve"]:
        lines.append(f"| {point['step']} | {point['metrics']['correct']}/24 | {point['metrics']['nll']:.4f} |")
    lines += ["", "每条样本实际训练 20 次；没有验证/测试选模。第 60 次出现退步，完整曲线保留；第 80 次是预先确定的终点。能够记住这些样本，不等于泛化。", "",
              "## 2. 表示与读出的小规模对照", "",
              "骨干固定在原 D-independent，提取相同样本的表示，各自用相同初始评分头做 512 次全批更新。头的初始权重此前训练于标记读出，两种读出并非各自优化后的最优配置。", "",
              "| 表示 | 开始答对 | 结束答对 | 结束 NLL |", "|---|---:|---:|---:|"]
    for name, arm in result["frozen_readout"]["arms"].items():
        lines.append(f"| {name} | {arm['initial']['metrics']['correct']}/24 | {arm['final']['metrics']['correct']}/24 | {arm['final']['metrics']['nll']:.5f} |")
    lines += ["", "`no-context` 把所有材料替换成相同内容。完整材料下能拟合、移除材料后退回固定类别水平，说明表示包含区分这些样本的信息；不证明可迁移语义。缓存标记表示的初始 NLL 与真实模型核对一致。", "",
              "## 3. 预热之后，回到原来的公开训练", "",
              "固定第 80 次预热权重在训练样本上达到预先规定的至少 23/24，随后从这份权重出发，完整执行原 seed-42 的 1,024 条混合数据训练。", "",
              "- 24 条预热样本全部属于原来的 1,024 条训练样本，没有增加独立样本。",
              "- 续训配置、样本和候选顺序、128 次更新、编码工作量与原 seed-42 核对一致；恢复 dropout=0.05。",
              "- 总预算增加到 80+128 次更新、480+1,024 次样本曝光；优化器在续训开始时重新初始化。",
              "- 预热评测完成后无条件执行续训，没有根据测试分数改流程或挑权重。",
              "- 同一批 192 条验证和 384 条开发测试；各阶段温度仅由验证集拟合。SNLI 参与训练，不是零样本。", "",
              "| 任务 | 原始起点 | 原一轮混合训练 | 仅 24 条预热 | 预热后再混合训练 |", "|---|---:|---:|---:|---:|"]
    evaluations = {"原始起点": result["initial"]["evaluation"], "原一轮混合训练": result["previous"],
                   "仅预热": result["warmup"]["evaluation"], "预热后续训": result["continued"]["evaluation"]}
    for source, title in names.items():
        values = [evaluation["test"]["raw"]["by_source"][source] for evaluation in evaluations.values()]
        lines.append(f"| {title} | " + " | ".join(f"{v['correct']}/{v['n']}（{v['accuracy']:.1%}）" for v in values) + " |")
    lines += ["", "### 推断任务的类别偏好", "",
              "下表每行是真实标签，每列是预测。三个真实类别各有 64 条；不只检查总准确率，也检查是否仍偏向固定候选。", "",
              "| 真实标签 | 预测矛盾 | 预测未知 | 预测支持 |", "|---|---:|---:|---:|"]
    for label, counts in result["snli_test_confusion"].items():
        lines.append(f"| {label} | " + " | ".join(str(counts.get(key, 0)) for key in ("contradiction", "neutral", "entailment")) + " |")
    lines += ["", "### 训练探针的重复曝光偏差", "",
              "最终训练探针并不代表所有训练样本；其中一部分在预热时已重复看到 20 次。分别报告，避免把记忆成绩当成整个训练集的拟合水平。", "",
              "| SNLI 训练探针子集 | 答对 / 样本 |", "|---|---:|"]
    for name, counts in result["snli_training_probe_exposure"].items():
        lines.append(f"| {name} | {counts['correct']}/{counts['n']} |")
    lines += ["", "### 相对原一轮混合训练的逐条变化", "",
              "| 任务 | 修正错误 | 引入错误 | 准确率变化 | 配对分组重采样 95% 区间 |", "|---|---:|---:|---:|---:|"]
    for source, title in names.items():
        p = result["paired_vs_previous"][source]
        low, high = p["paired_group_bootstrap_95_percentile"]
        lines.append(f"| {title} | {p['fixed']} | {p['regressed']} | {100*p['accuracy_change']:+.1f} 个百分点 | [{100*low:+.1f}, {100*high:+.1f}] |")
    lines += ["", "区间仅反映本批评测案例的条件不确定性，不包含种子、任务来源、超参数选择等总体不确定性，也未做多重比较校正。", "",
              "## 4. 概率质量", "", "| 任务 | 阶段 | 原始 NLL | 校准 NLL | 原始 Brier |", "|---|---|---:|---:|---:|"]
    for source, title in names.items():
        for stage, evaluation in evaluations.items():
            raw = evaluation["test"]["raw"]["by_source"][source]
            scaled = evaluation["test"]["temperature_scaled"]["by_source"][source]
            lines.append(f"| {title} | {stage} | {raw['nll']:.4f} | {scaled['nll']:.4f} | {raw['brier']:.4f} |")
    audit = result["continued"]["audit"]
    lines += ["", "## 5. 能说明什么", "",
              "- 能够拟合训练样本，说明现有结构和训练路径具备基本拟合能力；不能把过去的失败简单归因于参数没更新或标记表示完全没有信息。",
              "- 训练顺序、平衡采样、dropout 和额外预算共同改变。效果差异不能单独归因于某一项，也不是 RL 与直接求导的比较。",
              "- 本轮续训只有 seed-42 一个种子，使用已反复检查的开发集，不能声称已经达到通用零样本或生产可靠性。",
              f"- 最终模型换序稳定 {audit['stable_all_orders']}/{audit['cases']}，所有检查顺序都答对 {audit['correct_all_orders']}/{audit['cases']}；结构稳定性仍单独评价。", "",
              "完整记录见[evidence/learnability-v1.json](evidence/learnability-v1.json)。训练设计见[LEARNABILITY_DESIGN.zh-CN.md](LEARNABILITY_DESIGN.zh-CN.md)。", "",
              "```bash", "python scripts/overfit_sanity.py --device mps", "python scripts/frozen_readout_probe.py --device mps",
              "python scripts/run_curriculum_probe.py --device mps", "python scripts/learnability_report.py", "```", "",
              "每个实验需要新输出目录和既有公开起点；这些脚本不上传数据或权重。", ""]
    (root / "docs/LEARNABILITY_RESULTS.zh-CN.md").write_text("\n".join(lines))
    print(root / "docs/LEARNABILITY_RESULTS.zh-CN.md")


if __name__ == "__main__":
    main()
