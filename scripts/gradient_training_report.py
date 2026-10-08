"""Validate and report the complete real-model gradient-estimator study."""
import argparse
from collections import Counter
import json
from pathlib import Path

from decision_model.core import digest, write_json
from run_gradient_training_study import ARMS, CONTROL_KEYS

NAMES = {"bandit": "局部代码策略", "clinc": "意图排序", "goemotions": "情绪诊断",
         "json-schema": "Schema 规则", "snli": "SNLI 证据推断"}


def read(path):
    return json.loads(path.read_text())


def load_study(study):
    if read(study / "status.json")["state"] != "complete":
        raise ValueError("All three arms and audits must complete first")
    protocol = read(study / "protocol.json")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen configuration changed")
    initial = read(study / "initial.json")
    if initial["weight_sha256"] != protocol["initial_weight_sha256"] or initial["dataset_sha256"] != protocol["dataset_sha256"]:
        raise ValueError("Initial reference identity mismatch")
    controls = noise = audit_ids = None
    arms = {}
    for arm, mode in ARMS.items():
        directory = study / arm
        run = read(directory / "run.json")
        current = {key: run[key] for key in CONTROL_KEYS}
        if controls is None:
            controls = current
        if current != controls:
            raise ValueError("Matched training controls differ")
        if run["warm_start_weights_sha256"] != initial["weight_sha256"]:
            raise ValueError("Wrong warm-start weights")
        if run["dataset_sha256"] != protocol["dataset_sha256"] or digest(run["selected_ids"]) != run["selected_ids_sha256"]:
            raise ValueError("Dataset/selection mismatch")
        if any(run["selected_ids"][split] != initial["selected_ids"][split] for split in ("validation", "test")):
            raise ValueError("Starting and trained checkpoints used different evaluation cases")
        if digest((directory / "training-plan.jsonl").read_bytes()) != run["training_plan_sha256"]:
            raise ValueError("Candidate order plan changed")
        if run["config_sha256"] != digest((study / (arm + ".json")).read_bytes()):
            raise ValueError("Configuration mismatch")
        if read(directory / "model.json") != dict(protocol["base_config"], gradient_estimator=mode):
            raise ValueError("Model configuration mismatch")
        for name, expected in read(directory / "checksums.json").items():
            if digest((directory / name).read_bytes()) != expected:
                raise ValueError("Checkpoint integrity mismatch")
        for name, expected in run["source_files"].items():
            if expected != protocol["source_files"]["src/decision_model/" + name] or digest((directory / "source" / name).read_bytes()) != expected:
                raise ValueError("Training source snapshot mismatch")
        if run["objective"]["gradient_estimator"] != mode:
            raise ValueError("Objective mode mismatch")
        current_noise = run["objective"]["noise_plan_sha256"]
        if mode != "clean":
            if not current_noise:
                raise ValueError("Noisy training recorded no draws")
            if noise is None:
                noise = current_noise
            elif current_noise != noise:
                raise ValueError("Noisy arms used different Gaussian samples")
        elif current_noise is not None:
            raise ValueError("Clean arm unexpectedly sampled noise")
        audit = read(directory / "audit.json")
        ids = [row["id"] for row in audit["details"]]
        if audit_ids is None:
            audit_ids = ids
        if ids != audit_ids:
            raise ValueError("Audit examples differ")
        steps = [json.loads(line) for line in (directory / "optimizer-steps.jsonl").read_text().splitlines()]
        if len(steps) != run["updates"] or sum(step["clipped"] for step in steps) != run["gradient_clipping"]["clipped_updates"]:
            raise ValueError("Optimizer step record mismatch")
        arms[arm] = {"run": run, "evaluation": read(directory / "evaluation.json"),
                     "calibration": read(directory / "calibration.json"), "audit": audit,
                     "curve": read(directory / "curve.json"), "steps": steps,
                     "weight_sha256": digest((directory / "decision.safetensors").read_bytes())}
    # Before the first optimizer update, matched weights/dropout/noise should
    # produce the same sampled risk despite different backward estimators.
    first_risks = [arms[arm]["steps"][0]["running_objective_risk"] for arm in ("B-pathwise", "C-score-function")]
    if abs(first_risks[0] - first_risks[1]) > 1e-5:
        raise ValueError("Initial sampled risks differ despite matched B/C controls")
    return protocol, initial, controls, arms


def paired_changes(study, rows):
    predictions = {arm: {r["id"]: r for r in read(study / arm / "test-predictions.json")} for arm in ARMS}
    first, second = predictions["B-pathwise"], predictions["C-score-function"]
    if set(first) != set(second):
        raise ValueError("Paired test examples mismatch")
    paired = {}
    for identity in first:
        row = rows[identity]
        pair = [first[identity], second[identity]]
        if any(p["label"] != row["label"] for p in pair):
            raise ValueError("Paired ground truth mismatch")
        correctness = [max(p["probabilities"], key=p["probabilities"].get) == row["label"] for p in pair]
        kind = {(True, True): "both_correct", (False, False): "both_wrong",
                (False, True): "score_fixed", (True, False): "score_regressed"}[tuple(correctness)]
        counts = paired.setdefault(row["source"], dict.fromkeys(("both_correct", "both_wrong", "score_fixed", "score_regressed"), 0))
        counts[kind] += 1
    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/gradient-training-v1")
    parser.add_argument("--data", default="data/public-multitask-v2/cases.jsonl")
    parser.add_argument("--output", default="docs/GRADIENT_TRAINING_STUDY.zh-CN.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    protocol, initial, controls, arms = load_study(study)
    data = (root / args.data).read_bytes()
    if digest(data) != protocol["dataset_sha256"]:
        raise ValueError("Report dataset mismatch")
    rows = {row["id"]: row for row in map(json.loads, data.splitlines())}
    paired = paired_changes(study, rows)
    for source, counts in paired.items():
        for arm, count in (("B-pathwise", counts["both_correct"]+counts["score_regressed"]),
                           ("C-score-function", counts["both_correct"]+counts["score_fixed"])):
            metric = arms[arm]["evaluation"]["test"]["raw"]["by_source"][source]
            if metric["correct"] != count or metric["n"] != sum(counts.values()):
                raise ValueError("Paired counts disagree with aggregate accuracy")
    sources = sorted(paired)
    selected_train = arms["A-clean"]["run"]["selected_ids"]["train"]
    train_counts = dict(Counter(rows[identity]["source"] for identity in selected_train))
    prior_run = read(root / protocol["initial_checkpoint"] / "run.json")
    exposure = {"continuation_by_source": train_counts,
                "prior_selected_train_ids": prior_run["selected_ids"]["train"],
                "overlap_with_prior_training": len(set(selected_train) & set(prior_run["selected_ids"]["train"]))}
    evaluations = {"起点": initial["evaluation"], **{name: arm["evaluation"] for name, arm in arms.items()}}
    lines = ["# 真实模型上的概率目标与梯度方法对照", "",
             "从同一独立评分检查点继续训练三组权重：A 无噪声概率损失，B 带噪目标直接求导，C 同一带噪目标的策略梯度。", "",
             "**A/B 比较噪声影响，B/C 比较梯度方法。** 这是校准决策思路的独立实验，不是任何私有训练配方复现。算法推导和边界见[实验设计](GRADIENT_TRAINING_DESIGN.zh-CN.md)。", "",
             "## 固定条件", "",
             "- 同一训练起点、独立评分架构、192 条训练样本、1 轮、24 次更新；每请求一份视图。各组重新初始化优化器。",
             "- 96 条验证、192 条诊断测试；B/C 每请求 16 次高斯分数采样，σ=0.4，CE+0.5 Brier；同一独立噪声计划。",
             "- 起点权重、初始化、数据 ID、候选顺序和编码输入计数通过核对。最终更新预先确定，未按测试结果选权重。",
             "- 起点模型也使用本轮相同验证集拟合温度；所有模型以无噪声分数推理。",
             "- SNLI 是新增训练任务；情绪任务仍没有参与微调或温度拟合，但已是开发诊断。公开底座预训练污染未知。", "",
             "本轮实际续训来源：" + "、".join(f"{NAMES[source]} {count} 条" for source, count in sorted(train_counts.items())) + "。",
             f"其中 {exposure['overlap_with_prior_training']} 条已被起点模型的上一轮训练使用；这是旧任务复习加新任务适配，不是 192 条全新数据。", "",
             "本轮是单种子短程续训，没有进行方法各自的超参数搜索；相同学习率不代表各自的最优学习率。不能由本轮推出收敛后排名、普遍 RL 优劣或通用零样本能力。", "",
             "## 1. 同一测试集的准确率", "",
             "| 任务 | 起点 | A-clean | B-pathwise | C-score-function |", "|---|---:|---:|---:|---:|"]
    for source in sources:
        values = [e["test"]["raw"]["by_source"][source] for e in evaluations.values()]
        lines.append(f"| {NAMES[source]} | " + " | ".join(f"{v['correct']}/{v['n']}（{v['accuracy']:.1%}）" for v in values) + " |")
    lines += ["", "意图任务是抽样候选排序，SNLI 为筛选和平衡后的子集。不同任务不能混成一个数字宣称零样本效果。", "",
              "### B/C 的逐条得失", "", "| 任务 | 两者均对 | C 修正 B 的错误 | C 引入错误 | 两者均错 |", "|---|---:|---:|---:|---:|"]
    for source in sources:
        counts = paired[source]
        lines.append(f"| {NAMES[source]} | " + " | ".join(str(counts[k]) for k in ("both_correct", "score_fixed", "score_regressed", "both_wrong")) + " |")
    lines += ["", "## 2. 概率质量", "", "NLL、Brier 和 ECE 越低越好。温度仅由验证集拟合；ECE 使用固定 10 个分箱，小样本结果容易波动。", "",
              "| 任务 | 版本 | 原始 NLL | 校准 NLL | 原始 Brier | 校准 Brier | 原始 ECE | 校准 ECE |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for source in sources:
        for name, evaluation in evaluations.items():
            values = [evaluation["test"][mode]["by_source"][source][metric] for metric in ("nll", "brier", "ece_10") for mode in ("raw", "temperature_scaled")]
            lines.append(f"| {NAMES[source]} | {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "## 3. 更新过程与成本", "",
              "| 版本 | 训练分钟 | 编码序列 | 最后一轮平均实际成本 | 平均替代量 | 裁剪次数 | 平均裁剪前范数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, arm in arms.items():
        run, curve = arm["run"], arm["curve"][-1]
        clipping = run["gradient_clipping"]
        lines.append(f"| {name} | {run['training_seconds']/60:.2f} | {run['encoder_work']['encoder_sequences']} | {curve['mean_training_loss']:.4f} | {curve['mean_optimization_surrogate']:.4f} | {clipping['clipped_updates']}/{clipping['updates']} | {clipping['mean_pre_clip_norm']:.3f} |")
    lines += ["", "A 的成本是无噪声目标，B/C 是采样带噪目标，不能直接按这一列为三组排名。C 的替代量只用于反向传播，不是概率损失。", "",
              "梯度裁剪与 AdamW 改变了实际更新；不同轨迹上的范数或裁剪次数也不是梯度方差测量。梯度方差证据仍来自单独固定参数的数值实验。训练时间为同一台日常使用设备上的观察值，不是隔离硬件基准。", "",
              "## 4. 换序与重载", "", "| 版本 | 始终一致 | 所有检查顺序均正确 | 最大概率变化 | 重载最大差异 |", "|---|---:|---:|---:|---:|"]
    for name, arm in arms.items():
        audit = arm["audit"]
        delta = max(row["max_order_probability_difference"] for row in audit["details"])
        lines.append(f"| {name} | {audit['stable_all_orders']}/{audit['cases']} | {audit['correct_all_orders']}/{audit['cases']} | {delta:.3g} | {audit['max_reload_probability_difference']:.3g} |")
    lines += ["", "换序性质来自独立候选评分结构，不能算作策略梯度学到的能力。完全同分时的首次出现优先规则仍是边界。", "",
              "## 记录与复现", "", "```bash", "python scripts/run_gradient_training_study.py --device mps", "python scripts/gradient_training_report.py", "```", "",
              "要求新输出目录。代码、配置、逐步更新日志和模型均保存在本地。完整机器可读结果见[evidence/gradient-training-v1.json](evidence/gradient-training-v1.json)。", "",
              f"- 数据 SHA-256：`{protocol['dataset_sha256']}`。",
              f"- 起点权重 SHA-256：`{protocol['initial_weight_sha256']}`。",
              f"- 初始参数 SHA-256：`{controls['initial_trainable_sha256']}`。",
              f"- B/C 噪声计划 SHA-256：`{arms['B-pathwise']['run']['objective']['noise_plan_sha256']}`。", ""]
    output = root / args.output
    output.write_text("\n".join(lines))
    write_json(output.parent / "evidence/gradient-training-v1.json", {"protocol": protocol, "initial": initial,
               "matched_controls": controls, "arms": arms, "paired_changes_B_C": paired, "training_exposure": exposure})
    print(output)


if __name__ == "__main__":
    main()
