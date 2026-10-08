"""Verify a semantic expansion study and report both seeds, including regressions."""
import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path

from decision_model.core import digest, load_rows, write_json

ARMS = ("seed-42", "seed-43")
NAMES = {"snli": "SNLI 证据推断", "clinc": "意图排序", "json-schema": "Schema 规则",
         "bandit": "局部代码策略", "goemotions": "情绪（未参与微调）"}


def read(path):
    return json.loads(path.read_text())


def paired_diagnostics(rows, initial, trained):
    before = {p["id"]: p for p in initial}
    after = {p["id"]: p for p in trained}
    if set(before) != set(after) or set(before) != {r["id"] for r in rows}:
        raise ValueError("Paired evaluation IDs differ")
    output = {}
    for source in sorted({r["source"] for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        counts = Counter(dict(both_correct=0, both_wrong=0, fixed=0, regressed=0))
        confusion = {name: defaultdict(Counter) for name in ("initial", "trained")}
        groups = defaultdict(list)
        for row in subset:
            predictions = [before[row["id"]], after[row["id"]]]
            if any(p["label"] != row["label"] or set(p["probabilities"]) !=
                   {c["id"] for c in row["request"]["choices"]} for p in predictions):
                raise ValueError("Paired label/candidates mismatch")
            choices = [max(p["probabilities"], key=p["probabilities"].get) for p in predictions]
            for name, choice in zip(confusion, choices):
                confusion[name][row["label"]][choice] += 1
            correct = tuple(choice == row["label"] for choice in choices)
            counts[{(True, True): "both_correct", (False, False): "both_wrong",
                    (False, True): "fixed", (True, False): "regressed"}[correct]] += 1
            groups[row["group_id"]].append(int(correct[1]) - int(correct[0]))
        # Resample premise/schema groups, not individual rows from shared groups.
        # Conditional uncertainty over this finite evaluation population only.
        aggregates = [(sum(v), len(v)) for _, v in sorted(groups.items())]
        rng = random.Random(20260921)
        differences = []
        for _ in range(2000):
            sample = rng.choices(aggregates, k=len(aggregates))
            differences.append(sum(v[0] for v in sample) / sum(v[1] for v in sample))
        differences.sort()
        output[source] = {
            "n": len(subset), "groups": len(groups), **dict(counts),
            "accuracy_change": (counts["fixed"] - counts["regressed"]) / len(subset),
            "paired_group_bootstrap_95_percentile": [differences[49], differences[1949]],
            "confusion": {name: {label: dict(pred) for label, pred in matrix.items()}
                          for name, matrix in confusion.items()},
        }
    return output


def load_study(study):
    if read(study / "status.json")["state"] != "complete":
        raise ValueError("Both seeds and audits must finish before reporting/export")
    protocol = read(study / "protocol.json")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen study artifact changed: " + name)
    initial = read(study / "initial.json")
    if initial["weight_sha256"] != protocol["initial_weight_sha256"] or initial["dataset_sha256"] != protocol["dataset_sha256"]:
        raise ValueError("Starting reference mismatch")
    rows = load_rows(study / "cases.jsonl")
    test = [r for r in rows if r["split"] == "test"]
    arms = {}
    parameter_hash = audit_ids = None
    for arm in ARMS:
        directory = study / arm
        run = read(directory / "run.json")
        if run["selected_ids"] != protocol["selected_ids"] or run["updates"] != 128:
            raise ValueError("Training budget/selection differs")
        if run["warm_start_weights_sha256"] != protocol["initial_weight_sha256"] or run["dataset_sha256"] != protocol["dataset_sha256"]:
            raise ValueError("Training identity differs")
        if digest(run["selected_ids"]) != run["selected_ids_sha256"]:
            raise ValueError("Selected ID hash mismatch")
        if parameter_hash is None:
            parameter_hash = run["initial_trainable_sha256"]
        if run["initial_trainable_sha256"] != parameter_hash:
            raise ValueError("Different initial weights across seeds")
        config = read(study / (arm + ".json"))
        if read(directory / "model.json") != config or run["config_sha256"] != digest((study / (arm + ".json")).read_bytes()):
            raise ValueError("Model/config mismatch")
        for name, expected in read(directory / "checksums.json").items():
            if digest((directory / name).read_bytes()) != expected:
                raise ValueError("Checkpoint integrity mismatch")
        for name, expected in run["source_files"].items():
            if expected != protocol["source_files"]["src/decision_model/" + name] or digest((directory / "source" / name).read_bytes()) != expected:
                raise ValueError("Training source mismatch")
        plan = (directory / "training-plan.jsonl").read_bytes()
        if digest(plan) != run["training_plan_sha256"]:
            raise ValueError("Training plan changed")
        plan_rows = [identity for line in plan.splitlines() for identity in json.loads(line)["rows"]]
        if Counter(plan_rows) != Counter(run["selected_ids"]["train"]):
            raise ValueError("Training did not expose each selected row exactly once")
        if any(run["evaluation_ids"][s] != initial["selected_ids"][s] for s in ("validation", "test")):
            raise ValueError("Before/after evaluation mismatch")
        if len(run["evaluation_ids"]["train"]) != 128 or not set(run["evaluation_ids"]["train"]) <= set(run["selected_ids"]["train"]):
            raise ValueError("Training probe mismatch")
        audit = read(directory / "audit.json")
        ids = [r["id"] for r in audit["details"]]
        if audit_ids is None:
            audit_ids = ids
        if ids != audit_ids:
            raise ValueError("Audit cases differ")
        evaluation = read(directory / "evaluation.json")
        paired = paired_diagnostics(test, initial["predictions"], read(directory / "test-predictions.json"))
        for source, counts in paired.items():
            final_metric = evaluation["test"]["raw"]["by_source"][source]
            initial_metric = initial["evaluation"]["test"]["raw"]["by_source"][source]
            if final_metric["correct"] != counts["both_correct"] + counts["fixed"] or initial_metric["correct"] != counts["both_correct"] + counts["regressed"]:
                raise ValueError("Prediction accuracy disagrees with aggregate results")
        arms[arm] = {"run": run, "evaluation": evaluation, "audit": audit, "paired": paired,
                     "calibration": read(directory / "calibration.json"), "curve": read(directory / "curve.json"),
                     "weight_sha256": digest((directory / "decision.safetensors").read_bytes())}
    if arms[ARMS[0]]["run"]["training_plan_sha256"] == arms[ARMS[1]]["run"]["training_plan_sha256"]:
        raise ValueError("Different seeds unexpectedly used the same training plan")
    return protocol, initial, arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/semantic-scale-v1")
    parser.add_argument("--output", default="docs/SEMANTIC_SCALE_STUDY.zh-CN.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, initial, arms = load_study(root / args.study)
    evaluations = {"起点": initial["evaluation"], **{name: a["evaluation"] for name, a in arms.items()}}
    sources = ["snli", "clinc", "json-schema", "bandit", "goemotions"]
    lines = ["# 扩大公开语义训练：两个种子的实测", "",
             "## 本轮要回答什么", "",
             "固定独立候选评分结构和 CE + 0.5 Brier 训练方法，扩大 SNLI 推断训练，同时复习旧任务。",
             "方法固定用于控制变量，不是按上一轮测试成绩挑出的冠军。两个种子都报告，不选最好的一组。", "",
             "- 起点：上一轮架构实验的 D-independent；两个种子起始参数完全相同，优化器重新初始化。",
             "- 每个种子：768 条 SNLI + 128 条 CLINC + 96 条 Schema + 32 条 Bandit；一轮、128 次更新。",
             "- 256 条旧任务训练样本来自起点已使用的训练集；768 条 SNLI 未进入起点微调。",
             "- 验证：96 条 SNLI + 原来的 96 条旧任务；测试：192 条 SNLI + 原来的 192 条旧任务。",
             "- 训练前自动检查通过；每组另做真实模型换序、候选改名和权重重载检查。",
             "- 训练指标只测固定 128 条训练样本；实际训练完整使用 1,024 条，逐条训练计划已核对。", "",
             "相较起点，同时增加了新任务数据和更新次数，所以本轮回答“这套训练方案能否学起来”，不能单独归因于数据量。",
             "SNLI 已参与训练，其测试提升不能叫作零样本能力提升。情绪任务仍未参加微调/校准，但已被多轮开发查看。", "",
             "## 1. 固定测试集准确率", "",
             "| 任务 | 起点 | 种子 42 | 种子 43 |", "|---|---:|---:|---:|"]
    for source in sources:
        values = [e["test"]["raw"]["by_source"][source] for e in evaluations.values()]
        lines.append(f"| {NAMES[source]} | " + " | ".join(f"{v['correct']}/{v['n']}（{v['accuracy']:.1%}）" for v in values) + " |")
    lines += ["", "各任务独立报告。候选数、数据构造和难度不同，不以混合总分代替通用能力。", "",
              "### SNLI：训练探针、验证与测试", "",
              "训练探针来自训练集，仅用于检查是否学会训练样本；不参与模型选择，也不能称为泛化成绩。", "",
              "| 版本 | 划分 | 答对 / 样本 | 原始 NLL |", "|---|---|---:|---:|"]
    for name, arm in arms.items():
        for split, label in (("train", "训练探针"), ("validation", "验证"), ("test", "测试")):
            metric = arm["evaluation"][split]["raw"]["by_source"]["snli"]
            lines.append(f"| {name} | {label} | {metric['correct']}/{metric['n']} | {metric['nll']:.4f} |")
    lines += ["", "## 2. 推断任务的逐条变化", "",
              "| 版本 | 修正起点错误 | 引入新错误 | 准确率变化 | 按前提分组重采样的 95% 区间 |", "|---|---:|---:|---:|---:|"]
    for name, arm in arms.items():
        p = arm["paired"]["snli"]
        low, high = p["paired_group_bootstrap_95_percentile"]
        lines.append(f"| {name} | {p['fixed']} | {p['regressed']} | {100*p['accuracy_change']:+.1f} 个百分点 | [{100*low:+.1f}, {100*high:+.1f}] |")
    lines += ["", "区间来自 2,000 次同前提分组的配对重采样，只反映本评测样本构成的不确定性；不包含底座、数据来源、超参数和训练种子的总体不确定性，也未做多重比较校正。", "",
              "### 推断标签混淆", "", "| 版本 | 正确标签 | 预测标签及数量 |", "|---|---|---|"]
    matrices = {"起点": arms[ARMS[0]]["paired"]["snli"]["confusion"]["initial"],
                **{name: a["paired"]["snli"]["confusion"]["trained"] for name, a in arms.items()}}
    for name, matrix in matrices.items():
        for label, predictions in sorted(matrix.items()):
            lines.append(f"| {name} | {label} | " + "；".join(f"{k}: {v}" for k, v in sorted(predictions.items())) + " |")
    lines += ["", "## 3. 概率质量与校准", "",
              "NLL 和 Brier 越低越好。温度只由本轮验证集拟合，准确率不受这个正标量温度影响。", "",
              "| 任务 | 版本 | 原始 NLL | 校准 NLL | 原始 Brier | 校准 Brier |", "|---|---|---:|---:|---:|---:|"]
    for source in sources:
        for name, evaluation in evaluations.items():
            values = [evaluation["test"][mode]["by_source"][source][metric]
                      for metric in ("nll", "brier") for mode in ("raw", "temperature_scaled")]
            lines.append(f"| {NAMES[source]} | {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "三分类均匀概率的 NLL 约为 1.0986；只有准确率变化和概率损失一起看，才能避免把更自信误当成更强。", "",
              "## 4. 成本与结构稳定性", "",
              "| 版本 | 训练分钟 | 更新次数 | 发生裁剪的更新 | 换序稳定 | 所有顺序均正确 |", "|---|---:|---:|---:|---:|---:|"]
    for name, arm in arms.items():
        run, audit = arm["run"], arm["audit"]
        lines.append(f"| {name} | {run['training_seconds']/60:.2f} | {run['updates']} | {run['gradient_clipping']['clipped_updates']} | {audit['stable_all_orders']}/{audit['cases']} | {audit['correct_all_orders']}/{audit['cases']} |")
    lines += ["", "训练时间不含前后评测，也不是隔离硬件基准。换序稳定来自独立候选评分结构；同分时首次出现优先仍是边界。", "",
              "## 复现", "", "先按 SNLI 与公开混合数据说明生成固定数据，准备上一轮独立评分起点，再运行：", "", "```bash",
              "python scripts/run_semantic_scale_study.py --device mps --output runs/semantic-scale-reproduction",
              "python scripts/semantic_scale_report.py --study runs/semantic-scale-reproduction", "```", "",
              "每次使用新目录；配置、数据 ID、来源哈希、初始权重、训练顺序和代码快照全部保留。完整结果见[evidence/semantic-scale-v1.json](evidence/semantic-scale-v1.json)。", "",
              f"- 数据 SHA-256：`{protocol['dataset_sha256']}`。",
              f"- 起点权重 SHA-256：`{protocol['initial_weight_sha256']}`。",
              f"- 本次测试中与上轮梯度实验重合 {protocol['prior_gradient_test_overlap']} 条；扩大评测不等于新的盲测。", ""]
    output = root / args.output
    output.write_text("\n".join(lines))
    write_json(output.parent / "evidence/semantic-scale-v1.json", {"protocol": protocol, "initial": initial, "arms": arms})
    print(output)


if __name__ == "__main__":
    main()
