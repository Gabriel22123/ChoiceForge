"""Validate and report the common-start task-and-label stratification study."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from decision_model.core import digest, load_rows, metrics, write_json
from semantic_scale_report import paired_diagnostics
from task_balance_plan import make_orders


SEEDS = (42, 43)
MODES = ("mixed", "stratified")
ARMS = tuple(f"seed{seed}-{mode}" for seed in SEEDS for mode in MODES)
DIAGNOSTIC_ORDER = ("common-start", *ARMS)


def read(path):
    return json.loads(path.read_text())


def recalculate(rows, predictions):
    by_id = {point["id"]: point for point in predictions}
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Prediction IDs differ")
    labels, probabilities = [], []
    for row in rows:
        point = by_id[row["id"]]
        choices = [choice["id"] for choice in row["request"]["choices"]]
        if point["label"] != row["label"] or set(point["probabilities"]) != set(choices):
            raise ValueError("Prediction contract differs")
        labels.append(choices.index(row["label"]))
        probabilities.append([point["probabilities"][choice] for choice in choices])
    return metrics(labels, probabilities)


def compare_metrics(calculated, recorded):
    for key in ("n", "correct", "accuracy", "nll", "brier", "ece_10"):
        if abs(calculated[key] - recorded[key]) > 1e-9:
            raise ValueError("Recorded aggregate differs: " + key)


def choice_counts(rows, predictions):
    by_id = {row["id"]: row for row in rows}
    output = {}
    for source in sorted({row["source"] for row in rows}):
        truth, predicted = Counter(), Counter()
        for point in predictions:
            row = by_id[point["id"]]
            if row["source"] != source:
                continue
            truth[row["label"]] += 1
            predicted[max(point["probabilities"], key=point["probabilities"].get)] += 1
        output[source] = {"truth": dict(truth), "predicted": dict(predicted)}
    return output


def percentile(values, fraction):
    ordered = sorted(values); position = (len(ordered) - 1) * fraction
    lower = int(position); upper = min(lower + 1, len(ordered) - 1); weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def validated(root, study):
    protocol = read(study / "protocol.json")
    status = read(study / "status.json")
    if status != {"state": "complete", "models": list(DIAGNOSTIC_ORDER)}:
        raise ValueError("Study is not complete")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen protocol input changed: " + name)
    for name, expected in protocol["source_files"].items():
        if digest((root / name).read_bytes()) == expected:
            continue
        if name == "src/decision_model/__init__.py":
            snapshot = study / "seed42-mixed" / "source" / "__init__.py"
            frozen_lines = snapshot.read_text().splitlines()
            live_lines = (root / name).read_text().splitlines()
            if (digest(snapshot.read_bytes()) == expected and len(frozen_lines) == len(live_lines) and
                    all(a == b or (a.startswith("__version__ = ") and b.startswith("__version__ = "))
                        for a, b in zip(frozen_lines, live_lines))):
                continue
        raise ValueError("Frozen study source changed: " + name)
    start = root / protocol["initial_checkpoint"]
    if digest((start / "decision.safetensors").read_bytes()) != protocol["initial_weight_sha256"]:
        raise ValueError("Common start changed")
    rows = load_rows(study / "cases.jsonl")
    diagnostic_rows = load_rows(study / "diagnostic-cases.jsonl")
    if digest((study / "cases.jsonl").read_bytes()) != protocol["dataset_sha256"]:
        raise ValueError("Study rows changed")
    if digest((study / "diagnostic-cases.jsonl").read_bytes()) != protocol["diagnostic_dataset_sha256"]:
        raise ValueError("Diagnostic rows changed")
    test_rows = [row for row in rows if row["split"] == "test"]
    finals = read(study / "trained-checkpoints.json")
    initial = read(study / "initial.json")
    if initial["weight_sha256"] != protocol["initial_weight_sha256"]:
        raise ValueError("Initial evaluation checkpoint differs")
    compare_metrics(recalculate(test_rows, initial["predictions"]),
                    initial["evaluation"]["test"]["temperature_scaled"])

    arms = {}; initial_hash = None
    for arm in ARMS:
        directory = root / finals[arm]["path"]
        if digest((directory / "decision.safetensors").read_bytes()) != finals[arm]["weights_sha256"]:
            raise ValueError("Final weights changed: " + arm)
        config = read(study / f"{arm}.json"); run = read(directory / "run.json")
        evaluation = read(directory / "evaluation.json")
        predictions = read(directory / "test-predictions.json")
        if run["selected_ids"] != protocol["selected_ids"] or run["updates"] != 128:
            raise ValueError("Budget or selection differs: " + arm)
        if run["warm_start_weights_sha256"] != protocol["initial_weight_sha256"]:
            raise ValueError("Warm start differs: " + arm)
        if initial_hash is None:
            initial_hash = run["initial_trainable_sha256"]
        if run["initial_trainable_sha256"] != initial_hash:
            raise ValueError("Initial trainable parameters differ")
        expected_order = make_orders(rows, arm.split("-", 1)[1], int(arm[4:6]))[0]
        if config["epoch_row_orders"] != [expected_order]:
            raise ValueError("Frozen order differs: " + arm)
        executed = [identity for line in (directory / "training-plan.jsonl").read_text().splitlines()
                    for identity in json.loads(line)["rows"]]
        if executed != expected_order:
            raise ValueError("Executed row order differs: " + arm)
        compare_metrics(recalculate(test_rows, predictions), evaluation["test"]["temperature_scaled"])
        norms = [json.loads(line)["pre_clip_grad_norm"]
                 for line in (directory / "optimizer-steps.jsonl").read_text().splitlines()]
        if len(norms) != 128:
            raise ValueError("Optimizer-step count differs")
        arms[arm] = {
            "raw": evaluation["test"]["raw"],
            "predictions": predictions,
            "training": {"curve": read(directory / "curve.json"),
                         "gradient_clipping": run["gradient_clipping"],
                         "pre_clip_p95": percentile(norms, .95),
                         "training_seconds": run["training_seconds"],
                         "encoder_work": run["encoder_work"]},
        }

    manifest = read(study / "diagnostic-manifest.json")
    if manifest["order"] != list(DIAGNOSTIC_ORDER) or manifest["dataset_sha256"] != protocol["diagnostic_dataset_sha256"]:
        raise ValueError("Diagnostic manifest differs")
    diagnostic = {}
    for name in DIAGNOSTIC_ORDER:
        value = read(study / "diagnostic-results" / f"{name}.json")
        if value["dataset_sha256"] != protocol["diagnostic_dataset_sha256"]:
            raise ValueError("Diagnostic dataset differs")
        for source in protocol.get("diagnostic_counts", {"boolq": 384, "winogrande-1.1": 384}):
            subset = [row for row in diagnostic_rows if row["source"] == source]
            predictions = [point for point in value["predictions"] if point["id"] in {row["id"] for row in subset}]
            compare_metrics(recalculate(subset, predictions), value["raw"]["by_source"][source])
        diagnostic[name] = {"raw": value["raw"], "predictions": value["predictions"],
                            "choice_counts": choice_counts(diagnostic_rows, value["predictions"])}

    paired = {}; diagnostic_paired = {}
    for seed in SEEDS:
        paired[str(seed)] = paired_diagnostics(
            test_rows, arms[f"seed{seed}-mixed"]["predictions"], arms[f"seed{seed}-stratified"]["predictions"])
        diagnostic_paired[str(seed)] = paired_diagnostics(
            diagnostic_rows, diagnostic[f"seed{seed}-mixed"]["predictions"],
            diagnostic[f"seed{seed}-stratified"]["predictions"])
    return protocol, arms, diagnostic, paired, diagnostic_paired, read(study / "gradient-probe.json")


def pct(value):
    return f"{100 * value:.1f}%"


def points(value):
    return f"{100 * value:+.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/task-balance-v2")
    parser.add_argument("--report", default="docs/TASK_BALANCE_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/task-balance-v2.json")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; study = root / args.study
    protocol, arms, diagnostic, paired, diagnostic_paired, gradient = validated(root, study)
    evidence = {
        "protocol": protocol,
        "arms": {name: {"raw": value["raw"], "training": value["training"]} for name, value in arms.items()},
        "paired_stratified_minus_mixed": paired,
        "diagnostic": {name: {"raw": value["raw"], "choice_counts": value["choice_counts"]}
                       for name, value in diagnostic.items()},
        "diagnostic_paired_stratified_minus_mixed": diagnostic_paired,
        "gradient_probe": gradient,
    }
    write_json(root / args.evidence, evidence)

    lines = [
        "# 固定任务配比能否减少多任务干扰？", "", "## 结论", "",
        "固定 `3 PAWS + 3 SNLI + 2 旧任务` 的更新配比，在两个种子上都提高了 SNLI 和 PAWS 硬准确率；",
        "SNLI 分别提高 +2.6、+17.7 个百分点，PAWS 分别提高 +1.0、+2.1 个百分点。",
        "但 PAWS 原始 NLL 在两个种子都略差，BoolQ/WinoGrande 的分层相对混合变化方向不一致，不能据此宣称跨任务迁移改善。", "",
        "分层采样也没有稳定梯度：seed42 最大裁剪前范数从 111.7 增至 8929.9，seed43 从 51.2 增至 604.4；四组 128 次更新全部触发裁剪。",
        "起点梯度探针显示 PAWS 与 SNLI 的余弦均为正，冲突主要来自旧任务复习。下一步应显式做按任务梯度归一化或冲突投影，而不是继续仅调整采样比例。", "",
        "## 固定测试集：分层减去随机混合", "",
        "| 任务 | seed42 混合 | seed42 分层 | 差值 | seed43 混合 | seed43 分层 | 差值 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    sources = (("paws-wiki", "PAWS"), ("snli", "SNLI"), ("clinc", "CLINC"),
               ("json-schema", "Schema"), ("bandit", "代码策略"), ("goemotions", "情绪"))
    for source, title in sources:
        values = []
        for seed in SEEDS:
            before = arms[f"seed{seed}-mixed"]["raw"]["by_source"][source]["accuracy"]
            after = arms[f"seed{seed}-stratified"]["raw"]["by_source"][source]["accuracy"]
            values.extend((pct(before), pct(after), points(after - before)))
        lines.append(f"| {title} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | {values[5]} |")
    lines += ["", "PAWS 与 SNLI 的硬准确率在两个种子同向提高。旧任务并非全面改善：代码策略分别下降 10、30 个百分点，但它只有 10 条测试样本；Schema 和 CLINC 同向提高，情绪持平或小降。", "",
              "| 任务 / 种子 | 分层−混合修正 | 引入错误 | 配对组重采样 95% 区间 | 原始 NLL 变化 |", "|---|---:|---:|---:|---:|"]
    for source, title in (("paws-wiki", "PAWS"), ("snli", "SNLI")):
        for seed in SEEDS:
            item = paired[str(seed)][source]; lo, hi = item["paired_group_bootstrap_95_percentile"]
            nll = (arms[f"seed{seed}-stratified"]["raw"]["by_source"][source]["nll"] -
                   arms[f"seed{seed}-mixed"]["raw"]["by_source"][source]["nll"])
            lines.append(f"| {title} seed{seed} | {item['fixed']} | {item['regressed']} | [{points(lo)}, {points(hi)}] | {nll:+.4f} |")

    lines += ["", "## 已观察任务诊断", "",
              "| 任务 / 种子 | 混合 | 分层 | 分层−混合 | 修正 / 引入错误 | 配对区间 |", "|---|---:|---:|---:|---:|---:|"]
    for source, title in (("boolq", "BoolQ"), ("winogrande-1.1", "WinoGrande")):
        for seed in SEEDS:
            before = diagnostic[f"seed{seed}-mixed"]["raw"]["by_source"][source]
            after = diagnostic[f"seed{seed}-stratified"]["raw"]["by_source"][source]
            item = diagnostic_paired[str(seed)][source]; lo, hi = item["paired_group_bootstrap_95_percentile"]
            lines.append(f"| {title} seed{seed} | {before['correct']}/{before['n']}（{pct(before['accuracy'])}） | {after['correct']}/{after['n']}（{pct(after['accuracy'])}） | {points(item['accuracy_change'])} | {item['fixed']} / {item['regressed']} | [{points(lo)}, {points(hi)}] |")
    lines += ["", "BoolQ 的候选预测数量：", "", "| 模型 | yes | no |", "|---|---:|---:|"]
    for name in DIAGNOSTIC_ORDER:
        counts = diagnostic[name]["choice_counts"]["boolq"]["predicted"]
        lines.append(f"| {name} | {counts.get('yes', 0)} | {counts.get('no', 0)} |")

    lines += ["", "## 梯度与优化", "", "起点的八组无 dropout 梯度探针：", "",
              "| 任务对 | 平均余弦 | 负余弦次数 | 最小值 | 最大值 |", "|---|---:|---:|---:|---:|"]
    for key, value in gradient["aggregate"].items():
        lines.append(f"| {key} | {value['mean']:+.3f} | {value['negative']}/{value['n']} | {value['min']:+.3f} | {value['max']:+.3f} |")
    lines += ["", "| 种子 / 方案 | 最终训练风险 | 平均裁剪前范数 | P95 | 最大值 | 裁剪更新 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        training = arms[arm]["training"]; clipping = training["gradient_clipping"]
        lines.append(f"| {arm} | {training['curve'][-1]['mean_training_loss']:.4f} | {clipping['mean_pre_clip_norm']:.2f} | {training['pre_clip_p95']:.2f} | {clipping['max_pre_clip_norm']:.2f} | {clipping['clipped_updates']}/{clipping['updates']} |")

    lines += ["", "## 协议边界", "",
              "- 四组从同一个权重起点出发，使用完全相同的 1024 条训练行、一次曝光、128 次更新和优化器配置。",
              "- 随机混合组有 29 种更新级任务配比；分层组每次固定 3/3/2，并固定 SNLI 1/1/1 标签。",
              "- 相同 seed 不代表逐样本随机性完全相同：任务候选数、token 形状和行顺序会改变 RNG 消耗、padding 工作与 dropout 掩码归属。",
              "- BoolQ 与 WinoGrande 已在上一轮观察，本轮只作为未训练诊断，不能称为新的盲测；其标签未进入训练、校准或终点选择。",
              "- 两个种子和单一共同起点不足以给出通用结论；公共底座预训练污染仍无法排除。", "",
              "机器可读证据：[evidence/task-balance-v2.json](evidence/task-balance-v2.json)。", ""]
    (root / args.report).write_text("\n".join(lines))
    print(json.dumps({"snli_deltas": [paired[str(seed)]["snli"]["accuracy_change"] for seed in SEEDS],
                      "paws_deltas": [paired[str(seed)]["paws-wiki"]["accuracy_change"] for seed in SEEDS],
                      "boolq_deltas": [diagnostic_paired[str(seed)]["boolq"]["accuracy_change"] for seed in SEEDS]}), flush=True)


if __name__ == "__main__":
    main()
