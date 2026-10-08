"""Validate and report the two-seed PAWS task-expansion study."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from decision_model.core import digest, load_rows, metrics, write_json
from semantic_scale_report import paired_diagnostics


SEEDS = (42, 43)
BLIND_ORDER = ("seed42-start", "seed42-paws", "seed43-start", "seed43-paws")


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


def validated(root, study):
    status, protocol = read(study / "status.json"), read(study / "protocol.json")
    if status != {"state": "complete", "models": list(BLIND_ORDER)}:
        raise ValueError("Study is not complete")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen protocol artifact changed: " + name)
    for name, expected in protocol["source_files"].items():
        if digest((root / name).read_bytes()) == expected:
            continue
        # Packaging advances the public version after the frozen run. Validate the
        # byte-exact training snapshot, then allow only that one version assignment
        # to differ in the live package used to generate the release report.
        if name == "src/decision_model/__init__.py":
            snapshot = study / "seed-42-paws" / "source" / "__init__.py"
            frozen_lines = snapshot.read_text().splitlines()
            live_lines = (root / name).read_text().splitlines()
            if (digest(snapshot.read_bytes()) == expected and len(frozen_lines) == len(live_lines) and
                    all(a == b or (a.startswith("__version__ = ") and b.startswith("__version__ = "))
                        for a, b in zip(frozen_lines, live_lines))):
                continue
        raise ValueError("Frozen source changed: " + name)

    rows = load_rows(study / "cases.jsonl")
    blind_rows = load_rows(study / "blind-cases.jsonl")
    if digest((study / "cases.jsonl").read_bytes()) != protocol["study_dataset_sha256"]:
        raise ValueError("Study dataset changed")
    if digest((study / "blind-cases.jsonl").read_bytes()) != protocol["blind_dataset_sha256"]:
        raise ValueError("Blind dataset changed")
    test_rows = [row for row in rows if row["split"] == "test"]
    finals = read(study / "trained-checkpoints.json")
    arms, blind = {}, {}
    for seed in SEEDS:
        reference = read(study / f"reference-seed-{seed}.json")
        final_dir = root / finals[str(seed)]["path"]
        run, evaluation = read(final_dir / "run.json"), read(final_dir / "evaluation.json")
        if run["updates"] != 128 or run["selected_ids"] != protocol["selected_ids"]:
            raise ValueError("Training budget differs")
        if run["warm_start_weights_sha256"] != protocol["starts"][str(seed)]["weights_sha256"]:
            raise ValueError("Warm start differs")
        if digest((final_dir / "decision.safetensors").read_bytes()) != finals[str(seed)]["weights_sha256"]:
            raise ValueError("Final weights changed")
        initial_predictions = reference["predictions"]
        final_predictions = read(final_dir / "test-predictions.json")
        # Stored per-case probabilities use each run's validation-fitted temperature.
        # Positive temperature leaves hard predictions unchanged, so they remain valid
        # for the paired accuracy analysis below.
        for recorded, predictions in ((reference["evaluation"]["test"]["temperature_scaled"], initial_predictions),
                                      (evaluation["test"]["temperature_scaled"], final_predictions)):
            calculated = recalculate(test_rows, predictions)
            for key in ("n", "correct", "accuracy", "nll", "brier", "ece_10"):
                if abs(calculated[key] - recorded[key]) > 1e-9:
                    raise ValueError("Regular evaluation aggregate differs")
        arms[str(seed)] = {
            "reference": reference["evaluation"]["test"]["raw"],
            "final": evaluation["test"]["raw"],
            "paired": paired_diagnostics(test_rows, initial_predictions, final_predictions),
            "training": {"updates": run["updates"], "training_seconds": run["training_seconds"],
                         "gradient_clipping": run["gradient_clipping"], "encoder_work": run["encoder_work"]},
        }
    manifest = read(study / "blind-evaluation-manifest.json")
    if manifest["order"] != list(BLIND_ORDER) or manifest["blind_dataset_sha256"] != protocol["blind_dataset_sha256"]:
        raise ValueError("Blind manifest differs")
    for name in BLIND_ORDER:
        value = read(study / "blind-results" / f"{name}.json")
        record = manifest["checkpoints"][name]
        if (value["weights_sha256"] != record["weights_sha256"] or
                value["model_sha256"] != record["model_sha256"] or
                value["dataset_sha256"] != protocol["blind_dataset_sha256"]):
            raise ValueError("Blind checkpoint identity differs")
        for source in protocol["blind_counts"]:
            subset = [row for row in blind_rows if row["source"] == source]
            predictions = [point for point in value["predictions"] if point["id"] in {row["id"] for row in subset}]
            calculated = recalculate(subset, predictions)
            recorded = value["raw"]["by_source"][source]
            for key in ("n", "correct", "accuracy", "nll", "brier", "ece_10"):
                if abs(calculated[key] - recorded[key]) > 1e-9:
                    raise ValueError("Blind aggregate differs")
        blind[name] = {"raw": value["raw"], "predictions": value["predictions"],
                       "choice_counts": choice_counts(blind_rows, value["predictions"])}
    blind_paired = {}
    for seed in SEEDS:
        blind_paired[str(seed)] = paired_diagnostics(
            blind_rows, blind[f"seed{seed}-start"]["predictions"], blind[f"seed{seed}-paws"]["predictions"])
    return protocol, rows, blind_rows, arms, blind, blind_paired


def pct(value):
    return f"{100 * value:.1f}%"


def points(value):
    return f"{100 * value:+.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/task-expansion-v1")
    parser.add_argument("--report", default="docs/TASK_EXPANSION_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/task-expansion-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; study = root / args.study
    protocol, rows, blind_rows, arms, blind, blind_paired = validated(root, study)
    evidence = {"protocol": protocol, "arms": arms,
                "blind": {name: {"raw": value["raw"], "choice_counts": value["choice_counts"]}
                          for name, value in blind.items()},
                "blind_paired": blind_paired}
    write_json(root / args.evidence, evidence)

    paws_deltas = [arms[str(seed)]["paired"]["paws-wiki"]["accuracy_change"] for seed in SEEDS]
    boolq_deltas = [blind_paired[str(seed)]["boolq"]["accuracy_change"] for seed in SEEDS]
    wino_deltas = [blind_paired[str(seed)]["winogrande-1.1"]["accuracy_change"] for seed in SEEDS]
    lines = [
        "# 公开任务扩展：PAWS 训练能否带来跨任务迁移？", "", "## 结论", "",
        f"PAWS-Wiki 测试准确率在 seed42 提高 {points(paws_deltas[0])} 个百分点，在 seed43 提高 {points(paws_deltas[1])} 个百分点。",
        "新增任务在一个种子中学得明显，另一个种子只略有变化，训练稳定性仍不足。", "",
        f"新的冻结 BoolQ 任务变化为 {points(boolq_deltas[0])} 和 {points(boolq_deltas[1])} 个百分点，描述性平均 {points(mean(boolq_deltas))}；",
        f"WinoGrande 变化为 {points(wino_deltas[0])} 和 {points(wino_deltas[1])} 个百分点，描述性平均 {points(mean(wino_deltas))}。",
        "结果不支持 PAWS 训练已经形成稳定的跨任务迁移。seed42 的 BoolQ 起点把 384 条全部预测为 yes，准确率恰好等于 yes 标签占比；",
        "终点只产生 28 个 no 预测。seed43 则从 375 个 yes 大幅转向 260 个 no，导致总分下降。这里主要观察到候选偏好移动，而不是可复现的阅读理解提升。", "",
        "## 常规测试：学会了什么、遗忘了什么", "",
        "| 任务 | seed42 起点 | seed42 PAWS | 变化 | seed43 起点 | seed43 PAWS | 变化 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    sources = [("paws-wiki", "PAWS-Wiki"), ("snli", "SNLI"), ("clinc", "CLINC"),
               ("json-schema", "Schema"), ("bandit", "代码策略"), ("goemotions", "情绪")]
    for source, title in sources:
        values = []
        for seed in SEEDS:
            before = arms[str(seed)]["reference"]["by_source"][source]["accuracy"]
            after = arms[str(seed)]["final"]["by_source"][source]["accuracy"]
            values.extend((pct(before), pct(after), points(after - before)))
        lines.append(f"| {title} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | {values[5]} |")
    lines += ["", "PAWS 在两个种子都提高，但收益幅度没有稳定复现；SNLI 在两个种子分别下降 4.2 和 8.3 个百分点。Schema 点估计提高，其余旧任务有增有减。",
              "本实验没有同预算的非 PAWS 续训对照，因此不能把所有变化严格归因于数据内容；它测量的是整套继续训练方案相对各自起点的净变化。", "",
              "## 冻结任务族", "", "| 任务 / 种子 | 起点 | PAWS 终点 | 变化 | 修正 / 引入错误 | 配对组重采样 95% 区间 |", "|---|---:|---:|---:|---:|---:|"]
    for source, title in (("boolq", "BoolQ"), ("winogrande-1.1", "WinoGrande")):
        for seed in SEEDS:
            before = blind[f"seed{seed}-start"]["raw"]["by_source"][source]
            after = blind[f"seed{seed}-paws"]["raw"]["by_source"][source]
            paired = blind_paired[str(seed)][source]
            lo, hi = paired["paired_group_bootstrap_95_percentile"]
            lines.append(f"| {title} seed{seed} | {before['correct']}/{before['n']}（{pct(before['accuracy'])}） | {after['correct']}/{after['n']}（{pct(after['accuracy'])}） | {points(paired['accuracy_change'])} | {paired['fixed']} / {paired['regressed']} | [{points(lo)}, {points(hi)}] |")
    lines += ["", "BoolQ 冻结子集有 237 条 yes、147 条 no。各模型的硬预测数量如下：", "",
              "| 模型 | yes | no |", "|---|---:|---:|"]
    for name in BLIND_ORDER:
        counts = blind[name]["choice_counts"]["boolq"]["predicted"]
        lines.append(f"| {name} | {counts.get('yes', 0)} | {counts.get('no', 0)} |")
    lines += ["", "## 概率质量", "", "| 任务 / 种子 | 起点 NLL | PAWS 终点 NLL | 变化 |", "|---|---:|---:|---:|"]
    for source, title in (("paws-wiki", "PAWS"),):
        for seed in SEEDS:
            before = arms[str(seed)]["reference"]["by_source"][source]["nll"]
            after = arms[str(seed)]["final"]["by_source"][source]["nll"]
            lines.append(f"| {title} seed{seed} | {before:.4f} | {after:.4f} | {after-before:+.4f} |")
    for source, title in (("boolq", "BoolQ"), ("winogrande-1.1", "WinoGrande")):
        for seed in SEEDS:
            before = blind[f"seed{seed}-start"]["raw"]["by_source"][source]["nll"]
            after = blind[f"seed{seed}-paws"]["raw"]["by_source"][source]["nll"]
            lines.append(f"| {title} seed{seed} | {before:.4f} | {after:.4f} | {after-before:+.4f} |")
    lines += ["", "## 协议与边界", "",
              "- 每个种子从各自固定的标签均衡关系模型出发，训练 768 条 PAWS-Wiki 和 256 条旧任务复习样本；一轮、128 次更新、固定最终终点。",
              "- BoolQ 按 `SHA-256(question, passage)` 从可编码样本中选择 384 条，答案不参与选择；两个训练终点都完成后才统一推理。",
              "- WinoGrande 使用上轮完全相同的 384 条冻结集合，未进入训练；它已在上轮被观察，因此本轮新的分数盲任务是 BoolQ。",
              "- 两个种子不足以估计随机种子总体分布；公共底座预训练污染无法排除。所有 128 次更新在两个种子中都触发梯度裁剪。",
              "- PAWS 官方许可允许自由使用并建议署名；BoolQ 官方为 CC BY-SA 3.0。项目代码许可不覆盖这些数据。", "",
              f"机器可读证据：[evidence/task-expansion-v1.json](evidence/task-expansion-v1.json)。", ""]
    (root / args.report).write_text("\n".join(lines))
    print(json.dumps({"paws_deltas": paws_deltas, "boolq_deltas": boolq_deltas,
                      "winogrande_deltas": wino_deltas}), flush=True)


if __name__ == "__main__":
    main()
