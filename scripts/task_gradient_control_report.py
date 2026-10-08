"""Validate and report the matched task-gradient control study."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

from decision_model.core import digest, load_rows, metrics, write_json
try:
    from semantic_scale_report import paired_diagnostics
    from task_gradient_control import METHODS
except ModuleNotFoundError:  # Allows unit tests to import this file as scripts.*.
    from scripts.semantic_scale_report import paired_diagnostics
    from scripts.task_gradient_control import METHODS


SEEDS = (42, 43)
ARMS = tuple(f"seed{seed}-{method}" for seed in SEEDS for method in METHODS)


def read(path):
    return json.loads(Path(path).read_text())


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position); upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def prediction_probabilities(rows, predictions, temperature=None):
    by_id = {point["id"]: point for point in predictions}
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Prediction IDs differ")
    labels, probabilities = [], []
    for row in rows:
        point = by_id[row["id"]]
        choices = [choice["id"] for choice in row["request"]["choices"]]
        if point["label"] != row["label"] or set(point["probabilities"]) != set(choices):
            raise ValueError("Prediction label or candidate contract differs")
        values = [point["probabilities"][choice] for choice in choices]
        if temperature is not None:
            # Saved predictions are temperature-scaled.  Recover the raw
            # softmax distribution: p_raw is proportional to p_scaled ** T.
            logs = [temperature * math.log(max(value, 1e-45)) for value in values]
            offset = max(logs)
            values = [math.exp(value - offset) for value in logs]
            total = sum(values); values = [value / total for value in values]
        labels.append(choices.index(row["label"]))
        probabilities.append(values)
    return labels, probabilities


def recalculate(rows, predictions, temperature=None):
    labels, probabilities = prediction_probabilities(rows, predictions, temperature)
    result = metrics(labels, probabilities)
    result["by_source"] = {}
    for source in sorted({row["source"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["source"] == source]
        result["by_source"][source] = metrics(
            [labels[index] for index in indices], [probabilities[index] for index in indices])
    return result


def compare_metrics(calculated, recorded, tolerance=2e-6):
    for key in ("n", "correct", "accuracy", "nll", "brier", "ece_10"):
        if abs(calculated[key] - recorded[key]) > tolerance:
            raise ValueError("Recorded metric differs: " + key)
    if set(calculated.get("by_source", {})) != set(recorded.get("by_source", {})):
        raise ValueError("Recorded source metrics differ")
    for source in calculated.get("by_source", {}):
        compare_metrics(calculated["by_source"][source], recorded["by_source"][source], tolerance)


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


def finite_weights(path):
    import torch
    from safetensors.torch import load_file
    state = load_file(str(path))
    if not state or any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Checkpoint contains missing or non-finite trainable weights")
    return sorted(state)


def validate_live_source(root, study, name, expected):
    live = root / name
    if digest(live.read_bytes()) == expected:
        return
    if name == "src/decision_model/__init__.py":
        snapshot = study / "seed42-raw/source/src__decision_model____init__.py"
        frozen_lines = snapshot.read_text().splitlines()
        live_lines = live.read_text().splitlines()
        if (digest(snapshot.read_bytes()) == expected and len(frozen_lines) == len(live_lines) and
                all(left == right or (left.startswith("__version__ = ") and right.startswith("__version__ = "))
                    for left, right in zip(frozen_lines, live_lines))):
            return
    raise ValueError("Frozen study source changed: " + name)


def validated(root, study):
    protocol = read(study / "protocol.json")
    if protocol["arms"] != list(ARMS) or protocol["methods"] != list(METHODS):
        raise ValueError("Unexpected study arms")
    if read(study / "status.json") != {"state": "complete", "models": list(ARMS)}:
        raise ValueError("Study is incomplete")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen protocol input changed: " + name)
    for name, expected in protocol["source_files"].items():
        validate_live_source(root, study, name, expected)
    start = root / protocol["initial_checkpoint"]
    if (digest((start / "decision.safetensors").read_bytes()) != protocol["initial_weight_sha256"] or
            digest((start / "model.json").read_bytes()) != protocol["initial_model_sha256"]):
        raise ValueError("Common start changed")

    rows = load_rows(study / "cases.jsonl")
    diagnostic_rows = load_rows(study / "diagnostic-cases.jsonl")
    if digest((study / "cases.jsonl").read_bytes()) != protocol["dataset_sha256"]:
        raise ValueError("Study rows changed")
    if digest((study / "diagnostic-cases.jsonl").read_bytes()) != protocol["diagnostic_dataset_sha256"]:
        raise ValueError("Diagnostic rows changed")
    test_rows = [row for row in rows if row["split"] == "test"]
    finals = read(study / "trained-checkpoints.json")
    if set(finals) != set(ARMS):
        raise ValueError("Checkpoint manifest differs")

    arms, parameter_names, initial_hash = {}, None, None
    first_records = {}
    for arm in ARMS:
        seed_text, method = arm.split("-", 1)
        seed = int(seed_text.removeprefix("seed"))
        directory = root / finals[arm]["path"]
        if (digest((directory / "decision.safetensors").read_bytes()) != finals[arm]["weights_sha256"] or
                digest((directory / "model.json").read_bytes()) != finals[arm]["model_sha256"]):
            raise ValueError("Final checkpoint changed: " + arm)
        names = finite_weights(directory / "decision.safetensors")
        if parameter_names is None:
            parameter_names = names
        if names != parameter_names:
            raise ValueError("Checkpoint parameter sets differ")
        config = read(study / f"seed{seed}.json")
        run = read(directory / "run.json")
        if (read(directory / "model.json") != config or
                run["config_sha256"] != protocol["config_sha256"][str(seed)]):
            raise ValueError("Config differs: " + arm)
        if (run["method"] != method or run["seed"] != seed or run["engineering_smoke"] or
                run["updates"] != 128 or run["budget"]["task_forwards_per_update"] != 3 or
                run["selected_ids"] != protocol["selected_ids"] or
                run["warm_start_weights_sha256"] != protocol["initial_weight_sha256"]):
            raise ValueError("Training protocol differs: " + arm)
        if initial_hash is None:
            initial_hash = run["initial_trainable_sha256"]
        if run["initial_trainable_sha256"] != initial_hash:
            raise ValueError("Initial trainable parameters differ")
        for filename, expected in read(directory / "checksums.json").items():
            if digest((directory / filename).read_bytes()) != expected:
                raise ValueError("Checkpoint file checksum differs: " + arm)

        plan_raw = (directory / "training-plan.jsonl").read_bytes()
        if digest(plan_raw) != run["training_plan_sha256"]:
            raise ValueError("Training plan hash differs: " + arm)
        plans = [json.loads(line) for line in plan_raw.splitlines()]
        if len(plans) != 128:
            raise ValueError("Training plan length differs")
        exposed = []
        for index, plan in enumerate(plans, 1):
            if plan["update"] != index or {name: len(plan["tasks"][name]) for name in plan["tasks"]} != {
                    "paws": 3, "snli": 3, "replay": 2}:
                raise ValueError("Task update composition differs")
            exposed.extend(identity for name in ("paws", "snli", "replay") for identity in plan["tasks"][name])
        if Counter(exposed) != Counter(protocol["selected_ids"]["train"]):
            raise ValueError("Training exposure differs")

        steps = [json.loads(line) for line in (directory / "optimizer-steps.jsonl").read_text().splitlines()]
        if len(steps) != 128 or any(step["update"] != index for index, step in enumerate(steps, 1)):
            raise ValueError("Optimizer steps differ")
        for step in steps:
            numbers = [step["weighted_risk"], step["weighted_surrogate"],
                       step["combined_pre_clip_norm"], step["combined_diagnostic_norm"],
                       *step["raw_task_norms"].values(), *step["raw_task_cosines"].values()]
            if any(not math.isfinite(value) for value in numbers):
                raise ValueError("Non-finite optimizer diagnostic")
            if not math.isclose(step["combined_pre_clip_norm"], step["combined_diagnostic_norm"],
                                rel_tol=2e-5, abs_tol=2e-5):
                raise ValueError("Gradient norm implementations disagree")
            expected_kind = {"raw": "identity", "median_cap": "within_update_median_upper_cap",
                             "symmetric_project": "deterministic_symmetric_pairwise_projection"}[method]
            if step["transformation"]["kind"] != expected_kind:
                raise ValueError("Gradient transform differs")
            if method == "median_cap" and (step["transformation"]["amplified_tasks"] or
                    any(not 0 <= scale <= 1 for scale in step["transformation"]["scales"].values())):
                raise ValueError("Median cap amplified a task")
        first_records[arm] = steps[0]

        evaluation = read(directory / "evaluation.json")
        predictions = read(directory / "test-predictions.json")
        temperature = read(directory / "calibration.json")["temperature"]
        compare_metrics(recalculate(test_rows, predictions), evaluation["test"]["temperature_scaled"])
        compare_metrics(recalculate(test_rows, predictions, temperature), evaluation["test"]["raw"])
        norms = [step["combined_pre_clip_norm"] for step in steps]
        task_norms = {name: [step["raw_task_norms"][name] for step in steps]
                      for name in ("paws", "snli", "replay")}
        arms[arm] = {
            "raw": evaluation["test"]["raw"], "predictions": predictions,
            "training": {
                "training_seconds": run["training_seconds"],
                "gradient_clipping": run["gradient_clipping"],
                "combined_norm": {"mean": mean(norms), "p95": percentile(norms, .95),
                                  "max": max(norms)},
                "raw_task_norm": {name: {"mean": mean(values), "p95": percentile(values, .95),
                                          "max": max(values)} for name, values in task_norms.items()},
                "conflict_updates": sum(bool(step["transformation"].get("conflicting_pairs", []))
                                        for step in steps),
                "conflict_pairs": sum(len(step["transformation"].get("conflicting_pairs", []))
                                      for step in steps),
                "cap_updates": sum(any(scale < 1 for scale in step["transformation"].get("scales", {}).values())
                                   for step in steps),
                "minimum_cap_scale": min((scale for step in steps for scale in
                                           step["transformation"].get("scales", {}).values()), default=None),
            },
        }

    for seed in SEEDS:
        plans = [(root / finals[f"seed{seed}-{method}"]["path"] / "training-plan.jsonl").read_bytes()
                 for method in METHODS]
        if not plans[0] == plans[1] == plans[2]:
            raise ValueError("Candidate/row plans differ within seed")
        reference = first_records[f"seed{seed}-raw"]
        for method in METHODS[1:]:
            value = first_records[f"seed{seed}-{method}"]
            for key in ("task_risk", "task_surrogate", "raw_task_norms", "raw_task_cosines"):
                if value[key] != reference[key]:
                    raise ValueError("Common-start first-step diagnostic differs within seed")

    manifest = read(study / "diagnostic-manifest.json")
    if manifest["order"] != list(ARMS) or manifest["dataset_sha256"] != protocol["diagnostic_dataset_sha256"]:
        raise ValueError("Diagnostic manifest differs")
    diagnostic = {}
    for arm in ARMS:
        value = read(study / "diagnostic-results" / f"{arm}.json")
        if (value["dataset_sha256"] != protocol["diagnostic_dataset_sha256"] or
                value["weights_sha256"] != finals[arm]["weights_sha256"]):
            raise ValueError("Diagnostic identity differs: " + arm)
        compare_metrics(recalculate(diagnostic_rows, value["predictions"]), value["raw"])
        diagnostic[arm] = {"raw": value["raw"], "predictions": value["predictions"],
                           "choice_counts": choice_counts(diagnostic_rows, value["predictions"])}

    paired, diagnostic_paired = {}, {}
    for seed in SEEDS:
        raw = arms[f"seed{seed}-raw"]["predictions"]
        raw_diagnostic = diagnostic[f"seed{seed}-raw"]["predictions"]
        for method in METHODS[1:]:
            key = f"seed{seed}-{method}_minus_raw"
            paired[key] = paired_diagnostics(test_rows, raw, arms[f"seed{seed}-{method}"]["predictions"])
            diagnostic_paired[key] = paired_diagnostics(
                diagnostic_rows, raw_diagnostic, diagnostic[f"seed{seed}-{method}"]["predictions"])
    return protocol, arms, diagnostic, paired, diagnostic_paired


def pct(value):
    return f"{100 * value:.1f}%"


def points(value):
    return f"{100 * value:+.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/task-gradient-control-v1")
    parser.add_argument("--report", default="docs/TASK_GRADIENT_CONTROL_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/task-gradient-control-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    protocol, arms, diagnostic, paired, diagnostic_paired = validated(root, study)

    evidence = {
        "protocol": protocol,
        "arms": {name: {"raw": value["raw"], "training": value["training"]}
                 for name, value in arms.items()},
        "paired_method_minus_raw": paired,
        "diagnostic": {name: {"raw": value["raw"], "choice_counts": value["choice_counts"]}
                       for name, value in diagnostic.items()},
        "diagnostic_paired_method_minus_raw": diagnostic_paired,
    }
    write_json(root / args.evidence, evidence)

    lines = [
        "# 按任务控制梯度，能否稳定多任务训练？", "", "## 结论", "",
        "**中位削峰显著收紧梯度尾部，但当前强度会损伤 PAWS；单独冲突投影改善了 SNLI，却没有解决幅度爆点。两者都未满足预注册的默认方案标准。**", "",
        "中位削峰把 seed42/43 的合并梯度 P95 从 57.53/32.74 降到 29.67/29.25，最大值从 142.28/802.78 降到 70.86/46.20。",
        "但 PAWS 分别下降 3.6、0.5 个百分点；SNLI 分别提高 1.6、0.5 个百分点，因此未达到“两个已训任务在两个种子都保持或改善”的成功条件。", "",
        "对称冲突投影让 SNLI 在两个种子提高 1.6、1.0 个百分点，但 PAWS 为 +0.5、-3.1；其最大合并范数仍为 154.59、371.32。",
        "实验证明尺度失衡与方向冲突是两个独立问题。下一轮应先把削峰阈值放宽到中位数的 2 倍，再比较是否在削峰后追加投影。", "",
        "## 主任务结果：相对原始任务梯度加权", "",
        "| 种子 / 方法 | PAWS | 相对原始 | PAWS NLL | SNLI | 相对原始 | SNLI NLL | 总准确率 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"raw": "原始", "median_cap": "中位削峰", "symmetric_project": "冲突投影"}
    for seed in SEEDS:
        baseline = arms[f"seed{seed}-raw"]["raw"]["by_source"]
        for method in METHODS:
            value = arms[f"seed{seed}-{method}"]["raw"]
            paws = value["by_source"]["paws-wiki"]
            snli = value["by_source"]["snli"]
            lines.append(
                f"| seed{seed} {labels[method]} | {pct(paws['accuracy'])} | "
                f"{points(paws['accuracy'] - baseline['paws-wiki']['accuracy'])} | {paws['nll']:.4f} | "
                f"{pct(snli['accuracy'])} | {points(snli['accuracy'] - baseline['snli']['accuracy'])} | "
                f"{snli['nll']:.4f} | {pct(value['accuracy'])} |"
            )

    lines += ["", "配对变化（方法减原始）：", "",
              "| 方法 / 任务 / 种子 | 修正 | 引入错误 | 准确率变化 | 组重采样 95% 区间 |", "|---|---:|---:|---:|---:|"]
    for method in METHODS[1:]:
        for source, title in (("paws-wiki", "PAWS"), ("snli", "SNLI")):
            for seed in SEEDS:
                value = paired[f"seed{seed}-{method}_minus_raw"][source]
                lo, hi = value["paired_group_bootstrap_95_percentile"]
                lines.append(f"| {labels[method]} / {title} / seed{seed} | {value['fixed']} | {value['regressed']} | {points(value['accuracy_change'])} | [{points(lo)}, {points(hi)}] |")

    lines += ["", "## 梯度尾部", "",
              "| 种子 / 方法 | 平均合并范数 | P95 | 最大值 | 全局裁剪 | 削峰更新 / 冲突更新 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        training = arms[arm]["training"]
        gradient = training["combined_norm"]
        clipping = training["gradient_clipping"]
        intervention = (f"{training['cap_updates']}/128" if "median_cap" in arm else
                        f"{training['conflict_updates']}/128（{training['conflict_pairs']} 对）" if "symmetric_project" in arm else "—")
        lines.append(f"| {arm} | {gradient['mean']:.2f} | {gradient['p95']:.2f} | {gradient['max']:.2f} | {clipping['clipped_updates']}/{clipping['updates']} | {intervention} |")

    lines += ["", "中位削峰不会放大小梯度。两个种子都出现回放梯度接近零、其他任务仍有明显信号的更新；如果做普通单位范数归一化，会把这些近零方向放大多个数量级。", "",
              "## 旧任务与未训练任务", "",
              "| 种子 / 方法 | CLINC | Schema | 代码策略（n=10） | 情绪（未训练） |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        by = arms[arm]["raw"]["by_source"]
        lines.append(f"| {arm} | {pct(by['clinc']['accuracy'])} | {pct(by['json-schema']['accuracy'])} | {pct(by['bandit']['accuracy'])} | {pct(by['goemotions']['accuracy'])} |")

    lines += ["", "## 已观察迁移诊断", "",
              "BoolQ 与 WinoGrande 在此前实验中已经查看过，本轮只用于检查候选先验漂移，不能称为新盲测。", "",
              "| 种子 / 方法 | BoolQ | 相对原始 | WinoGrande | 相对原始 |", "|---|---:|---:|---:|---:|"]
    for seed in SEEDS:
        baseline = diagnostic[f"seed{seed}-raw"]["raw"]["by_source"]
        for method in METHODS:
            by = diagnostic[f"seed{seed}-{method}"]["raw"]["by_source"]
            lines.append(f"| seed{seed} {labels[method]} | {pct(by['boolq']['accuracy'])} | {points(by['boolq']['accuracy']-baseline['boolq']['accuracy'])} | {pct(by['winogrande-1.1']['accuracy'])} | {points(by['winogrande-1.1']['accuracy']-baseline['winogrande-1.1']['accuracy'])} |")
    lines += ["", "BoolQ 候选预测数量：", "", "| 模型 | yes | no |", "|---|---:|---:|"]
    for arm in ARMS:
        counts = diagnostic[arm]["choice_counts"]["boolq"]["predicted"]
        lines.append(f"| {arm} | {counts.get('yes', 0)} | {counts.get('no', 0)} |")
    lines += ["", "两个种子间的 `yes` 数量差从原始方案的 68 降到中位削峰的 13、冲突投影的 7，说明两种控制都可能降低候选先验对随机种子的敏感度；但准确率变化方向不一致，这仍只是已观察集上的诊断。"]

    lines += ["", "## 协议与边界", "",
              "- 六组从相同公共权重起点出发；每组使用同样的 1024 条训练行、一次曝光和 128 次更新。",
              "- 每次更新分别计算 3 条 PAWS、3 条 SNLI 和 2 条旧任务回放的梯度；三种方法都执行相同的三次任务级前向/反向传播。",
              "- 同一种子内，训练行顺序、候选排列、任务前向顺序和 dropout 随机种子完全一致；训练计划文件逐字节相同。",
              "- 原始方案、中位削峰和投影均按 3/8、3/8、2/8 合并任务梯度，之后统一做范数 1 的全局裁剪。",
              "- 对称投影是受 PCGrad 启发的确定性、无顺序版本，不声称等价复现随机 PCGrad。",
              "- 两个种子不足以估计完整分布；公共底座的预训练污染无法排除。BoolQ/WinoGrande 不是新盲测。", "",
              "机器可读证据：[evidence/task-gradient-control-v1.json](evidence/task-gradient-control-v1.json)。", ""]
    (root / args.report).write_text("\n".join(lines))
    print(json.dumps({
        "median_cap_p95": [arms[f"seed{seed}-median_cap"]["training"]["combined_norm"]["p95"] for seed in SEEDS],
        "median_cap_paws_delta": [paired[f"seed{seed}-median_cap_minus_raw"]["paws-wiki"]["accuracy_change"] for seed in SEEDS],
        "projection_snli_delta": [paired[f"seed{seed}-symmetric_project_minus_raw"]["snli"]["accuracy_change"] for seed in SEEDS],
    }), flush=True)


if __name__ == "__main__":
    main()
