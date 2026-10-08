"""Validate and report the frozen encoder/decoder comparison."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, median, quantiles

from decision_model.core import digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base
from decision_model.model import verify_local_base
try:
    from semantic_scale_report import paired_diagnostics
    from task_gradient_trust_report import compare_metrics, finite_weights, percentile, read, recalculate
except ModuleNotFoundError:
    from scripts.semantic_scale_report import paired_diagnostics
    from scripts.task_gradient_trust_report import compare_metrics, finite_weights, percentile, read, recalculate


SEEDS = (42, 43, 44)
ARCHITECTURES = ("encoder", "decoder")
ARMS = tuple(f"seed{seed}-{architecture}" for seed in SEEDS for architecture in ARCHITECTURES)
PRIMARY = ("paws-wiki", "snli")
GENERALIZATION = ("bandit", "clinc", "goemotions", "json-schema")
ARC_PRIMARY = ("arc-challenge", "arc-easy")


def validate_source(root, study, name, expected):
    live = root / name
    snapshot = study / "frozen-source" / name
    if digest(live.read_bytes()) == expected or (snapshot.is_file() and digest(snapshot.read_bytes()) == expected):
        return
    raise ValueError("Frozen architecture source changed: " + name)


def validated(root, study):
    protocol = read(study / "protocol.json")
    if (protocol["seeds"] != list(SEEDS) or protocol["architectures"] != list(ARCHITECTURES) or
            protocol["arms"] != list(ARMS)):
        raise ValueError("Unexpected architecture study arms")
    status = read(study / "status.json")
    if (status.get("state") != "complete" or status.get("models") != list(ARMS) or
            status.get("result_manifest_sha256") != digest((study / "result-manifest.json").read_bytes())):
        raise ValueError("Architecture study is incomplete")
    for name, expected in read(study / "result-manifest.json").items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or digest((study / relative).read_bytes()) != expected:
            raise ValueError("Architecture result changed: " + name)
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen architecture input changed: " + name)
    if read(study / "frozen-source-manifest.json") != {"source_files": protocol["source_files"]}:
        raise ValueError("Frozen architecture source manifest differs")
    for name, expected in protocol["source_files"].items():
        if digest((study / "frozen-source" / name).read_bytes()) != expected:
            raise ValueError("Frozen architecture source snapshot changed: " + name)
        validate_source(root, study, name, expected)
    for architecture, verifier in (("encoder", verify_local_base), ("decoder", verify_local_decoder_base)):
        base = protocol["bases"][architecture]
        lock = root / ("src/decision_model/base.lock.json" if architecture == "encoder" else
                       "src/decision_model/decoder_base.lock.json")
        if digest(lock.read_bytes()) != base["base_lock_sha256"]:
            raise ValueError("Architecture base lock changed: " + architecture)
        verifier(root / base["path"])

    rows = load_rows(study / "cases.jsonl")
    if digest((study / "cases.jsonl").read_bytes()) != protocol["dataset_sha256"]:
        raise ValueError("Architecture dataset changed")
    test_rows = [row for row in rows if row["split"] == "test"]
    finals = read(study / "trained-checkpoints.json")
    if set(finals) != set(ARMS):
        raise ValueError("Architecture checkpoint manifest differs")

    arms, parameter_names = {}, {}
    common_selected_ids = None
    for arm in ARMS:
        seed_text, architecture = arm.split("-", 1)
        seed = int(seed_text.removeprefix("seed"))
        directory = root / finals[arm]["path"]
        if (digest((directory / "decision.safetensors").read_bytes()) != finals[arm]["weights_sha256"] or
                digest((directory / "model.json").read_bytes()) != finals[arm]["model_sha256"] or
                digest((directory / "calibration.json").read_bytes()) != finals[arm]["calibration_sha256"] or
                digest((directory / "checksums.json").read_bytes()) != finals[arm]["checksums_sha256"] or
                digest((directory / "run.json").read_bytes()) != finals[arm]["run_sha256"] or
                digest((directory / "evaluation.json").read_bytes()) != finals[arm]["evaluation_sha256"] or
                digest((directory / "test-predictions.json").read_bytes()) != finals[arm]["test_predictions_sha256"] or
                digest((directory / "training-plan.jsonl").read_bytes()) != finals[arm]["training_plan_sha256"] or
                digest((directory / "optimizer-steps.jsonl").read_bytes()) != finals[arm]["optimizer_steps_sha256"]):
            raise ValueError("Architecture checkpoint changed: " + arm)
        names = finite_weights(directory / "decision.safetensors")
        parameter_names.setdefault(architecture, names)
        if names != parameter_names[architecture]:
            raise ValueError("Trainable parameter names differ within architecture")
        config = read(study / f"seed{seed}-{architecture}.json")
        run = read(directory / "run.json")
        if (read(directory / "model.json") != config or
                run["config_sha256"] != protocol["configs_sha256"][arm] or
                run["dataset_sha256"] != protocol["dataset_sha256"] or
                run["initialization"] != "public_pretrained_base_fresh_adapter_and_head" or
                run["seed"] != seed or run["updates"] != 128 or run["training_views"] != 1 or
                run["device"] != protocol["device"] or
                config["architecture"] != protocol["bases"][architecture]["architecture"] or
                config["model_id"] != protocol["bases"][architecture]["model_id"] or
                config["revision"] != protocol["bases"][architecture]["revision"] or
                run["over_token_budget_ids"] or
                {split: len(identities) for split, identities in run["selected_ids"].items()} !=
                {"train": 1024, "validation": 288, "test": 576}):
            raise ValueError("Architecture run protocol differs: " + arm)
        common_selected_ids = run["selected_ids"] if common_selected_ids is None else common_selected_ids
        if run["selected_ids"] != common_selected_ids:
            raise ValueError("Architecture endpoints selected different rows: " + arm)
        for filename, expected in read(directory / "checksums.json").items():
            if digest((directory / filename).read_bytes()) != expected:
                raise ValueError("Architecture checkpoint checksum differs: " + arm)
        plans = [json.loads(line) for line in (directory / "training-plan.jsonl").read_text().splitlines()]
        if len(plans) != 512 or sum(len(plan["rows"]) for plan in plans) != 1024:
            raise ValueError("Architecture training plan exposure differs: " + arm)
        if digest((directory / "training-plan.jsonl").read_bytes()) != run["training_plan_sha256"]:
            raise ValueError("Architecture training plan hash differs: " + arm)
        steps = [json.loads(line) for line in (directory / "optimizer-steps.jsonl").read_text().splitlines()]
        if len(steps) != 128 or any(step["update"] != index for index, step in enumerate(steps, 1)):
            raise ValueError("Architecture optimizer steps differ: " + arm)
        norms = [step["pre_clip_grad_norm"] for step in steps]
        if any(not math.isfinite(value) for value in norms):
            raise ValueError("Non-finite architecture gradient norm: " + arm)
        allocation = run["sampled_device_allocation"]
        samples = allocation["samples"]
        if (allocation["kind"] != "fixed_checkpoint_samples_not_exact_peak" or
                allocation["backend"] != protocol["device"] or len(samples) != 129 or
                samples[0]["stage"] != "model_loaded" or samples[0]["update"] is not None or
                any(sample["stage"] != "after_backward" or sample["update"] != index
                    for index, sample in enumerate(samples[1:], 1)) or
                any(not isinstance(sample["allocated_bytes"], int) or sample["allocated_bytes"] <= 0
                    for sample in samples) or
                allocation["maximum_sampled_bytes"] != max(sample["allocated_bytes"] for sample in samples)):
            raise ValueError("Architecture device allocation samples differ: " + arm)

        evaluation = read(directory / "evaluation.json")
        predictions = read(directory / "test-predictions.json")
        temperature = read(directory / "calibration.json")["temperature"]
        compare_metrics(recalculate(test_rows, predictions), evaluation["test"]["temperature_scaled"])
        compare_metrics(recalculate(test_rows, predictions, temperature), evaluation["test"]["raw"])
        audit = read(study / "audit-results" / f"{arm}.json")
        if audit["cases"] != 24 or audit["stable_all_orders"] > 24 or audit["correct_all_orders"] > 24:
            raise ValueError("Architecture audit counts differ: " + arm)
        if audit["max_reload_probability_difference"] > 2e-6 or audit["max_id_rename_probability_difference"] > 2e-6:
            raise ValueError("Architecture output contract differs: " + arm)
        benchmark = read(study / "benchmarks" / f"{arm}.json")
        timings = benchmark.get("timings", [])
        seconds = [point.get("seconds") for point in timings]
        timing_ids = Counter(point.get("id") for point in timings)
        rows_by_id = {row["id"]: row for row in test_rows}
        if (benchmark["checkpoint_weight_sha256"] != finals[arm]["weights_sha256"] or
                benchmark["dataset_sha256"] != protocol["dataset_sha256"] or
                benchmark["warmup_requests"] != 24 or benchmark["measurement_rounds"] != 3 or
                benchmark["summary"]["requests"] != 72 or len(timings) != 72 or
                len(timing_ids) != 24 or set(timing_ids.values()) != {3} or
                any(identity not in rows_by_id for identity in timing_ids) or
                any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                    for value in seconds) or
                any(point.get("repeat") not in (0, 1, 2) or
                    point.get("candidates") != len(rows_by_id[point["id"]]["request"]["choices"]) or
                    point.get("choice_id") not in {choice["id"] for choice in rows_by_id[point["id"]]["request"]["choices"]}
                    for point in timings)):
            raise ValueError("Architecture benchmark differs: " + arm)
        recalculated_summary = {"requests": 72, "median_seconds": median(seconds),
                                "p95_seconds": quantiles(seconds, n=100, method="inclusive")[94],
                                "mean_seconds": mean(seconds)}
        for key, expected in recalculated_summary.items():
            if not math.isclose(benchmark["summary"][key], expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Architecture benchmark summary differs: " + arm)
        arms[arm] = {
            "raw": evaluation["test"]["raw"], "predictions": predictions, "audit": audit,
            "benchmark": benchmark,
            "training": {
                "training_seconds": run["training_seconds"],
                "trainable_parameters": run["trainable_parameters"],
                "base_parameters": run["base_parameters"],
                "work": run["encoder_work"],
                "sampled_device_allocation": allocation,
                "gradient_norm": {"mean": mean(norms), "p95": percentile(norms, .95), "max": max(norms)},
                "clipped_updates": sum(step["clipped"] for step in steps),
            },
        }

    for seed in SEEDS:
        left = study / f"seed{seed}-encoder/training-plan.jsonl"
        right = study / f"seed{seed}-decoder/training-plan.jsonl"
        if left.read_bytes() != right.read_bytes():
            raise ValueError("Encoder and decoder plans differ within seed")
        if (read(study / f"seed{seed}-encoder/run.json")["selected_ids"] !=
                read(study / f"seed{seed}-decoder/run.json")["selected_ids"]):
            raise ValueError("Encoder and decoder selected rows differ within seed")

    blind = study / "arc-blind"
    blind_protocol = read(blind / "protocol.json")
    if (read(blind / "status.json") != {"state": "prepared"} or
            digest((blind / "cases.jsonl").read_bytes()) != blind_protocol["cases_sha256"] or
            blind_protocol["rows"] != 384 or not blind_protocol["selection_is_answer_blind"] or
            blind_protocol["exact_training_request_overlap"] != 0 or
            blind_protocol["dataset"] != read(study / "arc-source.json") or
            blind_protocol["dataset_source_record_sha256"] != digest((study / "arc-source.json").read_bytes()) or
            blind_protocol["preparation_script_sha256"] != protocol["source_files"]["scripts/prepare_arc_blind.py"] or
            blind_protocol["training_data_sha256"] != protocol["dataset_sha256"] or
            {name: value["selected_rows"] for name, value in blind_protocol["selection"].items()} !=
            {"ARC-Challenge": 192, "ARC-Easy": 192}):
        raise ValueError("ARC blind protocol differs")
    blind_rows = load_rows(blind / "cases.jsonl")
    if ({source: sum(row["source"] == source for row in blind_rows)
         for source in ("arc-challenge", "arc-easy")} != {"arc-challenge": 192, "arc-easy": 192} or
            {digest(row["request"]) for row in rows if row["split"] == "train"} &
            {digest(row["request"]) for row in blind_rows}):
        raise ValueError("ARC blind cases differ")
    arc = {}
    for arm in ARMS:
        value = read(study / "arc-results" / f"{arm}.json")
        directory = root / finals[arm]["path"]
        temperature = read(directory / "calibration.json")["temperature"]
        if (value["dataset_sha256"] != blind_protocol["cases_sha256"] or
                value["weights_sha256"] != finals[arm]["weights_sha256"] or value["split"] != "test"):
            raise ValueError("ARC result identity differs: " + arm)
        compare_metrics(recalculate(blind_rows, value["predictions"]), value["temperature_scaled"])
        compare_metrics(recalculate(blind_rows, value["predictions"], temperature), value["raw"])
        arc[arm] = {"raw": value["raw"], "predictions": value["predictions"]}

    paired, arc_paired = {}, {}
    for seed in SEEDS:
        key = f"seed{seed}-decoder_minus_encoder"
        paired[key] = paired_diagnostics(
            test_rows, arms[f"seed{seed}-encoder"]["predictions"], arms[f"seed{seed}-decoder"]["predictions"])
        arc_paired[key] = paired_diagnostics(
            blind_rows, arc[f"seed{seed}-encoder"]["predictions"], arc[f"seed{seed}-decoder"]["predictions"])
    return protocol, arms, blind_protocol, arc, paired, arc_paired


def pct(value):
    return f"{100 * value:.1f}%"


def pp(value):
    return f"{100 * value:+.1f}"


def decoder_default_checks(arms, arc):
    trained_accuracy_deltas, trained_nll_deltas = [], []
    generalization_accuracy_deltas, generalization_nll_deltas = [], []
    trained_combined_wins = 0
    arc_accuracy_deltas = []
    arc_combined_wins = 0
    for seed in SEEDS:
        encoder = arms[f"seed{seed}-encoder"]["raw"]["by_source"]
        decoder = arms[f"seed{seed}-decoder"]["raw"]["by_source"]
        trained = [decoder[source]["accuracy"] - encoder[source]["accuracy"] for source in PRIMARY]
        trained_accuracy_deltas.extend(trained)
        trained_nll_deltas.extend(decoder[source]["nll"] - encoder[source]["nll"] for source in PRIMARY)
        trained_combined_wins += mean(trained) > 0
        generalization_accuracy_deltas.extend(
            decoder[source]["accuracy"] - encoder[source]["accuracy"] for source in GENERALIZATION)
        generalization_nll_deltas.extend(
            decoder[source]["nll"] - encoder[source]["nll"] for source in GENERALIZATION)

        arc_encoder = arc[f"seed{seed}-encoder"]["raw"]["by_source"]
        arc_decoder = arc[f"seed{seed}-decoder"]["raw"]["by_source"]
        fresh = [arc_decoder[source]["accuracy"] - arc_encoder[source]["accuracy"] for source in ARC_PRIMARY]
        arc_accuracy_deltas.extend(fresh)
        arc_combined_wins += mean(fresh) > 0

    checks = {
        "trained_combined_accuracy_wins_at_least_two_seeds": trained_combined_wins >= 2,
        "no_trained_task_accuracy_loss_below_minus_1pp": min(trained_accuracy_deltas) >= -0.01,
        "trained_mean_nll_not_worse": mean(trained_nll_deltas) <= 0,
        "generalization_mean_accuracy_not_worse": mean(generalization_accuracy_deltas) >= 0,
        "generalization_mean_nll_not_worse": mean(generalization_nll_deltas) <= 0,
        "all_decoder_order_stability_complete": all(
            arms[f"seed{seed}-decoder"]["audit"]["stable_all_orders"] == 24 for seed in SEEDS),
        "arc_combined_accuracy_wins_at_least_two_seeds": arc_combined_wins >= 2,
        "arc_mean_accuracy_not_worse": mean(arc_accuracy_deltas) >= 0,
    }
    checks["eligible_as_default"] = all(checks.values())
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/architecture-comparison-v1")
    parser.add_argument("--report", default="docs/ENCODER_DECODER_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/architecture-comparison-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, arms, blind_protocol, arc, paired, arc_paired = validated(root, root / args.study)

    decoder_checks = decoder_default_checks(arms, arc)
    evidence = {
        "protocol": protocol,
        "arms": {name: {"raw": value["raw"], "training": value["training"],
                         "benchmark": {"summary": value["benchmark"]["summary"],
                                       "encoder_work": value["benchmark"]["encoder_work"]},
                         "audit": {key: value["audit"][key] for key in
                                   ("cases", "stable_all_orders", "correct_all_orders", "max_reload_probability_difference", "max_id_rename_probability_difference")}}
                 for name, value in arms.items()},
        "paired_decoder_minus_encoder": paired,
        "arc_protocol": blind_protocol,
        "arc": {name: value["raw"] for name, value in arc.items()},
        "arc_paired_decoder_minus_encoder": arc_paired,
        "decoder_default_rule": decoder_checks,
    }
    write_json(root / args.evidence, evidence)

    verdict = "满足" if decoder_checks["eligible_as_default"] else "不满足"
    lines = [
        "# 2B 级 encoder 与 decoder：同一候选决策接口的三种子比较", "", "## 结论", "",
        f"**decoder {verdict}预注册默认路线条件。**", "",
        "两条路线都不生成文本；模型只输出每个动态候选的标量分数，概率与固定 JSON 由程序构造。六个端点使用相同公开训练行、训练顺序、候选排列、更新数和适配层数。ARC 在六个端点和审计完成后才一次性启封，且全部端点都被评估。", "",
        "## 已训练任务", "",
        "| 种子 / 路线 | PAWS | SNLI | 总准确率 | PAWS NLL | SNLI NLL |", "|---|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for architecture in ARCHITECTURES:
            value = arms[f"seed{seed}-{architecture}"]["raw"]
            by = value["by_source"]
            lines.append(f"| seed{seed} {architecture} | {pct(by['paws-wiki']['accuracy'])} | {pct(by['snli']['accuracy'])} | {pct(value['accuracy'])} | {by['paws-wiki']['nll']:.4f} | {by['snli']['nll']:.4f} |")
    lines += ["", "## 保留任务与未见任务", "",
              "GoEmotions 在训练中未出现；其余三类是训练回放任务。这里单列，避免总体分数掩盖迁移或遗忘。", "",
              "| 种子 / 路线 | Bandit | CLINC | GoEmotions（未见） | JSON Schema |", "|---|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for architecture in ARCHITECTURES:
            by = arms[f"seed{seed}-{architecture}"]["raw"]["by_source"]
            lines.append(f"| seed{seed} {architecture} | {pct(by['bandit']['accuracy'])} | {pct(by['clinc']['accuracy'])} | {pct(by['goemotions']['accuracy'])} | {pct(by['json-schema']['accuracy'])} |")
    lines += ["", "## ARC 任务级盲测", "", "| 种子 / 路线 | ARC-Challenge | ARC-Easy | 总准确率 |", "|---|---:|---:|---:|"]
    for seed in SEEDS:
        for architecture in ARCHITECTURES:
            value = arc[f"seed{seed}-{architecture}"]["raw"]
            by = value["by_source"]
            lines.append(f"| seed{seed} {architecture} | {pct(by['arc-challenge']['accuracy'])} | {pct(by['arc-easy']['accuracy'])} | {pct(value['accuracy'])} |")
    lines += ["", "## 工程代价与输出稳定性", "", "| 种子 / 路线 | 可训练参数 | 训练秒数 | 采样设备占用 | 推理中位数 | 推理 P95 | 梯度 P95 | 换序稳定 | 全顺序正确 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for architecture in ARCHITECTURES:
            value = arms[f"seed{seed}-{architecture}"]
            training, audit = value["training"], value["audit"]
            latency = value["benchmark"]["summary"]
            memory_gib = training["sampled_device_allocation"]["maximum_sampled_bytes"] / 2**30
            lines.append(f"| seed{seed} {architecture} | {training['trainable_parameters']:,} | {training['training_seconds']:.1f} | {memory_gib:.2f} GiB | {1000*latency['median_seconds']:.1f} ms | {1000*latency['p95_seconds']:.1f} ms | {training['gradient_norm']['p95']:.2f} | {audit['stable_all_orders']}/24 | {audit['correct_all_orders']}/24 |")
    lines += ["", "## decoder 默认路线规则", ""]
    for name, passed in decoder_checks.items():
        lines.append(f"- {'通过' if passed else '未通过'}：`{name}`")
    lines += ["", "## 边界", "",
              "- 比较的是两个具体公开底座的端到端路线，不是注意力方向的纯消融。",
              "- tokenizer、预训练语料、隐藏宽度和总参数不同；相同训练行与更新数不代表 FLOPs 完全相同。",
              "- ARC 可能出现在公共底座的预训练语料中；本项目的封存协议只能排除本项目调参泄漏。",
              "- 三个随机种子仍不足以估计完整模型分布。",
              "- 机器可读证据：[evidence/architecture-comparison-v1.json](evidence/architecture-comparison-v1.json)。", ""]
    (root / args.report).write_text("\n".join(lines))
    print({"decoder_default_rule": decoder_checks})


if __name__ == "__main__":
    main()
