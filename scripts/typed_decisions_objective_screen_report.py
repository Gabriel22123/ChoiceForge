"""Verify and report the frozen-feature 4-fold x 3-seed objective screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_model.core import digest, write_json


ARMS = ("H-hard", "S-soft", "T-typed-rps", "R-rlcd-loo")
CONTRASTS = (("soft_minus_hard", "H-hard", "S-soft"),
             ("typed_rps_minus_soft", "S-soft", "T-typed-rps"),
             ("rlcd_loo_minus_typed_rps", "T-typed-rps", "R-rlcd-loo"))


def mean(values):
    return sum(values) / len(values)


def advancement(deltas):
    workflows = sorted({row["fold"] for row in deltas})
    workflow_ce = {workflow: mean([row["heldout_soft_ce"] for row in deltas
                                   if row["fold"] == workflow]) for workflow in workflows}
    values = {
        "heldout_ce_wins": sum(row["heldout_soft_ce"] < 0 for row in deltas),
        "endpoints": len(deltas),
        "mean_heldout_soft_ce_delta": mean([row["heldout_soft_ce"] for row in deltas]),
        "mean_seen_soft_ce_delta": mean([row["seen_soft_ce"] for row in deltas]),
        "mean_heldout_soft_brier_delta": mean([row["heldout_soft_brier"] for row in deltas]),
        "workflow_mean_heldout_soft_ce_delta": workflow_ce,
    }
    checks = {
        "heldout_ce_wins_at_least_9_of_12": values["heldout_ce_wins"] >= 9,
        "mean_heldout_ce_improves": values["mean_heldout_soft_ce_delta"] < 0,
        "no_workflow_mean_ce_regresses_over_0_01": max(workflow_ce.values()) <= .01,
        "mean_seen_ce_regression_at_most_0_005": values["mean_seen_soft_ce_delta"] <= .005,
        "mean_heldout_brier_nonregression": values["mean_heldout_soft_brier_delta"] <= 0,
    }
    values["checks"] = checks; values["advances_to_lora_validation"] = all(checks.values())
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/typed-decisions-objective-screen-v1")
    parser.add_argument("--output", default="docs/evidence/typed-decisions-objective-screen-v1.json")
    parser.add_argument("--report", default="docs/TYPED_DECISIONS_OBJECTIVE_SCREEN.zh-CN.md")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    study = root / args.study
    status = json.loads((study / "status.json").read_text())
    if status.get("state") != "complete" or status.get("endpoints") != 48:
        raise ValueError("Objective screen is not complete")
    protocol = json.loads((study / "protocol.json").read_text())
    if tuple(protocol["arms"]) != ARMS:
        raise ValueError("Objective-screen arms or order differ from the registered report")
    lock = json.loads((study / "test-lock.json").read_text())
    if lock.get("status") != "all_heads_and_validation_locked_before_test" or len(lock["endpoints"]) != 48:
        raise ValueError("Test lock is incomplete")
    if lock["protocol_sha256"] != digest((study / "protocol.json").read_bytes()):
        raise ValueError("Protocol changed after test lock")
    prepared = root / protocol["prepared_data"]
    if digest((prepared / "manifest.json").read_bytes()) != protocol["prepared_manifest_sha256"]:
        raise ValueError("Prepared-data manifest changed after the study")
    for fold, metadata in protocol["folds"].items():
        fold_root = prepared / ("heldout-" + fold)
        if (digest((fold_root / "fit.jsonl").read_bytes()) != metadata["fit_sha256"] or
                digest((fold_root / "test.jsonl").read_bytes()) != metadata["test_sha256"]):
            raise ValueError("Prepared fit or sealed test changed: " + fold)
    if digest((root / protocol["feature_cache"] / "manifest.json").read_bytes()) != protocol["feature_manifest_sha256"]:
        raise ValueError("Feature manifest changed after the study")
    if digest((root / protocol["feature_parity_evidence"]).read_bytes()) != protocol["feature_parity_evidence_sha256"]:
        raise ValueError("Feature parity evidence changed after the study")
    results = {}; rows = []
    for fold in sorted(protocol["folds"]):
        for seed in protocol["seeds"]:
            for arm in ARMS:
                endpoint = study / f"heldout-{fold}" / f"seed{seed}" / arm
                run = json.loads((endpoint / "run.json").read_text())
                locked = lock["endpoints"][f"{fold}/seed{seed}/{arm}"]
                if (run["head_weights_sha256"] != locked["head_weights_sha256"] or
                        digest((endpoint / "run.json").read_bytes()) != locked["run_sha256"] or
                        digest((endpoint / "checksums.json").read_bytes()) != locked["fit_checksums_sha256"] or
                        run["training_plan_sha256"] != locked["training_plan_sha256"]):
                    raise ValueError("Locked head differs")
                for filename, expected in json.loads((endpoint / "checksums.json").read_text()).items():
                    if digest((endpoint / filename).read_bytes()) != expected:
                        raise ValueError("Fit artifact changed")
                for filename, expected in json.loads((endpoint / "test-checksums.json").read_text()).items():
                    if digest((endpoint / filename).read_bytes()) != expected:
                        raise ValueError("Test artifact changed")
                test = json.loads((endpoint / "test-evaluation.json").read_text())["raw"]
                validation = json.loads((endpoint / "validation-evaluation.json").read_text())["raw"]
                item = {
                    "fold": fold, "seed": seed, "arm": arm,
                    "validation": {key: validation[key] for key in ("n", "argmax_agreement", "soft_cross_entropy", "soft_brier")},
                    "seen": {key: test["by_evaluation_regime"]["seen_task_new_examples"][key]
                             for key in ("n", "argmax_agreement", "soft_cross_entropy", "soft_brier")},
                    "heldout": {key: test["by_evaluation_regime"]["held_out_task"][key]
                                for key in ("n", "argmax_agreement", "soft_cross_entropy", "soft_brier")},
                    "weights_sha256": run["head_weights_sha256"], "objective": run["objective"],
                    "gradient_norm": run["gradient_norm"], "updates": run["updates"],
                }
                results[f"{fold}/seed{seed}/{arm}"] = item; rows.append(item)
    contrast_results = {}
    for name, control, treatment in CONTRASTS:
        deltas = []
        for fold in sorted(protocol["folds"]):
            for seed in protocol["seeds"]:
                left = results[f"{fold}/seed{seed}/{control}"]
                right = results[f"{fold}/seed{seed}/{treatment}"]
                deltas.append({
                    "fold": fold, "seed": seed,
                    "heldout_soft_ce": right["heldout"]["soft_cross_entropy"] - left["heldout"]["soft_cross_entropy"],
                    "heldout_soft_brier": right["heldout"]["soft_brier"] - left["heldout"]["soft_brier"],
                    "heldout_argmax": right["heldout"]["argmax_agreement"] - left["heldout"]["argmax_agreement"],
                    "seen_soft_ce": right["seen"]["soft_cross_entropy"] - left["seen"]["soft_cross_entropy"],
                })
        contrast_results[name] = {"control": control, "treatment": treatment,
                                  "paired_deltas": deltas, "advancement": advancement(deltas)}
    gradient_summary = {}
    arm_summary = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        arm_summary[arm] = {
            "mean_heldout_soft_ce": mean([row["heldout"]["soft_cross_entropy"] for row in arm_rows]),
            "mean_seen_soft_ce": mean([row["seen"]["soft_cross_entropy"] for row in arm_rows]),
            "mean_heldout_argmax_agreement": mean([
                row["heldout"]["argmax_agreement"] for row in arm_rows]),
        }
        gradient_summary[arm] = {
            "endpoint_mean_pre_clip_norm": mean([
                row["gradient_norm"]["mean"] for row in arm_rows]),
            "endpoint_max_pre_clip_norm": max(
                row["gradient_norm"]["max"] for row in arm_rows),
            "clipped_updates": sum(
                row["gradient_norm"]["clipped_updates"] for row in arm_rows),
            "updates": sum(row["updates"] for row in arm_rows),
        }
    evidence = {
        "status": "complete_representation_frozen_screen",
        "report_script_sha256": digest(Path(__file__).read_bytes()),
        "study_protocol_sha256": digest((study / "protocol.json").read_bytes()),
        "test_lock_sha256": digest((study / "test-lock.json").read_bytes()),
        "protocol": protocol, "test_lock": lock, "results": results,
        "contrasts": contrast_results, "arm_summary": arm_summary,
        "gradient_summary": gradient_summary,
        "boundary": "This isolates objectives on fixed public-base features. Advancement means eligible for end-to-end LoRA validation, not a final model win.",
    }
    lines = ["# Typed Decisions：固定 2B 表示的四折三种子目标筛选", "",
             "本实验固定 MiniCPM5-2B-Base 候选表示，只训练共享候选头，从而先隔离监督目标与梯度估计。",
             "通过只代表有资格进入端到端 LoRA 验证，不代表最终模型收益。", "",
             "## 预注册晋级结果", "",
             "| 对照 | 未见 CE 胜出端点 | 未见 CE 均值差 | 已见 CE 均值差 | 未见 Brier 均值差 | 晋级 LoRA |",
             "|---|---:|---:|---:|---:|---|"]
    for name, _, _ in CONTRASTS:
        value = contrast_results[name]["advancement"]
        lines.append(f"| {name} | {value['heldout_ce_wins']}/12 | {value['mean_heldout_soft_ce_delta']:+.5f} | {value['mean_seen_soft_ce_delta']:+.5f} | {value['mean_heldout_soft_brier_delta']:+.5f} | {'是' if value['advances_to_lora_validation'] else '否'} |")
    lines += ["", "## 绝对结果", "",
              "| 方法 | 未见 soft CE | 已见 soft CE | 未见 argmax agreement |",
              "|---|---:|---:|---:|"]
    for arm in ARMS:
        value = arm_summary[arm]
        lines.append(f"| {arm} | {value['mean_heldout_soft_ce']:.5f} | {value['mean_seen_soft_ce']:.5f} | {value['mean_heldout_argmax_agreement']:.1%} |")
    lines += ["", "## 判定细节", ""]
    for name, _, _ in CONTRASTS:
        value = contrast_results[name]["advancement"]
        failed = [check for check, passed in value["checks"].items() if not passed]
        workflow = value["workflow_mean_heldout_soft_ce_delta"]
        lines.append(f"- `{name}`：未通过 `{', '.join(failed)}`；各工作流未见 CE 均值差为 " +
                     "、".join(f"{key} {number:+.5f}" for key, number in workflow.items()) + "。")
    lines += ["", "三种增量方法均未通过预注册规则，因此没有方法进入完整 LoRA。",
              "本结果支持在可直接求导的教师概率监督上继续使用直接 proper loss；它不能否定 RLCD 在真实、不可微结果反馈上的价值。", "",
              "## 训练梯度诊断", "",
              "| 方法 | 端点平均裁剪前范数 | 端点最大范数 | 被裁剪更新 | 总更新 |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        value = gradient_summary[arm]
        lines.append(f"| {arm} | {value['endpoint_mean_pre_clip_norm']:.5f} | {value['endpoint_max_pre_clip_norm']:.5f} | {value['clipped_updates']} | {value['updates']} |")
    lines += ["", "## 边界", "",
              "四个工作流各自整族留出，种子固定为 42/43/44。48 个头和验证结果全部校验并锁定后，程序才首次解析密封 test 文件。",
              "特征缓存来自关闭 LoRA 适配器的固定公开底座；因此本实验不能发现目标函数与表示学习之间的相互作用。",
              "下一阶段只允许预注册通过的增量方法与其直接对照进入完整 LoRA；若没有增量方法通过，则保留更简单的直接监督基线。", "",
              f"机器证据：`{args.output}`。"]
    write_json(root / args.output, evidence)
    (root / args.report).write_text("\n".join(lines) + "\n")
    print(json.dumps({name: value["advancement"] for name, value in contrast_results.items()}, indent=2))


if __name__ == "__main__":
    main()
