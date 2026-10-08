#!/usr/bin/env python3
"""Train a residual expert first, then route the frozen expert semantically."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.feature_screen import RoutedResidualHead, parameter_sha256
from decision_model.training_objective import (TrainingObjective, context_advantage_loss,
                                               distillation_kl_loss)


_HELPER_PATH = Path(__file__).with_name("run_qasc_routed_residual_screen.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("qasc_routed_residual_helper", _HELPER_PATH)
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC); _HELPER_SPEC.loader.exec_module(_HELPER)
evaluate, load_features, make_plan = _HELPER.evaluate, _HELPER.load_features, _HELPER.make_plan
pad_batch, score, teacher_tensor = _HELPER.pad_batch, _HELPER.score, _HELPER.teacher_tensor


def read(path):
    return json.loads(Path(path).read_text())


def component_hash(module, component):
    value = getattr(module, component)
    return parameter_sha256(value)


def batch_loss(module, rows, keys, orders, features, manifest, config, stage):
    full, null, parent, parent_null, mask, targets = pad_batch(
        rows, keys, features, manifest, orders)
    if stage == "expert":
        correction = module.residual(full).squeeze(-1).masked_fill(~mask, 0.0)
        null_correction = module.residual(null).squeeze(-1).masked_fill(~mask, 0.0)
    elif stage == "router":
        correction, _ = module(full, mask)
        null_correction, _ = module(null, mask)
    else:
        raise ValueError("Unknown staged-router phase")
    logits, null_logits = parent + correction, parent_null + null_correction
    counts = [len(order) for order in orders]
    weights = torch.tensor([row["training_weight"] for row in rows])
    objective = TrainingObjective({"gradient_estimator": "clean",
                                   "brier_weight": config["brier_weight"]})
    loss, _ = objective(logits, targets, counts, [row.get("decision_type") for row in rows],
                        candidate_orders=orders, sample_weights=weights)
    full_teacher, full_active = teacher_tensor(
        rows, orders, logits.shape[1], "teacher_probabilities")
    null_teacher, null_active = teacher_tensor(
        rows, orders, logits.shape[1], "teacher_null_probabilities")
    loss = loss + config["distillation_weight"] * distillation_kl_loss(
        logits, full_teacher, counts, weights, full_active)
    loss = loss + config["null_distillation_weight"] * distillation_kl_loss(
        null_logits, null_teacher, counts, weights, null_active)
    loss = loss + config["context_advantage_weight"] * context_advantage_loss(
        logits, null_logits, targets, counts, config["context_advantage_gap"], weights)
    return loss


def train_stage(module, stage, rows, features, manifest, config, output):
    if stage == "expert":
        seed = config["expert_seed"]
        module.residual.requires_grad_(True); module.router.requires_grad_(False)
        parameters = list(module.residual.parameters())
    else:
        seed = config["router_seed"]
        module.residual.requires_grad_(False); module.router.requires_grad_(True)
        parameters = list(module.router.parameters())
    plan = make_plan(rows, seed)
    plan_path = output / f"{stage}-training-plan.jsonl"
    plan_path.write_text("".join(canonical(value) + "\n" for value in plan))
    by_id = {row["id"]: row for row in rows}
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], eps=1e-6)
    batches = [plan[index:index + config["batch_size"]]
               for index in range(0, len(plan), config["batch_size"])]
    norms, curve, updates = [], [], 0
    started = time.monotonic()
    module.train()
    for group_start in range(0, len(batches), config["accumulation"]):
        group = batches[group_start:group_start + config["accumulation"]]
        optimizer.zero_grad(set_to_none=True)
        size, group_loss = sum(map(len, group)), 0.0
        for batch_plan in group:
            batch = [by_id[item["row_id"]] for item in batch_plan]
            keys = [f"cases:{row['id']}" for row in batch]
            orders = [item["candidate_order"] for item in batch_plan]
            loss = batch_loss(module, batch, keys, orders, features, manifest, config, stage)
            scaled = loss * (len(batch) / size)
            scaled.backward(); group_loss += float(scaled.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
        optimizer.step(); updates += 1
        norms.append(float(norm)); curve.append({"update": updates, "loss": group_loss})
    if updates != config["updates_per_stage"]:
        raise ValueError("Staged-router optimizer update count differs")
    return {
        "seed": seed, "updates": updates, "seconds": time.monotonic() - started,
        "plan_sha256": digest(plan_path.read_bytes()),
        "gradient_norm": {"mean": sum(norms) / len(norms), "maximum": max(norms),
                          "clipped_updates": sum(value > config["gradient_clip"] for value in norms)},
        "curve": curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qasc-staged-router-screen-v1.json")
    parser.add_argument("--output", default="runs/qasc-staged-router-screen-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh staged-router study directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Staged-router protocol was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen staged-router source changed: " + relative)
    cache = root / config["feature_cache"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_manifest_sha256"]:
        raise ValueError("Frozen routed feature manifest changed")
    datasets = {name: load_rows(root / path) for name, path in config["data"].items()
                if not name.endswith("_sha256")}
    for name, rows in datasets.items():
        path = root / config["data"][name]
        if digest(path.read_bytes()) != config["data"][name + "_sha256"]:
            raise ValueError("Frozen staged-router dataset changed: " + name)
    keys = [f"{name}:{row['id']}" for name, rows in datasets.items() for row in rows]
    manifest, features = load_features(cache, keys)
    training = config["training"]
    torch.manual_seed(training["expert_seed"])
    module = RoutedResidualHead.build(
        manifest["hidden_size"], training["residual_width"], training["router_width"],
        training["dropout"], training["initial_gate_probability"])
    output.mkdir(parents=True); write_json(output / "status.json", {"state": "expert"})
    train_rows = [row for row in datasets["cases"] if row["split"] == "train"]
    initial = {"all": parameter_sha256(module), "residual": component_hash(module, "residual"),
               "router": component_hash(module, "router")}
    initial_logits, _, _ = score(module, train_rows[:24], "cases", features, manifest)
    parent_logits = [torch.tensor(
        manifest["rows"][f"cases:{row['id']}"]["parent_logits"], dtype=torch.float32).tolist()
        for row in train_rows[:24]]
    initial_max = max(abs(left - right)
                      for actual, parent in zip(initial_logits, parent_logits)
                      for left, right in zip(actual, parent))
    if initial_max != 0.0:
        raise ValueError("Staged residual does not initialize as the exact parent")
    expert = train_stage(module, "expert", train_rows, features, manifest, training, output)
    after_expert = {"residual": component_hash(module, "residual"),
                    "router": component_hash(module, "router")}
    if after_expert["residual"] == initial["residual"] or after_expert["router"] != initial["router"]:
        raise ValueError("Expert stage parameter isolation failed")
    save_file({name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()},
              str(output / "expert-stage.safetensors"))
    write_json(output / "status.json", {"state": "router"})
    router = train_stage(module, "router", train_rows, features, manifest, training, output)
    final = {"residual": component_hash(module, "residual"),
             "router": component_hash(module, "router")}
    if final["residual"] != after_expert["residual"] or final["router"] == after_expert["router"]:
        raise ValueError("Router stage parameter isolation failed")
    weights = output / "staged-router.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()},
              str(weights))
    reloaded = RoutedResidualHead.build(
        manifest["hidden_size"], training["residual_width"], training["router_width"],
        training["dropout"], training["initial_gate_probability"])
    reloaded.load_state_dict(load_file(str(weights)))
    run = {"initial": initial, "initial_max_logit_difference": initial_max,
           "after_expert": after_expert, "final": final,
           "expert": expert, "router": router, "weights_sha256": digest(weights.read_bytes()),
           "final_parameter_sha256": parameter_sha256(reloaded)}
    write_json(output / "run.json", run)
    write_json(output / "status.json", {"state": "evaluation"})
    cases = datasets["cases"]
    groups = {
        "seen": ([row for row in cases if row["split"] == "test"], "cases"),
        "qasc_validation": ([row for row in cases if row["split"] == "validation" and
                              row["source"] == "qasc"], "cases"),
        "development": (datasets["development"], "development"),
        "qasc_development": (datasets["qasc_development"], "qasc_development"),
        "replay_train": ([row for row in train_rows if row["source"] != "qasc"], "cases"),
        "qasc_train": ([row for row in train_rows if row["source"] == "qasc"], "cases"),
    }
    evaluations = {name: evaluate(reloaded, rows, dataset, features, manifest)
                   for name, (rows, dataset) in groups.items()}
    write_json(output / "evaluation.json", evaluations)
    rules = config["screening_rule"]
    context_deltas = {
        source: value["mean"] - evaluations["development"]["parent_context_advantage"][
            "by_source"][source]["mean"]
        for source, value in evaluations["development"]["context_advantage"]["by_source"].items()
    }
    qasc = evaluations["qasc_validation"]
    checks = {
        "mechanism.initial_exact": run["initial_max_logit_difference"] == 0.0,
        "mechanism.expert_stage_isolated": (after_expert["residual"] != initial["residual"] and
                                             after_expert["router"] == initial["router"]),
        "mechanism.router_stage_isolated": (final["residual"] == after_expert["residual"] and
                                             final["router"] != after_expert["router"]),
        "mechanism.router_separates": (evaluations["qasc_train"]["gates"]["all"]["mean"] -
                                        evaluations["replay_train"]["gates"]["all"]["mean"] >=
                                        rules["qasc_vs_replay_gate_min_delta"]),
        "seen.accuracy": evaluations["seen"]["raw"]["accuracy"] -
            evaluations["seen"]["parent_raw"]["accuracy"] >= rules["seen_accuracy_vs_parent_min_delta"],
        "seen.nll": evaluations["seen"]["raw"]["nll"] -
            evaluations["seen"]["parent_raw"]["nll"] <= rules["seen_nll_vs_parent_max_delta"],
        "development.accuracy": evaluations["development"]["raw"]["accuracy"] -
            evaluations["development"]["parent_raw"]["accuracy"] >=
            rules["development_accuracy_vs_parent_min_delta"],
        "development.context_retention": min(context_deltas.values()) >=
            rules["development_each_context_advantage_vs_parent_min_delta"],
        "qasc.accuracy": qasc["raw"]["accuracy"] - qasc["parent_raw"]["accuracy"] >=
            rules["qasc_accuracy_vs_parent_min_delta"],
        "qasc.nll": qasc["raw"]["nll"] - qasc["parent_raw"]["nll"] <=
            rules["qasc_nll_vs_parent_max_delta"],
        "qasc.context": evaluations["qasc_development"]["context_advantage"]["mean"] -
            evaluations["qasc_development"]["parent_context_advantage"]["mean"] >=
            rules["qasc_context_advantage_vs_parent_min_delta"],
    }
    summary = {
        "seen": {"accuracy": evaluations["seen"]["raw"]["accuracy"],
                 "parent_accuracy": evaluations["seen"]["parent_raw"]["accuracy"],
                 "nll": evaluations["seen"]["raw"]["nll"],
                 "parent_nll": evaluations["seen"]["parent_raw"]["nll"]},
        "development": {"accuracy": evaluations["development"]["raw"]["accuracy"],
                        "parent_accuracy": evaluations["development"]["parent_raw"]["accuracy"]},
        "qasc": {"accuracy": qasc["raw"]["accuracy"],
                 "parent_accuracy": qasc["parent_raw"]["accuracy"],
                 "nll": qasc["raw"]["nll"], "parent_nll": qasc["parent_raw"]["nll"]},
        "gates": {name: evaluations[name]["gates"]["all"]["mean"]
                  for name in ("replay_train", "qasc_train", "seen", "development",
                              "qasc_validation")},
    }
    evidence = {"format_version": 1, "study": config["study"],
                "role": "consumed single-seed two-stage mechanism screen",
                "protocol_sha256": digest(config_path.read_bytes()), "run": run,
                "summary": summary, "context_delta_vs_parent": context_deltas,
                "checks": checks, "advances_to_live_integration_audit": all(checks.values()),
                "limitations": ["All evaluation tasks are consumed.",
                                "Single seed and frozen-feature mechanism screen.",
                                "Live checkpoint and output-contract audits remain required."]}
    write_json(root / "docs/evidence/qasc-staged-router-screen-v1.json", evidence)
    gate_delta = summary["gates"]["qasc_train"] - summary["gates"]["replay_train"]
    lines = ["# QASC 两阶段语义路由筛选", "",
             "先训练非零专家，再冻结专家训练语义路由。全部任务均已消费，本结果只用于机制选择。", "",
             "| 指标 | 父模型 | 两阶段路由 | 差值 |", "|---|---:|---:|---:|",
             f"| 已见准确率 | {summary['seen']['parent_accuracy']:.4f} | {summary['seen']['accuracy']:.4f} | {summary['seen']['accuracy']-summary['seen']['parent_accuracy']:+.4f} |",
             f"| 已见 NLL | {summary['seen']['parent_nll']:.4f} | {summary['seen']['nll']:.4f} | {summary['seen']['nll']-summary['seen']['parent_nll']:+.4f} |",
             f"| 开发准确率 | {summary['development']['parent_accuracy']:.4f} | {summary['development']['accuracy']:.4f} | {summary['development']['accuracy']-summary['development']['parent_accuracy']:+.4f} |",
             f"| QASC 准确率 | {summary['qasc']['parent_accuracy']:.4f} | {summary['qasc']['accuracy']:.4f} | {summary['qasc']['accuracy']-summary['qasc']['parent_accuracy']:+.4f} |",
             f"| QASC NLL | {summary['qasc']['parent_nll']:.4f} | {summary['qasc']['nll']:.4f} | {summary['qasc']['nll']-summary['qasc']['parent_nll']:+.4f} |",
             "", "## 路由", "",
             f"- 回放门均值：`{summary['gates']['replay_train']:.6f}`。",
             f"- QASC 门均值：`{summary['gates']['qasc_train']:.6f}`。",
             f"- 门差：`{gate_delta:+.6f}`。", "", "## 判定", "",
             ("全部门槛通过；允许进入实时集成审计。" if all(checks.values()) else
              "至少一个门槛失败；不进入实时集成。"), "",
             "逐项门槛：" + ", ".join(f"`{key}`={'PASS' if value else 'FAIL'}"
                                        for key, value in checks.items()), "",
             "机器证据：`docs/evidence/qasc-staged-router-screen-v1.json`", ""]
    (root / "docs/QASC_STAGED_ROUTER_SCREEN.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(json.dumps({"state": "complete", "advances": all(checks.values()),
                      "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
