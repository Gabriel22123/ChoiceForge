#!/usr/bin/env python3
"""Counterfactual semantic routing with paired full/null gate consistency."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.feature_screen import RoutedResidualHead, parameter_sha256


_BASE_PATH = Path(__file__).with_name("run_qasc_counterfactual_router_screen.py")
_BASE_SPEC = importlib.util.spec_from_file_location("qasc_counterfactual_router_helper", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_BASE_SPEC); _BASE_SPEC.loader.exec_module(_BASE)
component_hash, derive_targets, evaluate = _BASE.component_hash, _BASE.derive_targets, _BASE.evaluate
load_features, make_plan, pad_batch = _BASE.load_features, _BASE.make_plan, _BASE.pad_batch
score, train_stage = _BASE.score, _BASE.train_stage


def read(path):
    return json.loads(Path(path).read_text())


def train_router(module, rows, targets, features, manifest, config, output):
    module.residual.requires_grad_(False); module.router.requires_grad_(True)
    parameters = list(module.router.parameters())
    plan = make_plan(rows, config["router_seed"])
    plan_path = output / "router-training-plan.jsonl"
    plan_path.write_text("".join(canonical(value) + "\n" for value in plan))
    by_id = {row["id"]: row for row in rows}
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], eps=1e-6)
    batches = [plan[index:index + config["batch_size"]]
               for index in range(0, len(plan), config["batch_size"])]
    curve, norms, updates = [], [], 0
    started = time.monotonic(); module.train()
    for group_start in range(0, len(batches), config["accumulation"]):
        group = batches[group_start:group_start + config["accumulation"]]
        optimizer.zero_grad(set_to_none=True)
        size, group_loss = sum(map(len, group)), 0.0
        for batch_plan in group:
            batch = [by_id[item["row_id"]] for item in batch_plan]
            keys = [f"cases:{row['id']}" for row in batch]
            orders = [item["candidate_order"] for item in batch_plan]
            full, null, _, _, mask, _ = pad_batch(batch, keys, features, manifest, orders)
            _, full_gate = module(full, mask); _, null_gate = module(null, mask)
            target = torch.tensor([targets[row["id"]] for row in batch])
            target_loss = 0.5 * ((full_gate - target).square().mean() +
                                 (null_gate - target).square().mean())
            consistency = (full_gate - null_gate).square().mean()
            loss = target_loss + config["full_null_gate_consistency_weight"] * consistency
            scaled = loss * (len(batch) / size)
            scaled.backward(); group_loss += float(scaled.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
        optimizer.step(); updates += 1
        norms.append(float(norm)); curve.append({"update": updates, "loss": group_loss})
    if updates != config["updates_per_stage"]:
        raise ValueError("Consistent router update count differs")
    return {"seed": config["router_seed"], "updates": updates,
            "seconds": time.monotonic() - started,
            "plan_sha256": digest(plan_path.read_bytes()),
            "full_null_gate_consistency_weight": config["full_null_gate_consistency_weight"],
            "gradient_norm": {"mean": sum(norms) / len(norms), "maximum": max(norms),
                              "clipped_updates": sum(value > config["gradient_clip"]
                                                     for value in norms)},
            "curve": curve}


def paired_gate_summary(module, rows, dataset, features, manifest):
    values, groups = [], defaultdict(list)
    module.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            keys = [f"{dataset}:{row['id']}" for row in batch]
            full, null, _, _, mask, _ = pad_batch(batch, keys, features, manifest)
            _, full_gate = module(full, mask); _, null_gate = module(null, mask)
            for row, full_value, null_value in zip(batch, full_gate.tolist(), null_gate.tolist()):
                item = {"full": full_value, "null": null_value,
                        "difference": full_value - null_value,
                        "absolute_difference": abs(full_value - null_value)}
                values.append(item); groups[row["source"]].append(item)
    aggregate = lambda rows: {"rows": len(rows),
                              "mean_full": sum(row["full"] for row in rows) / len(rows),
                              "mean_null": sum(row["null"] for row in rows) / len(rows),
                              "mean_difference": sum(row["difference"] for row in rows) / len(rows),
                              "mean_absolute_difference": sum(row["absolute_difference"] for row in rows) / len(rows)}
    return {"all": aggregate(values),
            "by_source": {source: aggregate(rows) for source, rows in sorted(groups.items())}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qasc-consistent-router-screen-v1.json")
    parser.add_argument("--output", default="runs/qasc-consistent-router-screen-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh consistent-router study directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Consistent-router protocol was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen consistent-router source changed: " + relative)
    cache = root / config["feature_cache"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_manifest_sha256"]:
        raise ValueError("Frozen routed feature manifest changed")
    datasets = {name: load_rows(root / path) for name, path in config["data"].items()
                if not name.endswith("_sha256")}
    for name in datasets:
        if digest((root / config["data"][name]).read_bytes()) != config["data"][name + "_sha256"]:
            raise ValueError("Frozen consistent-router dataset changed: " + name)
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
    initial_max = max(abs(left - right) for actual, parent in zip(initial_logits, parent_logits)
                      for left, right in zip(actual, parent))
    expert = train_stage(module, "expert", train_rows, features, manifest, training, output)
    after_expert = {"residual": component_hash(module, "residual"),
                    "router": component_hash(module, "router")}
    if (initial_max != 0 or after_expert["residual"] == initial["residual"] or
            after_expert["router"] != initial["router"]):
        raise ValueError("Consistent expert stage audit failed")
    targets, target_summary = derive_targets(module, train_rows, features, manifest, training, output)
    if len(targets) != 1024:
        raise ValueError("Consistent route target coverage differs")
    write_json(output / "target-summary.json", target_summary)
    router = train_router(module, train_rows, targets, features, manifest, training, output)
    final = {"residual": component_hash(module, "residual"),
             "router": component_hash(module, "router")}
    if final["residual"] != after_expert["residual"] or final["router"] == after_expert["router"]:
        raise ValueError("Consistent router stage audit failed")
    weights = output / "consistent-router.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()},
              str(weights))
    reloaded = RoutedResidualHead.build(
        manifest["hidden_size"], training["residual_width"], training["router_width"],
        training["dropout"], training["initial_gate_probability"])
    reloaded.load_state_dict(load_file(str(weights)))
    run = {"initial": initial, "initial_max_logit_difference": initial_max,
           "after_expert": after_expert, "final": final, "expert": expert,
           "target_summary": target_summary, "router": router,
           "weights_sha256": digest(weights.read_bytes()),
           "final_parameter_sha256": parameter_sha256(reloaded)}
    cases = datasets["cases"]
    groups = {
        "seen": ([row for row in cases if row["split"] == "test"], "cases"),
        "qasc_validation": ([row for row in cases if row["split"] == "validation" and row["source"] == "qasc"], "cases"),
        "development": (datasets["development"], "development"),
        "qasc_development": (datasets["qasc_development"], "qasc_development"),
        "replay_train": ([row for row in train_rows if row["source"] != "qasc"], "cases"),
        "qasc_train": ([row for row in train_rows if row["source"] == "qasc"], "cases"),
    }
    evaluations = {name: evaluate(reloaded, rows, dataset, features, manifest)
                   for name, (rows, dataset) in groups.items()}
    paired = {name: paired_gate_summary(reloaded, rows, dataset, features, manifest)
              for name, (rows, dataset) in groups.items()}
    write_json(output / "evaluation.json", evaluations); write_json(output / "paired-gates.json", paired)
    rules = config["screening_rule"]
    context_deltas = {source: value["mean"] - evaluations["development"]["parent_context_advantage"]["by_source"][source]["mean"] for source, value in evaluations["development"]["context_advantage"]["by_source"].items()}
    qasc = evaluations["qasc_validation"]
    checks = {
        "mechanism.initial_exact": initial_max == 0.0,
        "mechanism.expert_stage_isolated": after_expert["residual"] != initial["residual"] and after_expert["router"] == initial["router"],
        "mechanism.router_stage_isolated": final["residual"] == after_expert["residual"] and final["router"] != after_expert["router"],
        "mechanism.target_coverage": len(targets) == 1024,
        "mechanism.router_separates": evaluations["qasc_train"]["gates"]["all"]["mean"] - evaluations["replay_train"]["gates"]["all"]["mean"] >= rules["qasc_vs_replay_gate_min_delta"],
        "seen.accuracy": evaluations["seen"]["raw"]["accuracy"] - evaluations["seen"]["parent_raw"]["accuracy"] >= rules["seen_accuracy_vs_parent_min_delta"],
        "seen.nll": evaluations["seen"]["raw"]["nll"] - evaluations["seen"]["parent_raw"]["nll"] <= rules["seen_nll_vs_parent_max_delta"],
        "development.accuracy": evaluations["development"]["raw"]["accuracy"] - evaluations["development"]["parent_raw"]["accuracy"] >= rules["development_accuracy_vs_parent_min_delta"],
        "development.context_retention": min(context_deltas.values()) >= rules["development_each_context_advantage_vs_parent_min_delta"],
        "qasc.accuracy": qasc["raw"]["accuracy"] - qasc["parent_raw"]["accuracy"] >= rules["qasc_accuracy_vs_parent_min_delta"],
        "qasc.nll": qasc["raw"]["nll"] - qasc["parent_raw"]["nll"] <= rules["qasc_nll_vs_parent_max_delta"],
        "qasc.context": evaluations["qasc_development"]["context_advantage"]["mean"] - evaluations["qasc_development"]["parent_context_advantage"]["mean"] >= rules["qasc_context_advantage_vs_parent_min_delta"],
    }
    summary = {"seen": {"accuracy": evaluations["seen"]["raw"]["accuracy"], "parent_accuracy": evaluations["seen"]["parent_raw"]["accuracy"], "nll": evaluations["seen"]["raw"]["nll"], "parent_nll": evaluations["seen"]["parent_raw"]["nll"]}, "development": {"accuracy": evaluations["development"]["raw"]["accuracy"], "parent_accuracy": evaluations["development"]["parent_raw"]["accuracy"]}, "qasc": {"accuracy": qasc["raw"]["accuracy"], "parent_accuracy": qasc["parent_raw"]["accuracy"], "nll": qasc["raw"]["nll"], "parent_nll": qasc["parent_raw"]["nll"]}, "gates": {name: evaluations[name]["gates"]["all"]["mean"] for name in ("replay_train", "qasc_train", "seen", "development", "qasc_validation")}}
    run["paired_gates"] = paired
    evidence = {"format_version": 1, "study": config["study"], "role": "consumed single-seed paired-consistent routing screen", "protocol_sha256": digest(config_path.read_bytes()), "run": run, "summary": summary, "context_delta_vs_parent": context_deltas, "checks": checks, "advances_to_live_integration_audit": all(checks.values()), "limitations": ["All evaluation tasks are consumed.", "Single seed and frozen-feature mechanism screen.", "Live checkpoint and output-contract audits remain required."]}
    write_json(output / "run.json", run); write_json(root / "docs/evidence/qasc-consistent-router-screen-v1.json", evidence)
    gate_delta = summary["gates"]["qasc_train"] - summary["gates"]["replay_train"]
    lines = ["# QASC 成对一致反事实路由筛选", "", "反事实收益路由增加完整/空材料门一致性。全部评测任务均已消费。", "", "| 指标 | 父模型 | 成对一致路由 | 差值 |", "|---|---:|---:|---:|", f"| 已见准确率 | {summary['seen']['parent_accuracy']:.4f} | {summary['seen']['accuracy']:.4f} | {summary['seen']['accuracy']-summary['seen']['parent_accuracy']:+.4f} |", f"| 已见 NLL | {summary['seen']['parent_nll']:.4f} | {summary['seen']['nll']:.4f} | {summary['seen']['nll']-summary['seen']['parent_nll']:+.4f} |", f"| 开发准确率 | {summary['development']['parent_accuracy']:.4f} | {summary['development']['accuracy']:.4f} | {summary['development']['accuracy']-summary['development']['parent_accuracy']:+.4f} |", f"| QASC 准确率 | {summary['qasc']['parent_accuracy']:.4f} | {summary['qasc']['accuracy']:.4f} | {summary['qasc']['accuracy']-summary['qasc']['parent_accuracy']:+.4f} |", f"| QASC NLL | {summary['qasc']['parent_nll']:.4f} | {summary['qasc']['nll']:.4f} | {summary['qasc']['nll']-summary['qasc']['parent_nll']:+.4f} |", "", "## 路由", "", f"- 学得门均值（回放/QASC）：`{summary['gates']['replay_train']:.6f}` / `{summary['gates']['qasc_train']:.6f}`。", f"- 门差：`{gate_delta:+.6f}`。", f"- 开发集完整/空材料门平均绝对差：`{paired['development']['all']['mean_absolute_difference']:.6f}`。", "", "## 判定", "", ("全部门槛通过；允许进入实时集成审计。" if all(checks.values()) else "至少一个门槛失败；不进入实时集成。"), "", "逐项门槛：" + ", ".join(f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "", "机器证据：`docs/evidence/qasc-consistent-router-screen-v1.json`", ""]
    (root / "docs/QASC_CONSISTENT_ROUTER_SCREEN.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(json.dumps({"state": "complete", "advances": all(checks.values()), "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
