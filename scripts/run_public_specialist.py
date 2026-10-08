#!/usr/bin/env python3
"""Train a multi-seed residual specialist on frozen public-model features."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.counterfactual_routing import composite_gate_costs
from decision_model.feature_screen import RoutedResidualHead, parameter_sha256
from decision_model.route_utility import signed_route_utility
from decision_model.train import evaluate_rows


_HELPER_PATH = Path(__file__).with_name("run_qasc_routed_residual_screen.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("public_specialist_feature_helper", _HELPER_PATH)
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC); _HELPER_SPEC.loader.exec_module(_HELPER)
load_features, pad_batch = _HELPER.load_features, _HELPER.pad_batch


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    return sum(values) / len(values)


def select_data_rows(rows, splits, sources=()):
    split_set, source_set = set(splits), set(sources)
    if not split_set:
        raise ValueError("Specialist data selection needs at least one split")
    selected = [row for row in rows if row["split"] in split_set and
                (not source_set or row["source"] in source_set)]
    if not selected:
        raise ValueError("Specialist data selection is empty")
    return selected


def make_plan(rows, seed, epochs):
    plan = []
    for epoch in range(epochs):
        rng = random.Random(seed + epoch * 1_000_003)
        ordered = sorted(rows, key=lambda row: row["id"])
        rng.shuffle(ordered)
        for row in ordered:
            order = list(range(len(row["request"]["choices"])))
            rng.shuffle(order)
            plan.append({"epoch": epoch, "row_id": row["id"], "candidate_order": order})
    return plan


def specialist_loss(parent, parent_null, correction, null_correction, mask, targets,
                    brier_weight, context_weight, context_gap):
    logits = parent + correction
    null_logits = parent_null + null_correction
    log_probabilities = logits.log_softmax(-1)
    probabilities = logits.softmax(-1)
    one_hot = torch.nn.functional.one_hot(targets, logits.shape[1]).to(logits.dtype)
    cross_entropy = torch.nn.functional.nll_loss(log_probabilities, targets)
    brier = (((probabilities - one_hot) ** 2) * mask).sum(-1).mean()
    rows = torch.arange(len(targets), device=targets.device)
    advantage = log_probabilities[rows, targets] - null_logits.log_softmax(-1)[rows, targets]
    context = torch.relu(logits.new_tensor(context_gap) - advantage).mean()
    return cross_entropy + brier_weight * brier + context_weight * context


def score(module, rows, dataset, features, manifest, batch_size=64):
    logits, null_logits, corrections, null_corrections = [], [], [], []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            keys = [f"{dataset}:{row['id']}" for row in batch]
            full, null, parent, parent_null, mask, _ = pad_batch(
                batch, keys, features, manifest)
            correction = module.residual(full).squeeze(-1).masked_fill(~mask, 0.0)
            null_correction = module.residual(null).squeeze(-1).masked_fill(~mask, 0.0)
            for index, row in enumerate(batch):
                count = len(row["request"]["choices"])
                current = correction[index, :count]
                current_null = null_correction[index, :count]
                logits.append((parent[index, :count] + current).tolist())
                null_logits.append((parent_null[index, :count] + current_null).tolist())
                corrections.append(current.tolist()); null_corrections.append(current_null.tolist())
    return logits, null_logits, corrections, null_corrections


def context_summary(rows, full_logits, null_logits):
    groups = defaultdict(list)
    for row, full, null in zip(rows, full_logits, null_logits):
        target = [choice["id"] for choice in row["request"]["choices"]].index(row["label"])
        advantage = float(torch.log_softmax(torch.tensor(full), -1)[target] -
                          torch.log_softmax(torch.tensor(null), -1)[target])
        groups[row["source"]].append(advantage)
    return {source: {"rows": len(values), "mean": mean(values),
                     "positive": sum(value > 0 for value in values)}
            for source, values in sorted(groups.items())}


def evaluate(module, rows, dataset, features, manifest, cost):
    expert_logits, expert_null, corrections, null_corrections = score(
        module, rows, dataset, features, manifest)
    parent_logits = [manifest["rows"][f"{dataset}:{row['id']}"]["parent_logits"]
                     for row in rows]
    parent_null = [manifest["rows"][f"{dataset}:{row['id']}"]["parent_null_logits"]
                   for row in rows]
    parent_metrics, _ = evaluate_rows(rows, parent_logits, 1.0)
    expert_metrics, _ = evaluate_rows(rows, expert_logits, 1.0)
    gates = torch.tensor([0.0, 1.0], dtype=torch.float32)
    utility_records = []
    for row, parent, parent_n, correction, correction_n in zip(
            rows, parent_logits, parent_null, corrections, null_corrections):
        target = [choice["id"] for choice in row["request"]["choices"]].index(row["label"])
        costs = composite_gate_costs(
            torch.tensor(parent), torch.tensor(parent_n), torch.tensor(correction),
            torch.tensor(correction_n), target, gates,
            brier_weight=cost["brier_weight"],
            context_weight=cost["context_advantage_weight"],
            context_gap=cost["context_advantage_gap"],
            teacher=None, null_teacher=None,
            distillation_weight=0.0, null_distillation_weight=0.0)
        utility_records.append({"source": row["source"], **signed_route_utility(costs, gates)})
    by_source = {}
    for source in sorted({row["source"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["source"] == source]
        source_rows = [rows[index] for index in indices]
        parent, _ = evaluate_rows(source_rows, [parent_logits[index] for index in indices], 1.0)
        expert, _ = evaluate_rows(source_rows, [expert_logits[index] for index in indices], 1.0)
        utilities = [utility_records[index]["utility"] for index in indices]
        by_source[source] = {
            "rows": len(indices), "mean_utility": mean(utilities),
            "open_rate": mean([int(value > 0) for value in utilities]),
            "parent_accuracy": parent["accuracy"], "expert_accuracy": expert["accuracy"],
            "accuracy_delta": expert["accuracy"] - parent["accuracy"],
            "parent_nll": parent["nll"], "expert_nll": expert["nll"],
            "nll_delta": expert["nll"] - parent["nll"],
        }
    utilities = [record["utility"] for record in utility_records]
    return {
        "rows": len(rows), "mean_utility": mean(utilities),
        "open_rate": mean([int(value > 0) for value in utilities]),
        "parent_accuracy": parent_metrics["accuracy"],
        "expert_accuracy": expert_metrics["accuracy"],
        "accuracy_delta": expert_metrics["accuracy"] - parent_metrics["accuracy"],
        "parent_nll": parent_metrics["nll"], "expert_nll": expert_metrics["nll"],
        "nll_delta": expert_metrics["nll"] - parent_metrics["nll"],
        "context_advantage": context_summary(rows, expert_logits, expert_null),
        "parent_context_advantage": context_summary(rows, parent_logits, parent_null),
        "by_source": by_source,
    }


def train_seed(seed, rows, dataset, features, manifest, training, output):
    torch.manual_seed(seed)
    module = RoutedResidualHead.build(
        manifest["hidden_size"], training["residual_width"], training["router_width"],
        training["dropout"], training["initial_gate_probability"])
    module.router.requires_grad_(False)
    initial = parameter_sha256(module.residual)
    probe_logits, _, _, _ = score(module, rows[:24], dataset, features, manifest)
    parent_probe = [manifest["rows"][f"{dataset}:{row['id']}"]["parent_logits"]
                    for row in rows[:24]]
    initial_max = max(abs(left - right) for actual, parent in zip(probe_logits, parent_probe)
                      for left, right in zip(actual, parent))
    if initial_max != 0.0:
        raise ValueError("Specialist does not initialize as the exact parent")
    plan = make_plan(rows, seed, training["epochs"])
    plan_path = output / f"training-plan-seed-{seed}.jsonl"
    plan_path.write_text("".join(canonical(record) + "\n" for record in plan))
    by_id = {row["id"]: row for row in rows}
    optimizer = torch.optim.AdamW(module.residual.parameters(),
                                  lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"], eps=1e-6)
    batches = [plan[index:index + training["batch_size"]]
               for index in range(0, len(plan), training["batch_size"])]
    norms, curve, started = [], [], time.monotonic()
    module.train()
    for update, batch_plan in enumerate(batches, 1):
        batch = [by_id[record["row_id"]] for record in batch_plan]
        keys = [f"{dataset}:{row['id']}" for row in batch]
        orders = [record["candidate_order"] for record in batch_plan]
        full, null, parent, parent_null, mask, targets = pad_batch(
            batch, keys, features, manifest, orders)
        optimizer.zero_grad(set_to_none=True)
        correction = module.residual(full).squeeze(-1).masked_fill(~mask, 0.0)
        null_correction = module.residual(null).squeeze(-1).masked_fill(~mask, 0.0)
        loss = specialist_loss(
            parent, parent_null, correction, null_correction, mask, targets,
            training["brier_weight"], training["context_advantage_weight"],
            training["context_advantage_gap"])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.residual.parameters(),
                                              training["gradient_clip"])
        optimizer.step(); norms.append(float(norm))
        curve.append({"update": update, "loss": float(loss.detach())})
    if len(batches) != training["updates"]:
        raise ValueError("Specialist update count differs from frozen protocol")
    weights = output / f"specialist-seed-{seed}.safetensors"
    save_file({name: value.detach().cpu().contiguous()
               for name, value in module.state_dict().items()}, str(weights))
    reloaded = RoutedResidualHead.build(
        manifest["hidden_size"], training["residual_width"], training["router_width"],
        training["dropout"], training["initial_gate_probability"])
    reloaded.load_state_dict(load_file(str(weights)))
    final = parameter_sha256(reloaded.residual)
    if final == initial:
        raise ValueError("Specialist residual parameters did not change")
    run = {
        "seed": seed, "updates": len(batches), "seconds": time.monotonic() - started,
        "initial_residual_sha256": initial, "final_residual_sha256": final,
        "initial_max_logit_difference": initial_max,
        "weights_sha256": digest(weights.read_bytes()),
        "training_plan_sha256": digest(plan_path.read_bytes()),
        "gradient_norm": {"mean": mean(norms), "maximum": max(norms),
                          "clipped_updates": sum(value > training["gradient_clip"]
                                                 for value in norms)},
        "curve": curve,
    }
    return reloaded, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh specialist training directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Specialist protocol was not frozen before training")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen specialist source changed: " + relative)
    cache = root / config["feature_cache"]["path"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_cache"]["manifest_sha256"]:
        raise ValueError("Frozen specialist feature manifest changed")
    train_path, validation_path = (root / config["data"][name]
                                   for name in ("train", "validation"))
    for name, path in (("train", train_path), ("validation", validation_path)):
        if digest(path.read_bytes()) != config["data"][name + "_sha256"]:
            raise ValueError("Frozen specialist data changed: " + name)
    train_rows = select_data_rows(load_rows(train_path), config["data"]["train_splits"],
                                  config["data"].get("train_sources", []))
    validation_rows = select_data_rows(
        load_rows(validation_path), config["data"]["validation_splits"],
        config["data"].get("validation_sources", []))
    if (len(train_rows) != config["data"]["train_rows"] or
            len(validation_rows) != config["data"]["validation_rows"]):
        raise ValueError("Specialist row coverage differs")
    train_dataset = config["data"]["train_dataset"]
    validation_dataset = config["data"]["validation_dataset"]
    keys = ([f"{train_dataset}:{row['id']}" for row in train_rows] +
            [f"{validation_dataset}:{row['id']}" for row in validation_rows])
    manifest, features = load_features(cache, keys)
    output.mkdir(parents=True); write_json(output / "status.json", {"state": "training"})
    seeds = []
    for seed in config["training"]["seeds"]:
        module, run = train_seed(seed, train_rows, train_dataset, features, manifest,
                                 config["training"], output)
        evaluation = evaluate(module, validation_rows, validation_dataset, features,
                              manifest, config["cost"])
        seeds.append({"seed": seed, "run": run, "evaluation": evaluation})
        write_json(output / f"result-seed-{seed}.json", seeds[-1])
    rules = config["screening_rule"]
    checks = {
        "mechanism.initial_exact_each_seed": all(
            item["run"]["initial_max_logit_difference"] == 0.0 for item in seeds),
        "utility.combined_each_seed": all(
            item["evaluation"]["mean_utility"] >= rules["combined_mean_utility_min"]
            for item in seeds),
        "utility.each_source_each_seed": all(
            value["mean_utility"] >= rules["each_source_mean_utility_min"]
            for item in seeds for value in item["evaluation"]["by_source"].values()),
        "accuracy.combined_each_seed": all(
            item["evaluation"]["accuracy_delta"] >= rules["combined_accuracy_min_delta"]
            for item in seeds),
        "nll.combined_each_seed": all(
            item["evaluation"]["nll_delta"] <= rules["combined_nll_max_delta"]
            for item in seeds),
        "nll.each_source_each_seed": all(
            value["nll_delta"] <= rules["each_source_nll_max_delta"]
            for item in seeds for value in item["evaluation"]["by_source"].values()),
    }
    aggregate = {
        metric: {"mean": mean([item["evaluation"][metric] for item in seeds]),
                 "minimum": min(item["evaluation"][metric] for item in seeds),
                 "maximum": max(item["evaluation"][metric] for item in seeds)}
        for metric in ("mean_utility", "open_rate", "accuracy_delta", "nll_delta")
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "consumed multi-seed public-family specialist screen",
        "protocol_sha256": digest(config_path.read_bytes()),
        "seeds": [{"seed": item["seed"], "run": item["run"],
                   "evaluation": item["evaluation"]} for item in seeds],
        "aggregate": aggregate, "checks": checks,
        "qualifies_as_positive_utility_family": all(checks.values()),
        "limitations": config["limitations"],
    }
    write_json(output / "evidence.json", evidence)
    write_json(root / config["evidence_path"], evidence)
    rows = []
    for item in seeds:
        value = item["evaluation"]
        rows.append(f"| {item['seed']} | {value['mean_utility']:+.4f} | {value['open_rate']:.4f} | "
                    f"{value['parent_accuracy']:.4f} | {value['expert_accuracy']:.4f} | "
                    f"{value['accuracy_delta']:+.4f} | {value['nll_delta']:+.4f} |")
    lines = [
        f"# {config['display_name']} 多随机种子专家训练", "",
        f"在固定 2B 表示和父决策头上，为 {config['display_name']} 训练独立残差专家。"
        "验证集已消费，结果只用于机制筛选。", "",
        "| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 |",
        "|---:|---:|---:|---:|---:|---:|---:|", *rows, "", "## 判定", "",
        (f"全部门槛通过；{config['display_name']} 可计为一个已消费的正效用专家族。"
         if all(checks.values()) else
         f"至少一个门槛失败；{config['display_name']} 暂不计为正效用专家族。"), "",
        "逐项门槛：" + ", ".join(
            f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
        f"机器证据：`{config['evidence_path']}`", "",
    ]
    (root / config["report_path"]).write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "qualifies": all(checks.values())})
    print(canonical({"event": "complete", "qualifies": all(checks.values()),
                     "aggregate": aggregate, "checks": checks}))


if __name__ == "__main__":
    main()
