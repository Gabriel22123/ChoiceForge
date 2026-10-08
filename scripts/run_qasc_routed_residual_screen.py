#!/usr/bin/env python3
"""Train and evaluate a semantic request-routed residual on frozen features."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.feature_screen import RoutedResidualHead, parameter_sha256
from decision_model.train import evaluate_rows
from decision_model.training_objective import (TrainingObjective, context_advantage_loss,
                                               distillation_kl_loss)


def read(path):
    return json.loads(Path(path).read_text())


def load_features(cache, keys):
    from safetensors import safe_open
    manifest = read(cache / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Routed feature cache is incomplete")
    requested = set(keys)
    result = {}
    for shard in manifest["shards"]:
        selected = [key for key in shard["row_keys"] if key in requested]
        if not selected:
            continue
        if digest((cache / shard["file"]).read_bytes()) != shard["sha256"]:
            raise ValueError("Routed feature shard checksum mismatch")
        with safe_open(str(cache / shard["file"]), framework="pt", device="cpu") as values:
            for key in selected:
                entry = manifest["rows"][key]
                result[key] = (values.get_tensor(entry["full_key"]).float(),
                               values.get_tensor(entry["null_key"]).float())
    if set(result) != requested:
        raise ValueError("Routed feature cache coverage differs")
    return manifest, result


def pad_batch(rows, keys, feature_map, manifest, orders=None):
    import torch
    if orders is None:
        orders = [list(range(len(row["request"]["choices"]))) for row in rows]
    width = max(len(order) for order in orders)
    hidden = manifest["hidden_size"]
    full = torch.zeros(len(rows), width, hidden)
    null = torch.zeros_like(full)
    parent = torch.full((len(rows), width), -1e4)
    parent_null = torch.full_like(parent, -1e4)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    targets = torch.zeros(len(rows), dtype=torch.long)
    for index, (row, key, order) in enumerate(zip(rows, keys, orders)):
        count = len(order)
        if sorted(order) != list(range(count)):
            raise ValueError("Invalid routed candidate permutation")
        full_value, null_value = feature_map[key]
        entry = manifest["rows"][key]
        candidate_ids = [choice["id"] for choice in row["request"]["choices"]]
        if (list(full_value.shape) != entry["shape"] or
                list(null_value.shape) != entry["shape"] or
                entry["candidate_ids"] != candidate_ids or
                entry["request_sha256"] != digest(row["request"])):
            raise ValueError("Routed cached row identity differs")
        full[index, :count] = full_value[order]
        null[index, :count] = null_value[order]
        parent[index, :count] = torch.tensor(entry["parent_logits"])[order]
        parent_null[index, :count] = torch.tensor(entry["parent_null_logits"])[order]
        mask[index, :count] = True
        targets[index] = order.index(candidate_ids.index(row["label"]))
    return full, null, parent, parent_null, mask, targets


def teacher_tensor(rows, orders, width, field):
    import torch
    target = torch.zeros(len(rows), width)
    active = torch.zeros(len(rows))
    for index, (row, order) in enumerate(zip(rows, orders)):
        values = row.get(field)
        if values is None:
            continue
        choices = row["request"]["choices"]
        expected = {choice["id"] for choice in choices}
        if set(values) != expected or not math.isclose(sum(values.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Invalid routed teacher distribution")
        aligned = [values[choices[candidate]["id"]] for candidate in order]
        target[index, :len(order)] = torch.tensor(aligned)
        active[index] = 1
    return target, active


def make_plan(rows, seed):
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda row: row["id"])
    rng.shuffle(ordered)
    plan = []
    for row in ordered:
        order = list(range(len(row["request"]["choices"])))
        rng.shuffle(order)
        plan.append({"row_id": row["id"], "candidate_order": order})
    return plan


def context_summary(rows, full_logits, null_logits, temperature):
    values, groups = [], defaultdict(list)
    for row, full, null in zip(rows, full_logits, null_logits):
        target = [choice["id"] for choice in row["request"]["choices"]].index(row["label"])
        import torch
        advantage = (torch.log_softmax(torch.tensor(full) / temperature, -1)[target] -
                     torch.log_softmax(torch.tensor(null) / temperature, -1)[target]).item()
        values.append(advantage); groups[row["source"]].append(advantage)
    aggregate = lambda sequence: {"rows": len(sequence), "mean": sum(sequence) / len(sequence),
                                  "positive": sum(value > 0 for value in sequence)}
    return {**aggregate(values), "by_source": {key: aggregate(groups[key]) for key in sorted(groups)}}


def score(module, rows, dataset, feature_map, manifest, batch_size=32):
    import torch
    logits, null_logits, gates = [], [], []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            keys = [f"{dataset}:{row['id']}" for row in batch]
            full, null, parent, parent_null, mask, _ = pad_batch(
                batch, keys, feature_map, manifest)
            correction, gate = module(full, mask)
            null_correction, _ = module(null, mask)
            for index, row in enumerate(batch):
                count = len(row["request"]["choices"])
                logits.append((parent[index, :count] + correction[index, :count]).tolist())
                null_logits.append(
                    (parent_null[index, :count] + null_correction[index, :count]).tolist())
                gates.append(float(gate[index]))
    return logits, null_logits, gates


def gate_summary(rows, gates):
    groups = defaultdict(list)
    for row, gate in zip(rows, gates):
        groups[row["source"]].append(gate)
    aggregate = lambda values: {"rows": len(values), "mean": sum(values) / len(values),
                                "minimum": min(values), "maximum": max(values)}
    return {"all": aggregate(gates),
            "by_source": {key: aggregate(groups[key]) for key in sorted(groups)}}


def evaluate(module, rows, dataset, feature_map, manifest):
    logits, null_logits, gates = score(module, rows, dataset, feature_map, manifest)
    temperature = manifest["frozen"]["parent"]["temperature"]
    raw, _ = evaluate_rows(rows, logits, 1.0)
    scaled, _ = evaluate_rows(rows, logits, temperature)
    parent_logits = [manifest["rows"][f"{dataset}:{row['id']}"]["parent_logits"] for row in rows]
    parent_null = [manifest["rows"][f"{dataset}:{row['id']}"]["parent_null_logits"] for row in rows]
    parent_raw, _ = evaluate_rows(rows, parent_logits, 1.0)
    parent_scaled, _ = evaluate_rows(rows, parent_logits, temperature)
    return {
        "raw": raw, "temperature_scaled": scaled,
        "parent_raw": parent_raw, "parent_temperature_scaled": parent_scaled,
        "context_advantage": context_summary(rows, logits, null_logits, temperature),
        "parent_context_advantage": context_summary(rows, parent_logits, parent_null, temperature),
        "gates": gate_summary(rows, gates),
    }


def train(root, output, config, manifest, feature_map, case_rows):
    import torch
    seed = config["seed"]
    torch.manual_seed(seed)
    module = RoutedResidualHead.build(
        manifest["hidden_size"], config["residual_width"], config["router_width"],
        config["dropout"], config["initial_gate_probability"])
    train_rows = [row for row in case_rows if row["split"] == "train"]
    if len(train_rows) != 1024:
        raise ValueError("Routed screen expects 1,024 training rows")
    initial_hash = parameter_sha256(module)
    initial_probe, _, initial_gate = score(
        module, train_rows[:24], "cases", feature_map, manifest)
    parent_probe = [torch.tensor(
        manifest["rows"][f"cases:{row['id']}"]["parent_logits"], dtype=torch.float32).tolist()
        for row in train_rows[:24]]
    initial_max = max(abs(a - b) for row_a, row_b in zip(initial_probe, parent_probe)
                      for a, b in zip(row_a, row_b))
    if initial_max != 0.0:
        raise ValueError("Routed residual does not initialize as the exact parent")
    plan = make_plan(train_rows, seed)
    (output / "training-plan.jsonl").write_text(
        "".join(canonical(value) + "\n" for value in plan))
    by_id = {row["id"]: row for row in train_rows}
    objective = TrainingObjective({"gradient_estimator": "clean",
                                   "brier_weight": config["brier_weight"]})
    optimizer = torch.optim.AdamW(module.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], eps=1e-6)
    microbatch, accumulation = config["batch_size"], config["accumulation"]
    batches = [plan[index:index + microbatch] for index in range(0, len(plan), microbatch)]
    updates, gradient_norms, curve = 0, [], []
    gate_penalty_sum = 0.0
    started = time.monotonic()
    module.train()
    for group_start in range(0, len(batches), accumulation):
        group = batches[group_start:group_start + accumulation]
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        size = sum(len(batch) for batch in group)
        for batch_plan in group:
            rows = [by_id[item["row_id"]] for item in batch_plan]
            keys = [f"cases:{row['id']}" for row in rows]
            orders = [item["candidate_order"] for item in batch_plan]
            full, null, parent, parent_null, mask, targets = pad_batch(
                rows, keys, feature_map, manifest, orders)
            correction, gate = module(full, mask)
            null_correction, null_gate = module(null, mask)
            logits, null_logits = parent + correction, parent_null + null_correction
            counts = [len(order) for order in orders]
            weights = torch.tensor([row["training_weight"] for row in rows])
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
            replay = torch.tensor([row["source"] != "qasc" for row in rows])
            gate_penalty = ((gate[replay].sum() + null_gate[replay].sum()) /
                            max(1, 2 * int(replay.sum())))
            loss = loss + config["replay_gate_sparsity_weight"] * gate_penalty
            gate_penalty_sum += float(gate_penalty.detach()) * len(rows)
            scaled = loss * (len(rows) / size)
            scaled.backward(); group_loss += float(scaled.detach())
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), config["gradient_clip"])
        optimizer.step(); updates += 1
        gradient_norms.append(float(norm))
        curve.append({"update": updates, "loss": group_loss})
    if updates != config["updates"]:
        raise ValueError("Routed optimizer update count differs")
    from safetensors.torch import save_file, load_file
    weights_path = output / "routed-residual.safetensors"
    save_file({name: value.detach().cpu().contiguous()
               for name, value in module.state_dict().items()}, str(weights_path))
    reloaded = RoutedResidualHead.build(
        manifest["hidden_size"], config["residual_width"], config["router_width"],
        config["dropout"], config["initial_gate_probability"])
    reloaded.load_state_dict(load_file(str(weights_path)))
    after_hash = parameter_sha256(reloaded)
    if after_hash == initial_hash:
        raise ValueError("Routed residual parameters did not change")
    run = {
        "seed": seed, "updates": updates, "training_seconds": time.monotonic() - started,
        "initial_parameter_sha256": initial_hash, "final_parameter_sha256": after_hash,
        "weights_sha256": digest(weights_path.read_bytes()),
        "training_plan_sha256": digest((output / "training-plan.jsonl").read_bytes()),
        "initial_max_logit_difference": initial_max,
        "initial_gate_mean": sum(initial_gate) / len(initial_gate),
        "gradient_norm": {"mean": sum(gradient_norms) / len(gradient_norms),
                          "maximum": max(gradient_norms),
                          "clipped_updates": sum(value > config["gradient_clip"]
                                                 for value in gradient_norms)},
        "mean_replay_gate_penalty": gate_penalty_sum / len(train_rows),
        "curve": curve,
    }
    write_json(output / "run.json", run)
    return reloaded, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qasc-routed-residual-screen-v1.json")
    parser.add_argument("--output", default="runs/qasc-routed-residual-screen-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh routed-residual study directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_training":
        raise ValueError("Routed-residual protocol was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen routed-residual source changed: " + relative)
    cache = root / config["feature_cache"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_manifest_sha256"]:
        raise ValueError("Frozen routed feature manifest changed")
    output.mkdir(parents=True)
    write_json(output / "status.json", {"state": "training"})
    case_rows = load_rows(root / config["data"]["cases"])
    development = load_rows(root / config["data"]["development"])
    qasc_development = load_rows(root / config["data"]["qasc_development"])
    for name, rows in (("cases", case_rows), ("development", development),
                       ("qasc_development", qasc_development)):
        expected = config["data"][name + "_sha256"]
        if digest((root / config["data"][name]).read_bytes()) != expected:
            raise ValueError("Frozen routed dataset changed: " + name)
    keys = ([f"cases:{row['id']}" for row in case_rows] +
            [f"development:{row['id']}" for row in development] +
            [f"qasc_development:{row['id']}" for row in qasc_development])
    manifest, feature_map = load_features(cache, keys)
    module, run = train(root, output, config["training"], manifest, feature_map, case_rows)
    write_json(output / "status.json", {"state": "evaluation"})
    seen = [row for row in case_rows if row["split"] == "test"]
    qasc_validation = [row for row in case_rows
                       if row["split"] == "validation" and row["source"] == "qasc"]
    replay_train = [row for row in case_rows
                    if row["split"] == "train" and row["source"] != "qasc"]
    qasc_train = [row for row in case_rows
                  if row["split"] == "train" and row["source"] == "qasc"]
    evaluations = {
        "seen": evaluate(module, seen, "cases", feature_map, manifest),
        "qasc_validation": evaluate(module, qasc_validation, "cases", feature_map, manifest),
        "development": evaluate(module, development, "development", feature_map, manifest),
        "qasc_development": evaluate(
            module, qasc_development, "qasc_development", feature_map, manifest),
        "replay_train": evaluate(module, replay_train, "cases", feature_map, manifest),
        "qasc_train": evaluate(module, qasc_train, "cases", feature_map, manifest),
    }
    write_json(output / "evaluation.json", evaluations)
    rules = config["screening_rule"]
    dev_context = evaluations["development"]["context_advantage"]["by_source"]
    parent_dev_context = evaluations["development"]["parent_context_advantage"]["by_source"]
    context_deltas = {source: value["mean"] - parent_dev_context[source]["mean"]
                      for source, value in dev_context.items()}
    qasc = evaluations["qasc_validation"]
    checks = {
        "mechanism.initial_exact": run["initial_max_logit_difference"] == 0.0,
        "mechanism.router_separates": (
            evaluations["qasc_train"]["gates"]["all"]["mean"] -
            evaluations["replay_train"]["gates"]["all"]["mean"] >=
            rules["qasc_vs_replay_gate_min_delta"]),
        "seen.accuracy": (evaluations["seen"]["raw"]["accuracy"] -
                          evaluations["seen"]["parent_raw"]["accuracy"] >=
                          rules["seen_accuracy_vs_parent_min_delta"]),
        "seen.nll": (evaluations["seen"]["raw"]["nll"] -
                     evaluations["seen"]["parent_raw"]["nll"] <=
                     rules["seen_nll_vs_parent_max_delta"]),
        "development.accuracy": (evaluations["development"]["raw"]["accuracy"] -
                                 evaluations["development"]["parent_raw"]["accuracy"] >=
                                 rules["development_accuracy_vs_parent_min_delta"]),
        "development.context_retention": min(context_deltas.values()) >=
            rules["development_each_context_advantage_vs_parent_min_delta"],
        "qasc.accuracy": qasc["raw"]["accuracy"] - qasc["parent_raw"]["accuracy"] >=
            rules["qasc_accuracy_vs_parent_min_delta"],
        "qasc.nll": qasc["raw"]["nll"] - qasc["parent_raw"]["nll"] <=
            rules["qasc_nll_vs_parent_max_delta"],
        "qasc.context": (evaluations["qasc_development"]["context_advantage"]["mean"] -
                         evaluations["qasc_development"]["parent_context_advantage"]["mean"] >=
                         rules["qasc_context_advantage_vs_parent_min_delta"]),
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "consumed single-seed cached-feature mechanism screen",
        "protocol_sha256": digest(config_path.read_bytes()),
        "feature_manifest_sha256": digest((cache / "manifest.json").read_bytes()),
        "run": run, "checks": checks,
        "advances_to_live_integration_audit": all(checks.values()),
        "context_delta_vs_parent": context_deltas,
        "summary": {
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
        },
        "limitations": [
            "All evaluation tasks are consumed and cannot support a release claim.",
            "The cached screen audits the trainable routing mechanism; live prompt and checkpoint "
            "integration, ID-renaming behavior and output-contract audit remain required.",
            "Single seed; advancement only permits live integration and later fresh-task evaluation.",
        ],
    }
    write_json(root / "docs/evidence/qasc-routed-residual-screen-v1.json", evidence)
    summary = evidence["summary"]
    gate_delta = summary["gates"]["qasc_train"] - summary["gates"]["replay_train"]
    lines = [
        "# QASC 语义路由残差机制筛选", "",
        "本实验使用冻结父模型特征，只检验请求级语义路由能否在学习 QASC 时保留旧函数。"
        "全部评测任务均已消费，结果不构成发布或零样本泛化结论。", "",
        "| 指标 | 父模型 | 路由残差 | 差值 |", "|---|---:|---:|---:|",
        f"| 已见准确率 | {summary['seen']['parent_accuracy']:.4f} | "
        f"{summary['seen']['accuracy']:.4f} | "
        f"{summary['seen']['accuracy'] - summary['seen']['parent_accuracy']:+.4f} |",
        f"| 已见 NLL | {summary['seen']['parent_nll']:.4f} | "
        f"{summary['seen']['nll']:.4f} | "
        f"{summary['seen']['nll'] - summary['seen']['parent_nll']:+.4f} |",
        f"| 消费开发集准确率 | {summary['development']['parent_accuracy']:.4f} | "
        f"{summary['development']['accuracy']:.4f} | "
        f"{summary['development']['accuracy'] - summary['development']['parent_accuracy']:+.4f} |",
        f"| QASC 准确率 | {summary['qasc']['parent_accuracy']:.4f} | "
        f"{summary['qasc']['accuracy']:.4f} | "
        f"{summary['qasc']['accuracy'] - summary['qasc']['parent_accuracy']:+.4f} |",
        f"| QASC NLL | {summary['qasc']['parent_nll']:.4f} | "
        f"{summary['qasc']['nll']:.4f} | "
        f"{summary['qasc']['nll'] - summary['qasc']['parent_nll']:+.4f} |",
        "", "## 路由行为", "",
        f"- 旧任务回放门均值：`{summary['gates']['replay_train']:.6f}`。",
        f"- QASC 训练门均值：`{summary['gates']['qasc_train']:.6f}`。",
        f"- QASC − 回放门差：`{gate_delta:+.6f}`。", "",
        "## 判定", "",
        ("全部门槛通过；下一步只能进行实时模型集成与输出契约审计。"
         if evidence["advances_to_live_integration_audit"] else
         "至少一个门槛未通过；当前路由机制不进入实时模型集成。"), "",
        "逐项门槛：" + ", ".join(
            f"`{name}`={'PASS' if value else 'FAIL'}" for name, value in checks.items()), "",
        "机器证据：`docs/evidence/qasc-routed-residual-screen-v1.json`", "",
    ]
    (root / "docs/QASC_ROUTED_RESIDUAL_SCREEN.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(json.dumps({"state": "complete", "advances": all(checks.values()),
                      "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
