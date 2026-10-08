#!/usr/bin/env python3
"""Evaluate a lexicographically safe gold-cost oracle over frozen specialists."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import torch

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.train import evaluate_rows


_MATRIX_PATH = Path(__file__).with_name("run_specialist_cross_matrix.py")
_MATRIX_SPEC = importlib.util.spec_from_file_location("safe_oracle_matrix_helper", _MATRIX_PATH)
_MATRIX = importlib.util.module_from_spec(_MATRIX_SPEC); _MATRIX_SPEC.loader.exec_module(_MATRIX)
load_features, pad_batch = _MATRIX.load_features, _MATRIX.pad_batch
load_module = _MATRIX.load_module


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    return sum(values) / len(values)


def path_statistics(full_logits, null_logits, target, brier_weight=0.5):
    if (len(full_logits) != len(null_logits) or len(full_logits) < 2 or
            not 0 <= target < len(full_logits)):
        raise ValueError("Invalid safe-oracle path logits")
    full = torch.as_tensor(full_logits, dtype=torch.float64)
    null = torch.as_tensor(null_logits, dtype=torch.float64)
    if not torch.isfinite(full).all() or not torch.isfinite(null).all():
        raise ValueError("Safe-oracle logits must be finite")
    probabilities = full.softmax(-1)
    one_hot = torch.nn.functional.one_hot(torch.tensor(target), len(full)).to(full.dtype)
    nll = float(-full.log_softmax(-1)[target])
    brier = float(((probabilities - one_hot) ** 2).sum())
    context = float(full.log_softmax(-1)[target] - null.log_softmax(-1)[target])
    return {"correct": int(full.argmax()) == target, "nll": nll, "brier": brier,
            "context_advantage": context, "proper_cost": nll + brier_weight * brier}


def safe_improvement(parent, expert, tolerance=1e-12):
    eligible = (
        (not parent["correct"] or expert["correct"]) and
        expert["nll"] <= parent["nll"] + tolerance and
        expert["brier"] <= parent["brier"] + tolerance and
        expert["context_advantage"] + tolerance >= parent["context_advantage"]
    )
    improvement = parent["proper_cost"] - expert["proper_cost"]
    return {"eligible": bool(eligible and improvement > tolerance),
            "improvement": improvement}


def choose_safe(parent, experts):
    ranked = []
    for name, statistics in experts.items():
        result = safe_improvement(parent, statistics)
        if result["eligible"]:
            ranked.append((result["improvement"], name))
    if not ranked:
        return "parent", 0.0
    improvement, name = sorted(ranked, key=lambda value: (-value[0], value[1]))[0]
    return name, improvement


def collect_paths(module, rows, dataset, features, manifest):
    full_logits, null_logits = [], []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            keys = [f"{dataset}:{row['id']}" for row in batch]
            full, null, parent, parent_null, mask, _ = pad_batch(
                batch, keys, features, manifest)
            correction = module.residual(full).squeeze(-1).masked_fill(~mask, 0.0)
            null_correction = module.residual(null).squeeze(-1).masked_fill(~mask, 0.0)
            for index, row in enumerate(batch):
                count = len(row["request"]["choices"])
                full_logits.append((parent[index, :count] + correction[index, :count]).tolist())
                null_logits.append(
                    (parent_null[index, :count] + null_correction[index, :count]).tolist())
    return full_logits, null_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe-specialist-oracle-v1.json")
    parser.add_argument("--output", default="runs/safe-specialist-oracle-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh safe-specialist-oracle directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_evaluation":
        raise ValueError("Safe specialist oracle was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen safe-oracle source changed: " + relative)
    data_path = root / config["data"]["path"]
    if digest(data_path.read_bytes()) != config["data"]["sha256"]:
        raise ValueError("Frozen safe-oracle data changed")
    rows = [row for row in load_rows(data_path)
            if row["split"] in config["data"]["splits"] and
            row["source"] in config["data"]["sources"]]
    if len(rows) != config["data"]["rows"]:
        raise ValueError("Safe-oracle row coverage differs")
    cache = root / config["feature_cache"]["path"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_cache"]["manifest_sha256"]:
        raise ValueError("Frozen safe-oracle feature manifest changed")
    dataset = config["data"]["dataset"]
    keys = [f"{dataset}:{row['id']}" for row in rows]
    manifest, features = load_features(cache, keys)
    model = {**config["model"], "hidden_size": manifest["hidden_size"]}
    parent_full = [manifest["rows"][key]["parent_logits"] for key in keys]
    parent_null = [manifest["rows"][key]["parent_null_logits"] for key in keys]
    expert_paths = {}
    for name, endpoint in config["experts"].items():
        weights = root / endpoint["weights"]
        if digest(weights.read_bytes()) != endpoint["weights_sha256"]:
            raise ValueError("Frozen safe-oracle expert changed: " + name)
        expert_paths[name] = collect_paths(
            load_module(weights, model), rows, dataset, features, manifest)

    selected, selected_logits, selected_statistics = [], [], []
    audit = Counter()
    for index, row in enumerate(rows):
        target = [choice["id"] for choice in row["request"]["choices"]].index(row["label"])
        parent = path_statistics(parent_full[index], parent_null[index], target,
                                 config["proper_loss"]["brier_weight"])
        candidates = {
            name: path_statistics(paths[0][index], paths[1][index], target,
                                  config["proper_loss"]["brier_weight"])
            for name, paths in expert_paths.items()
        }
        choice, improvement = choose_safe(parent, candidates)
        statistics = parent if choice == "parent" else candidates[choice]
        selected.append({"source": row["source"], "path": choice,
                         "proper_improvement": improvement})
        selected_logits.append(parent_full[index] if choice == "parent"
                               else expert_paths[choice][0][index])
        selected_statistics.append(statistics)
        audit["parent_correct_lost"] += int(parent["correct"] and not statistics["correct"])
        audit["nll_regressed"] += int(statistics["nll"] > parent["nll"] + 1e-12)
        audit["brier_regressed"] += int(statistics["brier"] > parent["brier"] + 1e-12)
        audit["context_regressed"] += int(
            statistics["context_advantage"] + 1e-12 < parent["context_advantage"])

    parent_metrics, _ = evaluate_rows(rows, parent_full, 1.0)
    safe_metrics, _ = evaluate_rows(rows, selected_logits, 1.0)
    by_source = {}
    for source in sorted(config["data"]["sources"]):
        indices = [index for index, row in enumerate(rows) if row["source"] == source]
        source_rows = [rows[index] for index in indices]
        parent, _ = evaluate_rows(source_rows, [parent_full[index] for index in indices], 1.0)
        safe, _ = evaluate_rows(source_rows, [selected_logits[index] for index in indices], 1.0)
        records = [selected[index] for index in indices]
        by_source[source] = {
            "rows": len(indices),
            "mean_proper_improvement": mean([record["proper_improvement"] for record in records]),
            "selection": dict(Counter(record["path"] for record in records)),
            "accuracy_delta": safe["accuracy"] - parent["accuracy"],
            "nll_delta": safe["nll"] - parent["nll"],
            "mean_context_delta": mean([
                selected_statistics[index]["context_advantage"] -
                path_statistics(parent_full[index], parent_null[index],
                                [choice["id"] for choice in rows[index]["request"]["choices"]]
                                .index(rows[index]["label"]),
                                config["proper_loss"]["brier_weight"])["context_advantage"]
                for index in indices]),
        }
    summary = {
        "rows": len(rows),
        "mean_proper_improvement": mean([record["proper_improvement"] for record in selected]),
        "selection": dict(Counter(record["path"] for record in selected)),
        "parent_accuracy": parent_metrics["accuracy"], "safe_accuracy": safe_metrics["accuracy"],
        "accuracy_delta": safe_metrics["accuracy"] - parent_metrics["accuracy"],
        "parent_nll": parent_metrics["nll"], "safe_nll": safe_metrics["nll"],
        "nll_delta": safe_metrics["nll"] - parent_metrics["nll"],
        "audit": dict(audit), "by_source": by_source,
    }
    rules = config["screening_rule"]
    checks = {
        "safety.parent_correct": audit["parent_correct_lost"] == 0,
        "safety.nll": audit["nll_regressed"] == 0,
        "safety.brier": audit["brier_regressed"] == 0,
        "safety.context": audit["context_regressed"] == 0,
        "sources.accuracy": all(value["accuracy_delta"] >= -1e-12
                                for value in by_source.values()),
        "sources.nll": all(value["nll_delta"] <= 1e-12 for value in by_source.values()),
        "combined.accuracy": summary["accuracy_delta"] >= rules["accuracy_min_delta"],
        "combined.nll": summary["nll_delta"] <= rules["nll_max_delta"],
        "combined.proper_improvement": summary["mean_proper_improvement"] >=
            rules["proper_improvement_min"],
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "consumed lexicographically safe gold-cost oracle",
        "protocol_sha256": digest(config_path.read_bytes()),
        "summary": summary, "checks": checks,
        "advances_to_safe_router_design": all(checks.values()),
        "limitations": config["limitations"],
    }
    output.mkdir(parents=True); write_json(output / "evidence.json", evidence)
    write_json(root / "docs/evidence/safe-specialist-oracle-v1.json", evidence)
    lines = ["# 约束优先的安全专家 oracle", "",
             "先逐行过滤会破坏父模型正确性、NLL、Brier 或上下文优势的专家，再在安全集合内"
             "选择 proper-loss 改善最大的路径。", "", "## 汇总", "",
             f"- 平均 proper-loss 改善：`{summary['mean_proper_improvement']:+.4f}`。",
             f"- 准确率：`{summary['parent_accuracy']:.4f}` → `{summary['safe_accuracy']:.4f}` "
             f"(`{summary['accuracy_delta']:+.4f}`)。",
             f"- NLL：`{summary['parent_nll']:.4f}` → `{summary['safe_nll']:.4f}` "
             f"(`{summary['nll_delta']:+.4f}`)。",
             "- 路径选择：" + ", ".join(f"`{key}`={value}"
                                           for key, value in sorted(summary["selection"].items())) + "。",
             "- 安全违规：" + ", ".join(f"`{key}`={value}"
                                           for key, value in sorted(summary["audit"].items())) + "。",
             "", "## 判定", "",
             ("全部门槛通过，可进入安全多专家路由设计。" if all(checks.values())
              else "至少一个门槛失败，暂不训练安全多专家路由。"), "",
             "逐项门槛：" + ", ".join(
                 f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
             "机器证据：`docs/evidence/safe-specialist-oracle-v1.json`", ""]
    (root / "docs/SAFE_SPECIALIST_ORACLE_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(canonical({"event": "complete", "advances": all(checks.values()),
                     "summary": summary, "checks": checks}))


if __name__ == "__main__":
    main()
