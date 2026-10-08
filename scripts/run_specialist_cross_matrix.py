#!/usr/bin/env python3
"""Measure cross-family utility and the gold-cost oracle of a frozen expert bank."""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.counterfactual_routing import composite_gate_costs
from decision_model.feature_screen import RoutedResidualHead, parameter_sha256
from decision_model.route_utility import signed_route_utility
from decision_model.train import evaluate_rows


_HELPER_PATH = Path(__file__).with_name("run_qasc_routed_residual_screen.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("specialist_cross_feature_helper", _HELPER_PATH)
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC); _HELPER_SPEC.loader.exec_module(_HELPER)
load_features, pad_batch = _HELPER.load_features, _HELPER.pad_batch


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    return sum(values) / len(values)


def choose_oracle(utilities_by_expert):
    """Choose parent on ties, otherwise the highest positive-utility expert."""
    if not utilities_by_expert:
        raise ValueError("Oracle needs at least one expert")
    lengths = {len(values) for values in utilities_by_expert.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("Oracle utilities must be aligned and nonempty")
    choices = []
    for index in range(next(iter(lengths))):
        ranked = sorted(((float(values[index]), expert)
                         for expert, values in utilities_by_expert.items()),
                        key=lambda value: (-value[0], value[1]))
        utility, expert = ranked[0]
        choices.append((expert, utility) if utility > 0.0 else ("parent", 0.0))
    return choices


def load_module(path, model):
    module = RoutedResidualHead.build(
        model["hidden_size"], model["residual_width"], model["router_width"],
        model["dropout"], model["initial_gate_probability"])
    module.load_state_dict(load_file(str(path))); module.eval()
    return module


def expert_outputs(module, rows, dataset, features, manifest, cost):
    logits, utilities = [], []
    gates = torch.tensor([0.0, 1.0], dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            keys = [f"{dataset}:{row['id']}" for row in batch]
            full, null, parent, parent_null, mask, targets = pad_batch(
                batch, keys, features, manifest)
            correction = module.residual(full).squeeze(-1).masked_fill(~mask, 0.0)
            null_correction = module.residual(null).squeeze(-1).masked_fill(~mask, 0.0)
            for index, row in enumerate(batch):
                count = len(row["request"]["choices"])
                current = correction[index, :count]
                current_null = null_correction[index, :count]
                costs = composite_gate_costs(
                    parent[index, :count], parent_null[index, :count], current, current_null,
                    int(targets[index]), gates,
                    brier_weight=cost["brier_weight"],
                    context_weight=cost["context_advantage_weight"],
                    context_gap=cost["context_advantage_gap"], teacher=None,
                    null_teacher=None, distillation_weight=0.0,
                    null_distillation_weight=0.0)
                utilities.append(signed_route_utility(costs, gates)["utility"])
                logits.append((parent[index, :count] + current).tolist())
    return logits, utilities


def grouped_summary(rows, parent_logits, expert_logits, utilities):
    result = {}
    for source in sorted({row["source"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["source"] == source]
        selected = [rows[index] for index in indices]
        parent, _ = evaluate_rows(selected, [parent_logits[index] for index in indices], 1.0)
        expert, _ = evaluate_rows(selected, [expert_logits[index] for index in indices], 1.0)
        values = [utilities[index] for index in indices]
        result[source] = {
            "rows": len(indices), "mean_utility": mean(values),
            "open_rate": mean([int(value > 0.0) for value in values]),
            "accuracy_delta": expert["accuracy"] - parent["accuracy"],
            "nll_delta": expert["nll"] - parent["nll"],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/specialist-cross-matrix-v1.json")
    parser.add_argument("--output", default="runs/specialist-cross-matrix-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh specialist cross-matrix directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_evaluation":
        raise ValueError("Specialist cross-matrix protocol was not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen cross-matrix source changed: " + relative)
    data_path = root / config["data"]["path"]
    if digest(data_path.read_bytes()) != config["data"]["sha256"]:
        raise ValueError("Frozen cross-matrix data changed")
    all_rows = load_rows(data_path)
    rows = [row for row in all_rows if row["split"] in config["data"]["splits"] and
            row["source"] in config["data"]["sources"]]
    if len(rows) != config["data"]["rows"]:
        raise ValueError("Cross-matrix row coverage differs")
    cache = root / config["feature_cache"]["path"]
    if digest((cache / "manifest.json").read_bytes()) != config["feature_cache"]["manifest_sha256"]:
        raise ValueError("Frozen cross-matrix feature manifest changed")
    keys = [f"{config['data']['dataset']}:{row['id']}" for row in rows]
    manifest, features = load_features(cache, keys)
    model = {**config["model"], "hidden_size": manifest["hidden_size"]}
    parent_logits = [manifest["rows"][key]["parent_logits"] for key in keys]
    parent_metrics, _ = evaluate_rows(rows, parent_logits, 1.0)

    matrix, canonical_logits, canonical_utilities, parameter_hashes = {}, {}, {}, {}
    for expert, record in config["experts"].items():
        matrix[expert] = {}
        for endpoint in record["endpoints"]:
            weights = root / endpoint["weights"]
            if digest(weights.read_bytes()) != endpoint["weights_sha256"]:
                raise ValueError("Frozen expert weights changed: " + endpoint["weights"])
            module = load_module(weights, model)
            endpoint_key = str(endpoint["seed"])
            parameter_hashes[f"{expert}:{endpoint_key}"] = parameter_sha256(module)
            logits, utilities = expert_outputs(
                module, rows, config["data"]["dataset"], features, manifest, config["cost"])
            matrix[expert][endpoint_key] = grouped_summary(
                rows, parent_logits, logits, utilities)
            if endpoint["seed"] == record["canonical_seed"]:
                canonical_logits[expert], canonical_utilities[expert] = logits, utilities
        if expert not in canonical_logits:
            raise ValueError("Expert has no frozen canonical endpoint: " + expert)

    choices = choose_oracle(canonical_utilities)
    oracle_logits = []
    for index, (expert, _) in enumerate(choices):
        oracle_logits.append(parent_logits[index] if expert == "parent"
                             else canonical_logits[expert][index])
    oracle_metrics, _ = evaluate_rows(rows, oracle_logits, 1.0)
    oracle_by_source = {}
    for source in sorted(config["data"]["sources"]):
        indices = [index for index, row in enumerate(rows) if row["source"] == source]
        selected_rows = [rows[index] for index in indices]
        parent, _ = evaluate_rows(selected_rows, [parent_logits[index] for index in indices], 1.0)
        oracle, _ = evaluate_rows(selected_rows, [oracle_logits[index] for index in indices], 1.0)
        source_choices = [choices[index] for index in indices]
        oracle_by_source[source] = {
            "rows": len(indices), "mean_utility": mean([value for _, value in source_choices]),
            "selection": dict(Counter(expert for expert, _ in source_choices)),
            "accuracy_delta": oracle["accuracy"] - parent["accuracy"],
            "nll_delta": oracle["nll"] - parent["nll"],
        }
    oracle = {
        "rows": len(rows), "mean_utility": mean([value for _, value in choices]),
        "selection": dict(Counter(expert for expert, _ in choices)),
        "parent_accuracy": parent_metrics["accuracy"],
        "oracle_accuracy": oracle_metrics["accuracy"],
        "accuracy_delta": oracle_metrics["accuracy"] - parent_metrics["accuracy"],
        "parent_nll": parent_metrics["nll"], "oracle_nll": oracle_metrics["nll"],
        "nll_delta": oracle_metrics["nll"] - parent_metrics["nll"],
        "by_source": oracle_by_source,
    }
    rules = config["screening_rule"]
    own = {expert: matrix[expert][str(record["canonical_seed"])][record["home_source"]]
           for expert, record in config["experts"].items()}
    off_diagonal = [
        value["mean_utility"]
        for expert, record in config["experts"].items()
        for source, value in matrix[expert][str(record["canonical_seed"])].items()
        if source != record["home_source"]
    ]
    checks = {
        "experts.own_family_utility": all(
            value["mean_utility"] >= rules["each_home_utility_min"] for value in own.values()),
        "oracle.mean_utility": oracle["mean_utility"] >= rules["oracle_mean_utility_min"],
        "oracle.accuracy": oracle["accuracy_delta"] >= rules["oracle_accuracy_min_delta"],
        "oracle.nll": oracle["nll_delta"] <= rules["oracle_nll_max_delta"],
        "routing.needed": min(off_diagonal) < 0.0,
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "consumed frozen expert-bank cross-utility screen",
        "protocol_sha256": digest(config_path.read_bytes()),
        "parameter_sha256": parameter_hashes, "matrix": matrix, "oracle": oracle,
        "checks": checks, "advances_to_multi_expert_router_design": all(checks.values()),
        "limitations": config["limitations"],
    }
    output.mkdir(parents=True); write_json(output / "evidence.json", evidence)
    write_json(root / "docs/evidence/specialist-cross-matrix-v1.json", evidence)
    headers = sorted(config["data"]["sources"])
    table = ["| 专家 | " + " | ".join(headers) + " |",
             "|---|" + "---:|" * len(headers)]
    for expert, record in config["experts"].items():
        values = matrix[expert][str(record["canonical_seed"])]
        table.append("| " + expert + " | " + " | ".join(
            f"{values[source]['mean_utility']:+.4f}" for source in headers) + " |")
    lines = ["# 多专家交叉效用矩阵", "",
             "数值为固定专家相对父路径的平均反事实效用；正值表示完整打开专家更优。", "",
             *table, "", "## 金标成本 oracle", "",
             f"- 合并平均效用：`{oracle['mean_utility']:+.4f}`。",
             f"- 准确率：`{oracle['parent_accuracy']:.4f}` → `{oracle['oracle_accuracy']:.4f}` "
             f"(`{oracle['accuracy_delta']:+.4f}`)。",
             f"- NLL：`{oracle['parent_nll']:.4f}` → `{oracle['oracle_nll']:.4f}` "
             f"(`{oracle['nll_delta']:+.4f}`)。",
             "- 路径选择：" + ", ".join(f"`{key}`={value}"
                                           for key, value in sorted(oracle["selection"].items())) + "。",
             "", "## 判定", "",
             ("全部门槛通过，可进入多专家路由协议设计。" if all(checks.values())
              else "至少一个门槛失败，暂不进入多专家路由训练。"), "",
             "逐项门槛：" + ", ".join(
                 f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in checks.items()), "",
             "机器证据：`docs/evidence/specialist-cross-matrix-v1.json`", ""]
    (root / "docs/SPECIALIST_CROSS_MATRIX_RESULT.zh-CN.md").write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "advances": all(checks.values())})
    print(canonical({"event": "complete", "advances": all(checks.values()),
                     "oracle": oracle, "checks": checks}))


if __name__ == "__main__":
    main()
