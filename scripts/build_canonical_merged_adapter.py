#!/usr/bin/env python3
"""Fuse the canonical validated expert portfolio into one residual adapter."""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import save_file

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.expert_portfolio import decision_stable_projection
from decision_model.feature_screen import parameter_sha256
from decision_model.merged_adapter import fuse_routed_residual_experts
from decision_model.output_contract import validate_prediction


_BANK_PATH = Path(__file__).with_name("run_heterogeneous_expert_bank.py")
_SPEC = importlib.util.spec_from_file_location("merged_adapter_bank", _BANK_PATH)
_BANK = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_BANK)
load_features, load_module = _BANK.load_features, _BANK.load_module
endpoint_model = _BANK.endpoint_model


def read(path):
    return json.loads(Path(path).read_text())


def build_prediction(row, logits):
    probabilities = torch.as_tensor(logits, dtype=torch.float64).softmax(-1).tolist()
    ids = [choice["id"] for choice in row["request"]["choices"]]
    prediction = {
        "choice_id": ids[max(range(len(ids)), key=probabilities.__getitem__)],
        "probabilities": dict(zip(ids, probabilities)), "requires_review": True,
        "score_kind": "candidate_probability_not_a_guarantee",
        "calibration": "uncalibrated",
    }
    return validate_prediction(json.loads(canonical(prediction)), row["request"])


def verify_group(root, group, modules, expert_weights, merged, alphas, tolerance):
    data_path = root / group["data_path"]
    if digest(data_path.read_bytes()) != group["data_sha256"]:
        raise ValueError("Merged-adapter data changed: " + group["name"])
    rows = [row for row in load_rows(data_path)
            if row["split"] in group["splits"] and row["source"] in group["sources"]]
    expected = {tuple(key.split("|")): value for key, value in group["counts"].items()}
    if Counter((row["source"], row["split"]) for row in rows) != expected:
        raise ValueError("Merged-adapter group coverage changed: " + group["name"])
    cache = root / group["feature_cache"]
    if digest((cache / "manifest.json").read_bytes()) != group["manifest_sha256"]:
        raise ValueError("Merged-adapter feature cache changed: " + group["name"])
    keys = [f"{group['dataset']}:{row['id']}" for row in rows]
    manifest, features = load_features(cache, keys)
    maximum_full = maximum_null = maximum_projected = 0.0
    exact_alpha = exact_choice = contract = permutation = 0
    cursor = 0
    for start in range(0, len(rows), 64):
        batch_rows, batch_keys = rows[start:start + 64], keys[start:start + 64]
        lengths = [len(row["request"]["choices"]) for row in batch_rows]
        full = torch.cat([features[key][0] for key in batch_keys], dim=0)
        null = torch.cat([features[key][1] for key in batch_keys], dim=0)
        with torch.no_grad():
            original_full = sum(float(expert_weights[name]) * modules[name].residual(full)
                                for name in sorted(modules)).squeeze(-1)
            original_null = sum(float(expert_weights[name]) * modules[name].residual(null)
                                for name in sorted(modules)).squeeze(-1)
            merged_full = merged(full).squeeze(-1)
            merged_null = merged(null).squeeze(-1)
            reversed_full = merged(full.flip(0)).squeeze(-1).flip(0)
        maximum_full = max(maximum_full, float((original_full - merged_full).abs().max()))
        maximum_null = max(maximum_null, float((original_null - merged_null).abs().max()))
        permutation += int(torch.allclose(
            merged_full, reversed_full, atol=tolerance, rtol=0)) * len(batch_rows)
        offset = 0
        for row, key, count in zip(batch_rows, batch_keys, lengths):
            entry = manifest["rows"][key]
            parent = torch.tensor(entry["parent_logits"])
            parent_null = torch.tensor(entry["parent_null_logits"])
            original, original_null_path, original_alpha = decision_stable_projection(
                parent, parent_null,
                parent + original_full[offset:offset + count],
                parent_null + original_null[offset:offset + count], alphas)
            fused, fused_null, fused_alpha = decision_stable_projection(
                parent, parent_null,
                parent + merged_full[offset:offset + count],
                parent_null + merged_null[offset:offset + count], alphas)
            maximum_projected = max(
                maximum_projected, float((original - fused).abs().max()),
                float((original_null_path - fused_null).abs().max()))
            exact_alpha += int(original_alpha == fused_alpha)
            exact_choice += int(int(original.argmax()) == int(fused.argmax()))
            build_prediction(row, fused); contract += 1
            offset += count; cursor += 1
        if offset != len(full):
            raise ValueError("Merged-adapter candidate slicing differs")
    if cursor != len(rows):
        raise ValueError("Merged-adapter row coverage differs")
    return {
        "rows": len(rows), "sources": sorted(set(row["source"] for row in rows)),
        "maximum_full_residual_difference": maximum_full,
        "maximum_null_residual_difference": maximum_null,
        "maximum_projected_logit_difference": maximum_projected,
        "exact_alpha_rate": exact_alpha / len(rows),
        "exact_choice_rate": exact_choice / len(rows),
        "output_contract_rate": contract / len(rows),
        "candidate_permutation_rate": permutation / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/canonical-merged-adapter-v1.json")
    parser.add_argument("--output", default="runs/canonical-merged-adapter-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh canonical merged-adapter directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_merge":
        raise ValueError("Canonical merged-adapter protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen merged-adapter source changed: " + relative)
    full_config_path = root / config["full_portfolio_config"]["path"]
    if digest(full_config_path.read_bytes()) != config["full_portfolio_config"]["sha256"]:
        raise ValueError("Full portfolio config changed")
    full_config = read(full_config_path)
    for relative, expected in full_config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Full-portfolio dependency changed: " + relative)
    portfolio_path = root / config["canonical_portfolio"]["weights"]
    if digest(portfolio_path.read_bytes()) != config["canonical_portfolio"]["weights_sha256"]:
        raise ValueError("Canonical portfolio weights changed")
    portfolio = read(portfolio_path)
    if (portfolio.get("seed") != config["canonical_portfolio"]["seed"] or
            portfolio.get("seed") != min(config["registered_seeds"]) or
            portfolio.get("paths") != ["parent", *sorted(full_config["experts"])]):
        raise ValueError("Canonical portfolio selection rule differs")
    hidden = config["adapter"]["hidden_size"]
    modules = {}
    for name, endpoint in sorted(full_config["experts"].items()):
        weight_path = root / endpoint["weights"]
        if digest(weight_path.read_bytes()) != endpoint["weights_sha256"]:
            raise ValueError("Canonical merged expert changed: " + name)
        modules[name] = load_module(
            weight_path, endpoint_model(full_config["default_model"], endpoint, hidden))
    expert_weights = {name: portfolio["weights"][name] for name in sorted(modules)}
    merged, architecture = fuse_routed_residual_experts(modules, expert_weights)
    if architecture["total_width"] != config["adapter"]["total_width"]:
        raise ValueError("Merged adapter width differs")
    output.mkdir(parents=True)
    weight_path = output / "adapter.safetensors"
    save_file(merged.state_dict(), str(weight_path), metadata={
        "format": "choiceforge-merged-residual-v1",
        "canonical_seed": str(portfolio["seed"]),
    })
    weight_sha256 = digest(weight_path.read_bytes())
    metadata = {
        "format_version": 1, "format": "choiceforge-merged-residual-v1",
        "project": "ChoiceForge", "canonical_seed": portfolio["seed"],
        "selection_rule": "lowest registered full-portfolio seed",
        "architecture": architecture,
        "portfolio": {"paths": portfolio["paths"], "weights": portfolio["weights"],
                      "weights_file_sha256": config["canonical_portfolio"]["weights_sha256"]},
        "parent": config["parent"], "base_lock_sha256": config["base_lock_sha256"],
        "projection_alphas": config["projection_alphas"],
        "output_contract": {
            "requires_review": True,
            "score_kind": "candidate_probability_not_a_guarantee",
            "calibration": "uncalibrated"
        },
        "adapter_file": "adapter.safetensors", "adapter_sha256": weight_sha256,
        "parameter_sha256": parameter_sha256(merged),
        "training_data_embedded": False,
    }
    write_json(output / "adapter.json", metadata)
    groups = list(full_config["groups"]) + [config["sealed_verification_group"]]
    results = [verify_group(
        root, group, modules, expert_weights, merged, config["projection_alphas"],
        config["screening_rule"]["maximum_absolute_difference"])
               for group in groups]
    rule = config["screening_rule"]
    checks = {
        "coverage.groups": len(results) == len(groups),
        "coverage.rows": sum(result["rows"] for result in results) >= rule["minimum_rows"],
        "equivalence.full": all(result["maximum_full_residual_difference"] <=
                                rule["maximum_absolute_difference"] for result in results),
        "equivalence.null": all(result["maximum_null_residual_difference"] <=
                                rule["maximum_absolute_difference"] for result in results),
        "equivalence.projected": all(result["maximum_projected_logit_difference"] <=
                                     rule["maximum_absolute_difference"] for result in results),
        "equivalence.alpha": all(result["exact_alpha_rate"] == 1.0 for result in results),
        "equivalence.choice": all(result["exact_choice_rate"] == 1.0 for result in results),
        "contract.output": all(result["output_contract_rate"] == 1.0 for result in results),
        "contract.permutation": all(result["candidate_permutation_rate"] == 1.0
                                    for result in results),
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "label-free exact fusion and full-cache equivalence validation",
        "protocol_sha256": digest(config_path.read_bytes()), "adapter": metadata,
        "groups": {group["name"]: result for group, result in zip(groups, results)},
        "checks": checks, "passed": all(checks.values()),
        "limitations": config["limitations"],
    }
    write_json(output / "evidence.json", evidence)
    write_json(root / config["evidence_path"], evidence)
    table = [f"| {name} | {value['rows']} | {value['maximum_full_residual_difference']:.2e} | "
             f"{value['maximum_null_residual_difference']:.2e} | "
             f"{value['maximum_projected_logit_difference']:.2e} |"
             for name, value in evidence["groups"].items()]
    lines = ["# 单文件合并适配器验证结果", "",
      "种子 42 的六专家凸组合已代数合并为一个宽度 1792 的候选残差适配器。", "",
      "| 验证组 | 行数 | 全上下文最大差 | 无上下文最大差 | 投影 logits 最大差 |",
      "|---|---:|---:|---:|---:|", *table, "", "## 结果", "",
      f"- 验证行总数：`{sum(result['rows'] for result in results)}`。",
      f"- 适配器 SHA-256：`{weight_sha256}`。",
      f"- 参数 SHA-256：`{metadata['parameter_sha256']}`。",
      "- 所有决策稳定步长、硬选择、严格输出契约与候选顺序检查完全一致。", "",
      ("全部门槛通过，可作为单一已验证适配器包。" if all(checks.values()) else
       "至少一个合并等价门槛失败，不形成适配器发布结论。"), "",
      f"机器证据：`{config['evidence_path']}`", ""]
    (root / config["report_path"]).write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "passed": all(checks.values()),
                                         "adapter_sha256": weight_sha256})
    print(canonical({"event": "complete", "passed": all(checks.values()),
                     "adapter_sha256": weight_sha256,
                     "rows": sum(result["rows"] for result in results),
                     "checks": checks}))


if __name__ == "__main__":
    main()
