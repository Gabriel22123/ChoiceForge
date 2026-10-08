#!/usr/bin/env python3
"""Evaluate frozen full-source portfolios once on the sealed SciFact family."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from decision_model.core import canonical, digest, load_rows, target_distribution, write_json
from decision_model.expert_portfolio import decision_stable_projection
from decision_model.output_contract import validate_prediction


_BANK_PATH = Path(__file__).with_name("run_heterogeneous_expert_bank.py")
_SPEC = importlib.util.spec_from_file_location("scifact_final_bank", _BANK_PATH)
_BANK = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_BANK)
load_features, load_module, collect_paths = (
    _BANK.load_features, _BANK.load_module, _BANK.collect_paths)
endpoint_model = _BANK.endpoint_model


def read(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def score_paths(rows, full_logits, null_logits, brier_weight):
    if not rows or len(rows) != len(full_logits) or len(rows) != len(null_logits):
        raise ValueError("SciFact score inputs differ")
    accuracy = cross_entropy = brier = context = 0.0
    for row, full, null in zip(rows, full_logits, null_logits):
        target = torch.tensor(target_distribution(row), dtype=torch.float64)
        full = torch.as_tensor(full, dtype=torch.float64)
        null = torch.as_tensor(null, dtype=torch.float64)
        if (full.shape != target.shape or null.shape != target.shape or
                not torch.isfinite(full).all() or not torch.isfinite(null).all()):
            raise ValueError("SciFact path logits are invalid")
        probabilities = full.softmax(-1)
        full_log, null_log = full.log_softmax(-1), null.log_softmax(-1)
        accuracy += int(int(full.argmax()) == int(target.argmax()))
        cross_entropy -= float((target * full_log).sum())
        brier += float(((probabilities - target) ** 2).sum())
        context += float((target * (full_log - null_log)).sum())
    count = len(rows)
    result = {
        "rows": count, "accuracy": accuracy / count,
        "cross_entropy": cross_entropy / count, "brier": brier / count,
        "context_advantage": context / count,
    }
    result["proper_cost"] = result["cross_entropy"] + float(brier_weight) * result["brier"]
    return result


def build_prediction(row, logits):
    values = torch.as_tensor(logits, dtype=torch.float64).softmax(-1).tolist()
    ids = [choice["id"] for choice in row["request"]["choices"]]
    prediction = {
        "choice_id": ids[max(range(len(values)), key=values.__getitem__)],
        "probabilities": dict(zip(ids, values)),
        "requires_review": True,
        "score_kind": "candidate_probability_not_a_guarantee",
        "calibration": "uncalibrated",
    }
    # Exercise the exact serialized public contract, not only the Python object.
    return validate_prediction(json.loads(canonical(prediction)), row["request"])


def permutation_audit(parent, parent_null, portfolio, portfolio_null, alphas):
    selected, selected_null, alpha = decision_stable_projection(
        parent, parent_null, portfolio, portfolio_null, alphas)
    order = torch.arange(len(parent) - 1, -1, -1)
    permuted, permuted_null, permuted_alpha = decision_stable_projection(
        parent[order], parent_null[order], portfolio[order], portfolio_null[order], alphas)
    return bool(
        alpha == permuted_alpha and
        torch.allclose(selected, permuted[order], atol=1e-7, rtol=0) and
        torch.allclose(selected_null, permuted_null[order], atol=1e-7, rtol=0))


def validate_cache(root, config, protocol, rows):
    cache = root / config["feature_cache"]["path"]
    manifest = read(cache / "manifest.json")
    frozen = manifest.get("frozen", {})
    dataset = config["feature_cache"]["dataset"]
    expected_dataset = {
        "path": config["data"]["cases"],
        "sha256": protocol["cases_sha256"],
    }
    if (manifest.get("status") != "complete" or
            frozen.get("format") != config["feature_cache"]["format"] or
            frozen.get("datasets", {}).get(dataset) != expected_dataset or
            frozen.get("row_count") != len(rows) or
            frozen.get("extractor_source_sha256") !=
            config["source_files"]["scripts/cache_public_specialist_features.py"]):
        raise ValueError("SciFact final feature cache does not match the frozen protocol")
    keys = [f"{dataset}:{row['id']}" for row in rows]
    loaded_manifest, features = load_features(cache, keys)
    if loaded_manifest != manifest:
        raise ValueError("SciFact final feature manifest changed while loading")
    return manifest, features, keys


def load_expert_paths(root, config, rows, manifest, features):
    dataset = config["feature_cache"]["dataset"]
    output = {}
    for name, endpoint in sorted(config["experts"].items()):
        weights = root / endpoint["weights"]
        if digest(weights.read_bytes()) != endpoint["weights_sha256"]:
            raise ValueError("SciFact final expert changed: " + name)
        model = endpoint_model(config["default_model"], endpoint, manifest["hidden_size"])
        output[name] = collect_paths(
            load_module(weights, model), rows, dataset, features, manifest)
    return output


def evaluate_seed(rows, paths, parent_full, parent_null, expert_paths, record, config):
    if (record.get("paths") != list(paths) or set(record.get("weights", {})) != set(paths) or
            abs(sum(record["weights"].values()) - 1.0) > 1e-6 or
            any(value < 0 for value in record["weights"].values())):
        raise ValueError("SciFact final portfolio weight record is invalid")
    weights = torch.tensor([record["weights"][path] for path in paths])
    output_full, output_null, alphas = [], [], []
    contract_passed = permutation_passed = exact_decisions = 0
    for index, row in enumerate(rows):
        full = [torch.tensor(parent_full[index])] + [
            torch.tensor(expert_paths[name][0][index]) for name in paths[1:]]
        null = [torch.tensor(parent_null[index])] + [
            torch.tensor(expert_paths[name][1][index]) for name in paths[1:]]
        portfolio_full = (torch.stack(full) * weights[:, None]).sum(0)
        portfolio_null = (torch.stack(null) * weights[:, None]).sum(0)
        projected, projected_null, alpha = decision_stable_projection(
            full[0], null[0], portfolio_full, portfolio_null,
            config["projection_alphas"])
        output_full.append(projected); output_null.append(projected_null); alphas.append(alpha)
        exact_decisions += int(int(projected.argmax()) == int(full[0].argmax()))
        build_prediction(row, projected)
        contract_passed += 1
        permutation_passed += int(permutation_audit(
            full[0], null[0], portfolio_full, portfolio_null,
            config["projection_alphas"]))
    parent = score_paths(rows, parent_full, parent_null, config["brier_weight"])
    output = score_paths(rows, output_full, output_null, config["brier_weight"])
    count = len(rows)
    return {
        "seed": record["seed"], "rows": count,
        "parent_accuracy": parent["accuracy"], "accuracy": output["accuracy"],
        "accuracy_delta": output["accuracy"] - parent["accuracy"],
        "cross_entropy_delta": output["cross_entropy"] - parent["cross_entropy"],
        "brier_delta": output["brier"] - parent["brier"],
        "proper_gain": parent["proper_cost"] - output["proper_cost"],
        "context_advantage_delta": output["context_advantage"] - parent["context_advantage"],
        "mean_alpha": sum(alphas) / count, "minimum_alpha": min(alphas),
        "exact_decision_rate": exact_decisions / count,
        "output_contract_rate": contract_passed / count,
        "permutation_equivariance_rate": permutation_passed / count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact-final-evaluation-v1.json")
    parser.add_argument("--output", default="runs/scifact-final-evaluation-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    if output.exists():
        raise ValueError("Use a fresh SciFact final evaluation directory")
    config = read(config_path)
    if config.get("status") != "frozen_before_unsealing":
        raise ValueError("SciFact final evaluation protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen SciFact final source changed: " + relative)
    source_config = root / config["source"]["source_config"]
    if digest(source_config.read_bytes()) != config["source"]["source_config_sha256"]:
        raise ValueError("SciFact source metadata changed")
    archive = root / config["source"]["archive_path"]
    if (archive.stat().st_size != config["source"]["archive_bytes"] or
            file_sha256(archive) != config["source"]["archive_sha256"]):
        raise ValueError("SciFact sealed archive changed")
    cases, protocol_path = root / config["data"]["cases"], root / config["data"]["protocol"]
    protocol = read(protocol_path)
    if (protocol.get("study") != config["study"] or
            protocol.get("source_archive_sha256") != config["source"]["archive_sha256"] or
            protocol.get("source_config_sha256") != config["source"]["source_config_sha256"] or
            protocol.get("preparation_script_sha256") !=
            config["source_files"]["scripts/prepare_scifact_final.py"] or
            protocol.get("cases_sha256") != digest(cases.read_bytes()) or
            not protocol.get("answer_blind_pair_selection") or
            not protocol.get("answer_blind_candidate_order") or
            protocol.get("raw_text_release") is not False):
        raise ValueError("SciFact prepared data protocol differs")
    rows = load_rows(cases)
    if (len(rows) != protocol.get("rows") or not config["data"]["min_rows"] <= len(rows) <=
            config["data"]["max_rows"] or
            any(row.get("source") != "scifact" or row.get("evaluation_regime") !=
                "sealed_unseen_family" for row in rows)):
        raise ValueError("SciFact final rows differ from the frozen protocol")
    manifest, features, keys = validate_cache(root, config, protocol, rows)
    expert_paths = load_expert_paths(root, config, rows, manifest, features)
    names = tuple(sorted(expert_paths)); paths = ("parent", *names)
    parent_full = [manifest["rows"][key]["parent_logits"] for key in keys]
    parent_null = [manifest["rows"][key]["parent_null_logits"] for key in keys]
    runs = []
    for endpoint in config["portfolios"]:
        path = root / endpoint["weights"]
        if digest(path.read_bytes()) != endpoint["weights_sha256"]:
            raise ValueError("SciFact final portfolio weights changed")
        record = read(path)
        if record.get("seed") != endpoint["seed"]:
            raise ValueError("SciFact final portfolio seed differs")
        run = evaluate_seed(
            rows, paths, parent_full, parent_null, expert_paths, record, config["evaluation"])
        run["weights_file"] = endpoint["weights"]
        run["weights_file_sha256"] = endpoint["weights_sha256"]
        runs.append(run)
    rule = config["screening_rule"]
    checks = {
        "coverage.seeds": [run["seed"] for run in runs] == rule["required_seeds"],
        "integrity.rows": protocol["rows"] == len(rows),
        "integrity.cache": manifest["frozen"]["datasets"][
            config["feature_cache"]["dataset"]]["sha256"] == protocol["cases_sha256"],
        "decision.exact_each_seed": all(run["exact_decision_rate"] == 1.0 and
                                         run["accuracy_delta"] == 0.0 for run in runs),
        "calibration.cross_entropy_each_seed": all(
            run["cross_entropy_delta"] <= rule["cross_entropy_max_delta"] for run in runs),
        "calibration.brier_each_seed": all(
            run["brier_delta"] <= rule["brier_max_delta"] for run in runs),
        "utility.proper_each_seed": all(
            run["proper_gain"] >= rule["proper_gain_min"] for run in runs),
        "context.each_seed": all(
            run["context_advantage_delta"] >= rule["context_min_delta"] for run in runs),
        "contract.each_row_seed": all(run["output_contract_rate"] == 1.0 for run in runs),
        "permutation.each_row_seed": all(
            run["permutation_equivariance_rate"] == 1.0 for run in runs),
        "projection.material_each_seed": all(
            run["mean_alpha"] >= rule["mean_alpha_min"] for run in runs),
    }
    evidence = {
        "format_version": 1, "study": config["study"],
        "role": "one-time sealed unseen-family public evaluation",
        "protocol_sha256": digest(config_path.read_bytes()),
        "source_archive_sha256": config["source"]["archive_sha256"],
        "prepared_data": {"rows": len(rows), "cases_sha256": protocol["cases_sha256"],
                          "selection_sha256": protocol["selection_sha256"]},
        "feature_manifest_sha256": digest((root / config["feature_cache"]["path"] /
                                           "manifest.json").read_bytes()),
        "runs": runs, "checks": checks, "passed": all(checks.values()),
        "limitations": config["limitations"],
    }
    output.mkdir(parents=True)
    write_json(output / "evidence.json", evidence)
    write_json(root / config["evidence_path"], evidence)
    table = [
        f"| {run['seed']} | {run['accuracy_delta']:+.4f} | "
        f"{run['cross_entropy_delta']:+.4f} | {run['brier_delta']:+.4f} | "
        f"{run['proper_gain']:+.4f} | {run['context_advantage_delta']:+.4f} | "
        f"{run['mean_alpha']:.4f} |" for run in runs]
    lines = ["# SciFact 封存任务族最终评测", "",
      "在未参与开发的科学主张判断任务族上，一次性检验冻结的三组专家组合。", "",
      "| 种子 | 准确率差 | 交叉熵差 | Brier 差 | proper 收益 | 上下文收益差 | 平均组合步长 |",
      "|---:|---:|---:|---:|---:|---:|---:|", *table, "", "## 完整性", "",
      f"- 封存样本：`{len(rows)}` 行；原始与转换文本不进入发布包。",
      "- 每一行、每个种子均通过严格 JSON 输出契约与候选顺序等变性检查。",
      "- 决策稳定投影保证所有硬选择与冻结父模型一致。", "", "## 判定", "",
      ("全部预注册门槛通过。" if all(checks.values()) else
       "至少一个预注册门槛失败；结果保留，不调整门槛或重开封存任务族。"), "",
      "逐项门槛：" + ", ".join(
          f"`{name}`={'PASS' if passed else 'FAIL'}" for name, passed in checks.items()), "",
      f"机器证据：`{config['evidence_path']}`", ""]
    (root / config["report_path"]).write_text("\n".join(lines))
    write_json(output / "status.json", {"state": "complete", "passed": all(checks.values())})
    print(canonical({"event": "complete", "passed": all(checks.values()),
                     "rows": len(rows), "checks": checks,
                     "runs": [{key: run[key] for key in (
                         "seed", "accuracy_delta", "cross_entropy_delta", "brier_delta",
                         "proper_gain", "context_advantage_delta", "mean_alpha")}
                              for run in runs]}))


if __name__ == "__main__":
    main()
