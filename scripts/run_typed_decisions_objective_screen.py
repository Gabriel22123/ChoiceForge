"""Run the frozen-feature four-fold, three-seed objective screen."""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import random
import time
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.feature_screen import (CandidateHead, FeatureStore, aligned_targets,
                                           padded_logits, parameter_sha256)
from decision_model.train import calibrate, evaluate_rows
from decision_model.training_objective import TrainingObjective


def plan_seed(protocol_id, fold, seed):
    return f"{protocol_id}:{fold}:{seed}:row-and-candidate-plan"


def make_plan(rows, protocol, fold, seed):
    train = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["id"])
    rng = random.Random(plan_seed(protocol["protocol_id"], fold, seed))
    records = []
    for epoch in range(protocol["head"]["epochs"]):
        ordered = train.copy(); rng.shuffle(ordered)
        size = protocol["head"]["batch_size"]
        for start in range(0, len(ordered), size):
            batch = ordered[start:start + size]; permutations = []
            for row in batch:
                order = list(range(len(row["request"]["choices"]))); rng.shuffle(order)
                permutations.append(order)
            records.append({"epoch": epoch + 1, "rows": [row["id"] for row in batch],
                            "permutations": permutations})
    return records


def write_plan(path, records):
    raw = "".join(canonical(record) + "\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(raw)
    return digest(raw.encode())


def predict(head, rows, features, batch_size, device="cpu"):
    values = []
    head.eval()
    import torch
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            logits, _ = padded_logits(head, [features[row["id"]] for row in batch], device=device)
            for row, output in zip(batch, logits.cpu().tolist()):
                values.append(output[:len(row["request"]["choices"])] )
    return values


def verify_feature_identity(rows, feature_store):
    """Bind every cached tensor to the exact request and candidate order."""
    index = feature_store.manifest["rows"]
    for row in rows:
        if row["id"] not in index:
            raise ValueError("Feature cache misses row: " + row["id"])
        entry = index[row["id"]]
        candidate_ids = [choice["id"] for choice in row["request"]["choices"]]
        if (entry["candidate_ids"] != candidate_ids or
                entry["request_sha256"] != digest(row["request"])):
            raise ValueError("Feature cache request identity differs: " + row["id"])


def endpoint_checksums(endpoint):
    names = ("head.safetensors", "run.json", "validation-evaluation.json",
             "validation-predictions.json", "curve.json")
    return {name: digest((endpoint / name).read_bytes()) for name in names}


def verify_endpoint(endpoint):
    checksums = json.loads((endpoint / "checksums.json").read_text())
    for name, expected in checksums.items():
        if digest((endpoint / name).read_bytes()) != expected:
            raise ValueError("Endpoint artifact changed: " + str(endpoint / name))
    return json.loads((endpoint / "run.json").read_text())


def train_endpoint(endpoint, rows, feature_values, plan_path, plan_hash, protocol, fold, seed, arm):
    if endpoint.exists():
        return verify_endpoint(endpoint)
    import shutil
    temporary = endpoint.with_name(endpoint.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    import torch
    from safetensors.torch import save_file
    torch.manual_seed(seed)
    head = CandidateHead.build(protocol["hidden_size"], protocol["head"]["width"],
                               protocol["head"]["dropout"]).to("cpu")
    initial = parameter_sha256(head)
    objective_config = dict(protocol["objective"], noise_seed=seed)
    objective_config.update(protocol["arms"][arm])
    objective = TrainingObjective(objective_config)
    optimizer = torch.optim.AdamW(head.parameters(), lr=protocol["head"]["learning_rate"],
                                  weight_decay=protocol["head"]["weight_decay"], eps=1e-6)
    by_id = {row["id"]: row for row in rows}; curve = []; update = 0
    started = time.monotonic(); grad_norms = []
    for record in [json.loads(line) for line in plan_path.read_text().splitlines() if line]:
        batch = [by_id[row_id] for row_id in record["rows"]]
        permutations = record["permutations"]
        head.train(True); optimizer.zero_grad(set_to_none=True)
        logits, _ = padded_logits(head, [feature_values[row["id"]] for row in batch], permutations)
        mode = protocol["arms"][arm]["training_target_mode"]
        targets = aligned_targets(batch, permutations, logits.shape[1], mode)
        counts = [len(row["request"]["choices"]) for row in batch]
        kinds = [row["decision_type"] for row in batch]
        loss, risk = objective(logits, targets, counts, kinds, candidate_orders=permutations)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite feature-screen objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), protocol["head"]["gradient_clip"])
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite feature-screen gradient")
        optimizer.step(); update += 1; grad_norms.append(float(norm))
        curve.append({"epoch": record["epoch"], "update": update, "loss": float(loss.detach()),
                      "sampled_risk": float(risk), "pre_clip_grad_norm": float(norm)})
    validation = sorted((row for row in rows if row["split"] == "validation"), key=lambda row: row["id"])
    logits = predict(head, validation, feature_values, 256)
    calibration = calibrate(logits, validation)
    raw, predictions = evaluate_rows(validation, logits)
    calibrated, calibrated_predictions = evaluate_rows(validation, logits, calibration["temperature"])
    temporary.mkdir(parents=True)
    save_file({name: parameter.detach().cpu().contiguous()
               for name, parameter in head.state_dict().items()}, str(temporary / "head.safetensors"))
    write_json(temporary / "curve.json", curve)
    write_json(temporary / "validation-evaluation.json", {
        "raw": raw, "calibrated": calibrated, "calibration": calibration,
    })
    write_json(temporary / "validation-predictions.json", {
        "raw": predictions, "calibrated": calibrated_predictions,
    })
    run = {
        "status": "fit_complete_test_unread", "fold": fold, "seed": seed, "arm": arm,
        "initial_head_sha256": initial, "training_plan_sha256": plan_hash,
        "feature_manifest_sha256": protocol["feature_manifest_sha256"],
        "fit_sha256": protocol["folds"][fold]["fit_sha256"],
        "train_rows": sum(row["split"] == "train" for row in rows),
        "validation_rows": len(validation), "updates": update,
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "head_weights_sha256": digest((temporary / "head.safetensors").read_bytes()),
        "objective": objective.record(), "training_seconds": time.monotonic() - started,
        "gradient_norm": {"min": min(grad_norms), "max": max(grad_norms),
                          "mean": sum(grad_norms) / len(grad_norms),
                          "clipped_updates": sum(value > protocol["head"]["gradient_clip"] for value in grad_norms)},
        "test_parsed": False,
    }
    write_json(temporary / "run.json", run)
    write_json(temporary / "checksums.json", endpoint_checksums(temporary))
    endpoint.parent.mkdir(parents=True, exist_ok=True); temporary.replace(endpoint)
    return run


def load_head(endpoint, protocol):
    import torch
    from safetensors.torch import load_file
    head = CandidateHead.build(protocol["hidden_size"], protocol["head"]["width"],
                               protocol["head"]["dropout"])
    head.load_state_dict(load_file(str(endpoint / "head.safetensors"), device="cpu"), strict=True)
    head.eval(); return head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/typed-decisions-objective-screen-v1.json")
    parser.add_argument("--output", default="runs/typed-decisions-objective-screen-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / args.protocol; source = json.loads(protocol_path.read_text())
    import importlib.metadata
    import torch
    torch.set_num_threads(source["torch_threads"])
    torch.use_deterministic_algorithms(source["deterministic_algorithms"])
    output = root / args.output; feature_store = FeatureStore(root / source["feature_cache"])
    parity_path = root / source["feature_parity_evidence"]
    parity = json.loads(parity_path.read_text())
    if (parity.get("status") != "pass" or
            parity.get("cache_manifest_sha256") != feature_store.manifest_sha256):
        raise ValueError("Feature-cache parity evidence is missing, failed or stale")
    prepared = root / source["prepared_data"]
    prepared_manifest = json.loads((prepared / "manifest.json").read_text())
    if (prepared_manifest.get("protocol_id") != source["protocol_id"] or
            prepared_manifest.get("protocol_sha256") != digest(protocol_path.read_bytes())):
        raise ValueError("Prepared objective-screen data does not match the current protocol")
    cache_sources = feature_store.manifest["frozen"]["source_sha256"]
    for fold, metadata in prepared_manifest["folds"].items():
        if cache_sources.get(metadata["source"]) != metadata["source_sha256"]:
            raise ValueError("Feature cache and prepared fold source differ: " + fold)
    frozen = dict(source)
    frozen.update({
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_sha256": digest(protocol_path.read_bytes()),
        "feature_manifest_sha256": feature_store.manifest_sha256,
        "feature_parity_evidence_sha256": digest(parity_path.read_bytes()),
        "hidden_size": feature_store.manifest["hidden_size"],
        "prepared_manifest_sha256": digest((prepared / "manifest.json").read_bytes()),
        "folds": prepared_manifest["folds"],
        "source_files": {
            path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in (
                Path(__file__).resolve(), root / "src/decision_model/feature_screen.py",
                root / "src/decision_model/training_objective.py", root / "src/decision_model/objective.py",
                root / "src/decision_model/train.py", root / "src/decision_model/core.py", protocol_path)
        },
        "runtime": {"python": platform.python_version(), **{
            package: importlib.metadata.version(package)
            for package in ("torch", "safetensors", "numpy")}},
        "screen_boundary": "Frozen base representations isolate head/objective optimization. Passing is necessary but insufficient for end-to-end LoRA advancement.",
    })
    frozen_path = output / "protocol.json"
    if output.exists():
        existing = json.loads(frozen_path.read_text())
        # Creation time is historical state, not part of the protocol identity.
        frozen["created_utc"] = existing.get("created_utc")
        if existing != frozen:
            raise ValueError("Objective-screen frozen protocol changed")
    else:
        output.mkdir(parents=True); write_json(frozen_path, frozen)
        write_json(output / "status.json", {"state": "preparing_plans"})
    for name, expected in frozen["source_files"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Frozen objective-screen source changed: " + name)
    fit_rows = {}
    fit_ids = set()
    plans = {}
    for fold in sorted(frozen["folds"]):
        path = prepared / ("heldout-" + fold) / "fit.jsonl"
        if digest(path.read_bytes()) != frozen["folds"][fold]["fit_sha256"]:
            raise ValueError("Fit file changed: " + fold)
        rows = load_rows(path); fit_rows[fold] = rows
        verify_feature_identity(rows, feature_store)
        fit_ids.update(row["id"] for row in rows)
        for seed in frozen["seeds"]:
            plan_path = output / "plans" / f"heldout-{fold}-seed{seed}.jsonl"
            records = make_plan(rows, frozen, fold, seed)
            expected_raw = "".join(canonical(record) + "\n" for record in records)
            if plan_path.exists() and plan_path.read_text() != expected_raw:
                raise ValueError("Frozen training plan changed")
            plan_hash = write_plan(plan_path, records) if not plan_path.exists() else digest(plan_path.read_bytes())
            plans[(fold, seed)] = (plan_path, plan_hash)
    features = feature_store.load_subset(sorted(fit_ids))
    write_json(output / "status.json", {"state": "training", "test_files_parsed": False})
    controls = {}
    for fold in sorted(frozen["folds"]):
        for seed in frozen["seeds"]:
            plan_path, plan_hash = plans[(fold, seed)]
            for arm in frozen["arms"]:
                endpoint = output / f"heldout-{fold}" / f"seed{seed}" / arm
                write_json(output / "status.json", {"state": "training", "fold": fold,
                                                     "seed": seed, "arm": arm,
                                                     "test_files_parsed": False})
                run = train_endpoint(endpoint, fit_rows[fold], features, plan_path, plan_hash,
                                     frozen, fold, seed, arm)
                key = (fold, seed); value = (run["initial_head_sha256"], run["training_plan_sha256"], run["updates"])
                if key in controls and controls[key] != value:
                    raise ValueError("Matched head controls differ across arms")
                controls[key] = value
                print(canonical({"event": "fit_complete", "fold": fold, "seed": seed,
                                 "arm": arm, "updates": run["updates"]}), flush=True)
    endpoints = {}
    for fold in sorted(frozen["folds"]):
        for seed in frozen["seeds"]:
            for arm in frozen["arms"]:
                endpoint = output / f"heldout-{fold}" / f"seed{seed}" / arm
                run = verify_endpoint(endpoint)
                endpoints[f"{fold}/seed{seed}/{arm}"] = {
                    "head_weights_sha256": run["head_weights_sha256"],
                    "run_sha256": digest((endpoint / "run.json").read_bytes()),
                    "fit_checksums_sha256": digest((endpoint / "checksums.json").read_bytes()),
                    "training_plan_sha256": run["training_plan_sha256"],
                }
    lock = {"status": "all_heads_and_validation_locked_before_test",
            "protocol_sha256": digest(frozen_path.read_bytes()), "endpoints": endpoints,
            "test_files_parsed": False}
    lock_path = output / "test-lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Test lock changed")
    write_json(lock_path, lock)
    write_json(output / "status.json", {"state": "testing", "test_files_parsed": True})
    for fold in sorted(frozen["folds"]):
        test_path = prepared / ("heldout-" + fold) / "test.jsonl"
        if digest(test_path.read_bytes()) != frozen["folds"][fold]["test_sha256"]:
            raise ValueError("Sealed test file changed: " + fold)
        test = load_rows(test_path)
        verify_feature_identity(test, feature_store)
        test_features = feature_store.load_subset([row["id"] for row in test])
        for seed in frozen["seeds"]:
            for arm in frozen["arms"]:
                endpoint = output / f"heldout-{fold}" / f"seed{seed}" / arm
                target = endpoint / "test-evaluation.json"
                prediction_path = endpoint / "test-predictions.json"
                checksum_path = endpoint / "test-checksums.json"
                if target.exists() and prediction_path.exists() and checksum_path.exists():
                    saved = json.loads(checksum_path.read_text())
                    if (saved.get("test-evaluation.json") == digest(target.read_bytes()) and
                            saved.get("test-predictions.json") == digest(prediction_path.read_bytes())):
                        continue
                for partial in (target, prediction_path, checksum_path):
                    partial.unlink(missing_ok=True)
                head = load_head(endpoint, frozen); logits = predict(head, test, test_features, 256)
                validation = json.loads((endpoint / "validation-evaluation.json").read_text())
                temperature = validation["calibration"]["temperature"]
                raw, predictions = evaluate_rows(test, logits)
                calibrated, calibrated_predictions = evaluate_rows(test, logits, temperature)
                evaluation_tmp = endpoint / "test-evaluation.json.building"
                predictions_tmp = endpoint / "test-predictions.json.building"
                checksums_tmp = endpoint / "test-checksums.json.building"
                write_json(evaluation_tmp, {"raw": raw, "calibrated": calibrated,
                                            "temperature": temperature, "test_rows": len(test)})
                write_json(predictions_tmp, {
                    "raw": predictions, "calibrated": calibrated_predictions})
                write_json(checksums_tmp, {
                    "test-evaluation.json": digest(evaluation_tmp.read_bytes()),
                    "test-predictions.json": digest(predictions_tmp.read_bytes())})
                evaluation_tmp.replace(target); predictions_tmp.replace(prediction_path)
                checksums_tmp.replace(checksum_path)
                print(canonical({"event": "test_complete", "fold": fold,
                                 "seed": seed, "arm": arm}), flush=True)
    write_json(output / "status.json", {"state": "complete", "test_files_parsed": True,
                                         "endpoints": len(endpoints)})
    print(canonical({"event": "complete", "endpoints": len(endpoints)}))


if __name__ == "__main__":
    main()
