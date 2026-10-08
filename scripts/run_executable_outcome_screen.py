#!/usr/bin/env python3
"""Run the frozen-feature complete/chosen-only executable outcome screen."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, outcome_rewards, write_json
from decision_model.feature_screen import CandidateHead, FeatureStore, padded_logits, parameter_sha256
from decision_model.outcome_evaluation import outcome_metrics
from decision_model.outcome_objective import (
    behavior_policy, doubly_robust_loss, full_information_loss, ips_loo_loss,
)


ARMS = ("H-hard-winner", "O-full-information", "R-ips-loo", "D-doubly-robust")


def make_plan(rows, protocol, seed):
    train = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["id"])
    rng = random.Random(f'{protocol["study"]}:seed{seed}:row-plan')
    records = []
    for epoch in range(1, protocol["training"]["epochs"] + 1):
        ordered = train.copy(); rng.shuffle(ordered)
        size = protocol["training"]["batch_size"]
        for start in range(0, len(ordered), size):
            records.append({"epoch": epoch, "rows": [row["id"] for row in ordered[start:start + size]]})
    return records


def write_jsonl(path, records):
    raw = "".join(canonical(record) + "\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(raw)
    return digest(raw.encode())


def build_head(endpoint, hidden_size, width, dropout):
    import torch
    from safetensors.torch import load_file
    head = CandidateHead.build(hidden_size, width, dropout)
    saved = load_file(str(Path(endpoint) / "decision.safetensors"), device="cpu")
    state = {key.removeprefix("head."): value for key, value in saved.items()
             if key.startswith("head.")}
    head.load_state_dict(state, strict=True)
    return head.float()


def make_reward_head(hidden_size, width, dropout, seed):
    import torch
    torch.manual_seed(seed)
    head = CandidateHead.build(hidden_size, width, dropout).float()
    final = head[-1]
    with torch.no_grad():
        final.weight.zero_(); final.bias.zero_()
    return head


def predict(head, rows, features, batch_size):
    import torch
    values = []; head.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            logits, _ = padded_logits(head, [features[row["id"]] for row in batch])
            values.extend([value[:len(row["request"]["choices"])]
                           for row, value in zip(batch, logits.cpu().tolist())])
    return values


def make_behavior_logs(rows, features, initial_head, protocol, seed):
    import torch
    train = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["id"])
    initial_logits = predict(initial_head, train, features, 64)
    by_id = {row["id"]: (row, values) for row, values in zip(train, initial_logits)}
    generator = torch.Generator(device="cpu").manual_seed(seed + 84017)
    records = []
    samples = protocol["training"]["logged_actions_per_row_per_epoch"]
    exploration = protocol["training"]["behavior_exploration"]
    for epoch in range(1, protocol["training"]["epochs"] + 1):
        for row_id in sorted(by_id):
            row, values = by_id[row_id]
            count = len(values)
            mu = behavior_policy(torch.tensor([values]), [count], exploration)[0]
            actions = torch.multinomial(mu, samples, replacement=True, generator=generator).tolist()
            rewards = outcome_rewards(row)
            records.append({"epoch": epoch, "row": row_id, "actions": actions,
                            "rewards": [rewards[action] for action in actions],
                            "behavior_probabilities": [float(mu[action]) for action in actions]})
    return records


def verify_feature_identity(rows, store):
    for row in rows:
        entry = store.manifest["rows"].get(row["id"])
        if not entry or entry["request_sha256"] != digest(row["request"]) or entry["candidate_ids"] != [
                choice["id"] for choice in row["request"]["choices"]]:
            raise ValueError("feature identity differs: " + row["id"])


def train_arm(target, arm, rows, features, initial_head, plan, plan_hash, logs, log_hash,
              protocol, seed):
    if target.exists():
        checksums = json.loads((target / "checksums.json").read_text())
        if all(digest((target / name).read_bytes()) == value for name, value in checksums.items()):
            return json.loads((target / "run.json").read_text())
        raise ValueError("existing endpoint checksum differs: " + str(target))
    import torch
    from safetensors.torch import save_file
    temporary = target.with_name(target.name + ".building")
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    policy = CandidateHead.build(
        features[next(iter(features))].shape[1], protocol["training"]["policy_head_width"],
        protocol["training"]["policy_head_dropout"]).float()
    policy.load_state_dict(initial_head.state_dict())
    initial_hash = parameter_sha256(policy)
    reward_head = (make_reward_head(
        features[next(iter(features))].shape[1], protocol["training"]["reward_head_width"],
        protocol["training"]["reward_head_dropout"], seed + 1907)
                   if arm == "D-doubly-robust" else None)
    parameters = list(policy.parameters()) + (list(reward_head.parameters()) if reward_head else [])
    optimizer = torch.optim.AdamW(parameters, lr=protocol["training"]["learning_rate"],
                                  weight_decay=protocol["training"]["weight_decay"])
    by_id = {row["id"]: row for row in rows}
    log_index = {(record["epoch"], record["row"]): record for record in logs}
    curve = []; updates = 0; started = time.monotonic()
    for record in plan:
        batch = [by_id[row_id] for row_id in record["rows"]]
        policy.train(True); optimizer.zero_grad(set_to_none=True)
        feature_rows = [features[row["id"]] for row in batch]
        logits, _ = padded_logits(policy, feature_rows)
        counts = [len(row["request"]["choices"]) for row in batch]
        if arm == "H-hard-winner":
            labels = torch.tensor([[choice["id"] for choice in row["request"]["choices"]].index(row["label"])
                                   for row in batch])
            loss = torch.nn.functional.cross_entropy(logits, labels)
            detail = {}
        elif arm == "O-full-information":
            rewards = torch.zeros_like(logits)
            for index, row in enumerate(batch):
                values = outcome_rewards(row); rewards[index, :len(values)] = torch.tensor(values)
            loss = full_information_loss(logits, rewards, counts); detail = {}
        else:
            selected = [log_index[(record["epoch"], row["id"])] for row in batch]
            actions = torch.tensor([item["actions"] for item in selected], dtype=torch.long)
            observed = torch.tensor([item["rewards"] for item in selected], dtype=torch.float32)
            propensities = torch.tensor([item["behavior_probabilities"] for item in selected],
                                        dtype=torch.float32)
            if arm == "R-ips-loo":
                loss = ips_loo_loss(logits, actions, observed, propensities, counts); detail = {}
            else:
                reward_logits, _ = padded_logits(reward_head, feature_rows)
                estimates = torch.sigmoid(reward_logits)
                loss, detail = doubly_robust_loss(
                    logits, estimates, actions, observed, propensities, counts,
                    protocol["training"]["reward_model_weight"])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite outcome loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, protocol["training"]["gradient_clip"])
        if not torch.isfinite(grad_norm):
            raise ValueError("nonfinite outcome gradient")
        optimizer.step(); updates += 1
        curve.append({"epoch": record["epoch"], "update": updates, "loss": float(loss.detach()),
                      "gradient_norm": float(grad_norm),
                      **{key: float(value) for key, value in detail.items()}})
    splits, predictions, evaluations, split_logits = {}, {}, {}, {}
    for split in ("train", "validation"):
        selected = sorted((row for row in rows if row["split"] == split), key=lambda row: row["id"])
        values = predict(policy, selected, features, 64)
        split_logits[split] = values
        metric, records = outcome_metrics(selected, values)
        splits[split] = selected; predictions[split] = records; evaluations[split] = metric
    if arm in ("R-ips-loo", "D-doubly-robust"):
        final_by_id = {row["id"]: values for row, values in zip(splits["train"], split_logits["train"])}
        weights = []
        for record in logs:
            probabilities = torch.softmax(torch.tensor(final_by_id[record["row"]]), -1)
            weights.extend(float(probabilities[action] / propensity) for action, propensity in zip(
                record["actions"], record["behavior_probabilities"]))
        ordered = sorted(weights)
        evaluations["logged_feedback"] = {
            "queries": len(weights), "importance_weight_mean": sum(weights) / len(weights),
            "importance_weight_p95": ordered[int(.95 * (len(ordered) - 1))],
            "importance_weight_max": max(weights),
            "importance_effective_sample_ratio": (sum(weights) ** 2 /
                                                   (sum(value * value for value in weights) * len(weights))),
        }
    if reward_head:
        squared = []
        reward_head.eval()
        with torch.no_grad():
            validation = splits["validation"]
            for start in range(0, len(validation), 64):
                batch = validation[start:start + 64]
                reward_logits, _ = padded_logits(
                    reward_head, [features[row["id"]] for row in batch])
                estimates = torch.sigmoid(reward_logits)
                for index, row in enumerate(batch):
                    truth = outcome_rewards(row)
                    squared.extend((float(estimates[index, candidate]) - reward) ** 2
                                   for candidate, reward in enumerate(truth))
        evaluations["reward_model"] = {"validation_full_table_mse": sum(squared) / len(squared),
                                         "validation_candidates": len(squared)}
    tensors = {"policy." + key: value.detach().cpu().contiguous()
               for key, value in policy.state_dict().items()}
    if reward_head:
        tensors.update({"reward." + key: value.detach().cpu().contiguous()
                        for key, value in reward_head.state_dict().items()})
    save_file(tensors, str(temporary / "heads.safetensors"))
    write_json(temporary / "curve.json", curve)
    write_json(temporary / "evaluation.json", evaluations)
    write_json(temporary / "validation-predictions.json", predictions["validation"])
    run = {"status": "complete", "arm": arm, "seed": seed, "updates": updates,
           "initial_policy_head_sha256": initial_hash, "final_policy_head_sha256": parameter_sha256(policy),
           "training_plan_sha256": plan_hash, "behavior_log_sha256": log_hash,
           "behavior_log_visible_to_training": arm in ("R-ips-loo", "D-doubly-robust"),
           "complete_reward_table_visible_to_training": arm in ("H-hard-winner", "O-full-information"),
           "seconds": time.monotonic() - started}
    write_json(temporary / "run.json", run)
    names = ("heads.safetensors", "curve.json", "evaluation.json",
             "validation-predictions.json", "run.json")
    write_json(temporary / "checksums.json", {name: digest((temporary / name).read_bytes()) for name in names})
    temporary.replace(target)
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/executable-outcome-screen-v1.json")
    parser.add_argument("--feature-root", default="cache/mbpp-outcome-features-v2")
    parser.add_argument("--output", default="runs/executable-outcome-screen-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path, output = root / args.protocol, root / args.output
    protocol = json.loads(protocol_path.read_text())
    if tuple(protocol["arms"]) != ARMS:
        raise ValueError("frozen arm order differs from implementation")
    data_path = root / protocol["data"]["path"]
    if digest(data_path.read_bytes()) != protocol["data"]["sha256"]:
        raise ValueError("frozen outcome data changed")
    rows = load_rows(data_path)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"state": "training", "official_test_parsed": False})
    controls = {}
    for endpoint_record in protocol["representation_endpoints"]:
        endpoint_id, seed = endpoint_record["id"], endpoint_record["seed"]
        store = FeatureStore(root / args.feature_root / endpoint_id)
        if (store.manifest["frozen"].get("protocol_sha256") != digest(protocol_path.read_bytes()) or
                store.manifest["frozen"].get("endpoint_weights_sha256") != endpoint_record["weights_sha256"]):
            raise ValueError("feature cache differs from outcome protocol")
        verify_feature_identity(rows, store); features = store.load_all()
        endpoint = root / endpoint_record["path"]
        initial_head = build_head(
            endpoint, store.manifest["hidden_size"], protocol["training"]["policy_head_width"],
            protocol["training"]["policy_head_dropout"])
        initial_evaluation = {}
        for split in ("train", "validation"):
            selected = sorted((row for row in rows if row["split"] == split), key=lambda row: row["id"])
            values = predict(initial_head, selected, features, 64)
            initial_evaluation[split] = outcome_metrics(selected, values)[0]
        initial_path = output / endpoint_id / "initial-evaluation.json"
        if initial_path.exists() and json.loads(initial_path.read_text()) != initial_evaluation:
            raise ValueError("initial evaluation changed")
        write_json(initial_path, initial_evaluation)
        plan = make_plan(rows, protocol, seed)
        plan_path = output / endpoint_id / "training-plan.jsonl"
        plan_hash = write_jsonl(plan_path, plan) if not plan_path.exists() else digest(plan_path.read_bytes())
        if plan_hash != digest("".join(canonical(item) + "\n" for item in plan).encode()):
            raise ValueError("existing training plan changed")
        logs = make_behavior_logs(rows, features, initial_head, protocol, seed)
        log_path = output / endpoint_id / "behavior-log.jsonl"
        log_hash = write_jsonl(log_path, logs) if not log_path.exists() else digest(log_path.read_bytes())
        if log_hash != digest("".join(canonical(item) + "\n" for item in logs).encode()):
            raise ValueError("existing behavior log changed")
        for arm in ARMS:
            target = output / endpoint_id / arm
            run = train_arm(target, arm, rows, features, initial_head, plan, plan_hash,
                            logs, log_hash, protocol, seed)
            control = (run["initial_policy_head_sha256"], run["training_plan_sha256"])
            if endpoint_id in controls and controls[endpoint_id] != control:
                raise ValueError("matched initial policy or training plan differs")
            controls[endpoint_id] = control
            print(canonical({"event": "arm_complete", "endpoint": endpoint_id, "arm": arm}), flush=True)
    write_json(output / "status.json", {"state": "complete", "official_test_parsed": False,
                                         "endpoints": len(protocol["representation_endpoints"]),
                                         "arms": list(ARMS)})
    print(canonical({"event": "complete", "official_test_parsed": False}))


if __name__ == "__main__":
    main()
