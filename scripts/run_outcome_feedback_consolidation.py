#!/usr/bin/env python3
"""Replay fixed chosen-action logs and compare feedback consolidation."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, outcome_rewards, write_json
from decision_model.feature_screen import CandidateHead, FeatureStore, padded_logits, parameter_sha256
from decision_model.outcome_evaluation import outcome_metrics
from decision_model.outcome_feedback import consolidate_feedback
from decision_model.outcome_objective import doubly_robust_loss, ips_loo_loss, optimal_set_loss


ARMS = ("R-ips-loo-replay", "D-doubly-robust-replay", "A-consolidated-optimal-set")


def build_endpoint_head(endpoint, hidden_size, width):
    from safetensors.torch import load_file
    head = CandidateHead.build(hidden_size, width, 0.)
    state = load_file(str(Path(endpoint) / "decision.safetensors"), device="cpu")
    head.load_state_dict({key.removeprefix("head."): value for key, value in state.items()
                          if key.startswith("head.")}, strict=True)
    return head.float()


def reward_head(hidden_size, width, seed):
    import torch
    torch.manual_seed(seed)
    head = CandidateHead.build(hidden_size, width, 0.).float()
    with torch.no_grad(): head[-1].weight.zero_(); head[-1].bias.zero_()
    return head


def predict(head, rows, features):
    import torch
    head.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            logits, _ = padded_logits(head, [features[row["id"]] for row in batch])
            values.extend([value[:len(row["request"]["choices"])]
                           for row, value in zip(batch, logits.tolist())])
    return values


def make_plan(train, config, seed):
    rng = random.Random(f'{config["study"]}:seed{seed}:row-plan')
    plan = []
    for epoch in range(1, config["optimization"]["epochs"] + 1):
        ordered = sorted(train, key=lambda row: row["id"]); rng.shuffle(ordered)
        size = config["optimization"]["batch_size"]
        for start in range(0, len(ordered), size):
            plan.append({"epoch": epoch, "rows": [row["id"] for row in ordered[start:start + size]]})
    return plan


def evaluate(head, rows, features):
    values = predict(head, rows, features)
    return outcome_metrics(rows, values)[0]


def train_arm(target, arm, rows, features, initial, plan, plan_hash, logs, log_hash,
              consolidated, coverage, config, seed):
    if target.exists():
        checksums = json.loads((target / "checksums.json").read_text())
        if all(digest((target / name).read_bytes()) == value for name, value in checksums.items()):
            return json.loads((target / "run.json").read_text())
        raise ValueError("existing consolidation endpoint changed")
    import torch
    from safetensors.torch import save_file
    temporary = target.with_name(target.name + ".building")
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    hidden = features[next(iter(features))].shape[1]
    width = initial[1]
    policy = CandidateHead.build(hidden, width, 0.).float(); policy.load_state_dict(initial[0])
    q_head = reward_head(hidden, width, seed + 1907) if arm == "D-doubly-robust-replay" else None
    parameters = list(policy.parameters()) + (list(q_head.parameters()) if q_head else [])
    optimizer = torch.optim.AdamW(parameters, lr=config["optimization"]["learning_rate"],
                                  weight_decay=config["optimization"]["weight_decay"])
    by_id = {row["id"]: row for row in rows}
    log_index = {(record["epoch"], record["row"]): record for record in logs}
    train = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["id"])
    validation = sorted((row for row in rows if row["split"] == "validation"), key=lambda row: row["id"])
    checkpoints = set(config["optimization"]["checkpoint_epochs_for_diagnostics"])
    curve = [{"epoch": 0, "train": evaluate(policy, train, features),
              "validation": evaluate(policy, validation, features)}]
    updates = examples = 0
    current_epoch = 0; losses = []
    for record in plan:
        if record["epoch"] != current_epoch:
            if current_epoch in checkpoints and current_epoch:
                curve.append({"epoch": current_epoch, "mean_training_loss": sum(losses) / len(losses),
                              "train": evaluate(policy, train, features),
                              "validation": evaluate(policy, validation, features)})
            current_epoch = record["epoch"]; losses = []
        batch = [by_id[row_id] for row_id in record["rows"]]
        if arm == "A-consolidated-optimal-set":
            batch = [row for row in batch if row["id"] in consolidated]
            if not batch:
                continue
        policy.train(True); optimizer.zero_grad(set_to_none=True)
        feature_rows = [features[row["id"]] for row in batch]
        logits, _ = padded_logits(policy, feature_rows)
        counts = [len(row["request"]["choices"]) for row in batch]
        if arm == "A-consolidated-optimal-set":
            rewards = torch.zeros_like(logits)
            for index, row in enumerate(batch):
                values = consolidated[row["id"]]; rewards[index, :len(values)] = torch.tensor(values)
            loss = optimal_set_loss(logits, rewards, counts)
        else:
            log_epoch = (record["epoch"] - 1) % config["feedback"]["source_log_epochs"] + 1
            selected = [log_index[(log_epoch, row["id"])] for row in batch]
            actions = torch.tensor([item["actions"] for item in selected], dtype=torch.long)
            observed = torch.tensor([item["rewards"] for item in selected], dtype=torch.float32)
            propensities = torch.tensor([item["behavior_probabilities"] for item in selected],
                                        dtype=torch.float32)
            if arm == "R-ips-loo-replay":
                loss = ips_loo_loss(logits, actions, observed, propensities, counts)
            else:
                reward_logits, _ = padded_logits(q_head, feature_rows)
                loss = doubly_robust_loss(logits, torch.sigmoid(reward_logits), actions,
                                          observed, propensities, counts, 1.)[0]
        loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, config["optimization"]["gradient_clip"])
        optimizer.step(); updates += 1; examples += len(batch); losses.append(float(loss.detach()))
    if current_epoch in checkpoints:
        curve.append({"epoch": current_epoch, "mean_training_loss": sum(losses) / len(losses),
                      "train": evaluate(policy, train, features),
                      "validation": evaluate(policy, validation, features)})
    tensors = {"policy." + key: value.detach().contiguous() for key, value in policy.state_dict().items()}
    if q_head:
        tensors.update({"reward." + key: value.detach().contiguous() for key, value in q_head.state_dict().items()})
    save_file(tensors, str(temporary / "heads.safetensors"))
    write_json(temporary / "curve.json", curve)
    write_json(temporary / "evaluation.json", {"train": evaluate(policy, train, features),
                                                 "validation": evaluate(policy, validation, features)})
    pristine = CandidateHead.build(hidden, width, 0.).float(); pristine.load_state_dict(initial[0])
    run = {"status": "complete", "arm": arm, "seed": seed, "updates": updates,
           "training_examples": examples, "initial_policy_head_sha256": parameter_sha256(pristine),
           "final_policy_head_sha256": parameter_sha256(policy),
           "training_plan_sha256": plan_hash, "behavior_log_sha256": log_hash,
           "new_environment_queries": 0, "coverage": coverage,
           "complete_reward_table_visible_to_training": False,
           "training_feedback": ("consolidated_observed_rewards_only" if arm.startswith("A-")
                                 else "chosen_action_records_only")}
    write_json(temporary / "run.json", run)
    names = ("heads.safetensors", "curve.json", "evaluation.json", "run.json")
    write_json(temporary / "checksums.json", {name: digest((temporary / name).read_bytes()) for name in names})
    temporary.replace(target)
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/executable-outcome-replay-followup-v1.json")
    parser.add_argument("--output", default="runs/executable-outcome-feedback-consolidation-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, output = root / args.config, root / args.output
    config = json.loads(config_path.read_text())
    source_protocol = root / config["source_protocol"]["path"]
    if digest(source_protocol.read_bytes()) != config["source_protocol"]["sha256"]:
        raise ValueError("source protocol changed")
    data_path = root / config["data"]["path"]
    if digest(data_path.read_bytes()) != config["data"]["sha256"]:
        raise ValueError("outcome screen data changed")
    rows = load_rows(data_path); train = [row for row in rows if row["split"] == "train"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"state": "training", "official_test_parsed": False,
                                         "validation_is_fresh_confirmation": False})
    for endpoint in config["endpoints"]:
        log_path = root / endpoint["behavior_log"]
        if digest(log_path.read_bytes()) != endpoint["behavior_log_sha256"]:
            raise ValueError("behavior log changed")
        logs = [json.loads(line) for line in log_path.read_text().splitlines() if line]
        minimal = [{"id": row["id"], "request": row["request"]} for row in train]
        consolidated, incomplete, coverage = consolidate_feedback(minimal, logs)
        coverage["incomplete_row_ids_sha256"] = digest(sorted(incomplete))
        store = FeatureStore(root / endpoint["feature_cache"]); features = store.load_all()
        checkpoint = root / endpoint["checkpoint"]
        initial_head = build_endpoint_head(checkpoint, store.manifest["hidden_size"], 256)
        initial = ({key: value.detach().clone() for key, value in initial_head.state_dict().items()}, 256)
        plan = make_plan(train, config, endpoint["seed"])
        plan_path = output / endpoint["id"] / "training-plan.jsonl"
        raw = "".join(canonical(item) + "\n" for item in plan)
        if plan_path.exists() and plan_path.read_text() != raw:
            raise ValueError("training plan changed")
        plan_path.parent.mkdir(parents=True, exist_ok=True); plan_path.write_text(raw)
        plan_hash = digest(raw.encode())
        controls = set()
        for arm in ARMS:
            run = train_arm(output / endpoint["id"] / arm, arm, rows, features, initial,
                            plan, plan_hash, logs, endpoint["behavior_log_sha256"],
                            consolidated, coverage, config, endpoint["seed"])
            controls.add((run["initial_policy_head_sha256"], run["training_plan_sha256"]))
            print(canonical({"event": "feedback_arm_complete", "endpoint": endpoint["id"],
                             "arm": arm}), flush=True)
        if len(controls) != 1:
            raise ValueError("matched feedback controls differ")
    write_json(output / "status.json", {"state": "complete", "official_test_parsed": False,
                                         "validation_is_fresh_confirmation": False,
                                         "endpoints": len(config["endpoints"]), "arms": list(ARMS)})
    print(canonical({"event": "complete", "official_test_parsed": False}))


if __name__ == "__main__":
    main()
