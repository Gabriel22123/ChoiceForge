#!/usr/bin/env python3
"""Diagnose whether frozen candidate features can fit executable rewards at all."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, outcome_rewards, write_json
from decision_model.feature_screen import CandidateHead, FeatureStore, padded_logits, parameter_sha256
from decision_model.outcome_evaluation import outcome_metrics
from decision_model.outcome_objective import full_information_loss, optimal_set_loss


ARMS = ("hard-first-max", "optimal-set", "expected-reward", "full-reward-regression")


def build_endpoint_head(endpoint, hidden_size, width):
    from safetensors.torch import load_file
    head = CandidateHead.build(hidden_size, width, 0.)
    state = load_file(str(Path(endpoint) / "decision.safetensors"), device="cpu")
    head.load_state_dict({key.removeprefix("head."): value for key, value in state.items()
                          if key.startswith("head.")}, strict=True)
    return head.float()


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


def evaluate(head, rows, features, reward_regression=False):
    import torch
    logits = predict(head, rows, features)
    result = outcome_metrics(rows, logits)[0]
    if reward_regression:
        squared = []
        for row, values in zip(rows, logits):
            estimates = torch.sigmoid(torch.tensor(values)).tolist()
            squared.extend((estimate - reward) ** 2
                           for estimate, reward in zip(estimates, outcome_rewards(row)))
        result["full_reward_mse"] = sum(squared) / len(squared)
    return result


def train_arm(arm, initial, rows, features, epochs, batch_size, learning_rate, seed, checkpoints):
    import torch
    head = CandidateHead.build(features[rows[0]["id"]].shape[1], initial[1], 0.).float()
    head.load_state_dict(initial[0])
    if arm == "full-reward-regression":
        with torch.no_grad():
            head[-1].weight.zero_(); head[-1].bias.zero_()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=.01)
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    curve = [{"epoch": 0, "train": evaluate(head, train, features, arm == "full-reward-regression"),
              "validation": evaluate(head, validation, features, arm == "full-reward-regression")}]
    rng = random.Random(f"outcome-learnability:{seed}:{arm}")
    for epoch in range(1, epochs + 1):
        ordered = train.copy(); rng.shuffle(ordered); losses = []
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start:start + batch_size]
            head.train(True); optimizer.zero_grad(set_to_none=True)
            logits, _ = padded_logits(head, [features[row["id"]] for row in batch])
            counts = [len(row["request"]["choices"]) for row in batch]
            rewards = torch.zeros_like(logits)
            for index, row in enumerate(batch):
                values = outcome_rewards(row); rewards[index, :len(values)] = torch.tensor(values)
            if arm == "hard-first-max":
                labels = torch.tensor([[choice["id"] for choice in row["request"]["choices"]].index(
                    row["label"]) for row in batch])
                loss = torch.nn.functional.cross_entropy(logits, labels)
            elif arm == "optimal-set":
                loss = optimal_set_loss(logits, rewards, counts)
            elif arm == "expected-reward":
                loss = full_information_loss(logits, rewards, counts)
            else:
                estimates = torch.sigmoid(logits)
                mask = torch.arange(logits.shape[1])[None, :] < torch.tensor(counts)[:, None]
                loss = torch.nn.functional.mse_loss(estimates[mask], rewards[mask])
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in checkpoints:
            curve.append({"epoch": epoch, "mean_training_loss": sum(losses) / len(losses),
                          "train": evaluate(head, train, features, arm == "full-reward-regression"),
                          "validation": evaluate(head, validation, features, arm == "full-reward-regression")})
    return curve, parameter_sha256(head)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/executable-outcome-screen-v1.json")
    parser.add_argument("--feature-root", default="cache/mbpp-outcome-features-v2")
    parser.add_argument("--endpoint-id", default="seed42-decoder")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", default="docs/evidence/outcome-learnability-seed42-v1.json")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / args.protocol; protocol = json.loads(protocol_path.read_text())
    endpoint_record = next((item for item in protocol["representation_endpoints"]
                            if item["id"] == args.endpoint_id), None)
    if endpoint_record is None:
        raise ValueError("unknown endpoint")
    data_path = root / protocol["data"]["path"]
    if digest(data_path.read_bytes()) != protocol["data"]["sha256"]:
        raise ValueError("frozen data changed")
    rows = load_rows(data_path)
    store = FeatureStore(root / args.feature_root / args.endpoint_id)
    if store.manifest["frozen"]["protocol_sha256"] != digest(protocol_path.read_bytes()):
        raise ValueError("feature protocol changed")
    features = store.load_all(); hidden = store.manifest["hidden_size"]
    width = protocol["training"]["policy_head_width"]
    initial_head = build_endpoint_head(root / endpoint_record["path"], hidden, width)
    initial = ({key: value.detach().clone() for key, value in initial_head.state_dict().items()}, width)
    checkpoints = {1, 5, 20, 50, 100, args.epochs}
    results = {}
    for arm in ARMS:
        curve, final_hash = train_arm(
            arm, initial, rows, features, args.epochs, protocol["training"]["batch_size"],
            protocol["training"]["learning_rate"], endpoint_record["seed"], checkpoints)
        results[arm] = {"curve": curve, "final_head_sha256": final_hash}
        print(canonical({"event": "learnability_arm", "arm": arm,
                         "train_regret": curve[-1]["train"]["mean_decision_regret"],
                         "validation_regret": curve[-1]["validation"]["mean_decision_regret"]}), flush=True)
    evidence = {"kind": "outcome_learnability_diagnostic", "official_test_parsed": False,
                "protocol_sha256": digest(protocol_path.read_bytes()),
                "feature_manifest_sha256": store.manifest_sha256,
                "endpoint": endpoint_record, "epochs": args.epochs,
                "checkpoints": sorted(checkpoints), "results": results}
    write_json(root / args.output, evidence)
    print(canonical({"event": "complete", "official_test_parsed": False}))


if __name__ == "__main__":
    main()
