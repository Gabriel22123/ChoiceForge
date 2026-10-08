#!/usr/bin/env python3
"""Evaluate one locked endpoint on the sealed release suite and public contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.judge_factory import make_judge, verify_base_for_config
from decision_model.prior_evaluation import collect_evidence_and_prior_logits
from decision_model.train import evaluate_rows
from decision_model.typed_api import predict_typed, validate_typed_prediction
from release_evaluation import context_advantage


def read(path):
    return json.loads(Path(path).read_text())


def contract_payloads():
    return [
        {"type": "boolean", "task": "Determine whether the evidence satisfies the rule.",
         "context": "The submitted artifact has a valid signature.",
         "criteria": {"true": "The signature is valid.", "false": "The signature is invalid."}},
        {"type": "choice", "task": "Select the best supported action.",
         "context": "A service is healthy and the deployment checks passed.",
         "choices": [{"id": f"action_{index}", "description": f"Action {index}"}
                     for index in range(4)]},
        {"type": "score", "task": "Assign one ordered severity level.",
         "context": "A bounded contract audit case.",
         "levels": [{"id": f"level_{index}", "description": f"Severity level {index}"}
                    for index in range(10)]},
        {"type": "choice", "task": "Choose exactly one candidate from a large dynamic set.",
         "context": "Candidate 17 is explicitly identified as the applicable candidate.",
         "choices": [{"id": f"candidate_{index}",
                      "description": f"Candidate {index}; applicable only when named {index}."}
                     for index in range(32)]},
    ]


def audit_contract(judge):
    records = []
    for payload in contract_payloads():
        prediction = predict_typed(judge, payload)
        encoded = json.dumps(prediction, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
        validate_typed_prediction(decoded, payload)
        records.append({
            "type": payload["type"],
            "candidate_count": (2 if payload["type"] == "boolean" else
                                len(payload.get("choices", payload.get("levels", [])))),
            "exact_round_trip": decoded == prediction,
            "requires_review": decoded["requires_review"] is True,
            "unexpected_fields": [],
        })
    passed = all(item["exact_round_trip"] and item["requires_review"] and
                 not item["unexpected_fields"] for item in records)
    return {"cases": len(records), "passed": sum(
                item["exact_round_trip"] and item["requires_review"] and
                not item["unexpected_fields"] for item in records),
            "pass_rate": sum(
                item["exact_round_trip"] and item["requires_review"] and
                not item["unexpected_fields"] for item in records) / len(records),
            "all_passed": passed, "records": records}


def evaluate(args):
    checkpoint, data, output = map(Path, (args.checkpoint, args.data, args.output))
    locked = read(args.endpoint_manifest)
    name = checkpoint.name
    if locked.get("state") != "locked_before_blind" or name not in locked.get("endpoints", {}):
        raise ValueError("Endpoint is not in the pre-blind lock")
    blind_protocol = read(data.parent / "protocol.json")
    if (blind_protocol.get("prohibited_uses") != ["training", "calibration", "early stopping",
                                                   "endpoint selection", "hyperparameter selection"] or
            digest(data.read_bytes()) != blind_protocol.get("cases_sha256")):
        raise ValueError("Blind suite integrity differs")
    rows = load_rows(data)
    if len(rows) != blind_protocol["rows"] or any(row["split"] != "test" for row in rows):
        raise ValueError("Blind suite row contract differs")
    config = read(checkpoint / "model.json")
    verify_base_for_config(config, args.base_path)
    judge = make_judge(config, base_path=args.base_path, checkpoint=checkpoint, device=args.device)
    records, collection = collect_evidence_and_prior_logits(judge, rows)
    raw, _ = evaluate_rows(rows, [item["evidence_logits"] for item in records], 1.0)
    calibrated, predictions = evaluate_rows(
        rows, [item["evidence_logits"] for item in records], judge.temperature)
    advantage, advantage_rows = context_advantage(rows, records, judge.temperature)
    contract = audit_contract(judge)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "logits.private.json", records)
    write_json(output / "predictions.private.json", predictions)
    write_json(output / "context-advantage.private.json", advantage_rows)
    result = {
        "format_version": 1, "study": "release-candidate-v1", "endpoint": name,
        "checkpoint_sha256": digest((checkpoint / "decision.safetensors").read_bytes()),
        "endpoint_manifest_sha256": digest(Path(args.endpoint_manifest).read_bytes()),
        "blind_protocol_sha256": digest((data.parent / "protocol.json").read_bytes()),
        "cases_sha256": digest(data.read_bytes()), "rows": len(rows),
        "temperature": judge.temperature, "raw": raw, "temperature_scaled": calibrated,
        "context_advantage": advantage, "strict_typed_contract": contract,
        "collection": collection,
        "canary_release_exclusions": ["logits.private.json", "predictions.private.json",
                                       "context-advantage.private.json"],
        "training_or_selection_use": False,
    }
    write_json(output / "result.json", result)
    write_json(output / "checksums.json", {
        name: digest((output / name).read_bytes()) for name in
        ("result.json", "logits.private.json", "predictions.private.json",
         "context-advantage.private.json")})
    print(json.dumps({"endpoint": name, "rows": len(rows),
                      "accuracy": calibrated["accuracy"],
                      "contract_passed": contract["all_passed"]}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="runs/release-candidate-v1/blind-suite/cases.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint-manifest", default="runs/release-candidate-v1/locked-endpoints.json")
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
