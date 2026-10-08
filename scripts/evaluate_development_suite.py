#!/usr/bin/env python3
"""Evaluate a checkpoint on an explicitly consumed development suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.judge_factory import make_judge, verify_base_for_config
from decision_model.prior_evaluation import collect_evidence_and_prior_logits
from decision_model.train import evaluate_rows
from release_evaluation import context_advantage


def evaluate(args):
    checkpoint, data, output = map(Path, (args.checkpoint, args.data, args.output))
    if output.exists():
        raise ValueError("Use a fresh development-evaluation directory")
    rows = load_rows(data)
    if not rows or any(row["split"] != "test" for row in rows):
        raise ValueError("Development suite must contain test rows only")
    config = json.loads((checkpoint / "model.json").read_text())
    verify_base_for_config(config, args.base_path)
    judge = make_judge(
        config, base_path=args.base_path, checkpoint=checkpoint, device=args.device)
    records, collection = collect_evidence_and_prior_logits(judge, rows)
    logits = [item["evidence_logits"] for item in records]
    raw, _ = evaluate_rows(rows, logits, 1.0)
    scaled, predictions = evaluate_rows(rows, logits, judge.temperature)
    advantage, advantage_rows = context_advantage(rows, records, judge.temperature)
    result = {
        "format_version": 1,
        "role": "consumed development diagnostic; not a fresh or release-eligible blind test",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": digest((checkpoint / "decision.safetensors").read_bytes()),
        "cases_sha256": digest(data.read_bytes()),
        "rows": len(rows),
        "temperature": judge.temperature,
        "raw": raw,
        "temperature_scaled": scaled,
        "context_advantage": advantage,
        "collection": collection,
        "training_or_release_selection_use": False,
        "multi_seed_screening_use": True,
    }
    output.mkdir(parents=True)
    write_json(output / "result.json", result)
    write_json(output / "logits.private.json", records)
    write_json(output / "predictions.private.json", predictions)
    write_json(output / "context-advantage.private.json", advantage_rows)
    write_json(output / "checksums.json", {
        name: digest((output / name).read_bytes()) for name in (
            "result.json", "logits.private.json", "predictions.private.json",
            "context-advantage.private.json")
    })
    print(json.dumps({"checkpoint": checkpoint.name, "rows": len(rows),
                      "raw_accuracy": raw["accuracy"]}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
