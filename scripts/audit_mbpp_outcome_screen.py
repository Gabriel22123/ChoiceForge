#!/usr/bin/env python3
"""Verify the decoder-admitted outcome screen data against the frozen protocol."""
import argparse
import json
from collections import Counter
from pathlib import Path

from decision_model.core import digest, load_rows, outcome_rewards, write_json
from decision_model.decoder_model import verify_local_decoder_base
from decision_model.prompting import render_decoder_admission, render_decoder_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/executable-outcome-screen-v1.json")
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--output", default="docs/evidence/mbpp-outcome-screen-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path, base = root / args.protocol, root / args.base_path
    protocol = json.loads(protocol_path.read_text()); frozen = protocol["data"]
    data_path, manifest_path = root / frozen["path"], root / frozen["manifest_path"]
    if digest(data_path.read_bytes()) != frozen["sha256"] or digest(
            manifest_path.read_bytes()) != frozen["manifest_sha256"]:
        raise ValueError("screen data or manifest differs from protocol")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_sha256"] != frozen["source_outcome_data_sha256"]:
        raise ValueError("screen parent data hash differs")
    rows = load_rows(data_path)
    if Counter(row["split"] for row in rows) != Counter(
            train=frozen["train_rows"], validation=frozen["validation_rows"]):
        raise ValueError("screen split counts differ")
    verify_local_decoder_base(base)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    admissions, paths = [], []
    for row in rows:
        admissions.append(len(tokenizer.encode(render_decoder_admission(row["request"]),
                                               add_special_tokens=True)))
        paths.extend(len(tokenizer.encode(render_decoder_candidate(row["request"], index),
                                          add_special_tokens=True))
                     for index in range(len(row["request"]["choices"])))
    if max(admissions + paths) > frozen["request_token_limit"]:
        raise ValueError("screen contains an over-limit request")
    rewards = [reward for row in rows for reward in outcome_rewards(row)]
    evidence = {
        "format_version": 1,
        "protocol_sha256": digest(protocol_path.read_bytes()),
        "data_sha256": frozen["sha256"],
        "manifest_sha256": frozen["manifest_sha256"],
        "source_outcome_data_sha256": frozen["source_outcome_data_sha256"],
        "rows": len(rows),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "candidates": len(rewards),
        "mean_candidate_reward": sum(rewards) / len(rewards),
        "request_admission_max": max(admissions),
        "candidate_path_max": max(paths),
        "request_token_limit": frozen["request_token_limit"],
        "excluded_for_request_length": manifest["excluded"]["request_admission_over_limit"],
        "official_test_rows_present": False,
    }
    write_json(root / args.output, evidence)
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
