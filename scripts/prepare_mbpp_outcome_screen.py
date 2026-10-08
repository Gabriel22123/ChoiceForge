#!/usr/bin/env python3
"""Apply the frozen decoder request gate to executable MBPP outcome rows."""
import argparse
import json
from collections import Counter
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base
from decision_model.prompting import render_decoder_admission, render_decoder_candidate


def admitted(rows, tokenizer, max_tokens):
    kept, excluded, admission_lengths, path_lengths = [], Counter(), [], []
    for row in rows:
        admission = len(tokenizer.encode(render_decoder_admission(row["request"]),
                                         add_special_tokens=True))
        paths = [len(tokenizer.encode(render_decoder_candidate(row["request"], index),
                                      add_special_tokens=True))
                 for index in range(len(row["request"]["choices"]))]
        admission_lengths.append(admission); path_lengths.extend(paths)
        if admission > max_tokens:
            excluded["request_admission_over_limit"] += 1
        elif max(paths) > max_tokens:
            excluded["candidate_path_over_limit"] += 1
        else:
            kept.append(row)
    return kept, excluded, admission_lengths, path_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/mbpp-outcomes-v1/data.jsonl")
    parser.add_argument("--output", default="data/mbpp-outcome-screen-v1")
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()
    if args.max_tokens < 1:
        raise ValueError("max token count must be positive")
    root = Path(__file__).resolve().parents[1]
    source, output, base = root / args.source, root / args.output, root / args.base_path
    if output.exists():
        raise ValueError("output directory already exists")
    verify_local_decoder_base(base)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False)
    rows = load_rows(source)
    kept, excluded, admission_lengths, path_lengths = admitted(rows, tokenizer, args.max_tokens)
    if not kept:
        raise ValueError("decoder gate excluded every outcome row")
    output.mkdir(parents=True)
    data_path = output / "data.jsonl"
    data_path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in kept))
    verified = load_rows(data_path)
    ordered_admission, ordered_paths = sorted(admission_lengths), sorted(path_lengths)
    manifest = {
        "format_version": 1,
        "source_path": args.source,
        "source_sha256": digest(source.read_bytes()),
        "data_sha256": digest(data_path.read_bytes()),
        "base_lock_sha256": digest((root / "src/decision_model/decoder_base.lock.json").read_bytes()),
        "tokenizer_revision": "96a57cd572a02506b4500f54427dca24970c1bac",
        "max_tokens": args.max_tokens,
        "rows": len(verified),
        "splits": dict(sorted(Counter(row["split"] for row in verified).items())),
        "excluded": dict(sorted(excluded.items())),
        "token_lengths": {
            "request_admission_max_all": max(admission_lengths),
            "request_admission_p50_all": ordered_admission[len(ordered_admission) // 2],
            "request_admission_p95_all": ordered_admission[int(len(ordered_admission) * .95)],
            "candidate_path_max_all": max(path_lengths),
        },
        "official_test_rows_present": False,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
