#!/usr/bin/env python3
"""Prepare public executable outcome tables; excludes official test by default."""
import argparse
import hashlib
import json
from pathlib import Path

from decision_model.code_outcomes import prepare_mbpp_outcomes
from decision_model.core import load_rows, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Pinned upstream MBPP JSON or JSONL")
    parser.add_argument("--source-config", required=True, help="Reviewed source and split lock")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--include-official-test", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.source_config)
    config = json.loads(config_path.read_text())
    source_sha256 = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
    if config.get("format_version") != 1 or config.get("source", {}).get("sha256") != source_sha256:
        raise ValueError("source does not match the reviewed source config")
    if config.get("source", {}).get("license") != "CC-BY-4.0":
        raise ValueError("expected reviewed CC-BY-4.0 MBPP source")
    output = Path(args.output)
    if output.exists():
        raise ValueError("output directory already exists")
    allowed = ("train", "validation", "test") if args.include_official_test else ("train", "validation")
    rows, manifest = prepare_mbpp_outcomes(
        args.source, max_candidates=args.max_candidates,
        timeout_seconds=args.timeout_seconds, allowed_splits=allowed)
    manifest["source_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest["source_revision"] = config["source"]["revision"]
    if not rows:
        raise ValueError("no executable outcome rows were produced")
    output.mkdir(parents=True)
    data_path = output / "data.jsonl"
    data_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    load_rows(data_path)
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
