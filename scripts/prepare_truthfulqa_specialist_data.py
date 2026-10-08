#!/usr/bin/env python3
"""Prepare a deterministic, answer-blind TruthfulQA specialist split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, validate_request, write_json


TASK = "Choose the answer that is truthful and directly answers the question."
REQUIRED_COLUMNS = {
    "Type", "Category", "Question", "Best Answer", "Best Incorrect Answer",
    "Correct Answers", "Incorrect Answers", "Source",
}


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_raw(raw: dict) -> None:
    if set(raw) != REQUIRED_COLUMNS:
        raise ValueError("TruthfulQA columns changed")
    required = ("Question", "Best Answer", "Best Incorrect Answer", "Category", "Type")
    if any(not isinstance(raw[name], str) or not raw[name].strip() for name in required):
        raise ValueError("TruthfulQA required value is empty")
    if raw["Best Answer"].strip() == raw["Best Incorrect Answer"].strip():
        raise ValueError("TruthfulQA binary candidates must differ")


def selection_key(raw: dict) -> str:
    """Hash the input pair without retaining which candidate is correct."""
    validate_raw(raw)
    return digest({"question": raw["Question"].strip(),
                   "candidates": sorted((raw["Best Answer"].strip(),
                                         raw["Best Incorrect Answer"].strip()))})


def request_and_label(raw: dict) -> tuple[dict, str]:
    validate_raw(raw)
    question = raw["Question"].strip()
    correct = raw["Best Answer"].strip()
    values = sorted((correct, raw["Best Incorrect Answer"].strip()),
                    key=lambda answer: digest({"question": question, "candidate": answer}))
    choices = [{"id": f"option_{index}", "description": answer}
               for index, answer in enumerate(values)]
    request = {"task": TASK, "context": question, "choices": choices}
    validate_request(request)
    label = next(choice["id"] for choice in choices if choice["description"] == correct)
    return request, label


def make_case(raw: dict, split: str, revision: str, source_sha256: str) -> dict:
    if split not in ("train", "validation"):
        raise ValueError("TruthfulQA specialist accepts train or validation")
    request, label = request_and_label(raw)
    key = selection_key(raw)
    return {
        "id": digest({"source": "truthfulqa", "selection_key": key,
                      "revision": revision, "split": split}),
        "group_id": "truthfulqa:" + key,
        "source": "truthfulqa",
        "family": "truthfulness_binary",
        "split": split,
        "evaluation_regime": ("seen_task_training" if split == "train"
                              else "seen_task_new_examples"),
        "decision_type": "boolean",
        "request": request,
        "label": label,
        "provenance": [{
            "origin": "TruthfulQA official public repository",
            "source_url": "https://github.com/sylinrl/TruthfulQA",
            "revision": revision,
            "source_file_sha256": source_sha256,
            "selection_key": key,
            "category": raw["Category"].strip(),
            "type": raw["Type"].strip(),
            "license": "Apache-2.0",
            "transformation": ("used the official Best Answer and Best Incorrect Answer; "
                               "removed answer roles from the input; applied deterministic "
                               "answer-blind split and candidate permutation"),
        }],
    }


def split_rows(rows: list[dict], train_rows: int, validation_rows: int):
    keyed = []
    seen = set()
    for raw in rows:
        key = selection_key(raw)
        if key in seen:
            raise ValueError("Duplicate TruthfulQA answer-blind input")
        seen.add(key); keyed.append((key, raw))
    keyed.sort(key=lambda value: value[0])
    if train_rows < 1 or validation_rows < 1 or train_rows + validation_rows > len(keyed):
        raise ValueError("Invalid TruthfulQA split sizes")
    return (keyed[:train_rows], keyed[train_rows:train_rows + validation_rows],
            keyed[train_rows + validation_rows:])


def download(config: dict, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name, record in config["files"].items():
        target = cache / name
        with urllib.request.urlopen(record["url"], timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Downloaded TruthfulQA source differs: " + name)
        target.write_bytes(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache/truthfulqa-specialist-source")
    parser.add_argument("--source-config", default="configs/truthfulqa-specialist-source.json")
    parser.add_argument("--output", default="runs/truthfulqa-specialist-v1")
    parser.add_argument("--train-rows", type=int, default=512)
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache, output = root / args.cache, root / args.output
    config_path = root / args.source_config
    config = json.loads(config_path.read_text())
    if config.get("license") != "Apache-2.0" or len(config.get("revision", "")) != 40:
        raise ValueError("Unexpected TruthfulQA source identity")
    if args.download_only:
        download(config, cache)
        print(canonical({"event": "downloaded", "revision": config["revision"],
                         "files": len(config["files"])}))
        return
    if output.exists():
        raise ValueError("Use a fresh TruthfulQA specialist data directory")
    for name, record in config["files"].items():
        if file_sha256(cache / name) != record["sha256"]:
            raise ValueError("TruthfulQA source file changed: " + name)
    with (cache / "TruthfulQA.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) != config["files"]["TruthfulQA.csv"]["rows"]:
        raise ValueError("TruthfulQA source row count changed")
    train, validation, reserve = split_rows(rows, args.train_rows, args.validation_rows)
    source_hash = config["files"]["TruthfulQA.csv"]["sha256"]
    cases = ([make_case(raw, "train", config["revision"], source_hash) for _, raw in train] +
             [make_case(raw, "validation", config["revision"], source_hash)
              for _, raw in validation])
    output.mkdir(parents=True)
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(canonical(row) + "\n" for row in cases))
    loaded = load_rows(cases_path)
    by_split = {split: dict(Counter(row["label"] for row in loaded if row["split"] == split))
                for split in ("train", "validation")}
    categories = {
        split: dict(Counter(row["provenance"][0]["category"] for row in loaded
                            if row["split"] == split))
        for split in ("train", "validation")
    }
    write_json(output / "protocol.json", {
        "study": "truthfulqa-specialist-v1",
        "role": "public-family specialist train and consumed validation",
        "source_config": args.source_config,
        "source_config_sha256": digest(config_path.read_bytes()),
        "revision": config["revision"], "license": config["license"],
        "source_rows": len(rows), "train_rows": len(train),
        "validation_rows": len(validation), "unused_reserve_rows": len(reserve),
        "train_selection_sha256": digest([key for key, _ in train]),
        "validation_selection_sha256": digest([key for key, _ in validation]),
        "reserve_selection_sha256": digest([key for key, _ in reserve]),
        "label_positions": by_split, "categories": categories,
        "answer_blind_split": True, "answer_blind_candidate_order": True,
        "cases_sha256": digest(cases_path.read_bytes()),
        "preparation_script_sha256": digest(Path(__file__).read_bytes()),
        "release_policy": config["release_policy"],
    })
    print(canonical({"event": "prepared", "rows": len(loaded),
                     "splits": dict(Counter(row["split"] for row in loaded)),
                     "cases_sha256": digest(cases_path.read_bytes())}))


if __name__ == "__main__":
    main()
