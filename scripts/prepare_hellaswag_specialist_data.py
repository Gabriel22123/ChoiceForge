#!/usr/bin/env python3
"""Prepare deterministic, answer-blind HellaSwag specialist rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, validate_request, write_json


TASK = "Choose the most plausible continuation of the described situation."
FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
}
REQUIRED_FIELDS = {
    "ind", "activity_label", "ctx_a", "ctx_b", "ctx", "endings",
    "source_id", "split", "split_type", "label",
}


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_raw(raw: dict) -> None:
    if set(raw) != REQUIRED_FIELDS:
        raise ValueError("HellaSwag fields changed")
    if (not isinstance(raw["ind"], int) or
            any(not isinstance(raw[name], str) or not raw[name].strip()
                for name in ("activity_label", "ctx", "source_id", "split", "split_type")) or
            not isinstance(raw["endings"], list) or len(raw["endings"]) != 4 or
            any(not isinstance(value, str) or not value.strip() for value in raw["endings"]) or
            len({value.strip() for value in raw["endings"]}) != 4 or
            raw["label"] not in {"0", "1", "2", "3"}):
        raise ValueError("Malformed HellaSwag row")


def selection_key(raw: dict) -> str:
    """Use input text only; the source label cannot affect admission."""
    validate_raw(raw)
    return digest({
        "activity": raw["activity_label"].strip(),
        "context": raw["ctx"].strip(),
        "endings": sorted(value.strip() for value in raw["endings"]),
        "source_id": raw["source_id"].strip(),
    })


def request_and_label(raw: dict) -> tuple[dict, str]:
    validate_raw(raw)
    context = raw["ctx"].strip()
    correct = raw["endings"][int(raw["label"])].strip()
    ordered = sorted((value.strip() for value in raw["endings"]),
                     key=lambda value: digest({"context": context, "candidate": value}))
    choices = [{"id": f"option_{index}", "description": value}
               for index, value in enumerate(ordered)]
    request = {
        "task": TASK,
        "context": f"Activity: {raw['activity_label'].strip()}\nSituation: {context}",
        "choices": choices,
    }
    validate_request(request)
    return request, next(choice["id"] for choice in choices
                         if choice["description"] == correct)


def make_case(raw: dict, split: str, revision: str, source_hash: str) -> dict:
    request, label = request_and_label(raw)
    key = selection_key(raw)
    return {
        "id": digest({"source": "hellaswag", "selection_key": key,
                      "revision": revision, "split": split}),
        "group_id": "hellaswag:" + key,
        "source": "hellaswag",
        "family": "situational_continuation",
        "split": split,
        "evaluation_regime": ("seen_task_training" if split == "train"
                              else "seen_task_new_examples"),
        "decision_type": "choice",
        "request": request,
        "label": label,
        "provenance": [{
            "origin": "HellaSwag official public dataset via pinned Hugging Face conversion",
            "source_url": "https://github.com/rowanz/hellaswag",
            "revision": revision,
            "official_split": split,
            "source_id": raw["source_id"].strip(),
            "source_file_sha256": source_hash,
            "selection_key": key,
            "license": "MIT",
            "transformation": ("kept activity, context and all four endings; removed source "
                               "answer positions; selected and permuted candidates using "
                               "answer-blind hashes"),
        }],
    }


def select(rows: list[dict], count: int) -> list[tuple[str, dict]]:
    keyed, seen = [], set()
    for raw in rows:
        key = selection_key(raw)
        if key in seen:
            raise ValueError("Duplicate HellaSwag answer-blind input")
        seen.add(key); keyed.append((key, raw))
    keyed.sort(key=lambda value: value[0])
    if count < 1 or count > len(keyed):
        raise ValueError("Invalid HellaSwag selection size")
    return keyed[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache/hellaswag-source")
    parser.add_argument("--source-config", default="configs/hellaswag-specialist-source.json")
    parser.add_argument("--output", default="runs/hellaswag-specialist-v1")
    parser.add_argument("--train-rows", type=int, default=512)
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache, output, config_path = root / args.cache, root / args.output, root / args.source_config
    config = json.loads(config_path.read_text())
    if config.get("license") != "MIT" or len(config.get("revision", "")) != 40:
        raise ValueError("Unexpected HellaSwag source identity")
    if args.download_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=config["dataset_id"], repo_type="dataset",
                          revision=config["revision"], local_dir=cache,
                          allow_patterns=list(config["files"]))
        for relative, expected in config["files"].items():
            if file_sha256(cache / relative) != expected["sha256"]:
                raise ValueError("Downloaded HellaSwag source differs: " + relative)
        print(canonical({"event": "downloaded", "revision": config["revision"],
                         "files": len(config["files"])}))
        return
    if output.exists():
        raise ValueError("Use a fresh HellaSwag specialist data directory")
    import pyarrow.parquet as parquet
    raw = {}
    for split, relative in FILES.items():
        expected = config["files"][relative]
        path = cache / relative
        if file_sha256(path) != expected["sha256"]:
            raise ValueError("HellaSwag source file changed: " + relative)
        table = parquet.read_table(path)
        if table.num_rows != expected["rows"]:
            raise ValueError("HellaSwag source row count changed: " + relative)
        raw[split] = table.to_pylist()
    train = select(raw["train"], args.train_rows)
    validation = select(raw["validation"], args.validation_rows)
    train_sources = {row["source_id"] for _, row in train}
    if train_sources & {row["source_id"] for _, row in validation}:
        raise ValueError("HellaSwag source video leaked across selected splits")
    cases = []
    for split, selected in (("train", train), ("validation", validation)):
        relative = FILES[split]
        cases.extend(make_case(row, split, config["revision"],
                               config["files"][relative]["sha256"])
                     for _, row in selected)
    output.mkdir(parents=True)
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(canonical(row) + "\n" for row in cases))
    loaded = load_rows(cases_path)
    write_json(output / "protocol.json", {
        "study": "hellaswag-specialist-v1",
        "role": "public-family specialist train and consumed validation",
        "source_config": args.source_config,
        "source_config_sha256": digest(config_path.read_bytes()),
        "revision": config["revision"], "license": config["license"],
        "source_rows": {split: len(rows) for split, rows in raw.items()},
        "train_rows": len(train), "validation_rows": len(validation),
        "train_selection_sha256": digest([key for key, _ in train]),
        "validation_selection_sha256": digest([key for key, _ in validation]),
        "label_positions": {split: dict(Counter(row["label"] for row in loaded
                                                 if row["split"] == split))
                            for split in ("train", "validation")},
        "answer_blind_selection": True, "answer_blind_candidate_order": True,
        "source_group_overlap": 0,
        "cases_sha256": digest(cases_path.read_bytes()),
        "preparation_script_sha256": digest(Path(__file__).read_bytes()),
        "release_policy": config["release_policy"],
    })
    print(canonical({"event": "prepared", "rows": len(loaded),
                     "splits": dict(Counter(row["split"] for row in loaded)),
                     "cases_sha256": digest(cases_path.read_bytes())}))


if __name__ == "__main__":
    main()
