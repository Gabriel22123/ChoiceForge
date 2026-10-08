#!/usr/bin/env python3
"""Prepare answer-blind, source-balanced ARC specialist training rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, validate_request, write_json
from decision_model.prompting import render_decoder_admission, render_decoder_candidate


TASK = "Choose the answer that best resolves this grade-school science question."
SOURCE_FILES = {
    "arc-challenge": "ARC-Challenge/train-00000-of-00001.parquet",
    "arc-easy": "ARC-Easy/train-00000-of-00001.parquet",
}


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_items(raw: dict) -> list[tuple[str, str]]:
    choices = raw.get("choices")
    if not isinstance(choices, dict):
        raise ValueError("ARC choices must be a dictionary")
    labels, texts = choices.get("label"), choices.get("text")
    if (not isinstance(labels, list) or not isinstance(texts, list) or
            not 2 <= len(labels) == len(texts) <= 32 or len(set(labels)) != len(labels) or
            any(not isinstance(value, str) or not value.strip() for value in labels + texts) or
            len({value.strip() for value in texts}) != len(texts)):
        raise ValueError("ARC choices are malformed")
    if raw.get("answerKey") not in labels:
        raise ValueError("ARC answer key is absent from candidates")
    return list(zip(labels, (text.strip() for text in texts)))


def selection_key(raw: dict, source: str) -> str:
    """Hash only source input fields; the answer never affects admission or order."""
    items = source_items(raw)
    question = raw.get("question")
    identity = raw.get("id")
    if any(not isinstance(value, str) or not value.strip() for value in (question, identity)):
        raise ValueError("ARC id and question must be nonempty strings")
    return digest({"source": source, "id": identity.strip(), "question": question.strip(),
                   "choices": items})


def request_and_label(raw: dict) -> tuple[dict, str]:
    """Remove source letters and use a deterministic answer-blind candidate order."""
    question = raw["question"].strip()
    ordered = sorted(source_items(raw),
                     key=lambda item: digest({"question": question, "candidate": item[1]}))
    choices = [{"id": f"option_{index}", "description": text}
               for index, (_, text) in enumerate(ordered)]
    label = next(choice["id"] for choice, (source_label, _) in zip(choices, ordered)
                 if source_label == raw["answerKey"])
    request = {"task": TASK, "context": question, "choices": choices}
    validate_request(request)
    return request, label


def semantic_key(request: dict) -> str:
    return digest({"task": request["task"], "context": request["context"],
                   "choice_descriptions": sorted(choice["description"]
                                                 for choice in request["choices"])})


def case(raw: dict, source: str, revision: str, source_file_sha256: str) -> dict:
    request, label = request_and_label(raw)
    upstream_id = raw["id"].strip()
    return {
        "id": digest({"source": source, "upstream_id": upstream_id,
                      "revision": revision, "request": request}),
        "group_id": f"{source}:{upstream_id}",
        "source": source,
        "family": "grade_school_science_choice",
        "split": "train",
        "evaluation_regime": "seen_task_training",
        "decision_type": "choice",
        "request": request,
        "label": label,
        "provenance": [{
            "origin": "AI2 ARC official public dataset via pinned Hugging Face conversion",
            "source_url": "https://huggingface.co/datasets/allenai/ai2_arc",
            "revision": revision,
            "config": source,
            "official_split": "train",
            "upstream_id": upstream_id,
            "source_file_sha256": source_file_sha256,
            "license": "CC-BY-SA-4.0",
            "transformation": ("kept question and all answer texts; removed source answer letters; "
                               "applied deterministic answer-blind candidate permutation"),
        }],
    }


def select_rows(raw_by_source: dict[str, list[dict]], rows_per_source: int,
                blocked_semantics: set[str], admissible) -> tuple[list[tuple[str, dict]], dict]:
    if rows_per_source < 1:
        raise ValueError("rows_per_source must be positive")
    prepared = {}
    semantic_counts = Counter()
    for source, rows in raw_by_source.items():
        values = []
        seen_selection = set()
        for raw in rows:
            key = selection_key(raw, source)
            if key in seen_selection:
                raise ValueError("Duplicate ARC answer-blind selection key")
            seen_selection.add(key)
            request, _ = request_and_label(raw)
            value = (key, semantic_key(request), raw, request)
            values.append(value); semantic_counts[value[1]] += 1
        prepared[source] = values
    selected, summary = [], {}
    for source in sorted(prepared):
        eligible = [value for value in prepared[source]
                    if value[1] not in blocked_semantics and semantic_counts[value[1]] == 1
                    and admissible(value[3])]
        eligible.sort(key=lambda value: value[0])
        if len(eligible) < rows_per_source:
            raise ValueError(f"Not enough eligible ARC rows for {source}")
        chosen = eligible[:rows_per_source]
        selected.extend((source, raw) for _, _, raw, _ in chosen)
        summary[source] = {
            "source_rows": len(prepared[source]),
            "eligible_rows": len(eligible),
            "selected_rows": len(chosen),
            "selection_keys_sha256": digest([key for key, _, _, _ in chosen]),
        }
    return selected, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache/arc-specialist-source")
    parser.add_argument("--source-config", default="configs/arc-specialist-source.json")
    parser.add_argument("--validation", default="runs/architecture-comparison-v1/arc-blind/cases.jsonl")
    parser.add_argument("--output", default="runs/arc-specialist-v1")
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--rows-per-source", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache, config_path = root / args.cache, root / args.source_config
    output, validation_path = root / args.output, root / args.validation
    if output.exists():
        raise ValueError("Use a fresh ARC specialist data directory")
    config = json.loads(config_path.read_text())
    if config.get("revision") != "210d026faf9955653af8916fad021475a3f00453":
        raise ValueError("Unexpected ARC revision")
    if config.get("license") != "CC-BY-SA-4.0":
        raise ValueError("Unexpected ARC license")

    if args.download_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=config["dataset_id"], repo_type="dataset",
                          revision=config["revision"], local_dir=cache,
                          allow_patterns=list(config["files"]))
        for relative, expected in config["files"].items():
            path = cache / relative
            if not path.is_file() or file_sha256(path) != expected["sha256"]:
                raise ValueError("Downloaded ARC source differs: " + relative)
        print(canonical({"event": "downloaded", "revision": config["revision"],
                         "files": len(config["files"])}))
        return

    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer
    raw_by_source = {}
    for source, relative in SOURCE_FILES.items():
        path = cache / relative
        expected = config["files"][relative]
        if file_sha256(path) != expected["sha256"]:
            raise ValueError("ARC source file changed: " + relative)
        table = parquet.read_table(path)
        if table.num_rows != expected["rows"]:
            raise ValueError("ARC source row count changed: " + relative)
        raw_by_source[source] = table.to_pylist()

    validation = load_rows(validation_path)
    blocked = {semantic_key(row["request"]) for row in validation}
    tokenizer = AutoTokenizer.from_pretrained(root / args.base_path, local_files_only=True,
                                               trust_remote_code=False)
    def admissible(request):
        lengths = [len(tokenizer.encode(render_decoder_admission(request), add_special_tokens=True))]
        lengths.extend(len(tokenizer.encode(render_decoder_candidate(request, index),
                                            add_special_tokens=True))
                       for index in range(len(request["choices"])))
        return max(lengths) <= args.max_tokens

    selected, selection = select_rows(raw_by_source, args.rows_per_source,
                                      blocked, admissible)
    source_hashes = {source: config["files"][relative]["sha256"]
                     for source, relative in SOURCE_FILES.items()}
    rows = [case(raw, source, config["revision"], source_hashes[source])
            for source, raw in selected]
    if len(rows) != args.rows_per_source * len(SOURCE_FILES):
        raise ValueError("ARC selected row count differs")

    output.mkdir(parents=True)
    train_path = output / "train.jsonl"
    train_path.write_text("".join(canonical(row) + "\n" for row in rows))
    loaded = load_rows(train_path)
    training_semantics = {semantic_key(row["request"]) for row in loaded}
    if training_semantics & blocked:
        raise ValueError("ARC training overlaps validation inputs")
    labels = {source: dict(Counter(row["label"] for row in loaded if row["source"] == source))
              for source in sorted(SOURCE_FILES)}
    write_json(output / "protocol.json", {
        "study": "arc-specialist-v1",
        "role": "consumed public-family specialist training",
        "source_config": args.source_config,
        "source_config_sha256": digest(config_path.read_bytes()),
        "source_revision": config["revision"],
        "source_license": config["license"],
        "rows_per_source": args.rows_per_source,
        "rows": len(loaded),
        "selection": selection,
        "label_positions": labels,
        "validation_path": args.validation,
        "validation_sha256": digest(validation_path.read_bytes()),
        "validation_rows": len(validation),
        "train_sha256": digest(train_path.read_bytes()),
        "exact_semantic_overlap": 0,
        "answer_blind_selection": True,
        "answer_blind_candidate_order": True,
        "max_tokens": args.max_tokens,
        "preparation_script_sha256": digest(Path(__file__).read_bytes()),
        "release_policy": config["release_policy"],
    })
    print(canonical({"event": "prepared", "rows": len(loaded),
                     "sources": dict(Counter(row["source"] for row in loaded)),
                     "train_sha256": digest(train_path.read_bytes())}))


if __name__ == "__main__":
    main()
