"""Pin ARC source files, then seal an answer-blind held-out task subset.

Run ``--download-only`` first.  Do not run ``--seal-selection`` until model
endpoints and the evaluation protocol are fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from decision_model.core import digest, load_rows, render, write_json
from decision_model.prompting import render_decoder_admission, render_decoder_candidate


DATASET_ID = "allenai/ai2_arc"
REVISION = "210d026faf9955653af8916fad021475a3f00453"
SOURCE_FILES = (
    "README.md",
    "ARC-Challenge/validation-00000-of-00001.parquet",
    "ARC-Easy/validation-00000-of-00001.parquet",
)
ROWS_PER_TASK = 192
MAX_TOKENS = 768


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download_only(cache, source_record):
    snapshot_download(repo_id=DATASET_ID, repo_type="dataset", revision=REVISION,
                      local_dir=cache, allow_patterns=list(SOURCE_FILES))
    missing = [name for name in SOURCE_FILES if not (cache / name).is_file()]
    if missing:
        raise ValueError("Pinned ARC snapshot is missing: " + ", ".join(missing))
    write_json(source_record, {
        "name": "AI2 Reasoning Challenge (ARC)",
        "repository": f"https://huggingface.co/datasets/{DATASET_ID}",
        "revision": REVISION,
        "license": "CC-BY-SA-4.0",
        "official_rows": {"ARC-Challenge-validation": 299, "ARC-Easy-validation": 570},
        "files": {name: sha256(cache / name) for name in SOURCE_FILES},
        "selection": ("For each task, retain rows admitted by both pinned tokenizers, order by "
                      "SHA-256 of source/id/question/choice text without answerKey, then take 192."),
        "release_policy": "Raw and transformed ARC rows and per-case predictions are excluded from release bundles.",
    })
    print({"downloaded": len(SOURCE_FILES), "revision": REVISION})


def verify_source(cache, source_record):
    record = json.loads(source_record.read_text())
    if record["revision"] != REVISION or record["license"] != "CC-BY-SA-4.0":
        raise ValueError("ARC source record differs")
    for name, expected in record["files"].items():
        if sha256(cache / name) != expected:
            raise ValueError("ARC source file changed: " + name)
    return record


def make_request(question, texts):
    return {
        "task": "Choose the answer that best resolves this grade-school science question.",
        "context": question.strip(),
        "choices": [{"id": f"option_{index}", "description": text.strip()}
                    for index, text in enumerate(texts)],
    }


def seal(cache, source_record, output, eurobert_path, decoder_path, training_data):
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    if output.exists():
        raise ValueError("Use a fresh output directory")
    source = verify_source(cache, source_record)
    euro = AutoTokenizer.from_pretrained(eurobert_path, local_files_only=True, trust_remote_code=False)
    decoder = AutoTokenizer.from_pretrained(decoder_path, local_files_only=True, trust_remote_code=False)
    training = load_rows(training_data)
    training_inputs = {digest(row["request"]) for row in training if row["split"] == "train"}
    selected_rows, selection_records = [], {}
    for config, filename in (
        ("ARC-Challenge", "ARC-Challenge/validation-00000-of-00001.parquet"),
        ("ARC-Easy", "ARC-Easy/validation-00000-of-00001.parquet"),
    ):
        raw_rows = parquet.read_table(cache / filename).to_pylist()
        eligible = []
        for raw in raw_rows:
            labels, texts = raw["choices"]["label"], raw["choices"]["text"]
            if len(labels) != len(texts) or not 2 <= len(texts) <= 32 or raw["answerKey"] not in labels:
                raise ValueError("Malformed ARC row: " + raw["id"])
            request = make_request(raw["question"], texts)
            euro_ids = euro.encode(render(request, euro.mask_token), add_special_tokens=True)
            decoder_admission = len(decoder.encode(render_decoder_admission(request), add_special_tokens=True))
            decoder_lengths = [len(decoder.encode(render_decoder_candidate(request, index), add_special_tokens=True))
                               for index in range(len(texts))]
            if (len(euro_ids) > MAX_TOKENS or decoder_admission > MAX_TOKENS or
                    max(decoder_lengths) > MAX_TOKENS):
                continue
            selection_key = digest({"source": config, "id": raw["id"], "question": raw["question"],
                                    "choice_labels": labels, "choice_text": texts})
            eligible.append((selection_key, raw, request))
        eligible.sort(key=lambda value: value[0])
        if len(eligible) < ROWS_PER_TASK:
            raise ValueError("Not enough jointly admissible ARC rows: " + config)
        chosen = eligible[:ROWS_PER_TASK]
        selection_records[config] = {
            "source_rows": len(raw_rows), "eligible_rows": len(eligible),
            "selected_rows": ROWS_PER_TASK,
            "selected_input_sha256": digest([value[0] for value in chosen]),
            "selected_source_ids": [value[1]["id"] for value in chosen],
        }
        for _, raw, request in chosen:
            labels = raw["choices"]["label"]
            answer_index = labels.index(raw["answerKey"])
            if digest(request) in training_inputs:
                raise ValueError("ARC request overlaps a training request")
            selected_rows.append({
                "id": digest({"source": config, "source_id": raw["id"], "revision": REVISION}),
                "group_id": f"{config}:{raw['id']}", "source": config.lower(), "split": "test",
                "request": request, "label": f"option_{answer_index}",
                "provenance": {"dataset": DATASET_ID, "revision": REVISION,
                               "config": config, "split": "validation", "source_id": raw["id"]},
            })

    output.mkdir(parents=True)
    cases = output / "cases.jsonl"
    with cases.open("w") as stream:
        for row in sorted(selected_rows, key=lambda value: value["id"]):
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    # Reparse through the public contract before freezing anything.
    loaded = load_rows(cases)
    if len(loaded) != 2 * ROWS_PER_TASK:
        raise ValueError("Sealed ARC row count differs")
    script = Path(__file__).resolve()
    write_json(output / "protocol.json", {
        "question": "Do frozen decision endpoints transfer to unseen grade-school science choices?",
        "dataset": source,
        "dataset_source_record_sha256": sha256(source_record),
        "preparation_script_sha256": sha256(script),
        "training_data_sha256": sha256(training_data),
        "exact_training_request_overlap": 0,
        "selection_is_answer_blind": True,
        "selection": selection_records,
        "cases_sha256": sha256(cases),
        "rows": len(loaded),
        "role": "Fresh task-level blind test only; never training, calibration, early stopping or endpoint selection.",
        "evaluation_rule": "Evaluate every preregistered endpoint once, raw probabilities only; report all endpoints and paired group bootstrap intervals.",
    })
    write_json(output / "status.json", {"state": "prepared"})
    print({"prepared": str(output), "rows": len(loaded), "cases_sha256": sha256(cases)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache/arc-blind")
    parser.add_argument("--source-record", default="configs/arc-blind-source.json")
    parser.add_argument("--output", default="runs/arc-blind-v1")
    parser.add_argument("--eurobert-path", default="models/eurobert-2.1b")
    parser.add_argument("--decoder-path", default="models/minicpm5-2b-base")
    parser.add_argument("--training-data", default="runs/task-gradient-trust-v1/cases.jsonl")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--download-only", action="store_true")
    action.add_argument("--seal-selection", action="store_true")
    args = parser.parse_args()
    cache, source_record = Path(args.cache), Path(args.source_record)
    if args.download_only:
        download_only(cache, source_record)
    else:
        seal(cache, source_record, Path(args.output), Path(args.eurobert_path),
             Path(args.decoder_path), Path(args.training_data))


if __name__ == "__main__":
    main()
