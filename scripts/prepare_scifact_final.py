#!/usr/bin/env python3
"""Open the pinned SciFact archive under a frozen, answer-blind protocol."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, validate_request, write_json


TASK = "Determine whether the scientific evidence supports or contradicts the claim."
CHOICES = {
    "support": "The evidence supports the scientific claim.",
    "contradict": "The evidence contradicts the scientific claim.",
}


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl_bytes(value, name):
    rows = []
    for line_number, line in enumerate(io.StringIO(value.decode("utf-8")), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{name} line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(name + " is empty")
    return rows


def archive_member(archive, suffix):
    matches = [member for member in archive.getmembers()
               if member.isfile() and member.name.rstrip("/").endswith(suffix)]
    if len(matches) != 1 or matches[0].size > 64 * 1024 * 1024:
        raise ValueError("Unexpected SciFact archive member: " + suffix)
    stream = archive.extractfile(matches[0])
    if stream is None:
        raise ValueError("SciFact archive member cannot be read")
    return stream.read()


def load_archive(path):
    with tarfile.open(path, "r:gz") as archive:
        corpus = read_jsonl_bytes(archive_member(archive, "data/corpus.jsonl"), "corpus")
        claims = read_jsonl_bytes(
            archive_member(archive, "data/claims_dev.jsonl"), "claims_dev")
    return corpus, claims


def validate_corpus_row(row):
    if (set(row) != {"doc_id", "title", "abstract", "structured"} or
            not isinstance(row["doc_id"], int) or
            not isinstance(row["title"], str) or not row["title"].strip() or
            not isinstance(row["abstract"], list) or not row["abstract"] or
            any(not isinstance(value, str) or not value.strip() for value in row["abstract"]) or
            not isinstance(row["structured"], bool)):
        raise ValueError("Malformed SciFact corpus row")


def validate_claim_row(row):
    if (set(row) != {"id", "claim", "evidence", "cited_doc_ids"} or
            not isinstance(row["id"], int) or
            not isinstance(row["claim"], str) or not row["claim"].strip() or
            not isinstance(row["evidence"], dict) or
            not isinstance(row["cited_doc_ids"], list) or
            any(not isinstance(value, int) for value in row["cited_doc_ids"])):
        raise ValueError("Malformed SciFact claim row")


def selection_key(claim_id, doc_id):
    if not isinstance(claim_id, int) or not isinstance(doc_id, int):
        raise ValueError("SciFact selection IDs must be integers")
    return digest({"claim_id": claim_id, "doc_id": doc_id})


def ordered_choice_ids(claim_id, doc_id):
    return sorted(CHOICES, key=lambda identity: digest({
        "claim_id": claim_id, "doc_id": doc_id, "candidate": identity}))


def evidence_item(claim, doc_id, annotations, corpus):
    validate_claim_row(claim)
    if (not isinstance(annotations, list) or not annotations or
            any(not isinstance(item, dict) or set(item) != {"label", "sentences"}
                for item in annotations)):
        raise ValueError("Malformed SciFact evidence annotation")
    labels = {item["label"] for item in annotations}
    if not labels <= {"SUPPORT", "CONTRADICT"} or len(labels) != 1:
        raise ValueError("SciFact document labels are inconsistent")
    sentence_ids = sorted({index for item in annotations for index in item["sentences"]})
    document = corpus.get(doc_id)
    if (document is None or not sentence_ids or
            any(isinstance(index, bool) or not isinstance(index, int) or
                not 0 <= index < len(document["abstract"]) for index in sentence_ids)):
        raise ValueError("SciFact evidence sentences are invalid")
    choice_ids = ordered_choice_ids(claim["id"], doc_id)
    choices = [{"id": identity, "description": CHOICES[identity]}
               for identity in choice_ids]
    request = {
        "task": TASK,
        "context": (f"Claim: {claim['claim'].strip()}\n"
                    f"Scientific source: {document['title'].strip()}\n"
                    "Evidence: " + " ".join(
                        document["abstract"][index].strip() for index in sentence_ids)),
        "choices": choices,
    }
    validate_request(request)
    label = "support" if next(iter(labels)) == "SUPPORT" else "contradict"
    key = selection_key(claim["id"], doc_id)
    return key, {
        "id": digest({"source": "scifact", "selection_key": key}),
        "group_id": f"scifact-claim:{claim['id']}",
        "source": "scifact", "family": "scientific_claim_verification",
        "split": "validation", "evaluation_regime": "sealed_unseen_family",
        "decision_type": "choice", "request": request, "label": label,
        "provenance": [{
            "origin": "SciFact official public development split",
            "source_url": "https://github.com/allenai/scifact",
            "official_split": "dev", "claim_id": claim["id"], "doc_id": doc_id,
            "evidence_sentence_ids": sentence_ids, "selection_key": key,
            "claim_and_annotation_license": "CC-BY-4.0",
            "abstract_corpus_license": "ODC-By-1.0",
            "transformation": ("paired each annotated claim/document with the union of its "
                               "gold rationale sentences; selected pairs and ordered support/"
                               "contradict choices without reading the label"),
        }],
    }


def prepare_rows(corpus_rows, claim_rows, max_rows):
    corpus = {}
    for row in corpus_rows:
        validate_corpus_row(row)
        if row["doc_id"] in corpus:
            raise ValueError("Duplicate SciFact document")
        corpus[row["doc_id"]] = row
    keyed = []
    for claim in claim_rows:
        validate_claim_row(claim)
        for raw_doc_id, annotations in claim["evidence"].items():
            try:
                doc_id = int(raw_doc_id)
            except (TypeError, ValueError) as error:
                raise ValueError("SciFact evidence document ID is invalid") from error
            keyed.append(evidence_item(claim, doc_id, annotations, corpus))
    keys = [key for key, _ in keyed]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate SciFact claim/document pair")
    keyed.sort(key=lambda value: value[0])
    if max_rows < 1:
        raise ValueError("SciFact max rows must be positive")
    return [row for _, row in keyed[:max_rows]], [key for key, _ in keyed[:max_rows]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact-final-evaluation-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = json.loads(config_path.read_text())
    if config.get("status") != "frozen_before_unsealing":
        raise ValueError("SciFact final protocol is not frozen")
    for relative, expected in config["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen SciFact source changed: " + relative)
    archive = root / config["source"]["archive_path"]
    if (archive.stat().st_size != config["source"]["archive_bytes"] or
            file_sha256(archive) != config["source"]["archive_sha256"]):
        raise ValueError("Sealed SciFact archive changed")
    output = root / config["data"]["output"]
    if output.exists():
        raise ValueError("Use a fresh SciFact final data directory")
    corpus, claims = load_archive(archive)
    rows, selected_keys = prepare_rows(corpus, claims, config["data"]["max_rows"])
    if len(rows) < config["data"]["min_rows"]:
        raise ValueError("SciFact final selection is too small")
    output.mkdir(parents=True)
    cases = output / "cases.jsonl"
    cases.write_text("".join(canonical(row) + "\n" for row in rows))
    loaded = load_rows(cases)
    protocol = {
        "format_version": 1, "study": config["study"],
        "role": "sealed unseen-family final evaluation data",
        "source_archive_sha256": config["source"]["archive_sha256"],
        "source_config_sha256": digest((root / config["source"]["source_config"]).read_bytes()),
        "preparation_script_sha256": digest(Path(__file__).read_bytes()),
        "rows": len(loaded), "cases_sha256": digest(cases.read_bytes()),
        "selection_sha256": digest(selected_keys),
        "answer_blind_pair_selection": True, "answer_blind_candidate_order": True,
        "label_counts": dict(Counter(row["label"] for row in loaded)),
        "raw_text_release": False,
    }
    write_json(output / "protocol.json", protocol)
    print(canonical({"event": "unsealed_and_prepared", "rows": len(rows),
                     "cases_sha256": protocol["cases_sha256"]}))


if __name__ == "__main__":
    main()
