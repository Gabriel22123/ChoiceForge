"""Deterministic QASC conversion for evidence-conditioned candidate scoring."""
from __future__ import annotations

from collections import defaultdict

from decision_model.core import canonical, digest


REVISION = "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070"
TRAIN_SHA256 = "b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1"
VALIDATION_SHA256 = "d1ae34ae13c5fce2c55305372c203c6cdb789728d0d7e5ea2956d55bc33f40ae"
README_SHA256 = "c90ae4776b98cd6fc0a22558784c43b07e1f0b00a090dda99797d6b6efab4a3e"
TASK = (
    "Use the question and the two science facts in Context. Select the candidate "
    "that is best supported by the facts."
)
OPTION_IDS = tuple(f"option-{index:02d}" for index in range(1, 9))


def _source_items(row: dict) -> list[tuple[str, str]]:
    choices = row.get("choices")
    if not isinstance(choices, dict):
        raise ValueError("QASC choices must be a dictionary")
    labels, texts = choices.get("label"), choices.get("text")
    if (not isinstance(labels, list) or not isinstance(texts, list) or
            len(labels) != 8 or len(texts) != 8 or len(set(labels)) != 8 or
            any(not isinstance(value, str) or not value.strip()
                for value in labels + texts) or len(set(texts)) != 8):
        raise ValueError("QASC needs eight unique labeled candidate texts")
    if row.get("answerKey") not in labels:
        raise ValueError("QASC answer key is absent from candidates")
    return list(zip(labels, (text.strip() for text in texts)))


def selection_key(row: dict) -> str:
    """Answer-blind content key used before any label-balanced selection."""
    items = _source_items(row)
    value = {
        "id": row.get("id"),
        "question": row.get("question"),
        "choices": items,
        "facts": [row.get("fact1"), row.get("fact2")],
    }
    if any(not isinstance(value[key], str) or not value[key].strip()
           for key in ("id", "question")):
        raise ValueError("QASC id and question must be nonempty strings")
    if any(not isinstance(fact, str) or not fact.strip() for fact in value["facts"]):
        raise ValueError("QASC facts must be nonempty strings")
    return digest(value)


def qasc_request_and_label(row: dict) -> tuple[dict, str]:
    """Hide source letters and order candidates by an answer-blind text hash."""
    question = row["question"].strip()
    ordered = sorted(
        _source_items(row),
        key=lambda item: digest({"question": question, "candidate": item[1]}),
    )
    choices = [
        {"id": identity, "description": text}
        for identity, (_, text) in zip(OPTION_IDS, ordered)
    ]
    label = next(identity for identity, (source_label, _) in zip(OPTION_IDS, ordered)
                 if source_label == row["answerKey"])
    request = {
        "task": TASK,
        "context": canonical({
            "question": question,
            "facts": [row["fact1"].strip(), row["fact2"].strip()],
        }),
        "choices": choices,
    }
    return request, label


def qasc_case(row: dict, split: str, file_sha256: str) -> dict:
    if split not in ("train", "validation"):
        raise ValueError("QASC screen only accepts official train or validation")
    request, label = qasc_request_and_label(row)
    upstream_id = row["id"].strip()
    return {
        "id": digest({"source": "qasc", "upstream_id": upstream_id,
                      "split": split, "request": request}),
        "source": "qasc",
        "family": "science_evidence_composition_8way",
        "split": split,
        "evaluation_regime": (
            "seen_task_training" if split == "train" else "seen_task_new_examples"
        ),
        "group_id": "qasc:" + upstream_id,
        "decision_type": "choice",
        "request": request,
        "label": label,
        "provenance": [{
            "origin": "QASC official public dataset via pinned Hugging Face conversion",
            "source_url": "https://github.com/allenai/qasc",
            "mirror_url": "https://huggingface.co/datasets/allenai/qasc",
            "revision": REVISION,
            "upstream_id": upstream_id,
            "official_split": split,
            "source_file_sha256": file_sha256,
            "license": "CC-BY-4.0",
            "transformation": (
                "kept question, fact1, fact2 and all eight answer texts; excluded "
                "combinedfact; replaced source letters and applied an answer-blind "
                "candidate permutation"
            ),
        }],
    }


def select_position_balanced(rows: list[dict], per_position: int,
                             admissible=None) -> list[dict]:
    """Select equal correct positions after an answer-blind candidate permutation."""
    if not isinstance(per_position, int) or isinstance(per_position, bool) or per_position <= 0:
        raise ValueError("per_position must be a positive integer")
    buckets: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    seen = set()
    for row in rows:
        key = selection_key(row)
        if key in seen:
            raise ValueError("Duplicate QASC answer-blind selection key")
        seen.add(key)
        request, label = qasc_request_and_label(row)
        if admissible is None or admissible(request):
            buckets[label].append((key, row))
    if set(buckets) != set(OPTION_IDS) or any(
            len(buckets[identity]) < per_position for identity in OPTION_IDS):
        raise ValueError("Not enough admissible QASC rows for balanced positions")
    selected = []
    for identity in OPTION_IDS:
        selected.extend(row for _, row in sorted(buckets[identity])[:per_position])
    return sorted(selected, key=selection_key)

