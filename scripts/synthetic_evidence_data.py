"""Deterministic project-authored evidence flips for public training.

Every group keeps the task, claim and candidates fixed while changing only the
evidence.  Labels follow from the generator rules; no teacher model is used.
"""
from __future__ import annotations

from decision_model.core import canonical, digest


TASK = (
    "Decide whether the evidence establishes the claim. Use only the evidence. "
    "Choose Yes only when the claim follows; otherwise choose No."
)
CHOICES = [
    {"id": "no", "description": "No"},
    {"id": "yes", "description": "Yes"},
]

FIRST = (
    "Ada", "Ben", "Cora", "Dario", "Elena", "Farah", "Gavin", "Hana",
    "Iris", "Jonas", "Kira", "Liam", "Marta", "Nolan", "Orla", "Pavel",
    "Quinn", "Rina", "Soren", "Talia",
)
LAST = (
    "Arden", "Briar", "Calder", "Dover", "Ellis", "Frost", "Grove", "Hart",
    "Ivory", "Jasper", "Keats", "Lane", "Morrow", "North", "Oakley", "Pine",
    "Reed", "Stone", "Thorne", "Vale",
)
PROPERTIES = (
    "alert", "brisk", "calm", "dry", "eager", "faint", "gentle", "helpful",
    "idle", "jolly", "kind", "level", "mild", "neat", "open", "patient",
    "quiet", "ready", "steady", "warm", "young", "bright", "clear", "safe",
)
PLACES = (
    "archive", "bridge", "courtyard", "depot", "gallery", "harbor", "island",
    "library", "market", "orchard", "plaza", "station", "studio", "tower",
    "valley", "workshop",
)


def _subject(index: int) -> str:
    return FIRST[index % len(FIRST)] + " " + LAST[(index // len(FIRST)) % len(LAST)]


def _property(index: int, offset: int) -> str:
    return PROPERTIES[(index * 7 + offset * 5) % len(PROPERTIES)]


def _evidence_pair(index: int) -> tuple[str, list[str], list[str]]:
    subject = _subject(index)
    target = _property(index, 0)
    other = next(value for value in (_property(index, i) for i in range(1, 8))
                 if value != target)
    bridge = next(value for value in (_property(index, i) for i in range(8, 16))
                  if value not in {target, other})
    place = PLACES[(index * 3) % len(PLACES)]
    template = index % 4
    claim = f"{subject} is {target}."
    if template == 0:
        yes = [f"{subject} is {target}.", f"{subject} visited the {place}."]
        no = [f"{subject} is {other}.", f"{subject} visited the {place}."]
    elif template == 1:
        yes = [f"{subject} is {other}.", f"Every {other} person is {target}."]
        no = [f"{subject} is {bridge}.", f"Every {other} person is {target}."]
    elif template == 2:
        yes = [
            f"{subject} is {other}.",
            f"Every {other} person is {bridge}.",
            f"Every {bridge} person is {target}.",
        ]
        no = [
            f"{subject} is {other}.",
            f"Every {other} person is {bridge}.",
            f"Every {bridge} person is not {target}.",
        ]
    else:
        yes = [
            f"{subject} entered the {place}.",
            f"Everyone who entered the {place} became {target}.",
        ]
        no = [
            f"{subject} left the {place}.",
            f"Everyone who entered the {place} became {target}.",
        ]
    return claim, yes, no


def generated_cases(train_groups: int = 256, validation_groups: int = 64) -> list[dict]:
    if train_groups <= 0 or validation_groups <= 0:
        raise ValueError("Evidence group counts must be positive")
    rows = []
    for index in range(train_groups + validation_groups):
        split = "train" if index < train_groups else "validation"
        claim, yes_evidence, no_evidence = _evidence_pair(index)
        group_id = "generated-evidence:" + digest({"index": index, "claim": claim})
        for label, evidence in (("yes", yes_evidence), ("no", no_evidence)):
            request = {
                "task": TASK,
                "context": canonical({"claim": claim, "evidence": evidence}),
                "choices": [dict(choice) for choice in CHOICES],
            }
            rows.append({
                "id": digest({"source": "generated-evidence", "request": request}),
                "source": "generated-evidence",
                "family": "counterfactual_evidence_use",
                "split": split,
                "evaluation_regime": (
                    "seen_task_training" if split == "train" else
                    "seen_task_new_examples"
                ),
                "group_id": group_id,
                "request": request,
                "label": label,
                "provenance": [{
                    "origin": "project_authored_deterministic_generator",
                    "generator": "scripts/synthetic_evidence_data.py",
                    "license": "Apache-2.0",
                    "label_method": "deterministic_rule_evaluation",
                    "transformation": (
                        "same task, claim and candidates; evidence-only paired intervention"
                    ),
                }],
            })
    return rows


def pair_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    selected = {row["id"]: row for row in rows if row["source"] == "generated-evidence"}
    predicted = {row["id"]: row for row in predictions if row["id"] in selected}
    if set(predicted) != set(selected):
        raise ValueError("Generated-evidence prediction coverage differs")
    groups: dict[str, list[bool]] = {}
    correct = 0
    for identifier, row in selected.items():
        values = predicted[identifier]["probabilities"]
        if set(values) != {"no", "yes"}:
            raise ValueError("Generated-evidence prediction support differs")
        choice = max(values, key=values.get)
        hit = choice == row["label"]
        correct += hit
        groups.setdefault(row["group_id"], []).append(hit)
    if any(len(values) != 2 for values in groups.values()):
        raise ValueError("Generated-evidence groups must contain exactly two rows")
    complete = sum(all(values) for values in groups.values())
    return {
        "rows": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected),
        "groups": len(groups),
        "complete_groups": complete,
        "complete_group_rate": complete / len(groups),
    }
