"""Prepare fixed PAWS training data and answer-blind BoolQ evaluation rows."""
from collections import Counter
import hashlib
import json

from decision_model.core import canonical, digest


PAWS_REVISION = "161ece9501cf0a11f3e48bd356eaa82de46d6a09"
BOOLQ_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
PAWS_HASHES = {
    "train": "8dc9ad3e5f30ad9a86b290fe236d528ef23a5751fec9a35d99cbacf68ba277cf",
    "validation": "7760d829453764ba342a6f562809a8ed21c2c3eec3fd9ffa544089f145d42f6d",
    "test": "ae342ff12bb84b84b95f468abf5db6cb7c7bd578271299fe9c99be75b8132f4d",
}
BOOLQ_HASH = "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8"


def input_hash(*values):
    return hashlib.sha256(canonical(values).encode()).hexdigest()


def paws_selection_key(row):
    return input_hash(row["id"], row["sentence1"], row["sentence2"])


def boolq_selection_key(row):
    """The answer is deliberately absent from frozen evaluation selection."""
    return input_hash(row["question"], row["passage"])


def select_balanced_paws(rows, per_label):
    buckets = {0: [], 1: []}
    for row in rows:
        if row["label"] not in buckets:
            raise ValueError("Unexpected PAWS label")
        buckets[row["label"]].append(row)
    if any(len(bucket) < per_label for bucket in buckets.values()):
        raise ValueError("Not enough PAWS rows")
    selected = []
    for label in (0, 1):
        selected.extend(sorted(buckets[label], key=paws_selection_key)[:per_label])
    return sorted(selected, key=paws_selection_key)


def select_boolq(rows, count, admissible=None):
    if count <= 0:
        raise ValueError("Selection count must be positive")
    candidates = [row for row in rows if admissible is None or admissible(row)]
    if len(candidates) < count:
        raise ValueError("Not enough token-admissible BoolQ rows")
    return sorted(candidates, key=boolq_selection_key)[:count]


def paws_case(row, split, file_sha256):
    label = "paraphrase" if row["label"] == 1 else "different"
    request = {
        "task": "Determine whether the two sentences express the same meaning.",
        "context": canonical({"sentence1": row["sentence1"], "sentence2": row["sentence2"]}),
        "choices": [
            {"id": "different", "description": "The two sentences express different meanings."},
            {"id": "paraphrase", "description": "The two sentences express the same meaning."},
        ],
    }
    key = digest({"source": "paws-wiki", "split": split, "upstream_id": row["id"], "request": request})
    return {
        "id": key,
        "source": "paws-wiki",
        "family": "paraphrase_identification",
        "split": split,
        "evaluation_regime": "seen_task_new_examples",
        "group_id": "paws-wiki:" + input_hash(row["sentence1"], row["sentence2"]),
        "request": request,
        "label": label,
        "provenance": [{
            "url": "https://huggingface.co/datasets/google-research-datasets/paws",
            "revision": PAWS_REVISION,
            "file_sha256": file_sha256,
            "upstream_id": row["id"],
            "original_split": split,
            "license": "PAWS dataset permission: freely usable for any purpose; attribution appreciated",
            "label_method": "upstream_human_label",
            "transformation": "sentence_pair_to_two_self_contained_candidates",
        }],
    }


def boolq_request(row):
    """Build model input without reading or requiring the upstream answer."""
    return {
        "task": "Answer the question using only the passage.",
        "context": canonical({"passage": row["passage"], "question": row["question"]}),
        "choices": [
            {"id": "no", "description": "The passage supports answering no to the question."},
            {"id": "yes", "description": "The passage supports answering yes to the question."},
        ],
    }


def boolq_case(row, file_sha256=BOOLQ_HASH):
    request = boolq_request(row)
    return {
        "id": digest({"source": "boolq", "split": "validation", "request": request}),
        "source": "boolq",
        "family": "reading_comprehension_yes_no",
        "split": "test",
        "evaluation_regime": "untouched_task_family",
        "group_id": "boolq:" + boolq_selection_key(row),
        "request": request,
        "label": "yes" if row["answer"] else "no",
        "provenance": [{
            "url": "https://huggingface.co/datasets/google/boolq",
            "revision": BOOLQ_REVISION,
            "file_sha256": file_sha256,
            "original_split": "validation",
            "license": "CC-BY-SA-3.0",
            "label_method": "upstream_gold_label",
            "transformation": "passage_question_to_two_self_contained_candidates; hash selection excludes answer",
        }],
    }


def summarize_labels(rows):
    return dict(sorted(Counter(row["label"] for row in rows).items()))
