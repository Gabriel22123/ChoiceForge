"""Matched row orders for mixed versus task-and-label-stratified updates."""
from collections import Counter, defaultdict
import random


TASKS = ("paws", "snli", "replay")


def task_name(row):
    if row["source"] == "paws-wiki":
        return "paws"
    if row["source"] == "snli":
        return "snli"
    return "replay"


def _training_rows(rows):
    selected = [row for row in rows if row["split"] == "train"]
    counts = Counter(task_name(row) for row in selected)
    if counts != Counter({"paws": 384, "snli": 384, "replay": 256}):
        raise ValueError("Expected 384 PAWS, 384 SNLI and 256 replay rows")
    if Counter(row["label"] for row in selected if task_name(row) == "paws") != Counter(
            {"different": 192, "paraphrase": 192}):
        raise ValueError("Expected balanced PAWS labels")
    if Counter(row["label"] for row in selected if task_name(row) == "snli") != Counter(
            {"contradiction": 128, "neutral": 128, "entailment": 128}):
        raise ValueError("Expected balanced SNLI labels")
    return selected


def make_orders(rows, mode, seed):
    """Return one 1024-row epoch; each consecutive eight rows form one update."""
    if mode not in ("mixed", "stratified"):
        raise ValueError("Unknown task-balance mode")
    selected = _training_rows(rows)
    rng = random.Random(seed + 91001)
    if mode == "mixed":
        order = sorted((row["id"] for row in selected))
        rng.shuffle(order)
        return [order]

    by_task_label = defaultdict(list)
    replay = []
    snli_groups = defaultdict(dict)
    for row in selected:
        task = task_name(row)
        if task == "replay":
            replay.append(row["id"])
        elif task == "snli":
            snli_groups[row["group_id"]][row["label"]] = row["id"]
        else:
            by_task_label[(task, row["label"])].append(row["id"])
    labels = ("contradiction", "neutral", "entailment")
    if len(snli_groups) != 128 or any(set(value) != set(labels) for value in snli_groups.values()):
        raise ValueError("Expected 128 complete SNLI premise groups")
    for values in by_task_label.values():
        rng.shuffle(values)
    rng.shuffle(replay)
    groups = sorted(snli_groups)
    rng.shuffle(groups)

    paws_offsets = {"different": 0, "paraphrase": 0}
    order = []
    for update in range(128):
        paws_pattern = (("different", "different", "paraphrase") if update % 2 == 0
                        else ("different", "paraphrase", "paraphrase"))
        block = []
        for label in paws_pattern:
            offset = paws_offsets[label]
            block.append(by_task_label[("paws", label)][offset])
            paws_offsets[label] += 1
        # One row per SNLI label, with fixed rotations so a single update never
        # contains multiple hypotheses from the same premise.
        for label_index, label in enumerate(labels):
            group = groups[(update + label_index * 43) % 128]
            block.append(snli_groups[group][label])
        block.extend(replay[2 * update:2 * update + 2])
        rng.shuffle(block)
        order.extend(block)
    if Counter(order) != Counter(row["id"] for row in selected):
        raise ValueError("Stratified order changed row exposure")
    return [order]


def update_histogram(rows, order):
    by_id = {row["id"]: row for row in rows}
    output = Counter()
    label_output = Counter()
    for start in range(0, len(order), 8):
        block = [by_id[key] for key in order[start:start + 8]]
        output[tuple(sorted(Counter(task_name(row) for row in block).items()))] += 1
        label_output[tuple(sorted(Counter((task_name(row), row["label"]) for row in block).items()))] += 1
    return {
        "task_count_patterns": {str(key): value for key, value in sorted(output.items(), key=lambda item: str(item[0]))},
        "task_label_patterns": {str(key): value for key, value in sorted(label_output.items(), key=lambda item: str(item[0]))},
    }
