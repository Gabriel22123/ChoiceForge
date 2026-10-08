"""Prepare a bounded, provenance-preserving SNLI research dataset locally."""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import io
import json
from pathlib import Path
import unicodedata
import zipfile

from decision_model.core import canonical, digest, load_rows, summarize_rows, write_json

SPLITS = {"train": "train", "dev": "validation", "test": "test"}
DESCRIPTIONS = {
    "entailment": "The premise supports the hypothesis.",
    "contradiction": "The premise contradicts the hypothesis.",
    "neutral": "The premise does not determine whether the hypothesis is true or false.",
}
SOURCE_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def records(archive, skipped=None):
    with zipfile.ZipFile(archive) as stream:
        for original, split in SPLITS.items():
            member = f"snli_1.0/snli_1.0_{original}.jsonl"
            with stream.open(member) as raw:
                for index, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8")):
                    row = json.loads(line)
                    label = row["gold_label"]
                    if label not in DESCRIPTIONS:
                        if skipped is not None:
                            skipped["unknown_or_no_consensus_label_rows"] += 1
                        continue
                    premise, hypothesis = row["sentence1"], row["sentence2"]
                    if not premise.strip() or not hypothesis.strip():
                        raise ValueError("Empty SNLI sentence")
                    group = digest(normalized(premise))
                    semantic = digest([normalized(premise), normalized(hypothesis)])
                    yield split, original, member, index, row, group, semantic


def transform(archive, output, expected_sha256, per_label):
    if output.exists():
        raise ValueError("Use a new output directory")
    with archive.open("rb") as raw:
        archive_sha = hashlib.file_digest(raw, "sha256").hexdigest()
    if archive_sha != expected_sha256:
        raise ValueError("SNLI archive checksum mismatch")
    if any(per_label.get(split, 0) <= 0 for split in SPLITS.values()):
        raise ValueError("Each split needs a positive per-label cap")

    # Only compact fingerprints/label sets are retained for the full source.
    groups, labels = defaultdict(set), defaultdict(set)
    eligible, excluded = Counter(), Counter()
    for split, _, _, _, row, group, semantic in records(archive, excluded):
        groups[group].add(split)
        labels[semantic].add(row["gold_label"])
        eligible[split] += 1
    cross_groups = {group for group, splits in groups.items() if len(splits) > 1}
    conflicts = {key for key, values in labels.items() if len(values) > 1}
    del groups, labels

    heaps, seen = defaultdict(list), set()
    for split, original, member, index, raw, group, semantic in records(archive):
        if group in cross_groups:
            excluded["cross_split_premise_rows"] += 1
            continue
        if semantic in conflicts:
            excluded["conflicting_label_rows"] += 1
            continue
        if semantic in seen:
            excluded["duplicate_pair_rows"] += 1
            continue
        seen.add(semantic)
        label = raw["gold_label"]
        identity = digest(["snli-1.0", semantic])
        # Bottom-hash selection per official split and label. Not a natural prior.
        priority = int(identity, 16)
        heap = heaps[(split, label)]
        if len(heap) >= per_label[split] and priority >= -heap[0][0]:
            continue
        choices = [{"id": key, "description": value} for key, value in DESCRIPTIONS.items()]
        choices.sort(key=lambda choice: digest([identity, choice["id"]]))
        row = {
            "id": identity, "group_id": "snli:" + group, "source": "snli",
            "family": "natural_language_inference", "split": split,
            "evaluation_regime": "seen_task_new_examples", "label": label,
            "request": {
                "task": "Determine how the premise relates to the hypothesis. Use only the information stated in the premise.",
                "context": canonical({"premise": raw["sentence1"], "hypothesis": raw["sentence2"]}),
                "choices": choices,
            },
            "provenance": [{"url": SOURCE_URL, "revision": "SNLI-1.0",
                "file_sha256": archive_sha, "member": member, "index": index,
                "pair_id": raw.get("pairID"), "original_split": original,
                "license": "CC-BY-SA-4.0", "label_method": "upstream_gold_label",
                "transformation": "three_self_contained_candidates; normalized-premise group exclusion; bottom-hash label-balanced subset"}],
        }
        entry = (-priority, identity, row)
        if len(heap) < per_label[split]:
            heapq.heappush(heap, entry)
        else:
            heapq.heapreplace(heap, entry)

    rows = sorted((item[2] for heap in heaps.values() for item in heap), key=lambda row: row["id"])
    if any((split, label) not in heaps for split in SPLITS.values() for label in DESCRIPTIONS):
        raise ValueError("Filtering left a split without all three labels")
    output.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        (output / "UPSTREAM_README.txt").write_bytes(source.read("snli_1.0/README.txt"))
    target = output / "cases.jsonl"
    target.write_text("".join(canonical(row) + "\n" for row in rows))
    load_rows(target)
    manifest = {
        "source_url": SOURCE_URL, "source_version": "SNLI-1.0", "archive_sha256": archive_sha,
        "license": "CC-BY-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "source_page": "https://nlp.stanford.edu/projects/snli/",
        "data_sha256": digest(target.read_bytes()), "summary": summarize_rows(rows),
        "eligible_gold_label_rows": dict(eligible), "excluded": dict(excluded),
        "cross_split_premise_groups": len(cross_groups), "conflicting_pairs": len(conflicts),
        "per_label_cap": per_label,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "policy": "Retain official split names; remove every row in cross-split normalized-premise groups; exclude conflicting labels; deduplicate normalized ordered pairs; bottom-hash sample per split/label",
        "limitations": ["Label-balanced subset, not original benchmark or deployment prior",
            "Exact normalized grouping does not catch every semantic near-duplicate",
            "No universal zero-shot or pretraining-clean claim", "Prepared only; not trained by this command"],
    }
    write_json(output / "manifest.json", manifest)
    (output / "ATTRIBUTION.md").write_text(
        "# SNLI-derived research subset\n\n"
        "Source: Stanford Natural Language Inference Corpus 1.0, Stanford NLP Group.\n"
        "Samuel R. Bowman, Gabor Angeli, Christopher Potts and Christopher D. Manning (2015), "
        "A large annotated corpus for learning natural language inference.\n\n"
        "Official source: https://nlp.stanford.edu/projects/snli/\n"
        "Dataset license: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/\n"
        "Includes underlying caption sources credited by the official SNLI page and archive README.\n"
        "Changes: task/candidate formatting, duplicate/conflict and group-leakage filtering, "
        "label-balanced deterministic selection. See manifest.json and per-row provenance.\n"
        "This derived data remains under CC BY-SA 4.0; the project code license is separate.\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock", default="configs/snli-source.json")
    parser.add_argument("--train-per-label", type=int, default=1024)
    parser.add_argument("--validation-per-label", type=int, default=128)
    parser.add_argument("--test-per-label", type=int, default=256)
    args = parser.parse_args()
    lock = json.loads(Path(args.lock).read_text())
    result = transform(Path(args.archive), Path(args.output), lock["sha256"],
                       {"train": args.train_per_label, "validation": args.validation_per_label,
                        "test": args.test_per_label})
    print(json.dumps({"summary": result["summary"], "excluded": result["excluded"],
                      "data_sha256": result["data_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
