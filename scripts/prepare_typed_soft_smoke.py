"""Select a tiny deterministic real-data subset for soft-label integration checks."""
import argparse
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, summarize_rows, write_json


def select(rows):
    selected = []
    for split, limit in (("train", 6), ("validation", 3)):
        pool = [row for row in rows if row["split"] == split]
        families = sorted({row["family"] for row in pool})
        for index in range(limit):
            selected.append(next(row for row in pool
                                 if row["family"] == families[index % len(families)]
                                 and row not in selected))
    for family in sorted({row["family"] for row in rows if row["split"] == "test"}):
        selected.append(next(row for row in rows
                             if row["split"] == "test" and row["family"] == family))
    return selected


def prepare(source, output):
    source = Path(source); output = Path(output)
    if output.exists():
        raise ValueError("Use a new smoke-data output directory")
    selected = select(load_rows(source))
    output.mkdir(parents=True)
    target = output / "cases.jsonl"
    target.write_text("".join(canonical(row) + "\n" for row in selected))
    verified = load_rows(target)
    manifest = {
        "purpose": "real-data integration smoke only; not a quality experiment",
        "source": str(source), "source_sha256": digest(source.read_bytes()),
        "dataset_sha256": digest(target.read_bytes()), "rows": summarize_rows(verified),
        "selection": "first rows by sorted source order, round-robin seen family; one official-test row per family",
        "script_sha256": digest(Path(__file__).read_bytes()),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/typed-decisions-workflow-folds-v1/heldout-security_incidents/cases.jsonl")
    parser.add_argument("--output", default="data/typed-decisions-soft-smoke-v1")
    args = parser.parse_args()
    print(canonical(prepare(args.source, args.output)))


if __name__ == "__main__":
    main()
