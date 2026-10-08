"""Combine pinned public task datasets without changing labels or split roles."""
import argparse
import json
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, summarize_rows, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/public-multitask-v2-inputs.json")
    parser.add_argument("--output", default="data/public-multitask-v2")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new dataset directory")
    lock = json.loads(Path(args.lock).read_text())
    rows = []
    for entry in lock["inputs"]:
        path = Path(entry["path"])
        if digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError("Input differs from pinned public preparation: " + str(path))
        rows.extend(load_rows(path))
    output.mkdir(parents=True)
    target = output / "cases.jsonl"
    target.write_text("".join(canonical(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))
    verified = load_rows(target)  # Includes global group/input/ID leakage checks.
    manifest = {"inputs": lock["inputs"], "rows": summarize_rows(verified),
                "dataset_sha256": digest(target.read_bytes()), "script_sha256": digest(Path(__file__).read_bytes()),
                "transformation": "Concatenate verified inputs; retain labels, source provenance, groups and split roles; sort by ID",
                "license": "Each source retains its own license; SNLI-derived records are CC-BY-SA-4.0",
                "evaluation_note": "Previously inspected sources remain development diagnostics; composition does not create a fresh blind test"}
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
