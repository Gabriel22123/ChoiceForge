"""Separate fit and sealed-test files for the frozen objective screen."""
import argparse
import json
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, summarize_rows, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/typed-decisions-objective-screen-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / args.protocol; protocol = json.loads(protocol_path.read_text())
    source = root / protocol["source_folds"]; output = root / protocol["prepared_data"]
    if output.exists():
        raise ValueError("Use a new objective-screen data directory")
    output.mkdir(parents=True)
    folds = {}
    for path in sorted(source.glob("heldout-*/cases.jsonl")):
        name = path.parent.name.removeprefix("heldout-"); rows = load_rows(path)
        fit = [row for row in rows if row["split"] in ("train", "validation")]
        test = [row for row in rows if row["split"] == "test"]
        if len(fit) != 4500 or len(test) != 2000:
            raise ValueError("Unexpected full-fold row counts")
        if any(row.get("evaluation_regime") == "held_out_task" for row in fit):
            raise ValueError("Held-out workflow leaked into fit file")
        target = output / ("heldout-" + name); target.mkdir()
        for filename, values in (("fit.jsonl", fit), ("test.jsonl", test)):
            (target / filename).write_text("".join(canonical(row) + "\n" for row in values))
            load_rows(target / filename)
        manifest = {
            "held_out_workflow": name, "source": path.relative_to(root).as_posix(),
            "source_sha256": digest(path.read_bytes()),
            "fit_sha256": digest((target / "fit.jsonl").read_bytes()),
            "test_sha256": digest((target / "test.jsonl").read_bytes()),
            "fit": summarize_rows(fit),
            "sealed_test": {"rows": len(test), "groups": len({row["group_id"] for row in test}),
                            "families": {family: sum(row["family"] == family for row in test)
                                         for family in sorted({row["family"] for row in test})}},
            "test_policy": "deterministic separation only; training runner must not parse test.jsonl before global lock",
        }
        write_json(target / "manifest.json", manifest); folds[name] = manifest
    if len(folds) != 4:
        raise ValueError("Expected four held-out workflow folds")
    write_json(output / "manifest.json", {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": digest(protocol_path.read_bytes()),
        "preparation_script_sha256": digest(Path(__file__).read_bytes()),
        "folds": folds,
    })
    print(json.dumps({"folds": sorted(folds), "fit_rows_per_fold": 4500,
                      "sealed_test_rows_per_fold": 2000}, indent=2))


if __name__ == "__main__":
    main()
