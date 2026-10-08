"""Evaluate the fixed warm-start checkpoint on the new study's exact split rule."""
import argparse
import json
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.evaluate import load_judge
from decision_model.train import balanced_subset, calibrate, evaluate_rows, predict_rows


def main():
    parser = argparse.ArgumentParser()
    for option in ("checkpoint", "data", "config", "output", "base-path"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    judge = load_judge(args)
    tokens, rows = {}, []
    for row in load_rows(args.data):
        if row["split"] == "train":
            continue
        try:
            tokens[row["id"]] = judge.encode(row["request"])
        except ValueError as exc:
            if "Input too long" not in str(exc):
                raise
            continue
        rows.append(row)
    splits = {split: balanced_subset([r for r in rows if r["split"] == split], config["max_" + split + "_rows"])
              for split in ("validation", "test")}
    values = {split: predict_rows(judge, selected, tokens) for split, selected in splits.items()}
    calibration = calibrate(values["validation"], splits["validation"])
    result = {"weight_sha256": digest((Path(args.checkpoint)/"decision.safetensors").read_bytes()),
              "dataset_sha256": digest(Path(args.data).read_bytes()),
              "selected_ids": {split: [r["id"] for r in selected] for split, selected in splits.items()},
              "calibration": calibration, "evaluation": {},
              "note": "Unchanged starting weights; temperature refitted only on this study's validation set"}
    for split in splits:
        raw, _ = evaluate_rows(splits[split], values[split])
        scaled, predictions = evaluate_rows(splits[split], values[split], calibration["temperature"])
        result["evaluation"][split] = {"raw": raw, "temperature_scaled": scaled}
        if split == "test":
            result["predictions"] = predictions
    write_json(args.output, result)
    print(json.dumps({"event": "reference_complete", "test_rows": len(splits["test"])}))


if __name__ == "__main__":
    main()
