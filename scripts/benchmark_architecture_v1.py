"""Re-audit the frozen architecture study with the public benchmark protocol."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from decision_model.benchmark import benchmark
from decision_model.core import digest, write_json


THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]


def read(path):
    return json.loads(Path(path).read_text())


def prediction_file(source, target):
    value = read(source)
    if isinstance(value, dict):
        value = value["predictions"]
    converted = []
    for prediction in value:
        probabilities = prediction["probabilities"]
        converted.append({
            "id": prediction["id"],
            "choice_id": max(probabilities, key=probabilities.get),
            "probabilities": probabilities,
            "requires_review": True,
        })
    target.write_text(json.dumps({"predictions": converted}, indent=2) + "\n")


def run(study, output):
    study = Path(study)
    output = Path(output)
    protocol = read(study / "protocol.json")
    arms = protocol["arms"]
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = {}
    with tempfile.TemporaryDirectory(prefix="choiceforge-benchmark-") as temporary:
        temporary = Path(temporary)
        for arm in arms:
            reports[arm] = {}
            for name, data_path, source_path in (
                ("test", study / "cases.jsonl", study / arm / "test-predictions.json"),
                ("arc", study / "arc-blind/cases.jsonl", study / "arc-results" / f"{arm}.json"),
            ):
                predictions_path = temporary / f"{arm}-{name}-predictions.json"
                prediction_file(source_path, predictions_path)
                manifest_path = temporary / f"{arm}-{name}-manifest.json"
                manifest_path.write_text(json.dumps({
                    "name": f"architecture-comparison-v1-{name}",
                    "version": "0.1",
                    "data_sha256": digest(data_path.read_bytes()),
                    "study_protocol_sha256": digest((study / "protocol.json").read_bytes()),
                    "prediction_source_sha256": digest(source_path.read_bytes()),
                }))
                report_path = temporary / f"{arm}-{name}-report.json"
                benchmark(SimpleNamespace(
                    data=str(data_path), predictions=str(predictions_path),
                    output=str(report_path), manifest=str(manifest_path), split="test",
                    thresholds=THRESHOLDS,
                ))
                report = read(report_path)
                reports[arm][name] = {
                    "rows": report["rows"],
                    "split": report["split"],
                    "metrics": report["metrics"],
                    "by_family": report["by_family"],
                    "report_sha256": digest(report_path.read_bytes()),
                    "prediction_source_sha256": digest(source_path.read_bytes()),
                }
    result = {
        "protocol": "choiceforge-decision-benchmark",
        "benchmark_version": "0.1",
        "study": "architecture-comparison-v1",
        "study_protocol_sha256": digest((study / "protocol.json").read_bytes()),
        "study_result_manifest_sha256": digest((study / "result-manifest.json").read_bytes()),
        "arms": reports,
        "conversion": (
            "Existing frozen probability predictions were converted to the public output "
            "contract; choice_id is the probability argmax and requires_review is true."
        ),
        "boundary": (
            "This report audits an existing frozen architecture study. It does not retrain "
            "either base and does not establish a new model-selection result."
        ),
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/architecture-comparison-v1")
    parser.add_argument("--output", default="docs/evidence/architecture-benchmark-v1.json")
    args = parser.parse_args()
    run(args.study, args.output)


if __name__ == "__main__":
    main()
