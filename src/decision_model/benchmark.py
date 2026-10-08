"""Model-independent benchmark and audit reports for candidate decisions."""

import json
import math
from collections import Counter
from pathlib import Path

from .core import digest, load_rows, write_json

DEFAULT_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)


def _read_predictions(path):
    raw = Path(path).read_text()
    if Path(path).suffix == ".jsonl":
        value = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        value = json.loads(raw)
        if isinstance(value, dict):
            value = value.get("predictions")
    if not isinstance(value, list) or not value:
        raise ValueError("Predictions must be a nonempty JSON list or object with predictions")
    by_id = {}
    for prediction in value:
        if not isinstance(prediction, dict) or not isinstance(prediction.get("id"), str):
            raise ValueError("Every prediction needs a string id")
        if prediction["id"] in by_id:
            raise ValueError("Duplicate prediction id: " + prediction["id"])
        by_id[prediction["id"]] = prediction
    return by_id


def _validate_prediction(row, prediction):
    choice_ids = [choice["id"] for choice in row["request"]["choices"]]
    probabilities = prediction.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(choice_ids):
        raise ValueError("Prediction probabilities must match every candidate for " + row["id"])
    values = [probabilities[choice_id] for choice_id in choice_ids]
    if (any(not isinstance(value, (int, float)) or isinstance(value, bool) or
            not math.isfinite(value) or not 0 <= value <= 1 for value in values) or
            not math.isclose(sum(values), 1.0, abs_tol=1e-5)):
        raise ValueError("Invalid probability vector for " + row["id"])
    choice_id = prediction.get("choice_id")
    if choice_id not in choice_ids:
        raise ValueError("Prediction choice_id is not a candidate for " + row["id"])
    if "requires_review" in prediction and not isinstance(prediction["requires_review"], bool):
        raise ValueError("requires_review must be boolean for " + row["id"])
    return choice_ids, values


def _metrics(rows, predictions, thresholds):
    if not rows:
        return {"n": 0}
    selected_correct = 0
    argmax_correct = 0
    nll = brier = ece = 0.0
    confidences = []
    predicted = []
    labels = []
    soft_cross_entropy = soft_brier = 0.0
    has_soft = True
    review_count = 0
    for row in rows:
        prediction = predictions[row["id"]]
        choice_ids, values = _validate_prediction(row, prediction)
        label_index = choice_ids.index(row["label"])
        selected_index = choice_ids.index(prediction["choice_id"])
        predicted_index = max(range(len(values)), key=values.__getitem__)
        confidence = values[predicted_index]
        labels.append(label_index)
        predicted.append(predicted_index)
        confidences.append(confidence)
        selected_correct += selected_index == label_index
        argmax_correct += predicted_index == label_index
        nll -= math.log(max(values[label_index], 1e-12))
        brier += sum((value - int(index == label_index)) ** 2
                     for index, value in enumerate(values))
        if prediction.get("requires_review") is True:
            review_count += 1
        target = row.get("target_probabilities")
        if target is None:
            has_soft = False
        else:
            target_values = [float(target[choice_id]) for choice_id in choice_ids]
            soft_cross_entropy -= sum(
                target_value * math.log(max(value, 1e-12))
                for target_value, value in zip(target_values, values)
            )
            soft_brier += sum((value - target_value) ** 2
                              for target_value, value in zip(target_values, values))
    n = len(rows)
    for bucket in range(10):
        indices = [index for index, confidence in enumerate(confidences)
                   if min(int(confidence * 10), 9) == bucket]
        if indices:
            ece += abs(
                sum(confidences[index] for index in indices)
                - sum(predicted[index] == labels[index] for index in indices)
            ) / n
    selective = {}
    for threshold in thresholds:
        selected = [index for index, confidence in enumerate(confidences)
                    if confidence >= threshold]
        selected_count_correct = sum(
            predicted[index] == labels[index] for index in selected
        )
        selective[str(threshold)] = {
            "coverage": len(selected) / n,
            "risk": 1 - selected_count_correct / len(selected) if selected else None,
            "accuracy": selected_count_correct / len(selected) if selected else None,
            "selected_count": len(selected),
        }
    result = {
        "n": n,
        "selected_correct": selected_correct,
        "selection_accuracy": selected_correct / n,
        "argmax_correct": argmax_correct,
        "argmax_accuracy": argmax_correct / n,
        "nll": nll / n,
        "brier": brier / n,
        "ece_10": ece,
        "review_rate": review_count / n,
        "selective": selective,
    }
    if has_soft:
        result["target_cross_entropy"] = soft_cross_entropy / n
        result["target_brier"] = soft_brier / n
    return result


def _group_metrics(rows, predictions, thresholds, field):
    values = sorted({row.get(field) for row in rows if row.get(field) is not None})
    return {
        value: _metrics(
            [row for row in rows if row.get(field) == value], predictions, thresholds
        )
        for value in values
    }


def benchmark(args):
    data_path = Path(args.data)
    prediction_path = Path(args.predictions)
    rows = load_rows(data_path)
    predictions = _read_predictions(prediction_path)
    row_ids = {row["id"] for row in rows}
    prediction_ids = set(predictions)
    if row_ids != prediction_ids:
        missing = sorted(row_ids - prediction_ids)
        extra = sorted(prediction_ids - row_ids)
        raise ValueError(f"Prediction IDs differ from dataset; missing={missing[:3]}, extra={extra[:3]}")
    thresholds = tuple(args.thresholds or DEFAULT_THRESHOLDS)
    if (not thresholds or any(not 0 < threshold <= 1 for threshold in thresholds)
            or len(set(thresholds)) != len(thresholds)):
        raise ValueError("Thresholds must be unique values in (0, 1]")
    manifest = None
    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path:
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("Benchmark manifest must be a JSON object")
        expected_data = manifest.get("data_sha256")
        actual_data = digest(data_path.read_bytes())
        if expected_data is not None and expected_data != actual_data:
            raise ValueError("Benchmark manifest data_sha256 does not match dataset")
    result = {
        "protocol": {
            "name": "choiceforge-decision-benchmark",
            "version": "0.1",
            "manifest": manifest,
        },
        "data_sha256": digest(data_path.read_bytes()),
        "predictions_sha256": digest(prediction_path.read_bytes()),
        "rows": len(rows),
        "sources": dict(Counter(row.get("source") for row in rows)),
        "metrics": _metrics(rows, predictions, thresholds),
        "by_family": _group_metrics(rows, predictions, thresholds, "family"),
        "by_decision_type": _group_metrics(rows, predictions, thresholds, "decision_type"),
        "by_evaluation_regime": _group_metrics(
            rows, predictions, thresholds, "evaluation_regime"
        ),
        "thresholds": list(thresholds),
    }
    write_json(args.output, result)
    return {"rows": len(rows), "output": str(args.output), "data_sha256": result["data_sha256"]}
