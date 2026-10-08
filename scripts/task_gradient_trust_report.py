"""Independent validator for the three-seed gradient trust-region study."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

from decision_model.core import digest, load_rows, metrics
try:
    from semantic_scale_report import paired_diagnostics
    from task_gradient_trust_projection import METHODS
except ModuleNotFoundError:
    from scripts.semantic_scale_report import paired_diagnostics
    from scripts.task_gradient_trust_projection import METHODS


SEEDS = (42, 43, 44)
ARMS = tuple(f"seed{seed}-{method}" for seed in SEEDS for method in METHODS)
PRIMARY = ("paws-wiki", "snli")


def read(path):
    return json.loads(Path(path).read_text())


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def prediction_probabilities(rows, predictions, temperature=None):
    by_id = {point["id"]: point for point in predictions}
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Prediction IDs differ")
    labels, probabilities = [], []
    for row in rows:
        point = by_id[row["id"]]
        choices = [choice["id"] for choice in row["request"]["choices"]]
        if point["label"] != row["label"] or set(point["probabilities"]) != set(choices):
            raise ValueError("Prediction label or candidate contract differs")
        values = [point["probabilities"][choice] for choice in choices]
        if temperature is not None:
            logs = [temperature * math.log(max(value, 1e-45)) for value in values]
            offset = max(logs)
            values = [math.exp(value - offset) for value in logs]
            total = sum(values)
            values = [value / total for value in values]
        labels.append(choices.index(row["label"]))
        probabilities.append(values)
    return labels, probabilities


def recalculate(rows, predictions, temperature=None):
    labels, probabilities = prediction_probabilities(rows, predictions, temperature)
    result = metrics(labels, probabilities)
    result["by_source"] = {}
    for source in sorted({row["source"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["source"] == source]
        result["by_source"][source] = metrics(
            [labels[index] for index in indices],
            [probabilities[index] for index in indices],
        )
    return result


def compare_metrics(calculated, recorded, tolerance=2e-6, path="metrics"):
    """Compare the complete recorded metric tree, including selective outputs."""
    if isinstance(calculated, dict):
        if not isinstance(recorded, dict) or set(calculated) != set(recorded):
            raise ValueError("Recorded metric keys differ: " + path)
        for key in calculated:
            compare_metrics(calculated[key], recorded[key], tolerance, f"{path}.{key}")
        return
    if calculated is None or recorded is None:
        if calculated is not recorded:
            raise ValueError("Recorded metric differs: " + path)
        return
    if isinstance(calculated, (int, float)) and not isinstance(calculated, bool):
        if not isinstance(recorded, (int, float)) or isinstance(recorded, bool) or abs(calculated - recorded) > tolerance:
            raise ValueError("Recorded metric differs: " + path)
        return
    if calculated != recorded:
        raise ValueError("Recorded metric differs: " + path)


def choice_counts(rows, predictions):
    by_id = {row["id"]: row for row in rows}
    output = {}
    for source in sorted({row["source"] for row in rows}):
        truth, predicted = Counter(), Counter()
        for point in predictions:
            row = by_id[point["id"]]
            if row["source"] == source:
                truth[row["label"]] += 1
                predicted[max(point["probabilities"], key=point["probabilities"].get)] += 1
        output[source] = {"truth": dict(truth), "predicted": dict(predicted)}
    return output


def finite_weights(path):
    import torch
    from safetensors.torch import load_file
    state = load_file(str(path))
    if not state or any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Checkpoint contains missing or non-finite trainable weights")
    return sorted(state)


def validate_live_source(root, study, name, expected):
    live = root / name
    if digest(live.read_bytes()) == expected:
        return
    snapshot = study / "frozen-source" / name
    if snapshot.is_file() and digest(snapshot.read_bytes()) == expected:
        return
    if name == "src/decision_model/__init__.py":
        snapshot = study / "seed42-soft_cap_2x_project_trust/source/src__decision_model____init__.py"
        frozen_lines = snapshot.read_text().splitlines()
        live_lines = live.read_text().splitlines()
        if (digest(snapshot.read_bytes()) == expected and len(frozen_lines) == len(live_lines) and
                all(left == right or (left.startswith("__version__ = ") and right.startswith("__version__ = "))
                    for left, right in zip(frozen_lines, live_lines))):
            return
    raise ValueError("Frozen study source changed: " + name)


def _validate_cap(step):
    cap = step["transformation"]["cap"]
    expected_median = sorted(step["raw_task_norms"].values())[1]
    if (cap["amplified_tasks"] or cap["multiplier"] != 2.0 or
            not math.isclose(cap["median_norm"], expected_median, rel_tol=1e-9) or
            not math.isclose(cap["threshold_norm"], 2 * expected_median, rel_tol=1e-9) or
            any(not 0 <= scale <= 1 for scale in cap["scales"].values())):
        raise ValueError("Soft-cap identity differs")
    for name, raw_norm in step["raw_task_norms"].items():
        expected_scale = min(1.0, cap["threshold_norm"] / raw_norm) if raw_norm else 1.0
        if not math.isclose(cap["scales"][name], expected_scale, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Soft-cap scale differs")


def _validate_trust(step):
    trust = step["transformation"]["trust_region"]
    alpha = trust["alpha"]
    raw = trust["reference_raw_combined_norm"]
    full = trust["full_projected_combined_norm"]
    final = trust["final_combined_norm"]
    if not 0 <= alpha <= 1:
        raise ValueError("Trust interpolation is outside [0, 1]")
    if trust["active"] != (alpha < 1 - 1e-12):
        raise ValueError("Trust active flag differs")
    if not math.isclose(trust["bound_norm"], raw, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Trust bound is not the raw combined norm")
    if not math.isclose(final, step["combined_diagnostic_norm"], rel_tol=2e-5, abs_tol=2e-5):
        raise ValueError("Trust final norm differs from applied gradient")
    tolerance = max(1e-5, raw * 2e-6)
    if final > raw + tolerance:
        raise ValueError("Trust region violated its norm bound")
    if full <= raw + 1e-10 and alpha != 1:
        raise ValueError("Trust region rejected a non-amplifying full correction")
    if alpha == 1 and not math.isclose(final, full, rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("Full trust step does not match projected norm")
    if alpha == 0 and not math.isclose(final, raw, rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("Zero trust step does not match raw norm")
    if 0 < alpha < 1 and not math.isclose(final, raw, rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("Partial trust step does not reach the boundary")
    return final - raw


def validated(root, study):
    protocol = read(study / "protocol.json")
    if protocol["seeds"] != list(SEEDS) or protocol["methods"] != list(METHODS) or protocol["arms"] != list(ARMS):
        raise ValueError("Unexpected study arms")
    if read(study / "status.json") != {"state": "complete", "models": list(ARMS)}:
        raise ValueError("Study is incomplete")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen protocol input changed: " + name)
    frozen_manifest = read(study / "frozen-source-manifest.json")
    if frozen_manifest != {"source_files": protocol["source_files"]}:
        raise ValueError("Frozen source manifest differs")
    for name, expected in protocol["source_files"].items():
        if digest((study / "frozen-source" / name).read_bytes()) != expected:
            raise ValueError("Frozen source snapshot changed: " + name)
    for name, expected in protocol["source_files"].items():
        validate_live_source(root, study, name, expected)
    start = root / protocol["initial_checkpoint"]
    if (digest((start / "decision.safetensors").read_bytes()) != protocol["initial_weight_sha256"] or
            digest((start / "model.json").read_bytes()) != protocol["initial_model_sha256"]):
        raise ValueError("Common start changed")
    for directory, status_key, manifest_key in (
        ("runs/task-gradient-control-v1", "raw_parent_status_sha256", "raw_parent_manifest_sha256"),
        ("runs/task-gradient-composition-v1", "composition_parent_status_sha256", "composition_parent_manifest_sha256"),
    ):
        parent = root / directory
        if (digest((parent / "status.json").read_bytes()) != protocol[status_key] or
                digest((parent / "trained-checkpoints.json").read_bytes()) != protocol[manifest_key]):
            raise ValueError("Reusable parent manifest changed: " + directory)
    for name, record in protocol["reused_controls"].items():
        directory = root / record["path"]
        if (digest((directory / "decision.safetensors").read_bytes()) != record["weights_sha256"] or
                digest((directory / "model.json").read_bytes()) != record["model_sha256"]):
            raise ValueError("Reusable control changed: " + name)

    rows = load_rows(study / "cases.jsonl")
    diagnostic_rows = load_rows(study / "diagnostic-cases.jsonl")
    if digest((study / "cases.jsonl").read_bytes()) != protocol["dataset_sha256"]:
        raise ValueError("Study rows changed")
    if digest((study / "diagnostic-cases.jsonl").read_bytes()) != protocol["diagnostic_dataset_sha256"]:
        raise ValueError("Diagnostic rows changed")
    test_rows = [row for row in rows if row["split"] == "test"]
    finals = read(study / "trained-checkpoints.json")
    if set(finals) != set(ARMS):
        raise ValueError("Checkpoint manifest differs")

    arms, first_records = {}, {}
    parameter_names = initial_hash = None
    for arm in ARMS:
        seed_text, method = arm.split("-", 1)
        seed = int(seed_text.removeprefix("seed"))
        directory = root / finals[arm]["path"]
        if (digest((directory / "decision.safetensors").read_bytes()) != finals[arm]["weights_sha256"] or
                digest((directory / "model.json").read_bytes()) != finals[arm]["model_sha256"]):
            raise ValueError("Final checkpoint changed: " + arm)
        names = finite_weights(directory / "decision.safetensors")
        parameter_names = names if parameter_names is None else parameter_names
        if names != parameter_names:
            raise ValueError("Checkpoint parameter sets differ")
        config = read(study / f"seed{seed}.json")
        run = read(directory / "run.json")
        expected_version = ({"seed42-raw": "task-gradient-control-v1", "seed43-raw": "task-gradient-control-v1",
                             "seed42-soft_cap_2x_then_project": "task-gradient-composition-v1",
                             "seed43-soft_cap_2x_then_project": "task-gradient-composition-v1"}.get(
                                 arm, "task-gradient-trust-v1"))
        if (read(directory / "model.json") != config or run["config_sha256"] != protocol["config_sha256"][str(seed)] or
                run["method"] != method or run["method_version"] != expected_version or run["seed"] != seed or
                run["engineering_smoke"] or run["updates"] != 128 or
                run["budget"]["task_forwards_per_update"] != 3 or
                run["selected_ids"] != protocol["selected_ids"] or
                run["warm_start_weights_sha256"] != protocol["initial_weight_sha256"]):
            raise ValueError("Training protocol differs: " + arm)
        initial_hash = run["initial_trainable_sha256"] if initial_hash is None else initial_hash
        if run["initial_trainable_sha256"] != initial_hash:
            raise ValueError("Initial trainable parameters differ")
        for filename, expected in read(directory / "checksums.json").items():
            if digest((directory / filename).read_bytes()) != expected:
                raise ValueError("Checkpoint checksum differs: " + arm)

        plan_raw = (directory / "training-plan.jsonl").read_bytes()
        if digest(plan_raw) != run["training_plan_sha256"]:
            raise ValueError("Training plan hash differs: " + arm)
        plans = [json.loads(line) for line in plan_raw.splitlines()]
        exposed = []
        if len(plans) != 128:
            raise ValueError("Training plan length differs")
        for index, plan in enumerate(plans, 1):
            if plan["update"] != index or {name: len(plan["tasks"][name]) for name in plan["tasks"]} != {
                    "paws": 3, "snli": 3, "replay": 2}:
                raise ValueError("Task update composition differs")
            exposed.extend(identity for name in ("paws", "snli", "replay") for identity in plan["tasks"][name])
        if Counter(exposed) != Counter(protocol["selected_ids"]["train"]):
            raise ValueError("Training exposure differs")

        steps = [json.loads(line) for line in (directory / "optimizer-steps.jsonl").read_text().splitlines()]
        if len(steps) != 128 or any(step["update"] != index for index, step in enumerate(steps, 1)):
            raise ValueError("Optimizer steps differ")
        bound_deltas = []
        for step in steps:
            numbers = [step["weighted_risk"], step["weighted_surrogate"], step["combined_pre_clip_norm"],
                       step["combined_diagnostic_norm"], *step["raw_task_norms"].values(),
                       *step["raw_task_cosines"].values()]
            if any(not math.isfinite(value) for value in numbers):
                raise ValueError("Non-finite optimizer diagnostic")
            if not math.isclose(step["combined_pre_clip_norm"], step["combined_diagnostic_norm"], rel_tol=2e-5, abs_tol=2e-5):
                raise ValueError("Gradient norm implementations disagree")
            expected_kind = {"raw": "identity", "soft_cap_2x_then_project": "two_x_median_upper_cap_then_symmetric_projection",
                             "soft_cap_2x_project_trust": "two_x_median_cap_projection_trust_region"}[method]
            if step["transformation"]["kind"] != expected_kind:
                raise ValueError("Gradient transform differs")
            if method != "raw":
                _validate_cap(step)
            if method == "soft_cap_2x_project_trust":
                bound_deltas.append(_validate_trust(step))
        first_records[arm] = steps[0]

        evaluation = read(directory / "evaluation.json")
        predictions = read(directory / "test-predictions.json")
        temperature = read(directory / "calibration.json")["temperature"]
        compare_metrics(recalculate(test_rows, predictions), evaluation["test"]["temperature_scaled"])
        compare_metrics(recalculate(test_rows, predictions, temperature), evaluation["test"]["raw"])
        norms = [step["combined_pre_clip_norm"] for step in steps]
        task_norms = {name: [step["raw_task_norms"][name] for step in steps] for name in ("paws", "snli", "replay")}
        trust_records = [step["transformation"].get("trust_region") for step in steps]
        trust_records = [value for value in trust_records if value]
        arms[arm] = {
            "raw": evaluation["test"]["raw"], "predictions": predictions,
            "training": {
                "training_seconds": run["training_seconds"], "gradient_clipping": run["gradient_clipping"],
                "combined_norm": {"mean": mean(norms), "p95": percentile(norms, .95), "max": max(norms)},
                "raw_task_norm": {name: {"mean": mean(values), "p95": percentile(values, .95), "max": max(values)}
                                  for name, values in task_norms.items()},
                "conflict_updates": sum(bool(step["transformation"].get("projection", {}).get("conflicting_pairs", [])) for step in steps),
                "conflict_pairs": sum(len(step["transformation"].get("projection", {}).get("conflicting_pairs", [])) for step in steps),
                "cap_updates": sum(any(scale < 1 for scale in step["transformation"].get("cap", {}).get("scales", {}).values()) for step in steps),
                "minimum_cap_scale": min((scale for step in steps for scale in step["transformation"].get("cap", {}).get("scales", {}).values()), default=None),
                "trust_active_updates": sum(value["active"] for value in trust_records),
                "trust_partial_updates": sum(0 < value["alpha"] < 1 for value in trust_records),
                "trust_alpha_mean": mean(value["alpha"] for value in trust_records) if trust_records else None,
                "maximum_bound_delta": max(bound_deltas) if bound_deltas else None,
            },
        }

    for seed in SEEDS:
        plan_bytes = [(root / finals[f"seed{seed}-{method}"]["path"] / "training-plan.jsonl").read_bytes() for method in METHODS]
        if not plan_bytes[0] == plan_bytes[1] == plan_bytes[2]:
            raise ValueError("Candidate/row plans differ within seed")
        reference = first_records[f"seed{seed}-raw"]
        for method in METHODS[1:]:
            value = first_records[f"seed{seed}-{method}"]
            for key in ("task_risk", "task_surrogate", "raw_task_norms", "raw_task_cosines"):
                if value[key] != reference[key]:
                    raise ValueError("Common-start first-step diagnostic differs within seed")

    manifest = read(study / "diagnostic-manifest.json")
    if manifest["order"] != list(ARMS) or manifest["dataset_sha256"] != protocol["diagnostic_dataset_sha256"]:
        raise ValueError("Diagnostic manifest differs")
    diagnostic = {}
    for arm in ARMS:
        value = read(study / "diagnostic-results" / f"{arm}.json")
        if value["dataset_sha256"] != protocol["diagnostic_dataset_sha256"] or value["weights_sha256"] != finals[arm]["weights_sha256"]:
            raise ValueError("Diagnostic identity differs: " + arm)
        compare_metrics(recalculate(diagnostic_rows, value["predictions"]), value["raw"])
        diagnostic[arm] = {"raw": value["raw"], "predictions": value["predictions"],
                           "choice_counts": choice_counts(diagnostic_rows, value["predictions"])}

    paired, diagnostic_paired = {}, {}
    for seed in SEEDS:
        raw = arms[f"seed{seed}-raw"]["predictions"]
        raw_diagnostic = diagnostic[f"seed{seed}-raw"]["predictions"]
        for method in METHODS[1:]:
            key = f"seed{seed}-{method}_minus_raw"
            paired[key] = paired_diagnostics(test_rows, raw, arms[f"seed{seed}-{method}"]["predictions"])
            diagnostic_paired[key] = paired_diagnostics(
                diagnostic_rows, raw_diagnostic, diagnostic[f"seed{seed}-{method}"]["predictions"])
    return protocol, arms, diagnostic, paired, diagnostic_paired
