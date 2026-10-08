"""Run the frozen, inference-only candidate-prior correction study."""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
from pathlib import Path
import shutil

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base
from decision_model.judge_factory import make_judge
from decision_model.prior_correction import STRENGTH_GRID, predict_with_prior_correction, select_shared_strength
from decision_model.prior_evaluation import (candidate_marginal_gap,
                                             collect_evidence_and_prior_logits,
                                             evaluate_with_temperature,
                                             fit_and_evaluate)


SEEDS = (42, 43, 44)
DEVELOPMENT_SEEDS = (42, 43)
CONFIRMATION_SEED = 44
ARCHITECTURE_STUDY = "runs/architecture-comparison-v1"
STUDY = "runs/candidate-prior-v1"
BLIND = "logical-deduction-blind"
BASE_PATH = "models/minicpm5-2b-base"


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes(root):
    paths = [
        root / "src/decision_model/core.py",
        root / "src/decision_model/output_contract.py",
        root / "src/decision_model/prior_correction.py",
        root / "src/decision_model/prior_evaluation.py",
        root / "src/decision_model/decoder_model.py",
        root / "src/decision_model/judge_factory.py",
        root / "src/decision_model/train.py",
        root / "src/decision_model/decoder_base.lock.json",
        root / "scripts/prepare_logical_deduction_blind.py",
        root / "scripts/candidate_prior_report.py",
        Path(__file__).resolve(),
    ]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def checkpoint_manifest(root, architecture_study):
    finals = read(architecture_study / "trained-checkpoints.json")
    result = {}
    for seed in SEEDS:
        arm = f"seed{seed}-decoder"
        record = finals[arm]
        checkpoint = root / record["path"]
        if (digest((checkpoint / "decision.safetensors").read_bytes()) != record["weights_sha256"] or
                digest((checkpoint / "model.json").read_bytes()) != record["model_sha256"] or
                digest((checkpoint / "checksums.json").read_bytes()) != record["checksums_sha256"]):
            raise ValueError("Architecture checkpoint changed: " + arm)
        result[str(seed)] = record
    return result


def prepare(root, output, device):
    architecture = root / ARCHITECTURE_STUDY
    status = read(architecture / "status.json")
    if (status.get("state") != "complete" or
            status.get("result_manifest_sha256") != digest((architecture / "result-manifest.json").read_bytes())):
        raise ValueError("Architecture study is incomplete")
    blind = output / BLIND
    if not blind.is_dir() or read(blind / "status.json") != {"state": "prepared"}:
        raise ValueError("Prepare the logical-deduction blind set first")
    blind_protocol = read(blind / "protocol.json")
    if (blind_protocol["rows"] != 288 or not blind_protocol["selection_is_answer_blind"] or
            blind_protocol["exact_training_request_overlap"] != 0 or
            digest((blind / "cases.jsonl").read_bytes()) != blind_protocol["cases_sha256"]):
        raise ValueError("Logical-deduction blind protocol differs")
    if (output / "protocol.json").exists():
        raise ValueError("Candidate-prior protocol already exists")
    verify_local_decoder_base(root / BASE_PATH)
    checkpoints = checkpoint_manifest(root, architecture)
    hashes = source_hashes(root)
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question": ("Does subtracting a centered null-context candidate score reduce "
                     "candidate-prior drift without degrading evidence-conditioned decisions?"),
        "device": device, "seeds": list(SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS), "confirmation_seed": CONFIRMATION_SEED,
        "strength_grid": list(STRENGTH_GRID),
        "selection": ("Choose one shared strength by equal-seed mean validation temperature-scaled "
                      "NLL on seeds 42/43; ties choose the smaller strength."),
        "method": "adjusted = evidence - lambda * (null_context_prior - mean(null_context_prior))",
        "architecture_study": ARCHITECTURE_STUDY,
        "architecture_result_manifest_sha256": digest((architecture / "result-manifest.json").read_bytes()),
        "architecture_cases_sha256": digest((architecture / "cases.jsonl").read_bytes()),
        "checkpoints": checkpoints,
        "base_path": BASE_PATH,
        "base_lock_sha256": digest((root / "src/decision_model/decoder_base.lock.json").read_bytes()),
        "blind_protocol_sha256": digest((blind / "protocol.json").read_bytes()),
        "blind_cases_sha256": digest((blind / "cases.jsonl").read_bytes()),
        "source_files": hashes,
        "advancement_rule": [
            "selected strength must be greater than zero",
            "on seed44 PAWS and SNLI, neither accuracy delta may be below -1 point, mean accuracy delta must be nonnegative, and mean calibrated NLL delta must be nonpositive",
            "on seed44 retained Bandit/CLINC/GoEmotions/JSON-Schema, mean accuracy and calibrated NLL must not worsen",
            "on fresh logical deduction, corrected combined accuracy must beat raw in at least two of three seeds, mean accuracy delta across all nine seed/subtask comparisons must be positive, and mean calibrated NLL delta must be nonpositive",
            "all 12 real-model audit cases must preserve the selected semantic choice under every tested order and ID rename in all three seeds",
        ],
        "blind_boundary": ("The shared strength is written and hashed before any blind model "
                           "forward pass. Blind rows never select strength or temperature."),
        "release_boundary": ("Raw/transformed blind rows, collected logits and per-case "
                             "predictions are local evidence and excluded from release bundles."),
    }
    snapshot = output / "frozen-source"
    for name, expected in hashes.items():
        source = root / name
        if digest(source.read_bytes()) != expected:
            raise ValueError("Candidate-prior source changed while preparing: " + name)
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol-checksums.json", {
        "protocol.json": digest((output / "protocol.json").read_bytes()),
        f"{BLIND}/protocol.json": digest((blind / "protocol.json").read_bytes()),
        f"{BLIND}/status.json": digest((blind / "status.json").read_bytes()),
        f"{BLIND}/cases.jsonl": digest((blind / "cases.jsonl").read_bytes()),
    })
    write_json(output / "status.json", {"state": "prepared"})
    print({"prepared": str(output), "seeds": list(SEEDS), "blind_rows": 288})


def verify(root, output, protocol):
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen candidate-prior input changed: " + name)
    for name, expected in protocol["source_files"].items():
        live = root / name
        snapshot = output / "frozen-source" / name
        if digest(snapshot.read_bytes()) != expected:
            raise ValueError("Frozen candidate-prior source snapshot changed: " + name)
        if digest(live.read_bytes()) != expected:
            raise ValueError("Candidate-prior source changed: " + name)
    architecture = root / protocol["architecture_study"]
    if (digest((architecture / "result-manifest.json").read_bytes()) !=
            protocol["architecture_result_manifest_sha256"]):
        raise ValueError("Architecture study result manifest changed")
    checkpoint_manifest(root, architecture)
    if digest((root / "src/decision_model/decoder_base.lock.json").read_bytes()) != protocol["base_lock_sha256"]:
        raise ValueError("Decoder base lock changed")
    verify_local_decoder_base(root / protocol["base_path"])


def unload(judge):
    torch = judge.torch
    del judge
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def metric_only(value):
    return {key: item for key, item in value.items() if key != "predictions"}


def audit_corrected(judge, rows, strength, temperature):
    selected = sorted(rows, key=lambda row: digest({"audit": row["id"]}))[:12]
    stable, renamed_stable, maximum = 0, 0, 0.0
    for row in selected:
        request = row["request"]
        reference = predict_with_prior_correction(judge, request, strength, temperature)
        orders = [list(range(len(request["choices"]))),
                  list(reversed(range(len(request["choices"])))),
                  list(range(1, len(request["choices"]))) + [0]]
        order_ok = True
        for order in orders:
            permuted = dict(request, choices=[dict(request["choices"][index]) for index in order])
            value = predict_with_prior_correction(judge, permuted, strength, temperature)
            order_ok &= value["choice_id"] == reference["choice_id"]
            maximum = max(maximum, *(abs(value["probabilities"][identity] - probability)
                                      for identity, probability in reference["probabilities"].items()))
        stable += order_ok
        renamed = dict(request, choices=[{"id": f"renamed_{index}",
                                          "description": choice["description"]}
                                         for index, choice in enumerate(request["choices"])])
        renamed_value = predict_with_prior_correction(judge, renamed, strength, temperature)
        by_description = {choice["description"]: reference["probabilities"][choice["id"]]
                          for choice in request["choices"]}
        renamed_by_description = {choice["description"]: renamed_value["probabilities"][choice["id"]]
                                  for choice in renamed["choices"]}
        difference = max(abs(by_description[key] - renamed_by_description[key])
                         for key in by_description)
        maximum = max(maximum, difference)
        reference_description = next(choice["description"] for choice in request["choices"]
                                     if choice["id"] == reference["choice_id"])
        renamed_description = next(choice["description"] for choice in renamed["choices"]
                                   if choice["id"] == renamed_value["choice_id"])
        renamed_stable += reference_description == renamed_description
    return {"cases": len(selected), "stable_all_orders": stable,
            "stable_under_id_rename": renamed_stable,
            "max_probability_difference": maximum}


def run(root, output, device):
    protocol = read(output / "protocol.json")
    if device != protocol["device"]:
        raise ValueError("Candidate-prior study device differs from frozen protocol")
    if read(output / "status.json").get("state") not in (
            "prepared", "validation_collection", "selection_complete",
            "confirmation_evaluation", "failed"):
        raise ValueError("Candidate-prior study cannot resume from current state")
    verify(root, output, protocol)
    architecture = root / protocol["architecture_study"]
    rows = load_rows(architecture / "cases.jsonl")
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    blind_rows = load_rows(output / BLIND / "cases.jsonl")
    logits_dir = output / "logits"
    logits_dir.mkdir(exist_ok=True)
    validation_dir = output / "validation-grid"
    validation_dir.mkdir(exist_ok=True)

    validation_nll = {}
    validation_results = {}
    for seed in SEEDS:
        destination = logits_dir / f"seed{seed}-validation.json"
        grid_path = validation_dir / f"seed{seed}.json"
        if not destination.exists():
            write_json(output / "status.json", {"state": "validation_collection", "seed": seed})
            checkpoint = root / protocol["checkpoints"][str(seed)]["path"]
            config = read(checkpoint / "model.json")
            judge = make_judge(config, base_path=root / protocol["base_path"],
                               checkpoint=checkpoint, device=device)
            records, summary = collect_evidence_and_prior_logits(judge, validation_rows)
            write_json(destination, {"summary": summary, "records": records})
            unload(judge)
        stored = read(destination)
        grid = {str(strength): metric_only(fit_and_evaluate(
            validation_rows, stored["records"], strength)) for strength in STRENGTH_GRID}
        write_json(grid_path, grid)
        validation_results[str(seed)] = grid
        validation_nll[str(seed)] = {
            strength: grid[str(strength)]["temperature_scaled"]["nll"]
            for strength in STRENGTH_GRID}

    selection_path = output / "selection.json"
    if not selection_path.exists():
        selected = select_shared_strength({str(seed): validation_nll[str(seed)]
                                           for seed in DEVELOPMENT_SEEDS})
        selected["temperatures_by_seed"] = {
            str(seed): validation_results[str(seed)][str(selected["selected_strength"])]["calibration"]["temperature"]
            for seed in SEEDS}
        selected["raw_temperatures_by_seed"] = {
            str(seed): validation_results[str(seed)]["0.0"]["calibration"]["temperature"]
            for seed in SEEDS}
        write_json(selection_path, selected)
    selection_sha = digest(selection_path.read_bytes())
    write_json(output / "status.json", {"state": "selection_complete",
                                         "selection_sha256": selection_sha})
    selection = read(selection_path)
    strength = float(selection["selected_strength"])

    results_dir = output / "results"
    results_dir.mkdir(exist_ok=True)
    audit_dir = output / "audits"
    audit_dir.mkdir(exist_ok=True)
    for seed in SEEDS:
        result_path = results_dir / f"seed{seed}.json"
        audit_path = audit_dir / f"seed{seed}.json"
        if result_path.exists() and audit_path.exists():
            continue
        write_json(output / "status.json", {"state": "confirmation_evaluation", "seed": seed,
                                             "selection_sha256": selection_sha})
        checkpoint = root / protocol["checkpoints"][str(seed)]["path"]
        config = read(checkpoint / "model.json")
        judge = make_judge(config, base_path=root / protocol["base_path"],
                           checkpoint=checkpoint, device=device)
        split_records = {}
        for split, split_rows in (("test", test_rows), ("blind", blind_rows)):
            path = logits_dir / f"seed{seed}-{split}.json"
            if not path.exists():
                records, summary = collect_evidence_and_prior_logits(judge, split_rows)
                write_json(path, {"summary": summary, "records": records,
                                  "selection_sha256_before_collection": selection_sha})
            split_records[split] = read(path)["records"]
        raw_temperature = float(selection["raw_temperatures_by_seed"][str(seed)])
        corrected_temperature = float(selection["temperatures_by_seed"][str(seed)])
        result = {"seed": seed, "selected_strength": strength,
                  "selection_sha256": selection_sha, "splits": {}}
        for split, split_rows in (("test", test_rows), ("blind", blind_rows)):
            raw = evaluate_with_temperature(split_rows, split_records[split], 0.0, raw_temperature)
            corrected = evaluate_with_temperature(
                split_rows, split_records[split], strength, corrected_temperature)
            result["splits"][split] = {
                "raw": raw, "corrected": corrected,
                "raw_marginal_gap": candidate_marginal_gap(split_rows, raw["predictions"]),
                "corrected_marginal_gap": candidate_marginal_gap(split_rows, corrected["predictions"]),
            }
        write_json(result_path, result)
        write_json(audit_path, audit_corrected(
            judge, blind_rows, strength, corrected_temperature))
        unload(judge)

    result_files = ["selection.json"]
    result_files += [f"validation-grid/seed{seed}.json" for seed in SEEDS]
    result_files += [f"logits/seed{seed}-{split}.json" for seed in SEEDS
                     for split in ("validation", "test", "blind")]
    result_files += [f"results/seed{seed}.json" for seed in SEEDS]
    result_files += [f"audits/seed{seed}.json" for seed in SEEDS]
    manifest = {name: digest((output / name).read_bytes()) for name in result_files}
    write_json(output / "result-manifest.json", manifest)
    write_json(output / "status.json", {"state": "complete", "selection_sha256": selection_sha,
                                         "result_manifest_sha256": digest((output / "result-manifest.json").read_bytes())})
    print({"state": "complete", "selected_strength": strength,
           "selection_sha256": selection_sha}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=STUDY)
    parser.add_argument("--device", default="mps")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if args.prepare_only:
        prepare(root, output, args.device)
    else:
        run(root, output, args.device)


if __name__ == "__main__":
    main()
