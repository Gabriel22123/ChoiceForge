"""Run the preregistered three-seed train-time context-advantage study."""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from decision_model.context_advantage_evaluation import summarize_context_advantage
from decision_model.core import digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base
from decision_model.judge_factory import make_judge
from decision_model.prior_evaluation import collect_evidence_and_prior_logits


SEEDS = (42, 43, 44)
CONTROL_STUDY = "runs/architecture-comparison-v1"
BASE_PATH = "models/minicpm5-2b-base"
BLIND = "disambiguation-blind"
WEIGHT = 0.25
GAP = 0.2


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes(root):
    paths = [
        root / "src/decision_model/core.py",
        root / "src/decision_model/decoder_model.py",
        root / "src/decision_model/judge_factory.py",
        root / "src/decision_model/train.py",
        root / "src/decision_model/training_objective.py",
        root / "src/decision_model/prior_correction.py",
        root / "src/decision_model/prior_evaluation.py",
        root / "src/decision_model/context_advantage_evaluation.py",
        root / "src/decision_model/decoder_base.lock.json",
        root / "scripts/prepare_disambiguation_blind.py",
        root / "scripts/context_advantage_report.py",
        Path(__file__).resolve(),
    ]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def control_manifest(root, study):
    finals = read(study / "trained-checkpoints.json")
    controls = {}
    for seed in SEEDS:
        arm = f"seed{seed}-decoder"
        record = finals[arm]
        checkpoint = root / record["path"]
        for filename, field in (("decision.safetensors", "weights_sha256"),
                                ("model.json", "model_sha256"),
                                ("checksums.json", "checksums_sha256"),
                                ("run.json", "run_sha256"),
                                ("evaluation.json", "evaluation_sha256"),
                                ("training-plan.jsonl", "training_plan_sha256"),
                                ("optimizer-steps.jsonl", "optimizer_steps_sha256")):
            if digest((checkpoint / filename).read_bytes()) != record[field]:
                raise ValueError(f"Control checkpoint changed: {arm}/{filename}")
        controls[str(seed)] = record
    return controls


def prepare(root, output, device):
    if (output / "protocol.json").exists():
        raise ValueError("Context-advantage protocol already exists")
    control = root / CONTROL_STUDY
    status = read(control / "status.json")
    if (status.get("state") != "complete" or status.get("result_manifest_sha256") !=
            digest((control / "result-manifest.json").read_bytes())):
        raise ValueError("Completed architecture controls are required")
    blind = output / BLIND
    if not blind.is_dir() or read(blind / "status.json") != {"state": "prepared"}:
        raise ValueError("Prepare the disambiguation blind set before the protocol")
    blind_protocol = read(blind / "protocol.json")
    if (blind_protocol["rows"] != 192 or blind_protocol["selected_groups"] != 64 or
            not blind_protocol["selection_is_answer_blind"] or
            not blind_protocol["complete_group_selection"] or
            blind_protocol["exact_training_request_overlap"] != 0 or
            digest((blind / "cases.jsonl").read_bytes()) != blind_protocol["cases_sha256"]):
        raise ValueError("Disambiguation blind protocol differs")
    verify_local_decoder_base(root / BASE_PATH)
    rows = load_rows(control / "cases.jsonl")
    if len([row for row in rows if row["split"] == "train"]) != 1024:
        raise ValueError("Context-advantage study expects 1,024 training rows")
    controls = control_manifest(root, control)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(control / "cases.jsonl", output / "cases.jsonl")
    configs = {}
    for seed in SEEDS:
        checkpoint = root / controls[str(seed)]["path"]
        config = read(checkpoint / "model.json")
        if "context_advantage_weight" in config or "context_advantage_gap" in config:
            raise ValueError("Control unexpectedly contains context-advantage settings")
        config["context_advantage_weight"] = WEIGHT
        config["context_advantage_gap"] = GAP
        target = output / f"seed{seed}-treatment.json"
        write_json(target, config)
        control_run = read(checkpoint / "run.json")
        configs[str(seed)] = {
            "config_sha256": digest(target.read_bytes()),
            "control_initial_trainable_sha256": control_run["initial_trainable_sha256"],
            "control_training_plan_sha256": digest(
                (checkpoint / "training-plan.jsonl").read_bytes()),
            "control_selected_ids_sha256": control_run["selected_ids_sha256"],
        }
    hashes = source_hashes(root)
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question": ("Does a fixed train-time gold context-advantage margin improve evidence "
                     "dependence and cross-task generalization over matched decoder controls?"),
        "device": device, "seeds": list(SEEDS),
        "intervention": {
            "name": "stop-gradient context advantage",
            "formula": "relu(gap - (log p_gold(real_context) - stopgrad(log p_gold(null_context))))",
            "weight": WEIGHT, "gap": GAP,
            "null_context": "same task and candidate descriptions; context replaced by [NO CONTEXT PROVIDED]",
            "null_forward": "eval mode and no_grad; does not consume training dropout RNG",
            "selection": "single fixed intervention after one functionality-only smoke test; no outcome-based grid",
        },
        "control_study": CONTROL_STUDY,
        "control_result_manifest_sha256": digest((control / "result-manifest.json").read_bytes()),
        "controls": controls,
        "configs": configs,
        "dataset_sha256": digest((output / "cases.jsonl").read_bytes()),
        "base_path": BASE_PATH,
        "base_lock_sha256": digest((root / "src/decision_model/decoder_base.lock.json").read_bytes()),
        "blind_protocol_sha256": digest((blind / "protocol.json").read_bytes()),
        "blind_cases_sha256": digest((blind / "cases.jsonl").read_bytes()),
        "source_files": hashes,
        "matched": [
            "same public 1,024-row training set, one exposure and 128 optimizer updates",
            "same seed-specific row order, candidate permutations, base, LoRA/head initialization and optimizer",
            "same candidate-independent decoder scoring architecture and validation-only temperature calibration",
            "same fixed endpoint selection with no early stopping or test-based selection",
        ],
        "primary_endpoints": [
            "PAWS and SNLI raw accuracy and NLL across all three seeds",
            "retained Bandit, CLINC, GoEmotions and JSON Schema accuracy and NLL",
            "raw-logit gold context advantage and margin shortfall on the fixed public test split",
            "candidate-order and candidate-ID invariance",
            "fresh BIG-bench disambiguation accuracy, NLL, Ambiguous recall and complete-triple accuracy",
            "training time, forward work and gradient norms",
        ],
        "advancement_rule": [
            "mean PAWS+SNLI accuracy delta is positive in at least two of three seeds",
            "no PAWS or SNLI seed/task accuracy delta is below -1 percentage point",
            "mean PAWS+SNLI raw NLL delta across all six comparisons is nonpositive",
            "mean retained-task accuracy delta is nonnegative and mean raw NLL delta is nonpositive",
            "test gold context-advantage mean improves in at least two seeds and overall, while mean margin shortfall decreases overall",
            "all 24 treatment audit cases are stable under every candidate order and candidate-ID rename in every seed",
            "fresh blind combined accuracy improves in at least two seeds and on average, mean raw NLL is non-worse, and mean Ambiguous recall and complete-triple accuracy are non-worse",
            "the intervention advances only if every condition passes; engineering cost is reported separately",
        ],
        "blind_boundary": ("Blind cases are selected and hashed without target access before training. "
                           "No blind model forward pass occurs until all three treatment endpoints, "
                           "training-plan checks and audits are complete; then every control and "
                           "treatment endpoint is evaluated once regardless of prior outcomes."),
        "release_boundary": ("Raw/transformed blind rows, logits and per-case predictions remain "
                             "local evidence and are excluded from release bundles."),
    }
    snapshot = output / "frozen-source"
    for name, expected in hashes.items():
        source = root / name
        if digest(source.read_bytes()) != expected:
            raise ValueError("Context-advantage source changed while preparing: " + name)
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    write_json(output / "protocol.json", protocol)
    frozen = ["protocol.json", "cases.jsonl", f"{BLIND}/protocol.json",
              f"{BLIND}/status.json", f"{BLIND}/cases.jsonl"] + [
                  f"seed{seed}-treatment.json" for seed in SEEDS]
    write_json(output / "protocol-checksums.json", {
        name: digest((output / name).read_bytes()) for name in frozen})
    write_json(output / "frozen-source-manifest.json", {"source_files": hashes})
    write_json(output / "status.json", {"state": "prepared"})
    print({"prepared": str(output), "seeds": list(SEEDS), "blind_rows": 192})


def verify(root, output, protocol):
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen context-advantage input changed: " + name)
    for name, expected in protocol["source_files"].items():
        if digest((output / "frozen-source" / name).read_bytes()) != expected:
            raise ValueError("Frozen source snapshot changed: " + name)
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Live source changed after protocol freeze: " + name)
    control = root / protocol["control_study"]
    if digest((control / "result-manifest.json").read_bytes()) != protocol["control_result_manifest_sha256"]:
        raise ValueError("Control result manifest changed")
    if control_manifest(root, control) != protocol["controls"]:
        raise ValueError("Control checkpoint manifest changed")
    if digest((root / "src/decision_model/decoder_base.lock.json").read_bytes()) != protocol["base_lock_sha256"]:
        raise ValueError("Decoder base lock changed")
    verify_local_decoder_base(root / protocol["base_path"])


def run_command(root, output, protocol, stage, seed, command):
    verify(root, output, protocol)
    write_json(output / "status.json", {"state": stage, "seed": seed})
    print({"stage": stage, "seed": seed}, flush=True)
    try:
        with (output / f"seed{seed}-{stage}.log").open("w") as stream:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        write_json(output / "status.json", {"state": "failed", "stage": stage, "seed": seed})
        raise


def unload(judge):
    torch = judge.torch
    del judge
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run(root, output, device):
    protocol = read(output / "protocol.json")
    if device != protocol["device"]:
        raise ValueError("Device differs from frozen protocol")
    if read(output / "status.json").get("state") not in (
            "prepared", "training", "auditing", "mechanism", "blind_evaluation", "failed"):
        raise ValueError("Context-advantage study cannot resume from current state")
    verify(root, output, protocol)
    data = output / "cases.jsonl"
    control_study = root / protocol["control_study"]
    treatments = {}
    for seed in SEEDS:
        target = output / f"seed{seed}-treatment"
        if not target.exists():
            run_command(root, output, protocol, "training", seed, [
                sys.executable, "-m", "decision_model.cli", "train",
                "--data", str(data), "--config", str(output / f"seed{seed}-treatment.json"),
                "--output", str(target), "--base-path", str(root / protocol["base_path"]),
                "--device", device])
        if not (target / "checksums.json").is_file():
            raise ValueError("Incomplete treatment requires explicit removal: " + str(seed))
        run_record = read(target / "run.json")
        config = read(output / f"seed{seed}-treatment.json")
        control_record = protocol["controls"][str(seed)]
        control_dir = root / control_record["path"]
        expected = protocol["configs"][str(seed)]
        context = run_record.get("context_advantage", {})
        if (run_record["dataset_sha256"] != protocol["dataset_sha256"] or
                run_record["config_sha256"] != expected["config_sha256"] or
                read(target / "model.json") != config or run_record["seed"] != seed or
                run_record["updates"] != 128 or
                run_record["initialization"] != "public_pretrained_base_fresh_adapter_and_head" or
                run_record["initial_trainable_sha256"] != expected["control_initial_trainable_sha256"] or
                run_record["selected_ids_sha256"] != expected["control_selected_ids_sha256"] or
                (target / "training-plan.jsonl").read_bytes() !=
                (control_dir / "training-plan.jsonl").read_bytes() or
                context.get("weight") != WEIGHT or context.get("gap") != GAP or
                context.get("null_branch_gradient") != "stopped" or
                context.get("null_forward_mode") != "eval" or
                context.get("rows") != 1024 or
                not math.isfinite(context.get("mean_unweighted_training_penalty", math.nan)) or
                run_record.get("counterfactual_forward_sequences") != 1024 or
                run_record.get("adapter_probe", {}).get("absolute_change", 0) <= 0):
            raise ValueError("Treatment endpoint differs from protocol: " + str(seed))
        for filename, expected_hash in read(target / "checksums.json").items():
            if digest((target / filename).read_bytes()) != expected_hash:
                raise ValueError("Treatment checksum differs: " + str(seed))
        treatments[str(seed)] = {
            "path": str(target.relative_to(root)),
            "weights_sha256": digest((target / "decision.safetensors").read_bytes()),
            "model_sha256": digest((target / "model.json").read_bytes()),
            "calibration_sha256": digest((target / "calibration.json").read_bytes()),
            "checksums_sha256": digest((target / "checksums.json").read_bytes()),
            "run_sha256": digest((target / "run.json").read_bytes()),
            "evaluation_sha256": digest((target / "evaluation.json").read_bytes()),
            "test_predictions_sha256": digest((target / "test-predictions.json").read_bytes()),
            "training_plan_sha256": digest((target / "training-plan.jsonl").read_bytes()),
            "optimizer_steps_sha256": digest((target / "optimizer-steps.jsonl").read_bytes()),
        }
    write_json(output / "trained-checkpoints.json", treatments)

    audits = output / "audit-results"
    audits.mkdir(exist_ok=True)
    for seed in SEEDS:
        audit = audits / f"seed{seed}-treatment.json"
        if not audit.exists():
            run_command(root, output, protocol, "auditing", seed, [
                sys.executable, "-m", "decision_model.cli", "audit",
                "--checkpoint", str(output / f"seed{seed}-treatment"),
                "--data", str(data), "--output", str(audit), "--max-cases", "24",
                "--base-path", str(root / protocol["base_path"]), "--device", device])
        value = read(audit)
        if (value["cases"] != 24 or value["max_reload_probability_difference"] > 2e-6 or
                value["max_id_rename_probability_difference"] > 2e-6):
            raise ValueError("Treatment audit contract differs: " + str(seed))

    test_rows = [row for row in load_rows(data) if row["split"] == "test"]
    mechanism = output / "mechanism"
    mechanism.mkdir(exist_ok=True)
    for seed in SEEDS:
        for arm, checkpoint in (
                ("control", root / protocol["controls"][str(seed)]["path"]),
                ("treatment", output / f"seed{seed}-treatment")):
            target = mechanism / f"seed{seed}-{arm}.json"
            if target.exists():
                continue
            verify(root, output, protocol)
            write_json(output / "status.json", {"state": "mechanism", "seed": seed, "arm": arm})
            judge = make_judge(read(checkpoint / "model.json"),
                               base_path=root / protocol["base_path"],
                               checkpoint=checkpoint, device=device)
            records, collection = collect_evidence_and_prior_logits(judge, test_rows)
            summary = summarize_context_advantage(test_rows, records, GAP)
            write_json(target, {"seed": seed, "arm": arm,
                                "checkpoint_weight_sha256": digest(
                                    (checkpoint / "decision.safetensors").read_bytes()),
                                "dataset_sha256": protocol["dataset_sha256"],
                                "collection": collection, "summary": summary,
                                "records": records})
            unload(judge)

    # The first blind forward pass is below, after every treatment, plan check,
    # audit and non-blind mechanistic collection has completed.
    blind = output / BLIND
    blind_protocol = read(blind / "protocol.json")
    blind_results = output / "blind-results"
    blind_results.mkdir(exist_ok=True)
    for seed in SEEDS:
        for arm, checkpoint, expected_weight in (
                ("control", root / protocol["controls"][str(seed)]["path"],
                 protocol["controls"][str(seed)]["weights_sha256"]),
                ("treatment", output / f"seed{seed}-treatment",
                 treatments[str(seed)]["weights_sha256"])):
            target = blind_results / f"seed{seed}-{arm}.json"
            if not target.exists():
                run_command(root, output, protocol, "blind_evaluation", f"{seed}-{arm}", [
                    sys.executable, "-m", "decision_model.cli", "evaluate",
                    "--checkpoint", str(checkpoint), "--data", str(blind / "cases.jsonl"),
                    "--output", str(target), "--split", "test", "--max-cases", "0",
                    "--base-path", str(root / protocol["base_path"]), "--device", device])
            value = read(target)
            if (value["dataset_sha256"] != blind_protocol["cases_sha256"] or
                    value["weights_sha256"] != expected_weight or value["split"] != "test" or
                    value["raw"]["n"] != 192):
                raise ValueError(f"Blind evaluation identity differs: {seed}-{arm}")

    result_files = ["trained-checkpoints.json"]
    result_files += [f"audit-results/seed{seed}-treatment.json" for seed in SEEDS]
    result_files += [f"mechanism/seed{seed}-{arm}.json" for seed in SEEDS
                     for arm in ("control", "treatment")]
    result_files += [f"blind-results/seed{seed}-{arm}.json" for seed in SEEDS
                     for arm in ("control", "treatment")]
    manifest = {name: digest((output / name).read_bytes()) for name in result_files}
    write_json(output / "result-manifest.json", manifest)
    write_json(output / "status.json", {"state": "complete", "seeds": list(SEEDS),
                                         "result_manifest_sha256": digest(
                                             (output / "result-manifest.json").read_bytes())})
    print({"state": "complete", "seeds": list(SEEDS)}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/context-advantage-v1")
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
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
