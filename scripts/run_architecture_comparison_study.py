"""Prepare and run a fresh-adapter, three-seed encoder/decoder comparison."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

from decision_model.core import digest, load_rows, write_json
from decision_model.decoder_model import MODEL_ID as DECODER_ID, MODEL_REVISION as DECODER_REVISION
from decision_model.decoder_model import verify_local_decoder_base
from decision_model.model import MODEL_ID as ENCODER_ID, MODEL_REVISION as ENCODER_REVISION
from decision_model.model import verify_local_base


SEEDS = (42, 43, 44)
ARCHITECTURES = ("encoder", "decoder")
ARMS = tuple(f"seed{seed}-{architecture}" for seed in SEEDS for architecture in ARCHITECTURES)
PARENT = "runs/task-gradient-trust-v1"
BASES = {
    "encoder": {"architecture": "bidirectional_encoder", "model_id": ENCODER_ID,
                "revision": ENCODER_REVISION, "path": "models/eurobert-2.1b"},
    "decoder": {"architecture": "causal_decoder", "model_id": DECODER_ID,
                "revision": DECODER_REVISION, "path": "models/minicpm5-2b-base"},
}


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes(root):
    paths = [*sorted((root / "src/decision_model").glob("*.py")),
             root / "src/decision_model/base.lock.json",
             root / "src/decision_model/decoder_base.lock.json",
             root / "scripts/prepare_arc_blind.py",
             root / "scripts/benchmark_checkpoint.py",
             root / "scripts/architecture_comparison_report.py",
             Path(__file__).resolve()]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def prepare(root, output, device):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    parent = root / PARENT
    if read(parent / "status.json")["state"] != "complete":
        raise ValueError("Complete the preregistered gradient study before architecture comparison")
    parent_protocol = read(parent / "protocol.json")
    data = parent / "cases.jsonl"
    if digest(data.read_bytes()) != parent_protocol["dataset_sha256"]:
        raise ValueError("Parent public dataset changed")
    verify_local_base(root / BASES["encoder"]["path"])
    verify_local_decoder_base(root / BASES["decoder"]["path"])
    rows = load_rows(data)
    if len([row for row in rows if row["split"] == "train"]) != 1024:
        raise ValueError("Architecture study expects the fixed 1,024-row training set")

    output.mkdir(parents=True)
    shutil.copyfile(data, output / "cases.jsonl")
    shutil.copyfile(root / "configs/arc-blind-source.json", output / "arc-source.json")
    config_hashes = {}
    for seed in SEEDS:
        reference = read(parent / f"seed{seed}.json")
        for architecture in ARCHITECTURES:
            config = dict(reference)
            config.update(BASES[architecture])
            config.pop("path")
            target = output / f"seed{seed}-{architecture}.json"
            write_json(target, config)
            config_hashes[f"seed{seed}-{architecture}"] = digest(target.read_bytes())

    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question": "With fresh adapters, does a similarly sized bidirectional encoder or causal decoder make more stable dynamic-candidate decisions under matched data and updates?",
        "device": device,
        "seeds": list(SEEDS), "architectures": list(ARCHITECTURES), "arms": list(ARMS),
        "dataset_sha256": digest((output / "cases.jsonl").read_bytes()),
        "parent_protocol_sha256": digest((parent / "protocol.json").read_bytes()),
        "source_files": source_hashes(root), "configs_sha256": config_hashes,
        "bases": {
            architecture: {**record, "base_lock_sha256": digest(
                (root / ("src/decision_model/base.lock.json" if architecture == "encoder" else
                         "src/decision_model/decoder_base.lock.json")).read_bytes())}
            for architecture, record in BASES.items()
        },
        "matched": [
            "same 1,024 public training rows, one exposure and 128 optimizer updates",
            "same per-seed frozen row order and candidate permutation RNG",
            "independent per-candidate encoding and shared scalar decision head",
            "rank-8 LoRA on the final 32 transformer layers and head width 256",
            "same clean CE plus 0.5 Brier objective, optimizer settings and clipping",
            "same validation-only temperature fitting and fixed final checkpoint selection",
        ],
        "architecture_specific": [
            "encoder reads a reserved marker immediately before the candidate through bidirectional attention",
            "decoder reads the final token after the candidate through causal attention",
            "tokenizers, pretrained corpora, hidden sizes and base parameter counts differ and are part of the end-to-end route comparison",
        ],
        "primary_endpoints": [
            "PAWS and SNLI raw accuracy, NLL and Brier across all three seeds",
            "old-task accuracy and probability quality",
            "candidate-order stability and ID-rename invariance",
            "training time, inference work proxy, fixed-checkpoint sampled device allocation and exact trainable parameters",
        ],
        "advancement_rule": {
            "trained_tasks": [
                "decoder mean PAWS+SNLI accuracy delta is positive in at least two of three seeds",
                "decoder loses no more than 1 accuracy point on either PAWS or SNLI in any seed",
                "decoder mean NLL delta across the six seed/task comparisons is non-positive",
                "decoder mean accuracy and mean NLL across bandit, CLINC, GoEmotions and JSON Schema are not worse",
                "all 24 decoder audit cases are stable under every tested candidate order in every seed",
            ],
            "sealed_arc": [
                "decoder mean ARC-Challenge+ARC-Easy accuracy delta is positive in at least two of three seeds",
                "decoder mean accuracy delta across the six seed/task ARC comparisons is non-negative",
            ],
            "decision": "decoder is eligible as the default only when every trained-task and sealed-ARC condition passes; engineering cost is reported rather than hidden",
        },
        "blind_boundary": "ARC case selection/evaluation remains sealed until all six fixed endpoints and audits are complete; then every endpoint is evaluated once regardless of PAWS/SNLI outcomes. ARC never enters training, calibration, early stopping or endpoint selection",
        "checkpoint_selection": "All six fixed final endpoints; no early stopping and no test-based model selection",
    }
    for name, expected in protocol["source_files"].items():
        source = root / name
        if digest(source.read_bytes()) != expected:
            raise ValueError("Architecture source changed during preparation: " + name)
        target = output / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    write_json(output / "frozen-source-manifest.json", {"source_files": protocol["source_files"]})
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol-checksums.json", {
        path.name: digest(path.read_bytes()) for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    })
    write_json(output / "status.json", {"state": "prepared"})
    print({"prepared": str(output), "arms": list(ARMS)})


def verify(root, output, protocol):
    for name, expected in read(output / "protocol-checksums.json").items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError("Frozen architecture input changed: " + name)
    for name, expected in protocol["source_files"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Frozen architecture source changed: " + name)
    if digest((root / PARENT / "protocol.json").read_bytes()) != protocol["parent_protocol_sha256"]:
        raise ValueError("Parent protocol changed")
    verify_local_base(root / protocol["bases"]["encoder"]["path"])
    verify_local_decoder_base(root / protocol["bases"]["decoder"]["path"])


def run_command(root, output, protocol, stage, arm, command):
    verify(root, output, protocol)
    write_json(output / "status.json", {"state": stage, "arm": arm})
    print({"stage": stage, "arm": arm}, flush=True)
    try:
        with (output / f"{arm}-{stage}.log").open("w") as stream:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        write_json(output / "status.json", {"state": "failed", "stage": stage, "arm": arm})
        raise


def run(root, output, device):
    protocol = read(output / "protocol.json")
    if device != protocol["device"]:
        raise ValueError("Architecture study device differs from frozen protocol")
    if read(output / "status.json")["state"] not in ("prepared", "failed", "training", "auditing", "benchmarking", "sealing_blind", "blind_evaluation"):
        raise ValueError("Architecture study cannot resume from current state")
    finals = {}
    data = output / "cases.jsonl"
    selected_ids = None
    for arm in ARMS:
        seed_text, architecture = arm.split("-", 1)
        seed = int(seed_text.removeprefix("seed"))
        target = output / arm
        if not target.exists():
            run_command(root, output, protocol, "training", arm, [
                sys.executable, "-m", "decision_model.cli", "train",
                "--data", str(data), "--config", str(output / f"seed{seed}-{architecture}.json"),
                "--output", str(target), "--base-path", str(root / protocol["bases"][architecture]["path"]),
                "--device", device,
            ])
        if not (target / "checksums.json").is_file():
            raise ValueError("Incomplete endpoint requires explicit removal before resume: " + arm)
        run_record = read(target / "run.json")
        config = read(output / f"seed{seed}-{architecture}.json")
        if (run_record["dataset_sha256"] != protocol["dataset_sha256"] or
                run_record["config_sha256"] != protocol["configs_sha256"][arm] or
                run_record["initialization"] != "public_pretrained_base_fresh_adapter_and_head" or
                run_record["seed"] != seed or run_record["updates"] != 128 or
                read(target / "model.json") != config):
            raise ValueError("Architecture endpoint protocol differs: " + arm)
        selected_ids = run_record["selected_ids"] if selected_ids is None else selected_ids
        if run_record["selected_ids"] != selected_ids:
            raise ValueError("Architecture endpoints selected different rows")
        for filename, expected in read(target / "checksums.json").items():
            if digest((target / filename).read_bytes()) != expected:
                raise ValueError("Architecture endpoint changed: " + arm)
        finals[arm] = {"path": str(target.relative_to(root)),
                       "weights_sha256": digest((target / "decision.safetensors").read_bytes()),
                       "model_sha256": digest((target / "model.json").read_bytes()),
                       "calibration_sha256": digest((target / "calibration.json").read_bytes()),
                       "checksums_sha256": digest((target / "checksums.json").read_bytes()),
                       "run_sha256": digest((target / "run.json").read_bytes()),
                       "evaluation_sha256": digest((target / "evaluation.json").read_bytes()),
                       "test_predictions_sha256": digest((target / "test-predictions.json").read_bytes()),
                       "training_plan_sha256": digest((target / "training-plan.jsonl").read_bytes()),
                       "optimizer_steps_sha256": digest((target / "optimizer-steps.jsonl").read_bytes())}

    for seed in SEEDS:
        left = (output / f"seed{seed}-encoder/training-plan.jsonl").read_bytes()
        right = (output / f"seed{seed}-decoder/training-plan.jsonl").read_bytes()
        if left != right:
            raise ValueError("Encoder and decoder training plans differ within seed")
    write_json(output / "trained-checkpoints.json", finals)
    audits = output / "audit-results"
    audits.mkdir(exist_ok=True)
    for arm in ARMS:
        seed_text, architecture = arm.split("-", 1)
        target = output / arm
        audit = audits / f"{arm}.json"
        if not audit.exists():
            run_command(root, output, protocol, "auditing", arm, [
                sys.executable, "-m", "decision_model.cli", "audit",
                "--checkpoint", str(target), "--data", str(data), "--output", str(audit),
                "--max-cases", "24", "--base-path", str(root / protocol["bases"][architecture]["path"]),
                "--device", device,
            ])
        value = read(audit)
        if value["cases"] != 24 or value["max_reload_probability_difference"] > 2e-6 or value["max_id_rename_probability_difference"] > 2e-6:
            raise ValueError("Architecture audit contract differs: " + arm)

    benchmarks = output / "benchmarks"
    benchmarks.mkdir(exist_ok=True)
    for arm in ARMS:
        _, architecture = arm.split("-", 1)
        target = output / arm
        benchmark = benchmarks / f"{arm}.json"
        if not benchmark.exists():
            run_command(root, output, protocol, "benchmarking", arm, [
                sys.executable, "scripts/benchmark_checkpoint.py",
                "--checkpoint", str(target), "--data", str(data), "--output", str(benchmark),
                "--base-path", str(root / protocol["bases"][architecture]["path"]), "--device", device,
            ])
        value = read(benchmark)
        if (value["checkpoint_weight_sha256"] != finals[arm]["weights_sha256"] or
                value["dataset_sha256"] != protocol["dataset_sha256"] or
                value["warmup_requests"] != 24 or value["measurement_rounds"] != 3 or
                value["summary"]["requests"] != 72):
            raise ValueError("Architecture benchmark identity differs: " + arm)

    blind = output / "arc-blind"
    if not blind.exists():
        run_command(root, output, protocol, "sealing_blind", "all", [
            sys.executable, "scripts/prepare_arc_blind.py", "--seal-selection",
            "--cache", str(root / "cache/arc-blind"),
            "--source-record", str(output / "arc-source.json"),
            "--output", str(blind),
            "--eurobert-path", str(root / protocol["bases"]["encoder"]["path"]),
            "--decoder-path", str(root / protocol["bases"]["decoder"]["path"]),
            "--training-data", str(data),
        ])
    blind_protocol = read(blind / "protocol.json")
    if (read(blind / "status.json") != {"state": "prepared"} or blind_protocol["rows"] != 384 or
            blind_protocol["exact_training_request_overlap"] != 0 or not blind_protocol["selection_is_answer_blind"]):
        raise ValueError("Sealed ARC protocol differs")
    arc_results = output / "arc-results"
    arc_results.mkdir(exist_ok=True)
    for arm in ARMS:
        _, architecture = arm.split("-", 1)
        target = output / arm
        result = arc_results / f"{arm}.json"
        if not result.exists():
            run_command(root, output, protocol, "blind_evaluation", arm, [
                sys.executable, "-m", "decision_model.cli", "evaluate",
                "--checkpoint", str(target), "--data", str(blind / "cases.jsonl"),
                "--output", str(result), "--split", "test", "--max-cases", "0",
                "--base-path", str(root / protocol["bases"][architecture]["path"]),
                "--device", device,
            ])
        value = read(result)
        if (value["dataset_sha256"] != blind_protocol["cases_sha256"] or value["split"] != "test" or
                value["raw"]["n"] != 384 or value["weights_sha256"] != finals[arm]["weights_sha256"]):
            raise ValueError("ARC evaluation identity differs: " + arm)
    result_files = ["trained-checkpoints.json", "arc-blind/protocol.json", "arc-blind/status.json",
                    "arc-blind/cases.jsonl"]
    result_files += [f"audit-results/{arm}.json" for arm in ARMS]
    result_files += [f"benchmarks/{arm}.json" for arm in ARMS]
    result_files += [f"arc-results/{arm}.json" for arm in ARMS]
    result_manifest = {name: digest((output / name).read_bytes()) for name in result_files}
    write_json(output / "result-manifest.json", result_manifest)
    write_json(output / "status.json", {"state": "complete", "models": list(ARMS),
                                         "result_manifest_sha256": digest((output / "result-manifest.json").read_bytes())})
    print({"state": "complete", "models": list(ARMS)}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/architecture-comparison-v1")
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if not args.run_prepared:
        prepare(root, output, args.device)
    if not args.prepare_only:
        run(root, output, args.device)


if __name__ == "__main__":
    main()
