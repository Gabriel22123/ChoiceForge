"""Compare label-balanced dispersed updates with premise/replay-matched variable labels."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import subprocess
import sys

from decision_model.core import digest, load_rows, write_json
from relation_group_plan import LABELS, make_label_unbalanced_orders


SEEDS = (42, 43)
BALANCED_STUDIES = {42: "runs/relation-group-v1", 43: "runs/relation-group-seed43-v1"}


def read(path):
    return json.loads(path.read_text())


def source_hashes(root):
    paths = [*sorted((root / "src/decision_model").glob("*.py")),
             root / "scripts/relation_group_plan.py", Path(__file__).resolve()]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def prepare(root, output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    source = root / "data/public-relation-groups-v1"
    manifest = read(source / "manifest.json")
    data = source / "cases.jsonl"
    if digest(data.read_bytes()) != manifest["dataset_sha256"]:
        raise ValueError("Dataset checksum differs")
    rows = load_rows(data); by_id = {row["id"]: row for row in rows}
    selected_ids = {split: [row["id"] for row in rows if row["split"] == split]
                    for split in ("train", "validation", "test")}
    initial = root / "runs/architecture-study-v1/D-independent"
    initial_sha = "8d5475f12df673591ce10a189058e26ae805ce78425a0b3f7df2fc08ceadf10e"
    if digest((initial / "decision.safetensors").read_bytes()) != initial_sha:
        raise ValueError("Starting weights changed")
    controls = {}; configs = {}; histograms = {}
    for seed in SEEDS:
        study = root / BALANCED_STUDIES[seed]
        if read(study / "status.json")["state"] != "complete":
            raise ValueError("Balanced reference is incomplete")
        protocol = read(study / "protocol.json")
        if (protocol["seed"] != seed or protocol["dataset_sha256"] != manifest["dataset_sha256"] or
                protocol["selected_ids"] != selected_ids or protocol["initial_weight_sha256"] != initial_sha):
            raise ValueError("Balanced reference identity differs")
        balanced_config = read(study / "dispersed.json")
        unbalanced_orders = make_label_unbalanced_orders(rows, seed, balanced_config["epochs"])
        config = dict(balanced_config, epoch_row_orders=unbalanced_orders)
        configs[seed] = config
        pattern_counts = Counter()
        for balanced_order, unbalanced_order in zip(balanced_config["epoch_row_orders"], unbalanced_orders):
            if Counter(balanced_order) != Counter(unbalanced_order):
                raise ValueError("Training rows differ")
            for old, new in zip(balanced_order, unbalanced_order):
                if by_id[old]["source"] != by_id[new]["source"] or by_id[old]["group_id"] != by_id[new]["group_id"]:
                    raise ValueError("Premise/source position differs")
                if by_id[old]["source"] != "snli" and old != new:
                    raise ValueError("Replay position differs")
            for start in range(0, len(unbalanced_order), 8):
                counts = Counter(by_id[key]["label"] for key in unbalanced_order[start:start + 8]
                                 if by_id[key]["source"] == "snli")
                pattern_counts[tuple(counts.get(label, 0) for label in LABELS)] += 1
        histograms[str(seed)] = {"/".join(map(str, pattern)): count
                                 for pattern, count in sorted(pattern_counts.items())}
        reference = study / "dispersed"
        controls[str(seed)] = {
            "balanced_study": BALANCED_STUDIES[seed],
            "balanced_config_sha256": digest((study / "dispersed.json").read_bytes()),
            "balanced_weights_sha256": digest((reference / "decision.safetensors").read_bytes()),
            "balanced_run_sha256": digest((reference / "run.json").read_bytes()),
            "balanced_order_sha256": digest(balanced_config["epoch_row_orders"]),
            "unbalanced_order_sha256": digest(unbalanced_orders),
        }
    output.mkdir(parents=True)
    (output / "cases.jsonl").write_bytes(data.read_bytes())
    write_json(output / "data-manifest.json", manifest)
    for seed, config in configs.items():
        write_json(output / f"seed-{seed}.json", config)
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seeds": list(SEEDS), "arms": [f"seed-{seed}" for seed in SEEDS],
        "question": "At identical rows, premise positions and replay positions, does enforcing2/2/2 relation labels per update help?",
        "initial_checkpoint": "runs/architecture-study-v1/D-independent", "initial_weight_sha256": initial_sha,
        "dataset_sha256": manifest["dataset_sha256"], "data_manifest": manifest,
        "selected_ids": selected_ids, "source_files": source_hashes(root),
        "updates": 256, "epochs": 2, "example_exposures": 2048,
        "controls": controls, "variable_label_histograms": histograms,
        "treatment": "Permute each premise's three relation rows across its fixed dispersed positions; preserve every premise and replay ID at every position while allowing per-update label counts to vary",
        "checkpoint_selection": "Fixed second-epoch endpoints; no validation/test early stopping or winner selection",
        "primary_endpoints": ["Raw SNLI accuracy and per-class recall", "Raw SNLI NLL and Brier"],
        "secondary_endpoints": ["Neutral precision", "Old-task retention", "Candidate order stability", "Evidence-flip diagnostic"],
        "limitations": ["Only seeds42/43", "Treatment changes which hypothesis occupies each fixed premise position and dropout-mask assignment",
                        "Evaluation is repeatedly inspected", "SNLI is a trained task, not zero-shot"],
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol-checksums.json", {
        path.name: digest(path.read_bytes()) for path in output.iterdir() if path.is_file()})
    write_json(output / "status.json", {"state": "prepared"})
    print(json.dumps({"prepared": str(output), "histograms": histograms}), flush=True)


def run(root, output, device):
    protocol = read(output / "protocol.json")
    initial = root / protocol["initial_checkpoint"]
    data = output / "cases.jsonl"; base = root / "models/eurobert-2.1b"

    def check():
        for name, expected in protocol["source_files"].items():
            if digest((root / name).read_bytes()) != expected:
                raise ValueError("Frozen source changed: " + name)
        for name, expected in read(output / "protocol-checksums.json").items():
            if digest((output / name).read_bytes()) != expected:
                raise ValueError("Frozen input changed: " + name)
        if digest((initial / "decision.safetensors").read_bytes()) != protocol["initial_weight_sha256"]:
            raise ValueError("Starting weights changed")
        for seed in protocol["seeds"]:
            control = protocol["controls"][str(seed)]
            reference = root / control["balanced_study"] / "dispersed"
            if (digest((reference / "decision.safetensors").read_bytes()) != control["balanced_weights_sha256"] or
                    digest((reference / "run.json").read_bytes()) != control["balanced_run_sha256"]):
                raise ValueError("Balanced reference changed")

    def execute(arm, stage, command):
        check(); write_json(output / "status.json", {"state": stage, "arm": arm})
        print(json.dumps({"arm": arm, "stage": stage}), flush=True)
        try:
            with (output / f"{arm}-{stage}.log").open("w") as stream:
                subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            write_json(output / "status.json", {"state": "failed", "arm": arm, "stage": stage}); raise

    for seed in protocol["seeds"]:
        arm = f"seed-{seed}"; target = output / arm
        execute(arm, "train", [sys.executable, "-m", "decision_model.cli", "train", "--data", str(data),
                "--config", str(output / f"seed-{seed}.json"), "--output", str(target),
                "--init-from", str(initial), "--base-path", str(base), "--device", device])
        record = read(target / "run.json")
        if record["updates"] != protocol["updates"] or record["selected_ids"] != protocol["selected_ids"]:
            raise ValueError("Variable-label budget or selection differs")
        execute(arm, "audit", [sys.executable, "-m", "decision_model.cli", "audit", "--checkpoint", str(target),
                "--data", str(data), "--output", str(target / "audit.json"), "--max-cases", "25",
                "--base-path", str(base), "--device", device])
    write_json(output / "status.json", {"state": "complete", "arms": protocol["arms"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/relation-label-balance-v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; output = root / args.output
    if not output.exists(): prepare(root, output)
    if not args.prepare_only: run(root, output, args.device)


if __name__ == "__main__":
    main()
