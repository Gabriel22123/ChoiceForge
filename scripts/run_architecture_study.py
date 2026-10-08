"""Fresh joint vs independent candidate scoring, matched request/update budget."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from decision_model.core import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/architecture-study-v1")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if output.exists():
        raise ValueError("Use a new study directory")
    output.mkdir(parents=True)
    data = root / "data/public-multitask-v1/cases.jsonl"
    base = root / "models/eurobert-2.1b"
    config = json.loads((root / "configs/eurobert-public-pilot.json").read_text())
    config.update(brier_weight=0., consistency_weight=0., training_views=2, candidate_chunk_size=8)
    variants = {"J-joint": "joint", "D-independent": "independent"}
    paths = list((root / "src/decision_model").glob("*.py"))
    paths += [Path(__file__).resolve(), root / "scripts/benchmark_checkpoint.py"]

    def source_hashes():
        return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in sorted(paths)}

    frozen = source_hashes()
    protocol = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "dataset_sha256": digest(data.read_bytes()), "source_files": frozen,
                "base_config": config, "variants": variants,
                "hypothesis": "Independent candidate scoring removes other candidates and list position from each score; accuracy may change and repeated context costs more",
                "matched": ["fresh initial trainable parameters", "selected example IDs", "permutation plan",
                            "request views", "optimizer updates", "objective and hyperparameters"],
                "not_matched": ["encoder sequence count", "token work", "dropout draws across different input shapes", "wall-clock training budget"],
                "admission": "Both modes apply the full joint request token budget before encoding",
                "independent_inference": "One candidate per encoder call; no sorting or probability averaging",
                "limitations": ["one seed, 384 training examples", "previously inspected diagnostic test",
                                "independent scores cannot directly condition on other candidate meanings",
                                "exact score ties still use the interface's first-index argmax tie policy"],
                "metrics": ["per-source accuracy/NLL/Brier/ECE", "24-case permutation audit",
                            "training encoder work", "loaded-model latency on the same 24 requests, three rounds"],
                "checkpoint_selection": "fixed final epoch; no test-based selection"}
    write_json(output / "protocol.json", protocol)
    for name, encoding in variants.items():
        write_json(output / (name + ".json"), dict(config, candidate_encoding=encoding))
    write_json(output / "protocol-checksums.json",
               {p.name: digest(p.read_bytes()) for p in output.glob("*.json")})
    reference = None
    for name in variants:
        if source_hashes() != frozen:
            raise ValueError("Frozen training/benchmark source changed")
        for stage in ("train", "audit", "benchmark"):
            write_json(output / "status.json", {"state": stage, "arm": name})
            print(json.dumps({"event": stage, "arm": name}), flush=True)
            if stage == "train":
                command = [sys.executable, "-m", "decision_model.cli", "train", "--data", str(data),
                           "--config", str(output / (name + ".json")), "--output", str(output / name),
                           "--base-path", str(base), "--device", args.device]
            elif stage == "audit":
                command = [sys.executable, "-m", "decision_model.cli", "audit", "--checkpoint", str(output / name),
                           "--data", str(data), "--output", str(output / name / "audit.json"), "--max-cases", "24",
                           "--base-path", str(base), "--device", args.device]
            else:
                command = [sys.executable, "scripts/benchmark_checkpoint.py", "--checkpoint", str(output / name),
                           "--data", str(data), "--output", str(output / name / "benchmark.json"),
                           "--base-path", str(base), "--device", args.device]
            try:
                with (output / (name + "-" + stage + ".log")).open("w") as log:
                    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError:
                write_json(output / "status.json", {"state": "failed", "arm": name, "stage": stage})
                raise
            if stage == "train":
                run = json.loads((output / name / "run.json").read_text())
                controls = {key: run[key] for key in ("initial_trainable_sha256", "selected_ids_sha256",
                            "training_plan_sha256", "updates", "forward_sequences", "forward_batches", "training_views")}
                if reference is None:
                    reference = controls
                elif controls != reference:
                    raise ValueError("Matched-request comparison failed")
                write_json(output / "matched-controls.json", reference)
    write_json(output / "status.json", {"state": "complete", "arms": list(variants)})


if __name__ == "__main__":
    main()
