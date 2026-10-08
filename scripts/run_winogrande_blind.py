"""Run every predeclared checkpoint on the frozen WinoGrande blind subset."""
import argparse
import gc
import json
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.model import Judge, verify_local_base
from decision_model.train import evaluate_rows, predict_rows


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/winogrande-blind-v1")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    status, protocol = read(study / "status.json"), read(study / "protocol.json")
    if status["state"] not in ("prepared", "evaluating"):
        raise ValueError("Blind study is not runnable")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen blind input changed: " + name)
    if digest((study / "cases.jsonl").read_bytes()) != protocol["dataset_sha256"]:
        raise ValueError("Blind dataset identity changed")
    for relative, expected in protocol["source_files"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError("Frozen blind-evaluation source changed: " + relative)
    rows = load_rows(study / "cases.jsonl")
    if len(rows) != protocol["selection"]["rows"] or any(row["split"] != "test" for row in rows):
        raise ValueError("Blind row set changed")
    base = root / "models/eurobert-2.1b"
    verify_local_base(base)
    results = study / "results"
    if status["state"] == "prepared":
        if results.exists():
            raise ValueError("Unexpected blind results directory")
        results.mkdir()
        completed = []
        write_json(study / "status.json", {"state": "evaluating", "completed": completed})
    else:
        if not results.is_dir():
            raise ValueError("Missing resumable blind results directory")
        completed = list(status.get("completed", []))
        declared = [item["name"] for item in protocol["checkpoints"]]
        if completed != declared[:len(completed)]:
            raise ValueError("Completed blind models are not a declared prefix")
        for name in completed:
            prior = read(results / (name + ".json"))
            expected = next(item for item in protocol["checkpoints"] if item["name"] == name)
            if (prior["dataset_sha256"] != protocol["dataset_sha256"] or
                    prior["weights_sha256"] != expected["weights_sha256"]):
                raise ValueError("Completed blind result identity changed: " + name)
    for checkpoint in protocol["checkpoints"]:
        name, directory = checkpoint["name"], root / checkpoint["path"]
        if name in completed:
            continue
        if (digest((directory / "decision.safetensors").read_bytes()) != checkpoint["weights_sha256"] or
                digest((directory / "model.json").read_bytes()) != checkpoint["model_sha256"]):
            raise ValueError("Predeclared checkpoint changed: " + name)
        config = read(directory / "model.json")
        judge = Judge(config, base_path=base, checkpoint=directory, device=args.device)
        tokens = {row["id"]: judge.encode(row["request"]) for row in rows}
        logits = predict_rows(judge, rows, tokens)
        raw, predictions = evaluate_rows(rows, logits)
        inherited, _ = evaluate_rows(rows, logits, judge.temperature)
        result = {"name": name, "checkpoint": checkpoint["path"],
                  "weights_sha256": checkpoint["weights_sha256"],
                  "model_sha256": checkpoint["model_sha256"],
                  "dataset_sha256": protocol["dataset_sha256"],
                  "rows": len(rows), "raw": raw,
                  "inherited_temperature": judge.temperature,
                  "inherited_temperature_metrics": inherited,
                  "predictions": predictions, "encoder_work": dict(judge.work)}
        write_json(results / (name + ".json"), result)
        completed.append(name)
        write_json(study / "status.json", {"state": "evaluating", "completed": completed})
        print(json.dumps({"model": name, "state": "complete"}), flush=True)
        torch = judge.torch
        del judge, tokens, logits
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
    write_json(study / "status.json", {"state": "complete", "models": completed})


if __name__ == "__main__":
    main()
