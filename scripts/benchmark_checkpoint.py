"""Loaded-model request latency on a fixed public diagnostic subset."""
import argparse
import json
import statistics
import time
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.evaluate import load_judge
from decision_model.train import balanced_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    run = json.loads((checkpoint / "run.json").read_text())
    if digest(Path(args.data).read_bytes()) != run["dataset_sha256"]:
        raise ValueError("Benchmark dataset differs from checkpoint provenance")
    ids = set(run["selected_ids"]["test"])
    rows = balanced_subset([r for r in load_rows(args.data) if r["id"] in ids], 24)
    judge = load_judge(args)

    def synchronize():
        if judge.device.type == "mps":
            judge.torch.mps.synchronize()
        elif judge.device.type == "cuda":
            judge.torch.cuda.synchronize()

    for row in rows:
        judge.predict(row["request"])
    synchronize()
    judge.work = dict.fromkeys(judge.work, 0)
    timings = []
    for repeat in range(3):
        for row in rows:
            synchronize()
            started = time.perf_counter()
            prediction = judge.predict(row["request"])
            synchronize()
            timings.append({"id": row["id"], "repeat": repeat,
                            "candidates": len(row["request"]["choices"]),
                            "seconds": time.perf_counter() - started,
                            "choice_id": prediction["choice_id"]})

    def summarize(values):
        return {"requests": len(values), "median_seconds": statistics.median(values),
                "p95_seconds": statistics.quantiles(values, n=100, method="inclusive")[94],
                "mean_seconds": statistics.mean(values)}

    result = {"checkpoint_weight_sha256": digest((checkpoint / "decision.safetensors").read_bytes()),
              "dataset_sha256": run["dataset_sha256"], "device": str(judge.device),
              "candidate_encoding": judge.config.get("candidate_encoding", "joint"),
              "warmup_requests": len(rows), "measurement_rounds": 3,
              "scope": "24 repeated public diagnostic requests; model load excluded; tokenization and output assembly included; no output cache; not a production SLA",
              "encoder_work": judge.work, "summary": summarize([r["seconds"] for r in timings]),
              "by_candidate_count": {str(k): summarize([r["seconds"] for r in timings if r["candidates"] == k])
                                     for k in sorted({r["candidates"] for r in timings})},
              "timings": timings}
    write_json(args.output, result)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
