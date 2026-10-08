"""Compare independent causal paths with private-branch shared-prefix inference."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from decision_model.decoder_model import DecoderJudge


def synchronize(judge):
    if judge.device.type == "cuda":
        judge.torch.cuda.synchronize(judge.device)
    elif judge.device.type == "mps":
        judge.torch.mps.synchronize()


def reset_work(judge):
    for key in judge.work:
        judge.work[key] = 0


def timed(judge, encoded, mode, repeats):
    judge.config["inference_encoding"] = mode
    with judge.torch.inference_mode():
        judge.logits([encoded])
        synchronize(judge)
        reset_work(judge)
        elapsed = []
        output = None
        for _ in range(repeats):
            start = time.perf_counter()
            output = judge.logits([encoded])
            synchronize(judge)
            elapsed.append((time.perf_counter() - start) * 1000)
    return output.detach().float().cpu(), {
        "median_ms": statistics.median(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "work": dict(judge.work),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minicpm5-decoder-comparison.json")
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument("--context-repeats", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    judge = DecoderJudge(config, base_path=args.base_path, checkpoint=args.checkpoint,
                         device=args.device)
    judge.train(False)
    original = judge.config.get("inference_encoding")
    results = []
    for count in args.candidate_counts:
        if not 2 <= count <= 32:
            raise ValueError("candidate counts must stay within the public 2..32 contract")
        request = {
            "task": "Choose the candidate best supported by the supplied evidence.",
            "context": " ".join(["The shared evidence remains relevant to every candidate."]
                                * args.context_repeats),
            "choices": [{"id": f"candidate-{index}",
                         "description": f"Candidate {index} has its own private conclusion."}
                        for index in range(count)],
        }
        encoded = judge.encode(request)
        path_output, path_metrics = timed(judge, encoded, "path", args.repeats)
        shared_output, shared_metrics = timed(judge, encoded, "shared", args.repeats)
        results.append({
            "candidates": count,
            "path_tokens": sum(len(sequence) for sequence, _ in encoded),
            "common_prefix_tokens": judge._common_prefix(encoded),
            "max_abs_logit_difference": float((path_output - shared_output).abs().max()),
            "same_argmax": int(path_output.argmax()) == int(shared_output.argmax()),
            "path": path_metrics,
            "shared": shared_metrics,
        })
    if original is None:
        judge.config.pop("inference_encoding", None)
    else:
        judge.config["inference_encoding"] = original
    report = {
        "scope": "local synthetic throughput and numerical-parity benchmark; not decision quality",
        "model_id": config["model_id"],
        "device": str(judge.device),
        "repeats_after_warmup": args.repeats,
        "results": results,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
