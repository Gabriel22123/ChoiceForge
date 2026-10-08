"""Load the pinned real decoder and verify non-generative scoring/backprop."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from decision_model.core import digest, write_json
from decision_model.decoder_model import DecoderJudge, verify_local_decoder_base


REQUESTS = [
    {
        "task": "Choose the claim supported by the context.",
        "context": "The indicator is red and the machine is stopped.",
        "choices": [
            {"id": "supported", "description": "The machine is stopped."},
            {"id": "contradicted", "description": "The machine is running."},
            {"id": "unknown", "description": "The machine was built yesterday."},
        ],
    },
    {
        "task": "Choose the best action.",
        "context": "A required field is missing from the request.",
        "choices": [
            {"id": "accept", "description": "Accept it without review."},
            {"id": "review", "description": "Request the missing information."},
        ],
    },
]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minicpm5-decoder-comparison.json")
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--output", default="runs/decoder-backbone-smoke-v1")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a fresh output directory")
    verify_local_decoder_base(args.base_path)
    config = read(args.config)
    start = time.monotonic()
    judge = DecoderJudge(config, base_path=args.base_path, device=args.device)
    load_seconds = time.monotonic() - start

    predictions = [judge.predict(request) for request in REQUESTS]
    for request, prediction in zip(REQUESTS, predictions):
        ids = {choice["id"] for choice in request["choices"]}
        if (set(prediction) != {"choice_id", "probabilities", "requires_review", "score_kind", "calibration"} or
                prediction["choice_id"] not in ids or set(prediction["probabilities"]) != ids or
                not math.isclose(sum(prediction["probabilities"].values()), 1.0, abs_tol=1e-5)):
            raise ValueError("Decoder output contract differs")

    torch = judge.torch
    judge.train(True)
    encoded = [judge.encode(request) for request in REQUESTS]
    logits = judge.logits(encoded)
    targets = torch.tensor([0, 1], device=judge.device)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    gradients = {name: parameter.grad for name, parameter in judge.named_trainable()}
    finite = all(value is not None and torch.isfinite(value).all().item() for value in gradients.values())
    nonzero_lora = sum(value is not None and value.abs().sum().item() > 0
                       for name, value in gradients.items() if "lora_B" in name)
    nonzero_head = sum(value is not None and value.abs().sum().item() > 0
                       for name, value in gradients.items() if name.startswith("head."))
    if not finite or not nonzero_lora or not nonzero_head:
        raise ValueError("Decoder trainable parameters did not receive finite nonzero gradients")
    peak_memory = (torch.mps.driver_allocated_memory() if judge.device.type == "mps" else
                   torch.cuda.max_memory_allocated() if judge.device.type == "cuda" else None)
    output.mkdir(parents=True)
    report = {
        "state": "complete", "model_id": config["model_id"], "revision": config["revision"],
        "config_sha256": digest(Path(args.config).read_bytes()),
        "base_lock_sha256": digest((Path(args.base_path) / "base.lock.json").read_bytes()),
        "device": str(judge.device), "load_seconds": load_seconds,
        "trainable_parameters": sum(parameter.numel() for _, parameter in judge.named_trainable()),
        "base_parameters": sum(parameter.numel() for parameter in judge.encoder.parameters()),
        "loss": loss.item(), "finite_gradients": finite,
        "nonzero_lora_B_tensors": nonzero_lora, "nonzero_head_tensors": nonzero_head,
        "work": judge.work, "peak_device_bytes": peak_memory,
        "predictions": predictions,
    }
    write_json(output / "report.json", report)
    print({key: report[key] for key in ("state", "load_seconds", "trainable_parameters",
                                        "nonzero_lora_B_tensors", "peak_device_bytes")})


if __name__ == "__main__":
    main()
