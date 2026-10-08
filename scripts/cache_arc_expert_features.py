#!/usr/bin/env python3
"""Cache frozen-parent full and null-context features for the ARC utility screen."""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.decoder_model import verify_local_decoder_base
from decision_model.judge_factory import make_judge
from decision_model.prior_correction import prior_only_request


DATASETS = {
    "arc": "runs/architecture-comparison-v1/arc-blind/cases.jsonl",
}
PARENT = "runs/functional-retention-screen-v1/warm-functional-context"
FORMAT = "arc-parent-expert-utility-features-v1"


def read(path):
    return json.loads(Path(path).read_text())


def source_rows(root):
    result = []
    for dataset, relative in DATASETS.items():
        rows = load_rows(root / relative)
        seen = set()
        for row in rows:
            if row["id"] in seen:
                raise ValueError(f"Duplicate row ID in {dataset}: {row['id']}")
            seen.add(row["id"])
            result.append((f"{dataset}:{row['id']}", dataset, row))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cache/arc-expert-features-v1")
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--device", default="mps", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--shard-rows", type=int, default=32)
    parser.add_argument("--batch-rows", type=int, default=2)
    args = parser.parse_args()
    if args.shard_rows < 1 or args.batch_rows < 1:
        raise ValueError("Shard and batch sizes must be positive")
    root = Path(__file__).resolve().parents[1]
    output, base, parent = root / args.output, root / args.base_path, root / PARENT
    verify_local_decoder_base(base)
    rows = source_rows(root)
    parent_model = read(parent / "model.json")
    frozen = {
        "format": FORMAT,
        "formal": True,
        "feature_semantics": (
            "final-token candidate states from the pinned frozen parent; full and "
            "prior-only requests; before the frozen scalar head"),
        "row_count": len(rows),
        "row_keys_sha256": digest([key for key, _, _ in rows]),
        "datasets": {name: {"path": path, "sha256": digest((root / path).read_bytes())}
                     for name, path in DATASETS.items()},
        "parent": {
            "path": PARENT,
            "weights_sha256": digest((parent / "decision.safetensors").read_bytes()),
            "model_sha256": digest((parent / "model.json").read_bytes()),
            "calibration_sha256": digest((parent / "calibration.json").read_bytes()),
            "temperature": read(parent / "calibration.json")["temperature"],
        },
        "extractor_source_sha256": digest(Path(__file__).read_bytes()),
        "decoder_model_source_sha256": digest(
            (root / "src/decision_model/decoder_model.py").read_bytes()),
        "base_lock_sha256": digest(
            (root / "src/decision_model/decoder_base.lock.json").read_bytes()),
        "parent_config": parent_model,
        "shard_rows": args.shard_rows,
        "batch_rows": args.batch_rows,
    }
    manifest_path = output / "manifest.json"
    if output.exists():
        if not manifest_path.is_file():
            raise ValueError("Existing cache has no resumable manifest")
        manifest = read(manifest_path)
        if manifest.get("frozen") != frozen:
            raise ValueError("Existing cache uses a different frozen protocol")
        if manifest.get("status") == "complete":
            for shard in manifest["shards"]:
                if digest((output / shard["file"]).read_bytes()) != shard["sha256"]:
                    raise ValueError("Completed feature shard changed")
            print(canonical({"event": "already_complete", "rows": len(manifest["rows"])}))
            return
    else:
        output.mkdir(parents=True)
        manifest = {
            "status": "building",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "frozen": frozen, "rows": {}, "shards": [], "work": {}, "seconds": 0.0,
        }
        write_json(manifest_path, manifest)

    completed = {item["index"]: item for item in manifest["shards"]}
    judge = make_judge(parent_model, base_path=base, checkpoint=parent, device=args.device)
    judge.train(False)
    judge.temperature = 1.0
    torch = judge.torch
    from safetensors.torch import save_file
    started = time.monotonic()
    for shard_index, start in enumerate(range(0, len(rows), args.shard_rows)):
        selected = rows[start:start + args.shard_rows]
        filename = f"features-{shard_index:05d}.safetensors"
        if shard_index in completed:
            item = completed[shard_index]
            expected_keys = [value[0] for value in selected]
            if (item["file"] != filename or item["row_keys"] != expected_keys or
                    digest((output / filename).read_bytes()) != item["sha256"]):
                raise ValueError("Existing feature shard differs")
            continue
        tensors, entries = {}, {}
        with torch.no_grad():
            for offset in range(0, len(selected), args.batch_rows):
                batch = selected[offset:offset + args.batch_rows]
                requests = [row["request"] for _, _, row in batch]
                null_requests = [prior_only_request(request) for request in requests]
                full = judge.features([judge.encode(request) for request in requests]).cpu()
                null = judge.features([judge.encode(request) for request in null_requests]).cpu()
                for local, (row_key, dataset, row) in enumerate(batch):
                    position = start + offset + local
                    count = len(row["request"]["choices"])
                    full_value = full[local, :count].to(torch.float16).contiguous()
                    null_value = null[local, :count].to(torch.float16).contiguous()
                    full_key, null_key = f"f{position:08d}", f"n{position:08d}"
                    tensors[full_key], tensors[null_key] = full_value, null_value
                    with torch.no_grad():
                        full_logits = judge.head(full_value.float().to(judge.device)).squeeze(-1)
                        null_logits = judge.head(null_value.float().to(judge.device)).squeeze(-1)
                    entries[row_key] = {
                        "dataset": dataset, "row_id": row["id"], "shard": filename,
                        "full_key": full_key, "null_key": null_key,
                        "shape": list(full_value.shape),
                        "candidate_ids": [choice["id"] for choice in row["request"]["choices"]],
                        "request_sha256": digest(row["request"]),
                        "null_request_sha256": digest(prior_only_request(row["request"])),
                        "parent_logits": full_logits.float().cpu().tolist(),
                        "parent_null_logits": null_logits.float().cpu().tolist(),
                    }
        temporary = output / (filename + ".tmp")
        save_file(tensors, str(temporary)); temporary.replace(output / filename)
        item = {"index": shard_index, "file": filename,
                "row_keys": [value[0] for value in selected],
                "sha256": digest((output / filename).read_bytes())}
        manifest["shards"].append(item); manifest["rows"].update(entries)
        manifest["work"] = dict(judge.work)
        manifest["seconds"] += time.monotonic() - started; started = time.monotonic()
        write_json(manifest_path, manifest)
        print(canonical({"event": "feature_shard", "index": shard_index,
                         "completed": len(manifest["rows"]), "total": len(rows)}), flush=True)
    if len(manifest["rows"]) != len(rows):
        raise ValueError("Feature cache row count differs")
    hidden = {entry["shape"][1] for entry in manifest["rows"].values()}
    if len(hidden) != 1:
        raise ValueError("Feature hidden sizes differ")
    manifest["hidden_size"] = hidden.pop()
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(canonical({"event": "complete", "rows": len(rows),
                     "hidden_size": manifest["hidden_size"],
                     "seconds": manifest["seconds"]}))


if __name__ == "__main__":
    main()
