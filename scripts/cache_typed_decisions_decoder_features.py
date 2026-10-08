"""Cache pinned MiniCPM candidate features once for objective-only screening."""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import time
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.decoder_model import (MODEL_ID, MODEL_REVISION, DecoderJudge,
                                          verify_local_decoder_base)


IDENTITY_FIELDS = ("source", "family", "group_id", "decision_type", "request",
                   "label", "target_probabilities")


def collect_unique_rows(fold_root: Path):
    paths = sorted(fold_root.glob("heldout-*/cases.jsonl"))
    if len(paths) != 4:
        raise ValueError("Expected exactly four prepared workflow folds")
    unique = {}
    membership = {}
    for path in paths:
        fold = path.parent.name.removeprefix("heldout-")
        for row in load_rows(path):
            stable = {key: row.get(key) for key in IDENTITY_FIELDS}
            if row["id"] in unique and canonical(stable) != canonical(unique[row["id"]]):
                raise ValueError("Same row ID has different semantic content across folds: " + row["id"])
            unique[row["id"]] = stable
            membership.setdefault(row["id"], []).append(fold)
    return [(row_id, unique[row_id], sorted(membership[row_id])) for row_id in sorted(unique)], paths


def cache_config(seed=0):
    return {
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "architecture": "causal_decoder", "seed": seed,
        "lora_layers": 1, "lora_rank": 2, "head_width": 64, "dropout": 0.0,
        "max_tokens": 768, "candidate_encoding": "independent",
        "candidate_chunk_size": 8, "inference_encoding": "path",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", default="data/typed-decisions-workflow-folds-v1")
    parser.add_argument("--output", default="cache/typed-decisions-minicpm5-features-v1")
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--shard-rows", type=int, default=128)
    parser.add_argument("--batch-rows", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Smoke-only cap; a capped cache is marked non-formal")
    args = parser.parse_args()
    if args.shard_rows < 1 or args.batch_rows < 1 or args.max_rows < 0:
        raise ValueError("Row counts must be positive, with max-rows optionally zero")
    root = Path(__file__).resolve().parents[1]
    fold_root, output, base = root / args.fold_root, root / args.output, root / args.base_path
    verify_local_decoder_base(base)
    rows, fold_paths = collect_unique_rows(fold_root)
    if args.max_rows:
        rows = rows[:args.max_rows]
    source_hashes = {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in fold_paths}
    frozen = {
        "format": "typed-decisions-minicpm5-candidate-features-v1",
        "formal": args.max_rows == 0,
        "row_count": len(rows),
        "row_ids_sha256": digest([row_id for row_id, _, _ in rows]),
        "source_sha256": source_hashes,
        "base_revision": MODEL_REVISION,
        "base_lock_sha256": digest((root / "src/decision_model/decoder_base.lock.json").read_bytes()),
        "extractor_source_sha256": digest(Path(__file__).read_bytes()),
        "decoder_model_source_sha256": digest((root / "src/decision_model/decoder_model.py").read_bytes()),
        "feature_semantics": "final-token hidden state from the pinned base with LoRA adapters disabled; before scalar head",
        "config": cache_config(),
        "shard_rows": args.shard_rows,
        "batch_rows": args.batch_rows,
    }
    manifest_path = output / "manifest.json"
    if output.exists():
        if not manifest_path.is_file():
            raise ValueError("Existing feature directory has no resumable manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("frozen") != frozen:
            raise ValueError("Existing feature cache uses a different frozen protocol")
        if manifest.get("status") == "complete":
            for shard in manifest["shards"]:
                if digest((output / shard["file"]).read_bytes()) != shard["sha256"]:
                    raise ValueError("Completed feature shard changed: " + shard["file"])
            print(json.dumps(manifest, indent=2)); return
    else:
        output.mkdir(parents=True)
        manifest = {
            "status": "building", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "frozen": frozen, "rows": {}, "shards": [], "work": {}, "seconds": 0.0,
        }
        write_json(manifest_path, manifest)

    completed = {item["index"]: item for item in manifest["shards"]}
    judge = DecoderJudge(cache_config(), base_path=base, device=args.device)
    judge.train(False)
    disable = judge.encoder.disable_adapter if hasattr(judge.encoder, "disable_adapter") else contextlib.nullcontext
    started = time.monotonic()
    from safetensors.torch import save_file
    for shard_index, start in enumerate(range(0, len(rows), args.shard_rows)):
        selected = rows[start:start + args.shard_rows]
        filename = f"features-{shard_index:05d}.safetensors"
        if shard_index in completed:
            item = completed[shard_index]
            if item["file"] != filename or item["row_ids"] != [value[0] for value in selected]:
                raise ValueError("Existing shard membership differs")
            if digest((output / filename).read_bytes()) != item["sha256"]:
                raise ValueError("Existing feature shard changed: " + filename)
            continue
        tensors, entries = {}, {}
        with judge.torch.no_grad(), disable():
            for offset in range(0, len(selected), args.batch_rows):
                batch = selected[offset:offset + args.batch_rows]
                encoded = [judge.encode(stable["request"]) for _, stable, _ in batch]
                features = judge.features(encoded).cpu()
                for local, (row_id, stable, membership) in enumerate(batch):
                    count = len(stable["request"]["choices"])
                    key = f"f{start + offset + local:08d}"
                    tensor = features[local, :count].to(judge.torch.float16).contiguous()
                    tensors[key] = tensor
                    entries[row_id] = {
                        "shard": filename, "key": key,
                        "shape": list(tensor.shape),
                        "candidate_ids": [choice["id"] for choice in stable["request"]["choices"]],
                        "request_sha256": digest(stable["request"]), "fold_membership": membership,
                    }
        temporary = output / (filename + ".tmp")
        save_file(tensors, str(temporary)); temporary.replace(output / filename)
        item = {"index": shard_index, "file": filename,
                "row_ids": [value[0] for value in selected],
                "sha256": digest((output / filename).read_bytes())}
        manifest["shards"].append(item); manifest["rows"].update(entries)
        manifest["work"] = dict(judge.work)
        manifest["seconds"] += time.monotonic() - started; started = time.monotonic()
        write_json(manifest_path, manifest)
        print(canonical({"event": "feature_shard", "index": shard_index,
                         "rows": len(selected), "completed": len(manifest["rows"]),
                         "total": len(rows)}), flush=True)
    if len(manifest["rows"]) != len(rows):
        raise ValueError("Feature cache row count differs")
    hidden_sizes = {value["shape"][1] for value in manifest["rows"].values()}
    if len(hidden_sizes) != 1:
        raise ValueError("Feature hidden sizes differ")
    manifest["hidden_size"] = hidden_sizes.pop()
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(canonical({"event": "complete", "rows": len(rows),
                     "hidden_size": manifest["hidden_size"], "seconds": manifest["seconds"]}))


if __name__ == "__main__":
    main()
