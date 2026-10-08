#!/usr/bin/env python3
"""Cache candidate features from one locked decoder endpoint for the outcome screen."""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.decoder_model import DecoderJudge, verify_local_decoder_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/executable-outcome-screen-v1.json")
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--shard-rows", type=int, default=64)
    parser.add_argument("--batch-rows", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    if args.shard_rows < 1 or args.batch_rows < 1 or args.max_rows < 0:
        raise ValueError("invalid row limits")
    root = Path(__file__).resolve().parents[1]
    protocol_path, base, output = root / args.protocol, root / args.base_path, root / args.output
    protocol = json.loads(protocol_path.read_text())
    matches = [item for item in protocol["representation_endpoints"] if item["id"] == args.endpoint_id]
    if len(matches) != 1:
        raise ValueError("endpoint ID is not unique in the frozen protocol")
    endpoint_record = matches[0]
    endpoint = root / endpoint_record["path"]
    weights = endpoint / "decision.safetensors"
    if digest(weights.read_bytes()) != endpoint_record["weights_sha256"]:
        raise ValueError("endpoint weights differ from frozen protocol")
    verify_local_decoder_base(base)
    data_path = root / protocol["data"]["path"]
    if digest(data_path.read_bytes()) != protocol["data"]["sha256"]:
        raise ValueError("outcome data differs from frozen protocol")
    rows = sorted(load_rows(data_path), key=lambda row: row["id"])
    if args.max_rows:
        rows = rows[:args.max_rows]
    model_config = json.loads((endpoint / "model.json").read_text())
    model_config["dropout"] = 0.0
    model_config["inference_encoding"] = "path"
    frozen = {
        "format": "mbpp-outcome-minicpm5-candidate-features-v1",
        "formal": args.max_rows == 0,
        "row_count": len(rows),
        "row_ids_sha256": digest([row["id"] for row in rows]),
        "data_sha256": protocol["data"]["sha256"],
        "protocol_sha256": digest(protocol_path.read_bytes()),
        "endpoint_id": args.endpoint_id,
        "endpoint_weights_sha256": endpoint_record["weights_sha256"],
        "model_config_sha256": digest((endpoint / "model.json").read_bytes()),
        "decoder_model_source_sha256": digest((root / "src/decision_model/decoder_model.py").read_bytes()),
        "extractor_source_sha256": digest(Path(__file__).read_bytes()),
        "feature_semantics": "final-token candidate hidden state after the locked endpoint LoRA and before any scalar head",
        "shard_rows": args.shard_rows,
        "batch_rows": args.batch_rows,
    }
    manifest_path = output / "manifest.json"
    if output.exists():
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        if not manifest or manifest.get("frozen") != frozen:
            raise ValueError("existing feature directory uses another protocol")
        if manifest.get("status") == "complete":
            for shard in manifest["shards"]:
                if digest((output / shard["file"]).read_bytes()) != shard["sha256"]:
                    raise ValueError("completed feature shard changed")
            print(canonical(manifest)); return
    else:
        output.mkdir(parents=True)
        manifest = {"status": "building", "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(), "frozen": frozen, "rows": {},
            "shards": [], "work": {}, "seconds": 0.0}
        write_json(manifest_path, manifest)
    completed = {item["index"]: item for item in manifest["shards"]}
    judge = DecoderJudge(model_config, base_path=base, checkpoint=endpoint, device=args.device)
    judge.train(False)
    from safetensors.torch import save_file
    started = time.monotonic()
    for shard_index, start in enumerate(range(0, len(rows), args.shard_rows)):
        selected = rows[start:start + args.shard_rows]
        filename = f"features-{shard_index:05d}.safetensors"
        if shard_index in completed:
            item = completed[shard_index]
            if item["file"] != filename or item["row_ids"] != [row["id"] for row in selected]:
                raise ValueError("resumed shard membership differs")
            if digest((output / filename).read_bytes()) != item["sha256"]:
                raise ValueError("resumed feature shard changed")
            continue
        tensors, entries = {}, {}
        with judge.torch.no_grad():
            for offset in range(0, len(selected), args.batch_rows):
                batch = selected[offset:offset + args.batch_rows]
                encoded = [judge.encode(row["request"]) for row in batch]
                features = judge.features(encoded).cpu()
                for local, row in enumerate(batch):
                    count = len(row["request"]["choices"])
                    key = f"f{start + offset + local:08d}"
                    tensor = features[local, :count].to(judge.torch.float16).contiguous()
                    tensors[key] = tensor
                    entries[row["id"]] = {
                        "shard": filename, "key": key, "shape": list(tensor.shape),
                        "candidate_ids": [choice["id"] for choice in row["request"]["choices"]],
                        "request_sha256": digest(row["request"]),
                    }
        temporary = output / (filename + ".tmp")
        save_file(tensors, str(temporary)); temporary.replace(output / filename)
        item = {"index": shard_index, "file": filename,
                "row_ids": [row["id"] for row in selected],
                "sha256": digest((output / filename).read_bytes())}
        manifest["shards"].append(item); manifest["rows"].update(entries)
        manifest["work"] = dict(judge.work)
        manifest["seconds"] += time.monotonic() - started; started = time.monotonic()
        write_json(manifest_path, manifest)
        print(canonical({"event": "feature_shard", "endpoint": args.endpoint_id,
                         "rows": len(manifest["rows"]), "total": len(rows)}), flush=True)
    if len(manifest["rows"]) != len(rows):
        raise ValueError("feature cache row count differs")
    hidden = {entry["shape"][1] for entry in manifest["rows"].values()}
    if len(hidden) != 1:
        raise ValueError("feature hidden sizes differ")
    manifest["hidden_size"] = hidden.pop(); manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(canonical({"event": "complete", "endpoint": args.endpoint_id,
                     "rows": len(rows), "seconds": manifest["seconds"]}))


if __name__ == "__main__":
    main()
