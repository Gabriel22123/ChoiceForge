"""Fetch and checksum the pinned public decoder comparison backbone."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from decision_model.core import write_json


MODEL_ID = "openbmb/MiniCPM5-2B-Base"
REVISION = "96a57cd572a02506b4500f54427dca24970c1bac"
FILES = (
    "README.md",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/minicpm5-2b-base")
    args = parser.parse_args()
    output = Path(args.output)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=output,
        allow_patterns=list(FILES),
    )
    missing = [name for name in FILES if not (output / name).is_file()]
    if missing:
        raise ValueError("Pinned snapshot is missing: " + ", ".join(missing))
    expected = json.loads((Path(__file__).resolve().parents[1] / "src/decision_model/decoder_base.lock.json").read_text())
    observed = {name: sha256(output / name) for name in FILES}
    if expected["repository"] != f"https://huggingface.co/{MODEL_ID}" or expected["revision"] != REVISION:
        raise ValueError("Packaged decoder lock identity differs")
    if observed != expected["files"]:
        raise ValueError("Downloaded decoder files differ from the packaged lock")
    write_json(output / "base.lock.json", expected)
    print({"model": MODEL_ID, "revision": REVISION, "files": len(FILES),
           "bytes": sum((output / name).stat().st_size for name in FILES)})


if __name__ == "__main__":
    main()
