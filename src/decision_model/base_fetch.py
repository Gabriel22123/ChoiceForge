"""Download and verify the pinned public decoder base."""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path


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


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch_pinned_decoder_base(output):
    """Download the locked base and fail closed if any file differs."""
    from huggingface_hub import snapshot_download

    output = Path(output)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=output,
        allow_patterns=list(FILES),
    )
    missing = [name for name in FILES if not (output / name).is_file()]
    if missing:
        raise ValueError("Pinned snapshot is missing: " + ", ".join(missing))
    expected = json.loads(files("decision_model").joinpath("decoder_base.lock.json").read_text())
    observed = {name: _sha256(output / name) for name in FILES}
    if expected["repository"] != f"https://huggingface.co/{MODEL_ID}" or expected["revision"] != REVISION:
        raise ValueError("Packaged decoder lock identity differs")
    if observed != expected["files"]:
        raise ValueError("Downloaded decoder files differ from the packaged lock")
    (output / "base.lock.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    )
    return output
