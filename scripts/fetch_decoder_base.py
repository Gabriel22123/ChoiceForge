"""Fetch and checksum the pinned public decoder comparison backbone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_model.base_fetch import fetch_pinned_decoder_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/minicpm5-2b-base")
    args = parser.parse_args()
    output = Path(args.output)
    base = fetch_pinned_decoder_base(output)
    manifest = json.loads((base / "base.lock.json").read_text())
    print({"model": manifest["repository"], "revision": manifest["revision"],
           "files": len(manifest["files"]),
           "bytes": sum((base / name).stat().st_size for name in manifest["files"])})


if __name__ == "__main__":
    main()
