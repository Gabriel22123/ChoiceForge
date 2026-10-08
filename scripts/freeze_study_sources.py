"""Copy protocol-hashed study sources into an immutable local snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_model.core import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    protocol = json.loads((study / "protocol.json").read_text())
    destination = study / "frozen-source"
    manifest_path = study / "frozen-source-manifest.json"
    if destination.exists() or manifest_path.exists():
        raise ValueError("Study source snapshot already exists")
    for name, expected in protocol["source_files"].items():
        source = root / name
        if digest(source.read_bytes()) != expected:
            raise ValueError("Live source no longer matches frozen protocol: " + name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    write_json(manifest_path, {"source_files": protocol["source_files"]})
    print({"study": args.study, "source_files": len(protocol["source_files"])})


if __name__ == "__main__":
    main()
