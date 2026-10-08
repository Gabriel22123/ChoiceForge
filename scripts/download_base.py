"""Fetch only the pinned public backbone files, then verify all hashes."""
import argparse
import json
from importlib.resources import files
from pathlib import Path

from decision_model.model import MODEL_ID, verify_local_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/eurobert-2.1b")
    parser.add_argument("--offline", action="store_true", help="Only verify an already complete local checkpoint")
    args = parser.parse_args()
    output = Path(args.output)
    lock = json.loads(files("decision_model").joinpath("base.lock.json").read_text())
    complete = all((output / name).is_file() for name in lock["files"])
    if not complete:
        if args.offline:
            raise ValueError("Offline verification requires all five public base files")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID, revision=lock["revision"],
                          allow_patterns=list(lock["files"]), local_dir=str(output))
    verify_local_base(output)
    print(json.dumps({"status": "verified", "model": MODEL_ID,
                      "revision": lock["revision"], "path": str(output.resolve())}))


if __name__ == "__main__":
    main()
