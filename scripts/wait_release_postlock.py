#!/usr/bin/env python3
"""Wait for the frozen training run, then continue the one-shot release audit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/release-candidate-v1")
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-hours", type=float, default=12)
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 60 or not 1 <= args.max_hours <= 24:
        raise ValueError("Invalid bounded wait settings")
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    deadline = time.monotonic() + args.max_hours * 3600
    while time.monotonic() < deadline:
        locked = study / "locked-endpoints.json"
        if locked.is_file():
            print(json.dumps({"event": "endpoints-locked", "path": str(locked)}), flush=True)
            subprocess.run([sys.executable, "scripts/run_release_postlock.py",
                            "--study", args.study, "--device", args.device],
                           cwd=root, check=True)
            return
        status_path = study / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get("state") == "failed":
                raise RuntimeError("Frozen endpoint training failed: " + json.dumps(status))
        time.sleep(args.poll_seconds)
    raise TimeoutError("Frozen endpoint training did not lock within the bounded wait")


if __name__ == "__main__":
    main()
