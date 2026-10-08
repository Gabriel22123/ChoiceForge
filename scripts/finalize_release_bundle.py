#!/usr/bin/env python3
"""Add the successful project-completion audit to a staged release bundle."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


FILES = (
    "docs/PROJECT_COMPLETION_AUDIT.zh-CN.md",
    "docs/evidence/project-completion-v1.json",
)


def finalize(root, bundle):
    from verify_release_bundle import sha256, verify_bundle

    root, bundle = Path(root), Path(bundle)
    completion = json.loads((root / FILES[1]).read_text())
    if completion.get("complete") is not True or not all(completion["requirements"].values()):
        raise ValueError("Incomplete project audit cannot finalize a release bundle")
    verify_bundle(bundle, require_completion=False)
    manifest_path = bundle / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in FILES:
        source, target = root / name, bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest["files"][name] = {"sha256": sha256(target), "bytes": target.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify_bundle(bundle, require_completion=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="dist/decision-model-minicpm5-2b-release-candidate-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(finalize(root, root / args.bundle), sort_keys=True))


if __name__ == "__main__":
    main()
