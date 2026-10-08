"""Export aggregate blind-evaluation evidence without redistributing dataset text."""
import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

from decision_model.core import digest, write_json
from winogrande_blind_report import validated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/winogrande-blind-v1")
    parser.add_argument("--output", default="dist/winogrande-blind-v1/evaluation-evidence.tar.gz")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study, output = root / args.study, root / args.output
    if output.exists():
        raise ValueError("Use a new archive path")

    # Validation reads the frozen cases and per-case results locally. Neither is exported.
    validated(study)
    payload = {}
    fixed = (
        "LICENSE",
        "README.md",
        "README.zh-CN.md",
        "MODEL_CARD.md",
        "THIRD_PARTY.md",
        "requirements-tested.txt",
        "configs/winogrande-blind-source.json",
        "docs/DATA.md",
        "docs/FINDINGS.zh-CN.md",
        "docs/LOCAL_PLAN.md",
        "docs/WINOGRANDE_BLIND_RESULTS.zh-CN.md",
        "docs/evidence/winogrande-blind-v1.json",
        "third_party/winogrande-DATA-NOTICE",
        "scripts/prepare_winogrande_blind.py",
        "scripts/run_winogrande_blind.py",
        "scripts/winogrande_blind_report.py",
    )
    for name in fixed:
        payload[name] = (root / name).read_bytes()
    for name in ("protocol.json", "protocol-checksums.json", "status.json"):
        payload["study/" + name] = (study / name).read_bytes()

    forbidden_names = ("cases.jsonl", "predictions", ".zip", "decision.safetensors")
    if any(any(token in name.lower() for token in forbidden_names) for name in payload):
        raise ValueError("Export would include blind text, predictions, archive or weights")
    evidence = json.loads(payload["docs/evidence/winogrande-blind-v1.json"])
    if set(evidence) != {"protocol", "truth_option1_fraction", "models", "paired"}:
        raise ValueError("Unexpected aggregate evidence fields")

    payload["BUNDLE_README.md"] = (
        "# Frozen WinoGrande evaluation evidence\n\n"
        "This local bundle contains the frozen protocol, aggregate metrics, source code, "
        "provenance and license notices. It deliberately excludes WinoGrande text, transformed "
        "cases, per-case predictions, model weights and the public base model.\n\n"
        "Reproduction requires downloading the official archive at the pinned URL, verifying "
        "its SHA-256, preparing the frozen subset, and supplying the separately packaged local "
        "experimental checkpoints. The preparation script verifies the selection and exact "
        "fine-tuning-input overlap. No model winner is selected by this evaluation.\n"
    ).encode()
    payload["SHA256SUMS"] = "".join(
        f"{digest(raw)}  {name}\n" for name, raw in sorted(payload.items())
    ).encode()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name, raw in sorted(payload.items()):
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                info.mode = 0o644
                info.mtime = 0
                archive.addfile(info, io.BytesIO(raw))
    write_json(output.with_suffix(output.suffix + ".json"), {
        "sha256": digest(output.read_bytes()),
        "bytes": output.stat().st_size,
        "files": len(payload),
        "scope": "aggregate frozen evaluation evidence; no dataset text, per-case predictions, or weights",
    })
    print(output)


if __name__ == "__main__":
    main()
