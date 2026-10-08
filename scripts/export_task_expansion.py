"""Export the PAWS task-expansion adapters and aggregate evidence without dataset rows."""
import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

from decision_model.core import digest, write_json
from task_expansion_report import SEEDS, validated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/task-expansion-v1")
    parser.add_argument("--output", default="dist/task-expansion-v1/adapters-a10.tar.gz")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study, output = root / args.study, root / args.output
    if output.exists():
        raise ValueError("Use a new archive path")

    # This validates frozen rows and per-case results locally. Neither is exported.
    protocol, _, _, _, _, _ = validated(root, study)
    finals = json.loads((study / "trained-checkpoints.json").read_text())
    payload = {}
    fixed = (
        "LICENSE",
        "README.md",
        "README.zh-CN.md",
        "MODEL_CARD.md",
        "THIRD_PARTY.md",
        "pyproject.toml",
        "requirements-tested.txt",
        "requirements-local-lock.txt",
        "sources.lock.json",
        "configs/task-expansion-sources.json",
        "configs/winogrande-blind-source.json",
        "docs/DATA.md",
        "docs/FINDINGS.zh-CN.md",
        "docs/LOCAL_PLAN.md",
        "docs/TASK_EXPANSION_RESULTS.zh-CN.md",
        "docs/evidence/task-expansion-v1.json",
        "third_party/paws-DATA-LICENSE",
        "third_party/paws-UPSTREAM-README",
        "third_party/boolq-UPSTREAM-README",
        "third_party/winogrande-DATA-NOTICE",
        "scripts/task_expansion_data.py",
        "scripts/run_task_expansion_study.py",
        "scripts/task_expansion_report.py",
    )
    for name in fixed:
        payload[name] = (root / name).read_bytes()
    for name in (
        "protocol.json",
        "protocol-checksums.json",
        "status.json",
        "trained-checkpoints.json",
        "blind-evaluation-manifest.json",
    ):
        payload["study/" + name] = (study / name).read_bytes()

    adapter_files = (
        "decision.safetensors",
        "model.json",
        "checksums.json",
        "calibration.json",
        "run.json",
        "evaluation.json",
        "curve.json",
        "optimizer-steps.jsonl",
        "training-plan.jsonl",
    )
    weight_sha256 = {}
    for seed in SEEDS:
        name = f"seed{seed}-paws"
        directory = root / finals[str(seed)]["path"]
        for filename in adapter_files:
            payload[f"{name}/{filename}"] = (directory / filename).read_bytes()
        for path in sorted((directory / "source").glob("*.py")):
            payload[f"{name}/source/{path.name}"] = path.read_bytes()
        actual = digest((directory / "decision.safetensors").read_bytes())
        if actual != finals[str(seed)]["weights_sha256"]:
            raise ValueError(f"seed{seed} weight fingerprint differs")
        weight_sha256[name] = actual

    forbidden = (
        "cases.jsonl",
        "blind-results",
        "predictions",
        "training_state",
        ".parquet",
        ".zip",
    )
    bad = [name for name in payload if any(token in name.lower() for token in forbidden)]
    if bad:
        raise ValueError("Export contains forbidden row, prediction or optimizer-state files: " + str(bad))
    evidence = json.loads(payload["docs/evidence/task-expansion-v1.json"])
    if set(evidence) != {"protocol", "arms", "blind", "blind_paired"}:
        raise ValueError("Unexpected aggregate evidence fields")
    if evidence["protocol"]["study_dataset_sha256"] != protocol["study_dataset_sha256"]:
        raise ValueError("Evidence protocol differs")

    payload["BUNDLE_README.md"] = (
        "# Local PAWS task-expansion adapter bundle\n\n"
        "This bundle contains two loadable research checkpoints, source snapshots, frozen "
        "protocol metadata, aggregate evidence and source/license records. Neither checkpoint "
        "is selected as a production winner: PAWS learning differs sharply by seed and neither "
        "shows stable transfer to BoolQ or WinoGrande.\n\n"
        "Install the matching decision-model package and obtain the pinned public EuroBERT "
        "base separately. From the extracted directory, for example:\n\n"
        "```bash\n"
        "decision-model predict --checkpoint seed42-paws --base-path /path/to/eurobert-2.1b "
        "--input /path/to/request.json\n"
        "```\n\n"
        "The archive excludes base weights, raw or transformed PAWS/BoolQ/WinoGrande rows, "
        "per-case predictions and optimizer state. Download public sources separately and "
        "verify the pinned revisions and hashes before reproduction. SHA256SUMS covers every "
        "payload file other than itself. No remote publication occurs.\n"
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
        "arms": [f"seed{seed}-paws" for seed in SEEDS],
        "weight_sha256": weight_sha256,
        "scope": "local research adapters and aggregate evidence; not published",
        "excluded": ["base weights", "dataset rows", "per-case predictions", "optimizer state"],
    })
    print(output)


if __name__ == "__main__":
    main()
