"""Export task-balance adapters and aggregate evidence without dataset rows or predictions."""
import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

from decision_model.core import digest, write_json
from task_balance_report import ARMS, validated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/task-balance-v2")
    parser.add_argument("--output", default="dist/task-balance-v2/adapters-a11.tar.gz")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    study, output = root / args.study, root / args.output
    if output.exists():
        raise ValueError("Use a new archive path")
    validated(root, study)
    protocol = json.loads((study / "protocol.json").read_text())
    finals = json.loads((study / "trained-checkpoints.json").read_text())
    payload = {}
    fixed = (
        "LICENSE", "README.md", "README.zh-CN.md", "MODEL_CARD.md", "THIRD_PARTY.md",
        "pyproject.toml", "requirements-tested.txt", "requirements-local-lock.txt", "sources.lock.json",
        "configs/snli-source.json", "configs/task-expansion-sources.json", "configs/winogrande-blind-source.json",
        "docs/DATA.md", "docs/FINDINGS.zh-CN.md", "docs/LOCAL_PLAN.md",
        "docs/TASK_EXPANSION_RESULTS.zh-CN.md", "docs/TASK_BALANCE_RESULTS.zh-CN.md",
        "docs/evidence/task-expansion-v1.json", "docs/evidence/task-balance-v2.json",
        "third_party/paws-DATA-LICENSE", "third_party/paws-UPSTREAM-README",
        "third_party/boolq-UPSTREAM-README", "third_party/winogrande-DATA-NOTICE",
        "scripts/task_balance_plan.py", "scripts/task_balance_gradient.py",
        "scripts/run_task_balance_study.py", "scripts/task_balance_report.py",
    )
    for name in fixed:
        payload[name] = (root / name).read_bytes()
    for name in ("protocol.json", "protocol-checksums.json", "status.json",
                 "trained-checkpoints.json", "diagnostic-manifest.json", "gradient-probe.json"):
        payload["study/" + name] = (study / name).read_bytes()
    for arm in ARMS:
        payload[f"study/{arm}.json"] = (study / f"{arm}.json").read_bytes()

    adapter_files = ("decision.safetensors", "model.json", "checksums.json", "calibration.json",
                     "run.json", "evaluation.json", "curve.json", "optimizer-steps.jsonl",
                     "training-plan.jsonl")
    weight_sha256 = {}
    for arm in ARMS:
        directory = root / finals[arm]["path"]
        for filename in adapter_files:
            payload[f"{arm}/{filename}"] = (directory / filename).read_bytes()
        for path in sorted((directory / "source").glob("*.py")):
            payload[f"{arm}/source/{path.name}"] = path.read_bytes()
        actual = digest((directory / "decision.safetensors").read_bytes())
        if actual != finals[arm]["weights_sha256"]:
            raise ValueError("Weight fingerprint differs: " + arm)
        weight_sha256[arm] = actual

    forbidden = ("cases.jsonl", "diagnostic-results", "predictions", "training_state", ".parquet", ".zip")
    bad = [name for name in payload if any(token in name.lower() for token in forbidden)]
    if bad:
        raise ValueError("Export contains forbidden data, prediction or optimizer-state files: " + str(bad))
    evidence = json.loads(payload["docs/evidence/task-balance-v2.json"])
    if set(evidence) != {"protocol", "arms", "paired_stratified_minus_mixed", "diagnostic",
                         "diagnostic_paired_stratified_minus_mixed", "gradient_probe"}:
        raise ValueError("Unexpected aggregate evidence fields")
    if evidence["protocol"]["dataset_sha256"] != protocol["dataset_sha256"]:
        raise ValueError("Evidence protocol differs")

    payload["BUNDLE_README.md"] = (
        "# Local task-balance research adapter bundle\n\n"
        "This bundle contains four loadable endpoints from the common-start random-mixing "
        "versus task-and-label-stratification study. No endpoint is selected as a production "
        "winner. Stratification improves PAWS/SNLI hard accuracy but amplifies pre-clip "
        "gradient tails and does not produce stable BoolQ/WinoGrande diagnostic transfer.\n\n"
        "Install the matching decision-model package and obtain the pinned public EuroBERT "
        "base separately. From the extracted directory, for example:\n\n"
        "```bash\n"
        "decision-model predict --checkpoint seed42-stratified --base-path /path/to/eurobert-2.1b "
        "--input /path/to/request.json\n"
        "```\n\n"
        "The archive excludes base weights, raw or transformed dataset rows, per-case "
        "predictions and optimizer state. SHA256SUMS covers every payload file other than "
        "itself. No remote publication occurs.\n"
    ).encode()
    payload["SHA256SUMS"] = "".join(
        f"{digest(raw)}  {name}\n" for name, raw in sorted(payload.items())
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name, raw in sorted(payload.items()):
                info = tarfile.TarInfo(name); info.size = len(raw); info.mode = 0o644; info.mtime = 0
                archive.addfile(info, io.BytesIO(raw))
    write_json(output.with_suffix(output.suffix + ".json"), {
        "sha256": digest(output.read_bytes()), "bytes": output.stat().st_size,
        "files": len(payload), "arms": list(ARMS), "weight_sha256": weight_sha256,
        "scope": "local research adapters and aggregate evidence; not published",
        "excluded": ["base weights", "dataset rows", "per-case predictions", "optimizer state"],
    })
    print(output)


if __name__ == "__main__":
    main()
