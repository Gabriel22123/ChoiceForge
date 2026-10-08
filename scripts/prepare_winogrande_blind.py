"""Prepare a label-independent frozen WinoGrande 1.1 blind-evaluation subset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import urllib.request
import zipfile

from decision_model.core import canonical, digest, load_rows, write_json


ARCHIVE_URL = "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"
ARCHIVE_SHA256 = "3619ab104d8be2977b25c90ff420cb42d491707dcc75362a1e5d22bc082b7318"
DEV_MEMBER = "winogrande_1.1/dev.jsonl"
LABEL_MEMBER = "winogrande_1.1/dev-labels.lst"
TASK = (
    "Choose the option that best fills the blank in the sentence so the completed "
    "sentence is coherent and consistent with commonsense."
)
CHECKPOINTS = (
    ("initial", "runs/architecture-study-v1/D-independent"),
    ("seed42-one", "runs/relation-second-epoch-v1/seed-42"),
    ("seed42-variable", "runs/relation-label-balance-v1/seed-42"),
    ("seed42-balanced", "runs/relation-group-v1/dispersed"),
    ("seed42-grouped", "runs/relation-group-v1/grouped"),
    ("seed43-one", "runs/relation-second-epoch-v1/seed-43"),
    ("seed43-variable", "runs/relation-label-balance-v1/seed-43"),
    ("seed43-balanced", "runs/relation-group-seed43-v1/dispersed"),
    ("seed43-grouped", "runs/relation-group-seed43-v1/grouped"),
)


def read_archive(path, expected=ARCHIVE_SHA256):
    path = Path(path)
    if digest(path.read_bytes()) != expected:
        raise ValueError("WinoGrande archive checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if DEV_MEMBER not in names or LABEL_MEMBER not in names:
            raise ValueError("Official development files are missing")
        raw_rows = archive.read(DEV_MEMBER)
        raw_labels = archive.read(LABEL_MEMBER)
    if len(raw_rows) > 2_000_000 or len(raw_labels) > 100_000:
        raise ValueError("Unexpectedly large development files")
    records = [json.loads(line) for line in raw_rows.decode("utf-8").splitlines() if line.strip()]
    labels = [line.strip() for line in raw_labels.decode("utf-8").splitlines() if line.strip()]
    if len(records) != 1267 or len(labels) != len(records):
        raise ValueError("Unexpected WinoGrande 1.1 development size")
    seen = set()
    for index, (record, label) in enumerate(zip(records, labels)):
        if set(record) != {"qID", "sentence", "option1", "option2", "answer"}:
            raise ValueError("Unexpected WinoGrande record shape")
        if label not in ("1", "2") or record["answer"] != label:
            raise ValueError("Embedded and sidecar labels disagree")
        if record["qID"] in seen or record["sentence"].count("_") != 1:
            raise ValueError("Duplicate ID or malformed blank")
        if any(not isinstance(record[key], str) or not record[key].strip()
               for key in ("qID", "sentence", "option1", "option2")):
            raise ValueError("Empty WinoGrande field")
        if record["option1"].strip() == record["option2"].strip():
            raise ValueError("Duplicate WinoGrande options")
        seen.add(record["qID"])
        record["_official_index"] = index
    return records


def selection_key(record):
    """The upstream answer is deliberately absent from blind-subset selection."""
    return digest({key: record[key] for key in ("qID", "sentence", "option1", "option2")})


def select_rows(records, cap):
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= len(records):
        raise ValueError("cap must select a nonempty bounded subset")
    selected = sorted(records, key=lambda row: (selection_key(row), row["qID"]))[:cap]
    rows = []
    for record in selected:
        choices = [{"id": "option1", "description": record["option1"].strip()},
                   {"id": "option2", "description": record["option2"].strip()}]
        rows.append({
            "id": "winogrande:" + record["qID"],
            "group_id": "winogrande:" + record["qID"],
            "source": "winogrande-1.1",
            "family": "coreference_commonsense",
            "split": "test",
            "evaluation_regime": "untouched_task_family",
            "request": {"task": TASK, "context": record["sentence"].strip(), "choices": choices},
            "label": "option" + record["answer"],
            "provenance": [{
                "url": ARCHIVE_URL,
                "archive_sha256": ARCHIVE_SHA256,
                "member": DEV_MEMBER,
                "original_index": record["_official_index"],
                "qID": record["qID"],
                "license": "CC-BY (version unspecified by official repository README)",
                "label_method": "upstream development answer checked against dev-labels.lst",
                "transformation": "two-candidate dynamic ranking; input-hash blind subset",
            }],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="cache/winogrande/winogrande_1.1.zip")
    parser.add_argument("--output", default="runs/winogrande-blind-v1")
    parser.add_argument("--cap", type=int, default=384)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive, output = root / args.archive, root / args.output
    if output.exists():
        raise ValueError("Use a new blind-study directory")
    if not archive.exists():
        if args.offline:
            raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response:
            raw = response.read(5_000_000)
        if digest(raw) != ARCHIVE_SHA256:
            raise ValueError("Downloaded WinoGrande archive checksum mismatch")
        archive.write_bytes(raw)
    records = read_archive(archive)
    rows = select_rows(records, args.cap)
    output.mkdir(parents=True)
    cases = "".join(canonical(row) + "\n" for row in rows).encode()
    (output / "cases.jsonl").write_bytes(cases)
    # Reuse the full dataset contract so the frozen rows are independently checked.
    if load_rows(output / "cases.jsonl") != rows:
        raise ValueError("Written blind cases failed round-trip validation")
    checkpoints = []
    for name, relative in CHECKPOINTS:
        directory = root / relative
        checkpoints.append({"name": name, "path": relative,
                            "weights_sha256": digest((directory / "decision.safetensors").read_bytes()),
                            "model_sha256": digest((directory / "model.json").read_bytes())})
    prior_rows = load_rows(root / "runs/relation-group-v1/cases.jsonl")
    prior_train_inputs = {digest(row["request"]) for row in prior_rows if row["split"] == "train"}
    exact_finetuning_input_overlap = sum(digest(row["request"]) in prior_train_inputs for row in rows)
    if exact_finetuning_input_overlap:
        raise ValueError("WinoGrande blind input overlaps prior fine-tuning input")
    source_files = {}
    for relative in ("scripts/prepare_winogrande_blind.py", "scripts/run_winogrande_blind.py"):
        source_files[relative] = digest((root / relative).read_bytes())
    protocol = {
        "question": "Does relation-training recipe progress transfer to an untouched public coreference/commonsense task?",
        "source": {"name": "WinoGrande", "version": "1.1", "url": ARCHIVE_URL,
                   "archive_sha256": ARCHIVE_SHA256, "official_dev_rows": len(records),
                   "license": "Dataset CC-BY per official README; version unspecified; raw text is not redistributed in release bundles"},
        "selection": {"rows": len(rows), "method": f"first {args.cap} by SHA-256(qID, sentence, option1, option2); answer excluded",
                      "selection_key_sha256": digest([selection_key(r) for r in sorted(records, key=lambda r: (selection_key(r), r["qID"]))[:args.cap]]),
                      "selected_ids_sha256": digest([row["id"] for row in rows]),
                      "label_counts_recorded_after_freeze": dict(Counter(row["label"] for row in rows))},
        "dataset_sha256": digest(cases),
        "exact_prior_finetuning_input_overlap": exact_finetuning_input_overlap,
        "checkpoints": checkpoints,
        "primary_endpoint": "Raw accuracy on all frozen rows",
        "secondary_endpoints": ["Raw NLL", "Brier", "ECE", "inherited-temperature metrics", "paired case bootstrap"],
        "checkpoint_selection": "All nine predeclared endpoints retained; no tuning, calibration fitting, or winner selection on WinoGrande",
        "source_files": source_files,
        "limitations": [f"One {args.cap}-row label-independent subset of the public development split",
                        "Development labels are public; this is a locally frozen blind evaluation, not the hidden leaderboard test",
                        "Project fine-tuning has no exact input overlap; possible public-base pretraining contamination cannot be ruled out",
                        "Dataset license version is not specified by the official README, so raw text is excluded from release bundles"],
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol-checksums.json", {"cases.jsonl": digest(cases)})
    write_json(output / "status.json", {"state": "prepared", "models": [item["name"] for item in checkpoints]})
    print(json.dumps({"state": "prepared", "rows": len(rows), "dataset_sha256": protocol["dataset_sha256"]}))


if __name__ == "__main__":
    main()
