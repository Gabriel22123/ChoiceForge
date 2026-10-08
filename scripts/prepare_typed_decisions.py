"""Prepare pinned Typed Decisions data as leakage-resistant soft-label folds.

Only state, question instructions, candidate semantics and public teacher
probabilities enter prepared records.  Latent generation factors, rationales and
agreement diagnostics are deliberately excluded from model input and output.
"""
import argparse
import collections
import hashlib
import json
import math
import urllib.request
from pathlib import Path

from decision_model.core import canonical, digest, load_rows, summarize_rows, write_json


def json_cell(value, name):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {name}") from exc
    return value


def semantic(value, name):
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (dict, list)):
        return canonical(value)
    raise ValueError(f"{name} must be nonempty text or JSON")


def normalize_question(question):
    if not isinstance(question, dict) or not isinstance(question.get("instructions"), str):
        raise ValueError("Question needs textual instructions")
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is None:
            criteria = {"true": "TRUE", "false": "FALSE"}
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            raise ValueError("noul criteria must contain exactly true and false")
        pairs = [(key, semantic(criteria[key], f"{key} criterion")) for key in ("true", "false")]
        output_type = "boolean"
    elif kind == "choice":
        if isinstance(criteria, dict):
            pairs = [(str(key), semantic(value, "choice criterion")) for key, value in criteria.items()]
        elif isinstance(criteria, list):
            pairs = [(str(index), semantic(value, "choice criterion"))
                     for index, value in enumerate(criteria)]
        else:
            raise ValueError("choice criteria must be an object or array")
        if not 2 <= len(pairs) <= 32:
            raise ValueError("choice needs 2..32 criteria")
        output_type = "choice"
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("score criteria must be an ordered array of 2..10 levels")
        pairs = [(str(index), semantic(value, "score level")) for index, value in enumerate(criteria)]
        output_type = "score"
    else:
        raise ValueError("Unknown Typed Decisions question type")
    ids = [key for key, _ in pairs]
    descriptions = [description for _, description in pairs]
    if any(not key for key in ids) or len(ids) != len(set(ids)) or len(descriptions) != len(set(descriptions)):
        raise ValueError("Candidate IDs and descriptions must be distinct")
    return {
        "type": output_type,
        "task": question["instructions"].strip(),
        "choices": [{"id": key, "description": description} for key, description in pairs],
    }


def convert_case(case, split, upstream_split, revision, license_name, evaluation_regime):
    required = {"id", "workflow", "state", "questions", "gold"}
    if not isinstance(case, dict) or not required.issubset(case):
        raise ValueError("Typed Decisions case is missing required fields")
    questions = json_cell(case["questions"], "questions")
    gold = json_cell(case["gold"], "gold")
    state = semantic(json_cell(case["state"], "state"), "state")
    if not isinstance(questions, dict) or not questions or not isinstance(gold, dict):
        raise ValueError("Questions and gold must be nonempty objects")
    rows = []
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Question IDs must be nonempty strings")
        normalized = normalize_question(question)
        choice_ids = [choice["id"] for choice in normalized["choices"]]
        try:
            raw = gold[question_id]["probabilities"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Gold probabilities missing for question") from exc
        if not isinstance(raw, dict) or set(raw) != set(choice_ids):
            raise ValueError("Gold probability keys differ from candidate IDs")
        values = [float(raw[key]) for key in choice_ids]
        if any(value < 0 or not math.isfinite(value) for value in values) or sum(values) <= 0:
            raise ValueError("Gold probabilities must be finite nonnegative values with positive mass")
        total = sum(values)
        values = [value / total for value in values]
        label = choice_ids[max(range(len(values)), key=values.__getitem__)]
        rows.append({
            "id": f"typed-decisions:{case['workflow']}:{case['id']}:{question_id}",
            "group_id": f"typed-decisions:{case['workflow']}:{case['id']}",
            "source": "typed-decisions",
            "family": case["workflow"],
            "decision_type": normalized["type"],
            "split": split,
            "evaluation_regime": evaluation_regime,
            "label": label,
            "target_probabilities": dict(zip(choice_ids, values)),
            "request": {"task": normalized["task"], "context": state,
                        "choices": normalized["choices"]},
            "provenance": [{
                "dataset": "LocalLLaMA/typed-decisions",
                "url": "https://huggingface.co/datasets/LocalLLaMA/typed-decisions",
                "revision": revision,
                "license": license_name,
                "upstream_split": upstream_split,
                "upstream_case_id": case["id"],
                "workflow": case["workflow"],
                "question_id": question_id,
                "label_semantics": "public teacher probability distribution; not measured action success",
                "transformation": "semantic fields only; teacher probabilities normalized; first argmax retained as label",
            }],
        })
    return rows


def stable_order(cases, seed, workflow):
    return sorted(cases, key=lambda case: hashlib.sha256(
        f"{seed}:{workflow}:{case['id']}".encode()).hexdigest())


def build_fold(train_cases, test_cases, heldout, config):
    expected = config["expected"]
    workflows = expected["workflows"]
    if heldout not in workflows:
        raise ValueError("Unknown held-out workflow")
    by_train = {workflow: [case for case in train_cases if case["workflow"] == workflow]
                for workflow in workflows}
    by_test = {workflow: [case for case in test_cases if case["workflow"] == workflow]
               for workflow in workflows}
    for workflow in workflows:
        if len(by_train[workflow]) != expected["train_cases_per_workflow"]:
            raise ValueError(f"Unexpected train case count for {workflow}")
        if len(by_test[workflow]) != expected["test_cases_per_workflow"]:
            raise ValueError(f"Unexpected test case count for {workflow}")
    rows = []
    split_cases = collections.Counter()
    fit = expected["seen_train_cases_per_workflow"]
    for workflow in workflows:
        if workflow == heldout:
            continue
        ordered = stable_order(by_train[workflow], config["split_seed"], workflow)
        for split, cases in (("train", ordered[:fit]), ("validation", ordered[fit:])):
            for case in cases:
                rows.extend(convert_case(case, split, "train", config["revision"], config["license"],
                                         "seen_task_training" if split == "train" else "seen_task_validation"))
                split_cases[(split, workflow)] += 1
    for workflow in workflows:
        regime = "held_out_task" if workflow == heldout else "seen_task_new_examples"
        for case in by_test[workflow]:
            rows.extend(convert_case(case, "test", "test", config["revision"], config["license"], regime))
            split_cases[("test", workflow)] += 1
    expected_questions = expected["questions_per_case"]
    if any(count != expected_questions for count in collections.Counter(
            row["group_id"] for row in rows).values()):
        raise ValueError("Unexpected questions per case")
    return sorted(rows, key=lambda row: row["id"]), {
        split: {workflow: split_cases[(split, workflow)] for workflow in workflows
                if split_cases[(split, workflow)]}
        for split in ("train", "validation", "test")
    }


def verify_file(path, spec):
    if not path.is_file():
        raise ValueError(f"Missing pinned source file: {path}")
    raw = path.read_bytes()
    if len(raw) != spec["bytes"] or digest(raw) != spec["sha256"]:
        raise ValueError(f"Pinned source size or SHA-256 differs: {path}")


def download_file(path, spec):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify_file(path, spec)
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(spec["url"], timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        verify_file(temporary, spec)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_parquet(path):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Install the public-data extra: pip install -e '.[data]'") from exc
    return parquet.read_table(path).to_pylist()


def prepare(config_path, source_dir, output_dir, download=False):
    config = json.loads(Path(config_path).read_text())
    source_dir = Path(source_dir); output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Use a new Typed Decisions output directory")
    for spec in config["files"]:
        path = source_dir / spec["path"]
        if download:
            download_file(path, spec)
        verify_file(path, spec)
    train = read_parquet(source_dir / "all/train-00000-of-00001.parquet")
    test = read_parquet(source_dir / "all/test-00000-of-00001.parquet")
    output_dir.mkdir(parents=True)
    folds = {}
    for heldout in config["expected"]["workflows"]:
        rows, case_counts = build_fold(train, test, heldout, config)
        fold = output_dir / f"heldout-{heldout}"
        fold.mkdir()
        target = fold / "cases.jsonl"
        target.write_text("".join(canonical(row) + "\n" for row in rows))
        verified = load_rows(target)
        manifest = {
            "dataset": config["dataset"], "revision": config["revision"],
            "license": config["license"], "held_out_workflow": heldout,
            "split_seed": config["split_seed"], "case_counts": case_counts,
            "rows": summarize_rows(verified), "dataset_sha256": digest(target.read_bytes()),
            "source_files": [{key: spec[key] for key in ("path", "bytes", "sha256", "url")}
                             for spec in config["files"]],
            "test_policy": "official test only; never used for fitting, calibration or selection",
            "heldout_policy": "all train cases from the held-out workflow are excluded; its official test is zero-shot",
            "label_semantics": "agreement with a public teacher probability distribution, not objective action success",
            "excluded_fields": ["factors", "label_agreement", "teacher rationale", "all other upstream columns"],
            "script_sha256": digest(Path(__file__).read_bytes()),
        }
        write_json(fold / "manifest.json", manifest)
        folds[heldout] = {"path": str(fold), "dataset_sha256": manifest["dataset_sha256"],
                          "rows": len(rows), "case_counts": case_counts}
    write_json(output_dir / "manifest.json", {
        "study": "four-fold leave-one-workflow-out soft-target evaluation",
        "config_sha256": digest(Path(config_path).read_bytes()), "folds": folds,
        "selection_rule": "report every fold; no workflow may be selected after observing scores",
    })
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/typed-decisions-source.json")
    parser.add_argument("--source-dir", default="cache/typed-decisions-ea930645")
    parser.add_argument("--output", default="data/typed-decisions-workflow-folds-v1")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.source_dir, args.output, args.download), indent=2))


if __name__ == "__main__":
    main()
