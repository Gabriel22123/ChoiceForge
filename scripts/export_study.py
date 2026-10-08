"""Export a completed study's adapters with provenance and licenses, locally."""
import argparse
import gzip
import io
import tarfile
from pathlib import Path

from decision_model.core import digest, write_json
from study_report import ARMS, load_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/calibration-study-v1")
    parser.add_argument("--second-study")
    parser.add_argument("--output", default="dist/calibration-study-v1-adapters.tar.gz")
    parser.add_argument("--kind", choices=("calibration", "architecture", "gradient", "semantic", "curriculum", "matched", "relations", "relations-multiseed", "relation-ablations"), default="calibration")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study, output = root / args.study, root / args.output
    if output.exists():
        raise ValueError("Use a new archive path")
    arms = ARMS
    arm_paths = {}
    study_paths = None
    required_reports = ("docs/STUDY.zh-CN.md",)
    if args.kind == "relation-ablations":
        if not args.second_study:
            raise ValueError("--second-study is required for relation-ablations")
        from label_balance_report import load_experiment as load_label_balance
        from second_epoch_report import load_experiment as load_second_epoch
        label_study = root / args.second_study
        epoch_result = load_second_epoch(root, study)
        label_result = load_label_balance(root, label_study)
        epoch_protocol, label_protocol = epoch_result["protocol"], label_result["protocol"]
        for key in ("dataset_sha256", "initial_weight_sha256", "selected_ids"):
            if epoch_protocol[key] != label_protocol[key]:
                raise ValueError("Ablation studies differ on controlled field: " + key)
        if epoch_protocol["seeds"] != label_protocol["seeds"]:
            raise ValueError("Ablation studies use different seeds")
        arms = []
        for seed in epoch_protocol["seeds"]:
            arms.extend((f"seed{seed}-one", f"seed{seed}-variable", f"seed{seed}-balanced"))
            arm_paths[f"seed{seed}-one"] = study / f"seed-{seed}"
            arm_paths[f"seed{seed}-variable"] = label_study / f"seed-{seed}"
            control = label_protocol["controls"][str(seed)]
            arm_paths[f"seed{seed}-balanced"] = root / control["balanced_study"] / "dispersed"
        arms = tuple(arms)
        study_paths = (("second-epoch", study), ("label-balance", label_study))
        required_reports = ("docs/SECOND_EPOCH_RESULTS.zh-CN.md", "docs/LABEL_BALANCE_RESULTS.zh-CN.md")
    elif args.kind == "relations-multiseed":
        if not args.second_study:
            raise ValueError("--second-study is required for relations-multiseed")
        from relation_group_report import load_study as load_relation_study
        second = root / args.second_study
        first_result = load_relation_study(root, study)
        second_result = load_relation_study(root, second)
        first_seed, second_seed = first_result["protocol"]["seed"], second_result["protocol"]["seed"]
        if first_seed == second_seed:
            raise ValueError("Relation studies must use distinct seeds")
        for key in ("dataset_sha256", "initial_weight_sha256", "selected_ids", "updates", "epochs", "example_exposures"):
            if first_result["protocol"][key] != second_result["protocol"][key]:
                raise ValueError("Relation studies differ on controlled field: " + key)
        arms = tuple(f"seed{seed}-{arm}" for seed in (first_seed, second_seed) for arm in ("grouped", "dispersed"))
        arm_paths = {f"seed{first_seed}-{arm}": study / arm for arm in ("grouped", "dispersed")}
        arm_paths.update({f"seed{second_seed}-{arm}": second / arm for arm in ("grouped", "dispersed")})
        study_paths = ((f"seed{first_seed}", study), (f"seed{second_seed}", second))
        required_reports = ("docs/RELATION_GROUP_MULTISEED.zh-CN.md",)
    elif args.kind == "relations":
        from relation_group_report import load_study as load_relation_study
        result = load_relation_study(root, study)
        arms = tuple(result["arms"])
        required_reports = ("docs/RELATION_GROUP_RESULTS.zh-CN.md",)
    elif args.kind == "matched":
        from matched_budget_report import load_study as load_matched_study
        result = load_matched_study(root, study)
        arms = tuple(result["arms"])
        arm_paths = {name: root / value["arm"]["final"] for name, value in result["arms"].items()}
        required_reports = ("docs/MATCHED_BUDGET_RESULTS.zh-CN.md",)
    elif args.kind == "curriculum":
        from learnability_report import load_experiment
        load_experiment(root, study)
        arms = ("continued",)
        required_reports = ("docs/LEARNABILITY_RESULTS.zh-CN.md",)
    elif args.kind == "semantic":
        from semantic_scale_report import ARMS as semantic_arms, load_study as load_semantic_study
        load_semantic_study(study)
        arms = semantic_arms
        required_reports = ("docs/SEMANTIC_SCALE_STUDY.zh-CN.md",)
    elif args.kind == "gradient":
        from gradient_training_report import ARMS as gradient_arms, load_study as load_gradient_study
        load_gradient_study(study)
        arms = tuple(gradient_arms)
        required_reports = ("docs/GRADIENT_TRAINING_STUDY.zh-CN.md",)
    elif args.kind == "architecture":
        from architecture_report import ARMS as architecture_arms, load_study as load_architecture_study
        load_architecture_study(study)
        arms = architecture_arms
        required_reports = ("docs/ARCHITECTURE_STUDY.zh-CN.md",)
    else:
        load_study(study)
    payload = {}
    for name in ("LICENSE", "THIRD_PARTY.md", "MODEL_CARD.md"):
        payload[name] = (root / name).read_bytes()
    for directory in ("docs", "third_party"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in (".md", ".json", ""):
                payload[path.relative_to(root).as_posix()] = path.read_bytes()
    if any(report not in payload for report in required_reports):
        raise ValueError("Generate every complete measured study report before export")
    if study_paths:
        for prefix, directory in study_paths:
            for path in sorted(directory.glob("*.json")):
                payload["study/" + prefix + "/" + path.name] = path.read_bytes()
    else:
        for path in sorted(study.glob("*.json")):
            payload["study/" + path.name] = path.read_bytes()
    if args.kind == "matched":
        for path in sorted(study.glob("*-plan.jsonl")):
            payload["study/" + path.name] = path.read_bytes()
        for arm, value in result["arms"].items():
            warm = root / value["arm"]["phase1"]
            for name in ("run.json", "result.json", "protocol.json", "steps.jsonl", "model.json", "checksums.json"):
                if (warm/name).exists():
                    payload["study/"+arm+"-phase1/"+name] = (warm/name).read_bytes()
    for arm in arms:
        directory = arm_paths.get(arm, study / arm)
        for name in ("decision.safetensors", "model.json", "checksums.json", "calibration.json",
                     "run.json", "evaluation.json", "audit.json", "test-predictions.json", "training-plan.jsonl"):
            payload[arm + "/" + name] = (directory / name).read_bytes()
        for path in sorted((directory / "source").glob("*.py")):
            payload[arm + "/source/" + path.name] = path.read_bytes()
        if (directory / "benchmark.json").exists():
            payload[arm + "/benchmark.json"] = (directory / "benchmark.json").read_bytes()
        if (directory / "demo-tool-routing.json").exists():
            payload[arm + "/demo-tool-routing.json"] = (directory / "demo-tool-routing.json").read_bytes()
        if (directory / "optimizer-steps.jsonl").exists():
            payload[arm + "/optimizer-steps.jsonl"] = (directory / "optimizer-steps.jsonl").read_bytes()
        if (directory / "curve.json").exists():
            payload[arm + "/curve.json"] = (directory / "curve.json").read_bytes()
    payload["README.md"] = ("# Local experimental adapter bundle\n\n"
        f"Controlled study arms: {', '.join(arms)}. No winner is selected.\n"
        "These are research checkpoints, not validated production decision models.\n"
        f"See {', '.join(required_reports)} and MODEL_CARD.md for measured results and limitations.\n\n"
        "Install the matching decision-model source project and download/verify the pinned\n"
        "public EuroBERT base separately. From the extracted directory, for example:\n\n"
        f"```bash\ndecision-model predict --checkpoint {arms[0]} --base-path /path/to/eurobert-2.1b "
        "--input /path/to/request.json\n```\n\n"
        "The bundle contains adapter/head weights, calibration, source snapshots, provenance\n"
        "and licenses. It excludes base weights, raw datasets and optimizer states.\n"
        + ("For the matched study, phase1 directories under study/ contain metadata only;\n"
           "the four top-level arm directories contain the loadable final weights.\n" if args.kind == "matched" else "") +
        "Use the project's --init-from option for warm-start training with a new optimizer.\n"
        "SHA256SUMS lists every payload file other than itself. No remote publication occurs.\n").encode()
    payload["SHA256SUMS"] = "".join(f"{digest(raw)}  {name}\n" for name, raw in sorted(payload.items())).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name, raw in sorted(payload.items()):
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(raw))
    write_json(output.with_suffix(output.suffix + ".json"), {
        "sha256": digest(output.read_bytes()), "bytes": output.stat().st_size,
        "files": len(payload), "arms": list(arms), "scope": "local experimental adapters; not published"})
    print(output)


if __name__ == "__main__":
    main()
