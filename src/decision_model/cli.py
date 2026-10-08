"""Command-line entry points for public ChoiceForge workflows."""

import argparse
import json
from pathlib import Path

DEVICES = ("auto", "cpu", "cuda", "mps")


def _add_prepare_parser(commands):
    parser = commands.add_parser(
        "prepare", help="Fetch/verify pinned public sources and build grouped splits"
    )
    parser.add_argument("--lock", default="sources.lock.json")
    parser.add_argument("--cache", default=".cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--offline", action="store_true")


def _add_train_parser(commands):
    parser = commands.add_parser("train")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/eurobert-public-pilot.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-path", help="Optional verified original checkpoint already on disk")
    parser.add_argument("--init-from", help="Warm start adapter/head; starts a new optimizer")
    parser.add_argument("--device", default="auto", choices=DEVICES)


def _add_predict_parser(commands):
    parser = commands.add_parser("predict")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input", required=True, help="Canonical or typed Boolean/Choice/Score JSON"
    )
    parser.add_argument("--base-path")
    parser.add_argument("--device", default="auto", choices=DEVICES)


def _add_merged_parser(commands):
    parser = commands.add_parser("predict-merged", help="Predict with a verified merged bundle")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--input", required=True, help="Canonical task/context/choices JSON")
    parser.add_argument(
        "--base-path",
        help="Verified public MiniCPM base directory; downloads the pinned base when omitted",
    )
    parser.add_argument("--device", default="auto", choices=DEVICES)


def _add_evaluation_parser(commands, name):
    parser = commands.add_parser(name)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=24 if name == "audit" else 192)
    parser.add_argument("--base-path")
    parser.add_argument("--device", default="auto", choices=DEVICES)
    if name == "evaluate":
        parser.add_argument("--split", choices=("train", "validation", "test"), default="test")


def _add_benchmark_parser(commands):
    parser = commands.add_parser(
        "benchmark", help="Score a prediction file with the public audit protocol"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--threshold", dest="thresholds", action="append", type=float)


def _build_parser():
    parser = argparse.ArgumentParser(prog="decision-model")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_prepare_parser(commands)
    check = commands.add_parser("validate-data")
    check.add_argument("path")
    _add_train_parser(commands)
    _add_predict_parser(commands)
    _add_merged_parser(commands)
    _add_evaluation_parser(commands, "evaluate")
    _add_evaluation_parser(commands, "audit")
    _add_benchmark_parser(commands)
    return parser


def _run(args):
    if args.command == "prepare":
        from .data import build

        return build(args.lock, args.cache, args.output, args.offline)
    if args.command == "validate-data":
        from .core import load_rows, summarize_rows

        return summarize_rows(load_rows(args.path))
    if args.command == "train":
        from .train import train

        return train(args)
    if args.command in ("evaluate", "audit"):
        from .evaluate import audit, evaluate

        return (audit if args.command == "audit" else evaluate)(args)
    if args.command == "benchmark":
        from .benchmark import benchmark

        return benchmark(args)
    if args.command == "predict-merged":
        from .merged_runtime import load_merged_bundle

        base_path = args.base_path
        if base_path is None:
            base_path = "models/minicpm5-2b-base"
            if not (Path(base_path) / "base.lock.json").is_file():
                from .base_fetch import fetch_pinned_decoder_base

                fetch_pinned_decoder_base(base_path)
        model = load_merged_bundle(args.bundle, base_path=base_path, device=args.device)
        return model.predict(json.loads(Path(args.input).read_text()))

    from .judge_factory import make_judge, verify_base_for_config

    config = json.loads((Path(args.checkpoint) / "model.json").read_text())
    if args.base_path:
        verify_base_for_config(config, args.base_path)
    judge = make_judge(
        config,
        base_path=args.base_path,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    payload = json.loads(Path(args.input).read_text())
    if "type" in payload:
        from .typed_api import predict_typed

        return predict_typed(judge, payload)
    return judge.predict(payload)


def main(argv=None):
    args = _build_parser().parse_args(argv)
    result = _run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
