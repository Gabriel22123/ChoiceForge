import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="decision-model")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare",help="Fetch/verify pinned public sources and build grouped splits")
    data.add_argument("--lock",default="sources.lock.json")
    data.add_argument("--cache",default=".cache")
    data.add_argument("--output",required=True)
    data.add_argument("--offline",action="store_true")
    check = commands.add_parser("validate-data")
    check.add_argument("path")
    train = commands.add_parser("train")
    train.add_argument("--data",required=True)
    train.add_argument("--config",default="configs/eurobert-public-pilot.json")
    train.add_argument("--output",required=True)
    train.add_argument("--base-path",help="Optional verified original checkpoint already on disk")
    train.add_argument("--init-from",help="Warm start adapter/head; starts a new optimizer")
    train.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"])
    predict = commands.add_parser("predict")
    predict.add_argument("--checkpoint",required=True)
    predict.add_argument("--input",required=True,help="Canonical or typed Boolean/Choice/Score JSON")
    predict.add_argument("--base-path")
    predict.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"])
    merged = commands.add_parser("predict-merged", help="Predict with a verified merged bundle")
    merged.add_argument("--bundle", required=True)
    merged.add_argument("--input", required=True, help="Canonical task/context/choices JSON")
    merged.add_argument("--base-path", required=True, help="Verified public MiniCPM base directory")
    merged.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    for name in ("evaluate","audit"):
        sub=commands.add_parser(name)
        sub.add_argument("--checkpoint",required=True)
        sub.add_argument("--data",required=True)
        sub.add_argument("--output",required=True)
        sub.add_argument("--max-cases",type=int,default=24 if name=="audit" else 192)
        sub.add_argument("--base-path")
        sub.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"])
        if name=="evaluate":sub.add_argument("--split",choices=["train","validation","test"],default="test")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import build
        result = build(args.lock,args.cache,args.output,args.offline)
    elif args.command == "validate-data":
        from .core import load_rows,summarize_rows
        result = summarize_rows(load_rows(args.path))
    elif args.command == "train":
        from .train import train as run
        result = run(args)
    elif args.command in ("evaluate","audit"):
        from .evaluate import evaluate,audit
        result=(audit if args.command=="audit" else evaluate)(args)
    elif args.command == "predict-merged":
        from .merged_runtime import load_merged_bundle
        model = load_merged_bundle(
            args.bundle, base_path=args.base_path, device=args.device)
        result = model.predict(json.loads(Path(args.input).read_text()))
    else:
        from .judge_factory import make_judge, verify_base_for_config
        config = json.loads((Path(args.checkpoint)/"model.json").read_text())
        if args.base_path:
            verify_base_for_config(config,args.base_path)
        judge = make_judge(config,base_path=args.base_path,checkpoint=args.checkpoint,
                           device=args.device)
        payload = json.loads(Path(args.input).read_text())
        if "type" in payload:
            from .typed_api import predict_typed
            result = predict_typed(judge, payload)
        else:
            result = judge.predict(payload)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
