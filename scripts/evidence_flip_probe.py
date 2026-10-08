"""Small, authored counterfactual diagnostic; never a training data source."""
import argparse
import gc
import json
import os
from pathlib import Path
import re

from decision_model.core import canonical, digest, metrics, write_json
from decision_model.model import Judge, verify_local_base
from semantic_scale_report import load_study

CHOICES = [
    {"id": "contradiction", "description": "The premise contradicts the hypothesis."},
    {"id": "neutral", "description": "The premise does not determine whether the hypothesis is true or false."},
    {"id": "entailment", "description": "The premise supports the hypothesis."},
]
TASK = "Determine how the premise relates to the hypothesis. Use only the information stated in the premise."
# Within each group, task/hypothesis/choices stay fixed; only the premise changes.
# Wording is authored for this project; no private cases or teacher labels.
TRIPLES = [
    ("count", "Exactly two people are in the boat.",
     "Exactly two people are in the boat.", "Exactly three people are in the boat.", "Some people are in the boat."),
    ("color", "The only ball in the box is red.",
     "The box contains exactly one ball, and that ball is red.",
     "The box contains exactly one ball, and that ball is not red.", "The box contains exactly one ball."),
    ("transfer", "Alice handed the book to Bob.",
     "Alice handed the book to Bob.", "Alice did not hand the book to Bob.", "Alice and Bob discussed a book."),
    ("location", "The cup is on the table.",
     "The cup is on the table.", "The cup is not on the table.", "There is a cup in the room."),
    ("state", "The door is open.",
     "The door is open.", "The door is not open.", "The door is wooden."),
    ("quantifier", "Every box in the room is empty.",
     "All boxes in the room are empty.", "At least one box in the room contains a book.", "There are three boxes in the room."),
]


def cases():
    return [{"id": group + "-" + label, "group": group, "expected": label,
             "request": {"task": TASK, "context": canonical({"hypothesis": hypothesis, "premise": premise}),
                         "choices": [dict(c) for c in CHOICES]}}
            for group, hypothesis, *premises in TRIPLES
            for label, premise in zip(("entailment", "contradiction", "neutral"), premises)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/semantic-scale-v1")
    parser.add_argument("--output", default="runs/evidence-flips-v1")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--checkpoint", action="append", help="Explicit NAME=PATH; repeat to compare checkpoints instead of --study")
    parser.add_argument("--report", default="docs/EVIDENCE_FLIPS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/evidence-flips-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if output.exists():
        raise ValueError("Use a new diagnostic directory")
    if args.checkpoint:
        checkpoints = {}
        for value in args.checkpoint:
            name, separator, path = value.partition("=")
            if not separator or not path or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in checkpoints:
                raise ValueError("Use unique NAME=PATH checkpoint arguments")
            checkpoints[name] = root / path
        selection = "Explicit fixed checkpoints; diagnostic only, no fitting or checkpoint selection"
    else:
        protocol, _, arms = load_study(root / args.study)
        checkpoints = {"initial": root / protocol["initial_checkpoint"],
                       **{name: root / args.study / name for name in arms}}
        selection = "Both frozen final seeds and their common starting checkpoint"
    base = root / "models/eurobert-2.1b"
    verify_local_base(base)
    selected = cases()
    output.mkdir(parents=True)
    write_json(output / "cases.json", selected)
    experiment = {"kind": "Post-hoc, authored evidence-flip probe after seed-42 failure; not training or a blind benchmark",
                  "script_sha256": digest(Path(__file__).read_bytes()), "cases_sha256": digest(selected),
                  "cases": 18, "groups": 6, "selection": selection,
                  "inference": "Raw probabilities, no extra temperature fit", "models": {},
                  "limitations": ["Hand-authored English examples under the project code license; not an external internet benchmark",
                                  "Only six elementary relation groups; success would not establish general reasoning",
                                  "Counterfactual text intervention probes these specific cases, not whether every context is ignored"]}
    for name, checkpoint in checkpoints.items():
        config = json.loads((checkpoint / "model.json").read_text())
        judge = Judge(config, base_path=base, checkpoint=checkpoint, device=args.device)
        details, probabilities, labels = [], [], []
        with judge.torch.no_grad():
            for case in selected:
                p = judge.logits([judge.encode(case["request"])])[0].float().softmax(-1).cpu().tolist()
                ids = [c["id"] for c in case["request"]["choices"]]
                choice = ids[max(range(len(p)), key=p.__getitem__)]
                probabilities.append(p); labels.append(ids.index(case["expected"]))
                details.append({"id": case["id"], "group": case["group"], "expected": case["expected"],
                                "choice": choice, "probabilities": dict(zip(ids, p)), "correct": choice == case["expected"]})
        group_results = {group: {"all_three_correct": all(r["correct"] for r in details if r["group"] == group),
                                 "distinct_choices": len({r["choice"] for r in details if r["group"] == group})}
                         for group, *_ in TRIPLES}
        experiment["models"][name] = {"metrics": metrics(labels, probabilities), "groups": group_results,
                                      "details": details, "weight_sha256": digest((checkpoint / "decision.safetensors").read_bytes())}
        write_json(output / "result.json", experiment)
        print(json.dumps({"model": name, "correct": sum(r["correct"] for r in details),
                          "complete_groups": sum(g["all_three_correct"] for g in group_results.values())}), flush=True)
        torch = judge.torch
        del judge
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
    write_json(root / args.evidence, experiment)
    lines = ["# 证据翻转：材料改变后，答案会正确改变吗？", "",
             "观察到第一组推断失败后增加的诊断。18 条项目自写的英文逻辑样例，分为 6 组三元组；不是训练数据，也不是独立外部基准。",
             "每组保持任务说明、假设和候选含义不变，只改变前提，使正确关系依次为支持、矛盾、无法判断。三种材料都答对才算整组通过。", "",
             "| 模型 | 单题答对 | 三种材料全部答对的组 | 原始 NLL |", "|---|---:|---:|---:|"]
    for name, model in experiment["models"].items():
        lines.append(f"| {name} | {model['metrics']['correct']}/18 | {sum(g['all_three_correct'] for g in model['groups'].values())}/6 | {model['metrics']['nll']:.4f} |")
    lines += ["", "## 逐组决策", "", "下表每格按“支持材料 / 矛盾材料 / 信息不足材料”的顺序展示模型决策。", "",
              "| 关系 | " + " | ".join(experiment["models"]) + " |", "|" + "---|"*(1+len(experiment["models"]))]
    for group, *_ in TRIPLES:
        lines.append(f"| {group} | " + " | ".join(" / ".join(r["choice"] for r in model["details"] if r["group"] == group)
                                                   for model in experiment["models"].values()) + " |")
    lines += ["", "## 全部输入与标签依据", ""]
    for group, hypothesis, *premises in TRIPLES:
        lines += [f"### {group}", "", f"固定假设：{hypothesis}", ""]
        lines += [f"- {label}：{premise}" for label, premise in zip(("支持", "矛盾", "无法判断"), premises)]
        lines += [""]
    lines += ["这些样例可用于定位具体关系判断失败，不能当作所有语义任务的代表。概率变化与硬决策变化应分别看；即使没有正确翻转，也不能据此断言模型完全不读取材料。", "",
              "完整概率、权重指纹和各组结果见[机器可读记录](" + os.path.relpath(root / args.evidence, (root / args.report).parent) + ")。", ""]
    (root / args.report).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
