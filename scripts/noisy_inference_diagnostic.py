"""Post-study CPU diagnostic: clean probabilities vs averaged noisy probabilities."""
import argparse
import json
from pathlib import Path

import torch

from decision_model.core import digest, metrics, write_json
from gradient_training_report import NAMES, load_study, read


def reconstruct(predictions, temperature):
    """Positive temperature-scaled probabilities determine logits up to a shift."""
    rows = []
    for prediction in predictions:
        ids = list(prediction["probabilities"])
        values = torch.tensor(list(prediction["probabilities"].values()), dtype=torch.float64)
        if not torch.isfinite(values).all() or (values <= 0).any() or abs(values.sum().item()-1) > 1e-5:
            raise ValueError("Cannot recover logits from nonpositive/invalid stored probabilities")
        rows.append((temperature*values.log(), ids.index(prediction["label"])))
    return rows


def grouped(labels, probabilities, sources):
    return {source: metrics([y for y, s in zip(labels, sources) if s == source],
                            [p for p, s in zip(probabilities, sources) if s == source]) for source in sorted(set(sources))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/gradient-training-v1")
    parser.add_argument("--data", default="data/public-multitask-v2/cases.jsonl")
    parser.add_argument("--output", default="runs/noisy-inference-v1")
    parser.add_argument("--samples", type=int, default=8192)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("Use at least two samples")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    study, output = root / args.study, root / args.output
    if output.exists():
        raise ValueError("Use a new output directory")
    protocol, initial, _, arms = load_study(study)
    raw = (root / args.data).read_bytes()
    if digest(raw) != protocol["dataset_sha256"]:
        raise ValueError("Dataset mismatch")
    rows = {r["id"]: r for r in map(json.loads, raw.splitlines())}
    cases = {"initial": {"predictions": initial["predictions"], "calibration": initial["calibration"], "evaluation": initial["evaluation"]}}
    cases.update({arm: dict(value, predictions=read(study / arm / "test-predictions.json")) for arm, value in arms.items()})
    sigma = protocol["base_config"]["noise_sigma"]
    result = {"kind": "post-hoc probability diagnostic; not part of frozen primary model comparison",
              "samples_per_case_per_seed": args.samples, "seeds": [173, 419], "sigma": sigma,
              "dataset_sha256": protocol["dataset_sha256"], "script_sha256": digest(Path(__file__).read_bytes()),
              "method": "Recover logits up to a constant using T*log(saved calibrated p); verify raw NLL; average softmax of Gaussian-perturbed logits on CPU",
              "limitations": ["No checkpoint selection or additional temperature fit",
                              "Finite Monte Carlo can flip nearly tied decisions; this is not an accuracy-improvement method",
                              "Averaged probabilities and expected training loss are different objects"], "models": {}}
    reference_ids = [row["id"] for row in initial["predictions"]]
    for name, case in cases.items():
        predictions = case["predictions"]
        if [row["id"] for row in predictions] != reference_ids:
            raise ValueError("Prediction order mismatch")
        reconstructed = reconstruct(predictions, case["calibration"]["temperature"])
        labels = [label for _, label in reconstructed]
        sources = [rows[p["id"]]["source"] for p in predictions]
        clean = [logits.softmax(-1).tolist() for logits, _ in reconstructed]
        clean_metrics = grouped(labels, clean, sources)
        for source, values in clean_metrics.items():
            expected = case["evaluation"]["test"]["raw"]["by_source"][source]
            if abs(values["nll"] - expected["nll"]) > 1e-5 or values["correct"] != expected["correct"]:
                raise ValueError("Recovered logits do not reproduce saved raw metrics")
        noisy = {}
        for seed in result["seeds"]:
            generator = torch.Generator().manual_seed(seed)
            probabilities = [(logits + sigma*torch.randn(args.samples, len(logits), generator=generator, dtype=torch.float64)).softmax(-1).mean(0).tolist()
                             for logits, _ in reconstructed]
            noisy[str(seed)] = grouped(labels, probabilities, sources)
        result["models"][name] = {"clean": clean_metrics, "averaged_noisy": noisy}
    output.mkdir(parents=True)
    write_json(output / "result.json", result)
    write_json(root / "docs/evidence/noisy-inference-v1.json", result)
    lines = ["# 补充观察：无噪声概率与平均带噪概率", "",
             "这是在三组训练完成后增加的 CPU 概率诊断。主实验仍使用无噪声推理，不据此重新挑选权重或拟合温度。", "",
             "从保存的校准概率按 `T × log(p)` 恢复分数（差一个无关常数），先核对恢复后的原始 NLL 与准确率，再加入标准差 0.4 的高斯噪声，对 softmax 概率求平均。",
             f"每条请求、每个种子使用 {args.samples:,} 次采样；两次独立种子用于观察采样波动，各模型采用同一噪声序列。以下都是未加温度后处理的概率。", "",
             "| 任务 | 权重 | 无噪声 NLL | 平均带噪 NLL：种子 173 | 平均带噪 NLL：种子 419 |", "|---|---|---:|---:|---:|"]
    for source in sorted(set(sources)):
        for name, model in result["models"].items():
            lines.append(f"| {NAMES[source]} | {name} | {model['clean'][source]['nll']:.4f} | {model['averaged_noisy']['173'][source]['nll']:.4f} | {model['averaged_noisy']['419'][source]['nll']:.4f} |")
    lines += ["", "NLL 越低越好。概率平均不等于训练中的损失平均；它不会自动解决语义错误或证明校准。有限采样还可能改变近乎平局的硬决策，因此不把偶然准确率波动作为改进证据。", "",
              "完整 Brier/ECE 与覆盖率保存在[evidence/noisy-inference-v1.json](evidence/noisy-inference-v1.json)。这是已检查诊断集上的后续分析，不是新盲测。", ""]
    (root / "docs/NOISY_INFERENCE.zh-CN.md").write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
