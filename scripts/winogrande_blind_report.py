"""Verify and report the frozen WinoGrande zero-shot transfer evaluation."""
import argparse
import json
from pathlib import Path
from statistics import mean

from decision_model.core import digest, load_rows, metrics, write_json
from semantic_scale_report import paired_diagnostics


ORDER = ("initial", "seed42-one", "seed42-variable", "seed42-balanced", "seed42-grouped",
         "seed43-one", "seed43-variable", "seed43-balanced", "seed43-grouped")
TITLES = {"initial": "公共起点", "seed42-one": "seed42 一轮", "seed42-variable": "seed42 两轮自然波动",
          "seed42-balanced": "seed42 两轮均衡打散", "seed42-grouped": "seed42 两轮均衡成组",
          "seed43-one": "seed43 一轮", "seed43-variable": "seed43 两轮自然波动",
          "seed43-balanced": "seed43 两轮均衡打散", "seed43-grouped": "seed43 两轮均衡成组"}
COMPARISONS = (
    ("seed42-variable", "seed42-balanced", "seed42 标签均衡"),
    ("seed43-variable", "seed43-balanced", "seed43 标签均衡"),
    ("seed42-one", "seed42-balanced", "seed42 第二轮"),
    ("seed43-one", "seed43-balanced", "seed43 第二轮"),
    ("initial", "seed42-balanced", "seed42 均衡训练相对起点"),
    ("initial", "seed43-balanced", "seed43 均衡训练相对起点"),
    ("seed42-balanced", "seed42-grouped", "seed42 前提成组"),
    ("seed43-balanced", "seed43-grouped", "seed43 前提成组"),
)


def read(path):
    return json.loads(path.read_text())


def validated(study):
    status, protocol = read(study / "status.json"), read(study / "protocol.json")
    if status["state"] != "complete" or status["models"] != list(ORDER):
        raise ValueError("All predeclared blind models must finish")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen blind input changed: " + name)
    rows = load_rows(study / "cases.jsonl")
    if digest((study / "cases.jsonl").read_bytes()) != protocol["dataset_sha256"]:
        raise ValueError("Blind dataset identity changed")
    models = {}
    declared = {item["name"]: item for item in protocol["checkpoints"]}
    for name in ORDER:
        value = read(study / "results" / (name + ".json"))
        expected = declared[name]
        if (value["name"] != name or value["dataset_sha256"] != protocol["dataset_sha256"] or
                value["weights_sha256"] != expected["weights_sha256"] or
                value["model_sha256"] != expected["model_sha256"] or value["rows"] != len(rows)):
            raise ValueError("Blind result identity mismatch: " + name)
        by_id = {point["id"]: point for point in value["predictions"]}
        if set(by_id) != {row["id"] for row in rows}:
            raise ValueError("Blind prediction IDs differ: " + name)
        labels, probabilities = [], []
        for row in rows:
            point = by_id[row["id"]]
            choices = [choice["id"] for choice in row["request"]["choices"]]
            if point["label"] != row["label"] or set(point["probabilities"]) != set(choices):
                raise ValueError("Blind prediction contract mismatch: " + name)
            labels.append(choices.index(row["label"]))
            probabilities.append([point["probabilities"][choice] for choice in choices])
        recalculated = metrics(labels, probabilities)
        for key in ("n", "correct", "accuracy", "nll", "brier", "ece_10"):
            if abs(recalculated[key] - value["raw"][key]) > 1e-9:
                raise ValueError("Blind aggregate mismatch: " + name + ":" + key)
        value["predicted_option1_fraction"] = sum(
            max(point["probabilities"], key=point["probabilities"].get) == "option1"
            for point in value["predictions"]) / len(rows)
        models[name] = value
    paired = {}
    for before, after, title in COMPARISONS:
        paired[title] = paired_diagnostics(rows, models[before]["predictions"], models[after]["predictions"])["winogrande-1.1"]
    return protocol, rows, models, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/winogrande-blind-v1")
    parser.add_argument("--report", default="docs/WINOGRANDE_BLIND_RESULTS.zh-CN.md")
    parser.add_argument("--evidence", default="docs/evidence/winogrande-blind-v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, rows, models, paired = validated(root / args.study)
    truth_option1 = sum(row["label"] == "option1" for row in rows) / len(rows)
    balance_deltas = [paired[f"seed{seed} 标签均衡"]["accuracy_change"] for seed in (42, 43)]
    epoch_deltas = [paired[f"seed{seed} 第二轮"]["accuracy_change"] for seed in (42, 43)]
    transfer_deltas = [paired[f"seed{seed} 均衡训练相对起点"]["accuracy_change"] for seed in (42, 43)]
    balance_nll_deltas = [models[f"seed{seed}-balanced"]["raw"]["nll"] -
                          models[f"seed{seed}-variable"]["raw"]["nll"] for seed in (42, 43)]
    evidence = {"protocol": protocol, "truth_option1_fraction": truth_option1,
                "models": {name: {key: value[key] for key in ("weights_sha256", "raw", "inherited_temperature",
                                                                  "inherited_temperature_metrics", "predicted_option1_fraction",
                                                                  "encoder_work")} for name, value in models.items()},
                "paired": paired}
    write_json(root / args.evidence, evidence)
    balance_direction = "同向提高" if all(delta > 0 for delta in balance_deltas) else "没有在两个种子中同向提高"
    epoch_direction = "点估计都略高于" if all(delta > 0 for delta in epoch_deltas) else "没有都高于"
    transfer_direction = "点估计都略高于" if all(delta > 0 for delta in transfer_deltas) else "没有都高于"
    lines = ["# WinoGrande 冻结盲测：训练收益能否迁移？", "", "## 结论", "",
             f"在看分数前冻结的 384 条 WinoGrande 1.1 子集上，批内标签均衡{balance_direction}自然波动终点；"
             f"两个种子的描述性平均变化为 {100*mean(balance_deltas):+.1f} 个百分点。",
             f"第二轮训练{epoch_direction}一轮终点，描述性平均变化为 {100*mean(epoch_deltas):+.1f} 个百分点。",
             f"两个均衡打散终点{transfer_direction}公共起点，描述性平均变化为 {100*mean(transfer_deltas):+.1f} 个百分点。", "",
             "所有上述准确率配对区间都包含 0，九个终点的准确率仅为 49.0%–52.1%。本轮没有证据表明 SNLI 上的训练收益已经迁移为 WinoGrande 常识指代能力。",
             f"均衡相对自然波动的原始 NLL 在 seed42/43 分别变化 {balance_nll_deltas[0]:+.4f}、{balance_nll_deltas[1]:+.4f}；准确率持平不代表概率质量改善。", "",
             "这些结论只描述该冻结子集。WinoGrande 没有进入项目微调，但公共底座可能在预训练时见过相关文本，无法排除。", "",
             "## 九个预先登记的终点", "",
             "| 模型 | 正确 | 准确率 | 原始 NLL | Brier | ECE | 预测 option1 | 继承温度 NLL |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ORDER:
        value, raw = models[name], models[name]["raw"]
        inherited = value["inherited_temperature_metrics"]
        lines.append(f"| {TITLES[name]} | {raw['correct']}/{raw['n']} | {raw['accuracy']:.1%} | {raw['nll']:.4f} | "
                     f"{raw['brier']:.4f} | {raw['ece_10']:.4f} | {value['predicted_option1_fraction']:.1%} | {inherited['nll']:.4f} |")
    lines += ["", f"冻结子集真实 option1 比例为 {truth_option1:.1%}。原始概率是主指标；“继承温度”来自旧任务验证集，WinoGrande 标签未参与拟合。", "",
              "## 配对变化", "", "| 对照（后者减前者） | 准确率变化 | 修正 / 引入错误 | 逐案例重采样 95% 区间 |",
              "|---|---:|---:|---:|"]
    for _, _, title in COMPARISONS:
        value = paired[title]; low, high = value["paired_group_bootstrap_95_percentile"]
        lines.append(f"| {title} | {100*value['accuracy_change']:+.1f} 个百分点 | {value['fixed']} / {value['regressed']} | "
                     f"[{100*low:+.1f}, {100*high:+.1f}] |")
    lines += ["", "区间仅表示该固定 384 条案例上的条件不确定性，不包含模型种子、数据集选择或多重比较不确定性。", "",
              "## 冻结方式与边界", "",
              "- 子集按 `SHA-256(qID, sentence, option1, option2)` 选择，答案不进入抽样；九个 checkpoint 在首次推理前全部登记。",
              "- 使用官方 1.1 归档及固定 SHA-256；开发集 1,267 条中选择 384 条。没有在 WinoGrande 上训练、调参、拟合温度或选择赢家。",
              "- 与项目微调输入的精确重合为 0；语义近重复和公共底座预训练污染无法完全排除。",
              "- 官方仓库只写明数据为 CC-BY，未注明版本；发布包不包含原始文本，只保留下载、校验、转换代码和聚合证据。",
              "- 所有模型预测 option1 的比例为 65.4%–77.9%，而真实比例为 50.3%。独立候选编码保证换序等变，因此这里反映的是对官方 option1 内容分布的偏好，不是候选 ID 被模型读取。",
              "- 这是二选一英文代词指代/常识任务，不能代表所有新任务。", "",
              f"机器可读记录：[evidence/{Path(args.evidence).name}](evidence/{Path(args.evidence).name})。", ""]
    (root / args.report).write_text("\n".join(lines))
    print(json.dumps({"balance_deltas": balance_deltas, "epoch_deltas": epoch_deltas,
                      "transfer_deltas": transfer_deltas, "balance_nll_deltas": balance_nll_deltas}, ensure_ascii=False))


if __name__ == "__main__":
    main()
