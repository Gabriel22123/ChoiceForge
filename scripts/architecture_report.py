"""Render and preserve the completed joint-vs-independent study evidence."""
import argparse
import json
from pathlib import Path

from decision_model.core import digest, write_json

ARMS = ("J-joint", "D-independent")
NAMES = {"bandit": "局部代码策略", "clinc": "意图候选排序",
         "goemotions": "未参与微调的情绪任务", "json-schema": "Schema 规则"}
CONTROL_KEYS = ("initial_trainable_sha256", "selected_ids_sha256", "training_plan_sha256",
                "updates", "forward_sequences", "forward_batches", "training_views")


def read(path):
    return json.loads(path.read_text())


def paired_changes(study, rows):
    """Descriptive paired counts, not a post-hoc significance test."""
    predictions = [read(study / arm / "test-predictions.json") for arm in ARMS]
    maps = [{row["id"]: row for row in arm} for arm in predictions]
    if set(maps[0]) != set(maps[1]) or any(len(m) != len(p) for m, p in zip(maps, predictions)):
        raise ValueError("Paired prediction IDs differ or repeat")
    result = {}
    for identity in maps[0]:
        data = rows[identity]
        pair = [m[identity] for m in maps]
        if any(row["label"] != data["label"] for row in pair):
            raise ValueError("Paired ground truth mismatch")
        if any(set(row["probabilities"]) != {c["id"] for c in data["request"]["choices"]} for row in pair):
            raise ValueError("Paired candidates mismatch")
        correct = [max(row["probabilities"], key=row["probabilities"].get) == row["label"] for row in pair]
        key = {(True, True): "both_correct", (False, False): "both_wrong",
               (False, True): "fixed", (True, False): "regressed"}[tuple(correct)]
        counts = result.setdefault(data["source"], dict.fromkeys(("both_correct", "both_wrong", "fixed", "regressed"), 0))
        counts[key] += 1
    return result


def load_study(study):
    if read(study / "status.json")["state"] != "complete":
        raise ValueError("Wait for both training, audit and benchmark stages")
    protocol = read(study / "protocol.json")
    for name, expected in read(study / "protocol-checksums.json").items():
        if digest((study / name).read_bytes()) != expected:
            raise ValueError("Frozen configuration changed")
    arms = {}
    controls = audit_ids = benchmark_ids = None
    for arm in ARMS:
        directory = study / arm
        run = read(directory / "run.json")
        current = {k: run[k] for k in CONTROL_KEYS}
        if controls is None:
            controls = current
        if current != controls:
            raise ValueError("Unmatched experiment controls")
        if digest(run["selected_ids"]) != current["selected_ids_sha256"]:
            raise ValueError("Sample selection changed")
        if digest((directory / "training-plan.jsonl").read_bytes()) != current["training_plan_sha256"]:
            raise ValueError("Training plan changed")
        if run["dataset_sha256"] != protocol["dataset_sha256"]:
            raise ValueError("Dataset mismatch")
        if run["config_sha256"] != digest((study / (arm + ".json")).read_bytes()):
            raise ValueError("Configuration mismatch")
        if read(directory / "model.json") != read(study / (arm + ".json")):
            raise ValueError("Saved model configuration mismatch")
        if any(value != protocol["source_files"]["src/decision_model/" + name]
               for name, value in run["source_files"].items()):
            raise ValueError("Training source mismatch")
        for name, expected in read(directory / "checksums.json").items():
            if digest((directory / name).read_bytes()) != expected:
                raise ValueError("Checkpoint checksum mismatch")
        audit, benchmark = read(directory / "audit.json"), read(directory / "benchmark.json")
        current_ids = [row["id"] for row in audit["details"]]
        measured_ids = [(row["id"], row["repeat"]) for row in benchmark["timings"]]
        if audit_ids is None:
            audit_ids, benchmark_ids = current_ids, measured_ids
        if current_ids != audit_ids or measured_ids != benchmark_ids:
            raise ValueError("Audit or benchmark requests differ")
        if benchmark["checkpoint_weight_sha256"] != digest((directory / "decision.safetensors").read_bytes()):
            raise ValueError("Benchmark checkpoint mismatch")
        arms[arm] = {"run": run, "evaluation": read(directory / "evaluation.json"),
                     "audit": audit, "benchmark": benchmark, "calibration": read(directory / "calibration.json"),
                     "weight_sha256": benchmark["checkpoint_weight_sha256"]}
    return protocol, controls, arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/architecture-study-v1")
    parser.add_argument("--data", default="data/public-multitask-v1/cases.jsonl")
    parser.add_argument("--output", default="docs/ARCHITECTURE_STUDY.zh-CN.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root / args.study
    protocol, controls, arms = load_study(study)
    raw_data = (root / args.data).read_bytes()
    if digest(raw_data) != protocol["dataset_sha256"]:
        raise ValueError("Report dataset mismatch")
    rows = [json.loads(line) for line in raw_data.splitlines() if line.strip()]
    paired = paired_changes(study, {row["id"]: row for row in rows})
    for source, counts in paired.items():
        expected = (counts["both_correct"] + counts["regressed"],
                    counts["both_correct"] + counts["fixed"])
        for arm, correct in zip(ARMS, expected):
            metric = arms[arm]["evaluation"]["test"]["raw"]["by_source"][source]
            if metric["correct"] != correct or metric["n"] != sum(counts.values()):
                raise ValueError("Paired predictions disagree with accuracy report")
    sources = sorted(arms[ARMS[0]]["evaluation"]["test"]["raw"]["by_source"])
    joint, independent = (arms[arm] for arm in ARMS)
    lines = ["# 候选编码架构对照：联合编码与独立评分", "",
             "本轮只改变候选编码方式，比较同一公开 EuroBERT-2.1B 底座上的两组新训练权重。", "",
             "- **联合编码**：任务、材料和所有候选一起编码，在各候选标记处评分。",
             "- **独立评分**：每个候选分别与任务、材料一起编码，再对分数做 softmax。评分头和可训练参数集合相同。",
             "- 推理时独立评分每次编码一个候选，避免其他候选改变批次长度或填充。没有候选排序、重复换序投票或输出缓存。", "",
             "## 实验控制与边界", "",
             "两组均从相同可训练参数初始化出发，使用同样的 384 条训练、96 条校准、192 条诊断测试样本；2 轮、96 次更新、两份换序视图、相同 CE 目标，不加 Brier 或 JS。初始化、样本 ID、逐批换序计划及请求视图次数均通过核对。",
             "两种模式先执行同一完整请求长度检查，避免独立编码额外接纳样本。", "",
             "**这是相同样本和更新次数的对照，不是相同计算预算的对照。** 独立评分重复编码材料，编码序列数、批次形状、dropout 抽样及耗时会变化。", "",
             "单随机种子、小样本、已用于开发观察的诊断集；本轮不能证明通用零样本优势或生产可用性。", "",
             "## 1. 按来源判断准确率", "",
             "| 任务 | 联合编码 | 独立评分 |", "|---|---:|---:|"]
    for source in sources:
        values = [arms[arm]["evaluation"]["test"]["raw"]["by_source"][source] for arm in ARMS]
        lines.append(f"| {NAMES[source]} | " + " | ".join(f"{m['correct']}/{m['n']}（{m['accuracy']:.1%}）" for m in values) + " |")
    lines += ["", "意图任务是抽样候选排序，情绪任务是七类单标签子集。情绪任务未进入微调或温度拟合，但不能排除公开数据出现在底座预训练中。", "",
              "### 同一条请求上的得失", "",
              "| 任务 | 两者均正确 | 独立版修正 | 独立版退步 | 两者均错误 |", "|---|---:|---:|---:|---:|"]
    for source in sources:
        counts = paired[source]
        lines.append(f"| {NAMES[source]} | " + " | ".join(str(counts[key]) for key in ("both_correct", "fixed", "regressed", "both_wrong")) + " |")
    lines += ["", "“修正”表示联合版错误、独立版正确；“退步”反之。这里只描述已观察的固定测试请求，不作事后显著性或泛化保证。", "",
              "## 2. 同一批 24 条请求的换序与重载检查", "",
              "| 模式 | 所有检查顺序一致 | 所有检查顺序均正确 | 最大换序概率差 | 最大重载概率差 |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        audit = arms[arm]["audit"]
        delta = max(r["max_order_probability_difference"] for r in audit["details"])
        lines.append(f"| {arm} | {audit['stable_all_orders']}/{audit['cases']} | {audit['correct_all_orders']}/{audit['cases']} | {delta:.3g} | {audit['max_reload_probability_difference']:.3g} |")
    lines += ["", "| 任务 | 模式 | 检查数 | 始终一致 | 始终正确 |", "|---|---|---:|---:|---:|"]
    for source in sources:
        for arm in ARMS:
            rows = [r for r in arms[arm]["audit"]["details"] if r["source"] == source]
            lines.append(f"| {NAMES[source]} | {arm} | {len(rows)} | {sum(r['stable'] for r in rows)} | {sum(r['correct_all_orders'] for r in rows)} |")
    lines += ["", "独立评分的结构性质是每个候选的原始分数不依赖列表位置；概率仍存在浮点归约误差。完全相同分数时，现有接口按首次出现的位置打破平局，因此不声称所有输入的硬决策都绝对不变。", "",
              "## 3. 概率质量", "",
              "| 任务 | 模式 | 原始 NLL | 校准 NLL | 原始 Brier | 校准 Brier | 原始 ECE | 校准 ECE |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for source in sources:
        for arm in ARMS:
            evaluation = arms[arm]["evaluation"]["test"]
            values = [evaluation[mode]["by_source"][source][key]
                      for key in ("nll", "brier", "ece_10") for mode in ("raw", "temperature_scaled")]
            lines.append(f"| {NAMES[source]} | {arm} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "三种概率指标越低越好；ECE 使用固定 10 个分箱，小样本易波动。温度仅由验证集拟合。全部阈值覆盖率保存在机器可读结果中。", "",
              "## 4. 训练计算量与推理成本", "",
              "| 模式 | 训练分钟 | 训练请求视图 | 编码序列 | 编码调用 | 填充后 token 槽位 |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        run = arms[arm]["run"]
        work = run["encoder_work"]
        lines.append(f"| {arm} | {run['training_seconds']/60:.1f} | {run['forward_sequences']:,} | {work['encoder_sequences']:,} | {work['encoder_calls']:,} | {work['padded_token_slots']:,} |")
    lines += ["", "编码计数记录模型前向输入，不包含梯度检查点在反向时的重算；token 槽位和 attention_token_pairs 是输入形状代理指标，不是实测 FLOPs。", "",
              "### 已加载模型的端到端请求延迟", "",
              "同一设备上顺序测试同样 24 条请求：先预热一遍，再测三遍。计入编码、推理和输出组装，排除加载权重。无推理输出缓存。", "",
              "| 模式 | 测量请求数 | 中位数（毫秒） | P95（毫秒） |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        summary = arms[arm]["benchmark"]["summary"]
        lines.append(f"| {arm} | {summary['requests']} | {1000*summary['median_seconds']:.1f} | {1000*summary['p95_seconds']:.1f} |")
    lines += ["", "| 候选数 | 模式 | 请求数 | 中位数（毫秒） |", "|---|---|---:|---:|"]
    for count in joint["benchmark"]["by_candidate_count"]:
        for arm in ARMS:
            summary = arms[arm]["benchmark"]["by_candidate_count"][count]
            lines.append(f"| {count} | {arm} | {summary['requests']} | {1000*summary['median_seconds']:.1f} |")
    ratio = independent["benchmark"]["summary"]["median_seconds"] / joint["benchmark"]["summary"]["median_seconds"]
    lines += ["", f"本次混合请求中位延迟比为 {ratio:.2f} 倍。重复小样本测量不代表生产分布或 SLA；候选数与材料长度会改变成本。独立模式推理逐候选编码是当前实现选择，未来可以研究批处理，但需重新验证数值稳定性和延迟。",
              "测量在同一台正在使用的本机顺序执行，并非隔离、随机交错的硬件基准；不同候选数的请求也有不同任务和长度，不能只按该表推断候选数量的独立因果影响。", "",
              "## 使用限制", "",
              "独立评分看不到其他候选的内容。候选应具有自包含含义；“以上都不是”“A 和 B 都成立”等依赖其他选项的描述，不能假设与联合编码语义等价。它可能降低跨候选比较能力，即使输出更稳定。",
              "结构稳定不代表语义正确，也不代表概率已校准。当前接口仍返回 requires_review=true。", "",
              "## 复现与记录", "",
              "```bash", "python scripts/run_architecture_study.py --device mps", "python scripts/architecture_report.py", "```", "",
              "实验要求新输出目录。报告命令只重建派生文档与证据。两组权重分别位于 runs/architecture-study-v1/J-joint 和 D-independent。",
              "完整数值与来源：[architecture-study-v1.json](evidence/architecture-study-v1.json)。", "",
              f"- 数据 SHA-256：`{protocol['dataset_sha256']}`。",
              f"- 初始可训练参数 SHA-256：`{controls['initial_trainable_sha256']}`。",
              f"- 逐批候选顺序 SHA-256：`{controls['training_plan_sha256']}`。", ""]
    output = root / args.output
    (output.parent / "evidence").mkdir(parents=True, exist_ok=True)
    write_json(output.parent / "evidence/architecture-study-v1.json",
               {"protocol": protocol, "matched_controls": controls, "arms": arms, "paired_changes": paired})
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
