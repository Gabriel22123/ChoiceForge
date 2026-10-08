"""Report all frozen study arms; refuse incomplete or mismatched comparisons."""
import argparse
import json
import statistics
from pathlib import Path

from decision_model.core import digest, write_json

ARMS = ("A-ce", "B-proper", "C-consistency")
NAMES = {"bandit": "局部代码策略", "clinc": "意图候选排序",
         "goemotions": "未参与微调的情绪任务", "json-schema": "Schema 规则"}
CONTROL_KEYS = ("initial_trainable_sha256", "selected_ids_sha256", "training_plan_sha256",
                "updates", "forward_sequences", "forward_batches", "training_views")


def read(path):
    return json.loads(path.read_text())


def load_study(study):
    if read(study / "status.json")["state"] != "complete":
        raise ValueError("All three arms and audits must finish before reporting")
    protocol = read(study / "protocol.json")
    for filename, expected in read(study / "protocol-checksums.json").items():
        if digest((study / filename).read_bytes()) != expected:
            raise ValueError("Frozen protocol changed: " + filename)
    arms = {}
    reference = audit_ids = None
    for name in ARMS:
        directory = study / name
        run = read(directory / "run.json")
        if run["config_sha256"] != digest((study / (name + ".json")).read_bytes()):
            raise ValueError("Configuration hash differs: " + name)
        if read(directory / "model.json") != read(study / (name + ".json")):
            raise ValueError("Saved model configuration differs: " + name)
        if run["initialization"] != "public_pretrained_base_fresh_adapter_and_head":
            raise ValueError("Comparison requires fresh initialization")
        if digest(run["selected_ids"]) != run["selected_ids_sha256"]:
            raise ValueError("Selected ID metadata changed")
        if digest((directory / "training-plan.jsonl").read_bytes()) != run["training_plan_sha256"]:
            raise ValueError("Training plan changed")
        controls = {key: run[key] for key in CONTROL_KEYS}
        if reference is None:
            reference = controls
        if controls != reference or run["source_files"] != protocol["source_files"]:
            raise ValueError("Unmatched controls or sources: " + name)
        if run["dataset_sha256"] != protocol["dataset_sha256"]:
            raise ValueError("Dataset changed: " + name)
        for filename, expected in read(directory / "checksums.json").items():
            if digest((directory / filename).read_bytes()) != expected:
                raise ValueError("Checkpoint artifact changed: " + name + "/" + filename)
        audit = read(directory / "audit.json")
        ids = [row["id"] for row in audit["details"]]
        if audit_ids is None:
            audit_ids = ids
        if ids != audit_ids:
            raise ValueError("Audit examples differ")
        predictions = read(directory / "test-predictions.json")
        if [p["id"] for p in predictions] != run["selected_ids"]["test"]:
            raise ValueError("Prediction IDs differ from selected IDs")
        arms[name] = {"run": run, "evaluation": read(directory / "evaluation.json"),
                      "calibration": read(directory / "calibration.json"), "audit": audit,
                      "predictions": predictions}
    return protocol, reference, arms


def paired_counts(first, second):
    if [r["id"] for r in first] != [r["id"] for r in second]:
        raise ValueError("Pairing requires identical examples and order")
    counts = {"both_correct": 0, "first_only_correct": 0,
              "second_only_correct": 0, "both_wrong": 0}
    for a, b in zip(first, second):
        if a["label"] != b["label"]:
            raise ValueError("Paired labels differ")
        correct_a = max(a["probabilities"], key=a["probabilities"].get) == a["label"]
        correct_b = max(b["probabilities"], key=b["probabilities"].get) == b["label"]
        key = ("both_correct" if correct_a and correct_b else "first_only_correct" if correct_a
               else "second_only_correct" if correct_b else "both_wrong")
        counts[key] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="runs/calibration-study-v1")
    parser.add_argument("--gradient", default="runs/gradient-estimators-v1")
    parser.add_argument("--noise", default="runs/noise-optimum-v1")
    parser.add_argument("--normalization", default="runs/normalization-v1")
    parser.add_argument("--projected", default="runs/gradient-projected-v1")
    parser.add_argument("--output", default="docs/STUDY.zh-CN.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study, gradient = root / args.study, root / args.gradient
    protocol, controls, arms = load_study(study)
    gradient_result = read(gradient / "results.json")
    noise_result = read(root / args.noise / "results.json")
    normalization_result = read(root / args.normalization / "results.json")
    projected_result = read(root / args.projected / "results.json")
    projected_ratios = [r["estimators"]["score_function_loo_projected"]["variance_trace"] /
                        r["estimators"]["pathwise"]["variance_trace"] for r in projected_result["results"]]
    ratios = [r["variance_ratio_score_over_path"] for r in gradient_result["results"]]
    first_run = arms[ARMS[0]]["run"]
    sources = sorted(arms[ARMS[0]]["evaluation"]["test"]["raw"]["by_source"])
    evidence = {"protocol": protocol, "matched_controls": controls, "arms": {}, "paired": {},
                "gradient_results_sha256": digest((gradient / "results.json").read_bytes()),
                "gradient_variance_ratio": {"minimum": min(ratios), "median": statistics.median(ratios), "maximum": max(ratios)}}
    for name, result in arms.items():
        evidence["arms"][name] = {"training_seconds": result["run"]["training_seconds"],
                                  "trainable_parameters": result["run"]["trainable_parameters"],
                                  "calibration": result["calibration"], "evaluation": result["evaluation"],
                                  "audit": result["audit"],
                                  "artifact_hashes": {file: digest((study / name / file).read_bytes()) for file in
                                                     ("decision.safetensors", "run.json", "evaluation.json", "audit.json")}}
    lines = ["# 校准决策研究：第一轮受控实验", "",
             "本轮研究公开可描述的校准决策目标，比较独立实现的训练目标与梯度估计方法。不是任何私有训练方案复刻。", "",
             "## 实验约定", "",
             "| 组 | 概率训练目标 | 每批换序视图 | 一致性惩罚 |",
             "|---|---|---:|---:|", "| A | 交叉熵 CE | 2 | 0 |",
             "| B | CE + 0.5 Brier | 2 | 0 |", "| C | CE + 0.5 Brier | 2 | 0.05 JS |", "",
             f"三组均从公开 EuroBERT-2.1B 的同一初始适配器与评分头开始，使用相同的 384 条训练、96 条校准、192 条诊断测试数据。训练 2 轮、{controls['updates']} 次更新，每组 {controls['forward_sequences']} 个训练序列视图。",
             "初始化、数据 ID、逐批候选换序计划及前向次数均通过哈希/数值核对；配置和源码在运行前冻结，三组串行使用同一设备。", "",
             "**本轮仅一个随机种子、少量样本，测试样本已经在首轮试验中看过。下面是开发诊断结果，不能作为最终盲测、显著性结论或通用零样本能力证明。**", "",
             "## 1. 按任务看判断准确率", "",
             "| 任务 | 样本数 | 随机期望 | A：CE | B：加 Brier | C：再加一致性 |",
             "|---|---:|---:|---:|---:|---:|"]
    for source in sources:
        metrics = [arms[name]["evaluation"]["test"]["raw"]["by_source"][source] for name in ARMS]
        cells = [f"{m['correct']}/{m['n']}（{m['accuracy']:.1%}）" for m in metrics]
        lines.append(f"| {NAMES[source]} | {metrics[0]['n']} | {metrics[0]['uniform_random_accuracy']:.1%} | " + " | ".join(cells) + " |")
    lines += ["", "意图任务只在抽样 2/4/8 个候选中选择，不能等同于 CLINC150 原始 150 类分类成绩；情绪任务是七类单标签子集。情绪任务未进入微调或温度拟合，但不能排除底座预训练见过公开数据。", "",
              "温度缩放不改变最大概率候选，故原始与校准后的准确率相同。", "",
              "### 同一批样本上的成败变化", "",
              "| 比较（前组 → 后组） | 两组都对 | 前组对、后组错 | 前组错、后组对 | 两组都错 |",
              "|---|---:|---:|---:|---:|"]
    for a, b in zip(ARMS, ARMS[1:]):
        counts = paired_counts(arms[a]["predictions"], arms[b]["predictions"])
        evidence["paired"][f"{a}_to_{b}"] = counts
        lines.append(f"| {a} → {b} | " + " | ".join(str(v) for v in counts.values()) + " |")
    lines += ["", "成败变化只描述这 192 条混合任务样本，不作独立同分布假设或显著性检验。", "",
              "## 2. 概率质量：先看训练目标，再看后处理", "",
              "NLL 衡量给真实答案分配的概率；Brier 衡量整组概率与真实结果的差距；ECE 比较分箱后的信心与正确率。三者越低越好。小样本 ECE 容易波动，不能单凭一个较小数值断言已校准。", "",
              "| 任务 | 组 | 原始 NLL | 校准 NLL | 原始 Brier | 校准 Brier | 原始 ECE | 校准 ECE |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for source in sources:
        for name in ARMS:
            evaluation = arms[name]["evaluation"]["test"]
            raw, cal = (evaluation[k]["by_source"][source] for k in ("raw", "temperature_scaled"))
            values = [v for key in ("nll", "brier", "ece_10") for v in (raw[key], cal[key])]
            lines.append(f"| {NAMES[source]} | {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "| 组 | 拟合温度 | 验证集原始 NLL | 验证集校准 NLL | 训练分钟 |",
              "|---|---:|---:|---:|---:|"]
    for name, result in arms.items():
        c = result["calibration"]
        lines.append(f"| {name} | {c['temperature']:.4f} | {c['identity_validation_nll']:.4f} | {c['validation_nll']:.4f} | {result['run']['training_seconds']/60:.1f} |")
    lines += ["", "温度只在同一验证集上拟合。原始概率对比用于观察训练效果，校准后对比用于判断训练是否仍优于简单后处理。跨任务可靠性需要单独看，不能由验证集 NLL 降低推导。", "",
              "### 高信心覆盖：温度校准后、阈值固定为 0.9", "",
              "| 任务 | 组 | 达到阈值的样本数 | 覆盖率 | 其中正确率 |",
              "|---|---|---:|---:|---:|"]
    for source in sources:
        for name in ARMS:
            m = arms[name]["evaluation"]["test"]["temperature_scaled"]["by_source"][source]["selective"]["0.9"]
            accuracy = "无样本" if m["accuracy"] is None else f"{m['accuracy']:.1%}"
            lines.append(f"| {NAMES[source]} | {name} | {m['selected_count']} | {m['coverage']:.1%} | {accuracy} |")
    lines += ["", "这是固定阈值的描述性检查，不代表达到 90% 实际准确率，也不是可直接上线的自动决策阈值。0.5/0.7 阈值明细保存在随附机器可读结果中。", "",
              "## 3. 相同 24 条样本的候选换序检查", "",
              "每条检查全部（2–3 候选）或六种（更多候选）顺序，均按候选语义比较。", "",
              "| 组 | 所有顺序答案一致 | 所有顺序均正确 | 重载最大概率差 | 仅重命名 ID 的最大差 |",
              "|---|---:|---:|---:|---:|"]
    for name, result in arms.items():
        audit = result["audit"]
        lines.append(f"| {name} | {audit['stable_all_orders']}/{audit['cases']} | {audit['correct_all_orders']}/{audit['cases']} | {audit['max_reload_probability_difference']:.3g} | {audit['max_id_rename_probability_difference']:.3g} |")
    lines += ["", "| 任务 | 组 | 换序检查样本数 | 始终一致 | 始终正确 |",
              "|---|---|---:|---:|---:|"]
    for source in sources:
        for name in ARMS:
            details = [row for row in arms[name]["audit"]["details"] if row["source"] == source]
            lines.append(f"| {NAMES[source]} | {name} | {len(details)} | {sum(row['stable'] for row in details)} | {sum(row['correct_all_orders'] for row in details)} |")
    lines += ["", "判断一致不等于判断正确，故同时报告两项。ID 未输入模型，重命名应不影响概率；候选描述在序列中的位置仍会影响编码。", "",
              "## 4. 已知可微奖励下的梯度估计", "",
              "两种方法优化同一个带高斯噪声的 CE + 0.5 Brier 期望目标：一组采用带独立留一基线的策略梯度，一组通过采样过程直接求导。两组复用相同噪声样本，不做奖励标准差归一化。", "",
              "测试候选数 3/8/16、噪声 0.1/0.4、每组采样 4/16/64，以及犹豫和高信心选错两种初始状态，共 36 组。每组重复 2,048 次，另用 131,072 个独立样本估计参考梯度。", "",
              f"策略梯度与直接求导的方差比：最小 **{min(ratios):,.1f} 倍**、中位数 **{statistics.median(ratios):,.1f} 倍**、最大 **{max(ratios):,.1f} 倍**。本实验中直接求导的方差均较低。", "",
              "以下固定噪声 0.4、每组采样 16，展示各候选规模的绝对方差，避免只看比值。方差为所有 logit 维度方差之和。", "",
              "| 候选数 | 初始状态 | 策略梯度方差 | 直接求导方差 | 方差比 |",
              "|---|---|---:|---:|---:|"]
    for row in gradient_result["results"]:
        if row["sigma"] == .4 and row["samples"] == 16:
            estimates = row["estimators"]
            lines.append(f"| {row['candidates']} | {row['condition']} | {estimates['score_function_loo']['variance_trace']:.6g} | {estimates['pathwise']['variance_trace']:.6g} | {row['variance_ratio_score_over_path']:,.1f} |")
    lines += ["", f"补充检查：softmax 不受所有 logits 同幅平移影响，因此将策略梯度投影到分量和为零的子空间，去掉这部分无效方差。使用相同种子重做 36 组后，投影策略梯度与直接求导的方差比最小 {min(projected_ratios):,.1f}、中位数 {statistics.median(projected_ratios):,.1f}、最大 {max(projected_ratios):,.1f}。这只是一个基础控制，未穷尽策略梯度的降方差方法。", "",
              "**边界：**这是 logit 空间的合成实验，不是完整骨干模型的 RL 对照。采样数相同不等于计算成本相同；方差降低倍数不等于训练提速或准确率提升倍数。对一般问题，直接求导也没有普遍低方差保证。不可微、延迟或外部交互奖励仍可能需要强化学习。", "",
              "带噪声目标不等于三组模型使用的无噪声监督目标。不能把此结果拼接成“直接训练已经战胜 RLCD”。有限差分、软标签自动微分与 Monte Carlo 均值检查通过，只验证当前实现。", "",
              "## 5. 训练噪声与推理概率的差别", "",
              "另一个独立实验直接求解二分类总体交叉熵最优点：真实事件概率已知，两个 logits 各自加入独立高斯噪声。最优点约束扰动后概率的期望，并不要求去掉噪声后的概率等于真实概率。", "",
              "| 真实概率 | 每个 logit 的噪声标准差 | 最优点无噪声概率 | 扰动后平均概率 | 无噪声偏差（百分点） |",
              "|---|---:|---:|---:|---:|"]
    for row in noise_result["results"]:
        lines.append(f"| {row['true_probability']:.1%} | {row['sigma_per_logit']:.1f} | {row['clean_inference_probability']:.4%} | {row['noise_averaged_probability']:.4%} | {100*row['clean_probability_error']:+.4f} |")
    lines += ["", "使用 96 与 192 阶高斯求积交叉验证，各最优概率之差小于 1e-8。这是指定噪声模型与总体交叉熵下的反例，不是训练骨干模型的结果，也不用于推断任何外部实现。它提示：目标和推理概率的定义需要一致。", "",
              "## 6. 奖励标准化可能改变最优点", "",
              "在已知真实概率的二分类实验中，先求出带噪交叉熵的总体最优点，再测量不同更新方法在该点的平均梯度。两种标签分别计算组内损失与标准化，再按真实概率加权；没有奖励裁剪或额外 CE 锚点。", "",
              "取真实概率 0.9、每个 logit 噪声标准差 0.4、每组 16 个样本，独立重复 16,384 组。此时真实风险梯度为零，应是原目标的驻点。", "",
              "| 更新方法 | 平均梯度 | 均值标准误 |", "|---|---:|---:|"]
    representative = next(row for row in normalization_result["results"] if row["true_probability"] == .9
                          and row["sigma_per_logit"] == .4 and row["samples"] == 16)
    for name, measurement in representative["estimators"].items():
        lines.append(f"| {name} | {measurement['mean_gradient']:+.6f} | {measurement['standard_error']:.6f} |")
    lines += ["", "留一基线策略梯度、只减组均值的策略梯度与直接求导均接近零；组内再除以损失标准差后出现明显非零梯度。这里最小化损失，负梯度会使梯度下降继续提高第一个类别的分数，离开原目标最优点。", "",
              "所以，采用适当评分规则作奖励，不足以保证经过标准化的实际更新仍保持其校准最优点。这里只检验这个具体更新规则，不代表所有 RL 方法，也不等同于对外部实现的完整诊断。完整 18 组结果见随附 normalization-v1.json。", "",
              "## 复现与证据", "",
              "方法与控制说明：[RESEARCH.md](RESEARCH.md)。完整数值：[calibration-study-v1.json](evidence/calibration-study-v1.json)；36 组梯度结果：[gradient-estimators-v1.json](evidence/gradient-estimators-v1.json)。", "",
              "```bash", "python scripts/gradient_study.py --output runs/gradient-estimators-v1",
              "python scripts/gradient_study.py --projection-control --output runs/gradient-projected-v1",
              "python scripts/run_ablation.py --output runs/calibration-study-v1 --device mps",
              "python scripts/noise_optimum_study.py --output runs/noise-optimum-v1",
              "python scripts/normalization_study.py --output runs/normalization-v1",
              "python scripts/study_report.py", "```", "",
              "实验命令要求新的输出目录，避免覆盖已有记录。不同硬件或库版本可能产生数值差异。", "",
              f"- 数据 SHA-256：`{protocol['dataset_sha256']}`。",
              f"- 初始可训练参数 SHA-256：`{controls['initial_trainable_sha256']}`。",
              f"- 逐批换序计划 SHA-256：`{controls['training_plan_sha256']}`。",
              f"- 每组可训练参数：{first_run['trainable_parameters']:,}。", ""]
    output = root / args.output
    evidence_dir = output.parent / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    write_json(evidence_dir / "calibration-study-v1.json", evidence)
    write_json(evidence_dir / "gradient-estimators-v1.json", gradient_result)
    write_json(evidence_dir / "noise-optimum-v1.json", noise_result)
    write_json(evidence_dir / "normalization-v1.json", normalization_result)
    write_json(evidence_dir / "gradient-projected-v1.json", projected_result)
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
