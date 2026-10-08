"""Verify and report the fixed one-seed RLCD decomposition pilot."""
import argparse
import json
from pathlib import Path

from decision_model.core import digest, write_json


ARMS=("H-hard","S-soft","T-typed-rps","R-rlcd-loo")


def compact(metrics):
    return {key:metrics[key] for key in (
        "n","argmax_agreement","soft_cross_entropy","soft_brier","total_variation",
        "expected_top_label_mass")}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--study",default="runs/typed-decisions-rlcd-pilot-v1")
    parser.add_argument("--output",default="docs/evidence/typed-decisions-rlcd-pilot-v1.json")
    parser.add_argument("--report",default="docs/TYPED_DECISIONS_RLCD_PILOT.zh-CN.md")
    args=parser.parse_args();study=Path(args.study)
    if json.loads((study/"status.json").read_text()).get("state")!="complete":
        raise ValueError("RLCD pilot is not complete")
    protocol=json.loads((study/"protocol.json").read_text())
    for name,expected in json.loads((study/"protocol-checksums.json").read_text()).items():
        if name!="protocol-checksums.json" and digest((study/name).read_bytes())!=expected:
            raise ValueError("Frozen pilot input changed: "+name)
    controls=json.loads((study/"matched-controls.json").read_text())
    results={};objectives={};weights={}
    for arm in ARMS:
        run=json.loads((study/arm/"run.json").read_text())
        evaluation=json.loads((study/arm/"evaluation.json").read_text())
        checksums=json.loads((study/arm/"checksums.json").read_text())
        for name,expected in checksums.items():
            if digest((study/arm/name).read_bytes())!=expected:raise ValueError(f"{arm}/{name} changed")
        current={key:run[key] for key in controls}
        if current!=controls:raise ValueError("Matched controls differ for "+arm)
        test=evaluation["test"]["raw"]
        results[arm]={
            "validation":compact(evaluation["validation"]["raw"]),
            "test_all":compact(test),
            "test_seen":compact(test["by_evaluation_regime"]["seen_task_new_examples"]),
            "test_heldout":compact(test["by_evaluation_regime"]["held_out_task"]),
            "test_by_type":{kind:compact(value) for kind,value in test["by_decision_type"].items()},
        }
        objectives[arm]=run["objective"]
        weights[arm]=checksums["decision.safetensors"]
    def delta(treatment,control,scope,key):
        return results[treatment][scope][key]-results[control][scope][key]
    contrasts={name:{scope:{
        "argmax_agreement_delta":delta(treatment,control,scope,"argmax_agreement"),
        "soft_cross_entropy_delta":delta(treatment,control,scope,"soft_cross_entropy"),
        "soft_brier_delta":delta(treatment,control,scope,"soft_brier"),
    } for scope in ("validation","test_seen","test_heldout")}
        for name,control,treatment in (("soft_minus_hard","H-hard","S-soft"),
                                       ("typed_rps_minus_soft","S-soft","T-typed-rps"),
                                       ("rlcd_loo_minus_typed_rps","T-typed-rps","R-rlcd-loo"))}
    evidence={"status":"complete_pilot_no_method_advancement","protocol":protocol,
              "matched_controls":controls,"weights_sha256":weights,
              "objectives":objectives,"results":results,"contrasts":contrasts,
              "interpretation_rule":"Directional implementation evidence only; do not select or reject an arm from this one-seed 20-question test.",
              "next_gate":"Freeze all four workflow folds and seeds 42/43/44 only after checking numerical stability, gradient norms and target alignment here."}
    write_json(args.output,evidence)

    lines=["# Typed Decisions：RLCD 原理解构先导实验","",
           "本实验只检查四条训练路径能否在同一真实 2B 图上公平运行，并观察数值方向。",
           "只有一个种子、30 条训练题和 20 条测试题，**不能据此选模型或声称收益**。","",
           "## 四个对照","",
           "- `H-hard`：教师分布压成 argmax 硬标签，精确 CE+Brier。",
           "- `S-soft`：完整教师分布，精确 soft CE+Brier。",
           "- `T-typed-rps`：在 S 上仅给有序 Score 加严格适当 RPS，仍直接求导。",
           "- `R-rlcd-loo`：在 T 的风险上采四个高斯动作，用独立留一基线策略梯度，并加 1.0 soft CE 锚点；不做优势标准差归一化。","",
           "## 原始结果","",
           "| arm | 验证 soft CE | 已见测试 CE | 未见安全工作流 CE | 未见 argmax | 未见 Brier |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        value=results[arm]
        lines.append(f"| {arm} | {value['validation']['soft_cross_entropy']:.4f} | {value['test_seen']['soft_cross_entropy']:.4f} | {value['test_heldout']['soft_cross_entropy']:.4f} | {value['test_heldout']['argmax_agreement']:.1%} | {value['test_heldout']['soft_brier']:.4f} |")
    lines += ["","## 判断边界","",
              "所有 arm 的起始可训练权重、样本 ID、逐批行与候选排列、encoder 工作量和更新数均一致。",
              "R arm 只在 logits 上采样，不重复 encoder；它与 T 的差值同时包含平滑风险、策略梯度方差和 CE 锚点，正式实验仍需拆分。",
              "本项目的驻点实验显示，按组内优势标准差归一化可能移动 proper-risk 最优点，因此 R 改用独立留一基线且不除标准差。",
              "下一步是否进入完整四折三种子，只根据实现稳定性，不根据这 20 条测试题挑选超参数。","",
              f"机器证据：`{args.output}`。"]
    Path(args.report).write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":evidence["status"],"contrasts":contrasts},indent=2))


if __name__=="__main__":main()
