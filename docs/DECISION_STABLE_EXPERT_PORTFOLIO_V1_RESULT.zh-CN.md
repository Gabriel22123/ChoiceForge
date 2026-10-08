# 决策稳定专家组合 V1 结果

外层来源及其同名专家同时留出；推理投影保证父模型最终选择不翻转。

| 留出来源 | 准确率差 | CE 差 | Brier 差 | proper 收益 | 上下文差 | 平均步长 |
|---|---:|---:|---:|---:|---:|---:|
| clinc | +0.0000 | -0.0003 | -0.0000 | +0.0003 | -0.0386 | 0.9940 |
| gsm8k | +0.0000 | -0.0218 | -0.0089 | +0.0262 | +0.0246 | 0.8789 |
| hellaswag | +0.0000 | -0.0233 | -0.0094 | +0.0280 | +0.0308 | 0.9551 |
| json-schema | +0.0000 | +0.0133 | +0.0124 | -0.0195 | +0.0310 | 0.9405 |
| paws-wiki | +0.0000 | -0.0167 | -0.0097 | +0.0216 | +0.0122 | 0.9948 |
| qasc | +0.0000 | -0.0279 | -0.0072 | +0.0315 | +0.0203 | 1.0000 |
| snli | +0.0000 | -0.0244 | -0.0174 | +0.0331 | +0.0215 | 0.9740 |
| truthfulqa | +0.0000 | -0.0310 | +0.0040 | +0.0291 | +0.0413 | 0.9502 |
| typed-decisions | +0.0000 | -0.0120 | -0.0028 | +0.0134 | +0.0574 | 0.9840 |

## 宏平均

- 准确率差：`+0.0000`。
- 交叉熵差：`-0.0160`。
- Brier 差：`-0.0043`。
- proper-loss 收益：`+0.0182`。
- 上下文优势差：`+0.0223`。
- 平均投影步长：`0.9635`。

## 判定

全部冻结门槛通过；允许训练全来源组合并冻结最终评测协议。

逐项门槛：`coverage.folds_seeds`=PASS, `coverage.no_outer_data`=PASS, `coverage.no_matching_expert`=PASS, `decision.exact_each_run`=PASS, `calibration.cross_entropy`=PASS, `calibration.brier`=PASS, `utility.proper_gain`=PASS, `context.aggregate`=PASS, `sources.cross_entropy`=PASS, `sources.proper_gain`=PASS, `sources.context`=PASS, `coverage.material_projection`=PASS

机器证据：`docs/evidence/decision-stable-expert-portfolio-v1.json`
