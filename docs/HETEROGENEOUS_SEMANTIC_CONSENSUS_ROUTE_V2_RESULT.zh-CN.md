# 六专家九来源语义一致性路由 V2结果

六个异构专家在九个已消费来源上执行三种子一致性路由；每个外层来源完全留出。

| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |
|---|---:|---:|---:|---:|---:|
| clinc | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| gsm8k | 0.0039 | 1.0000 | +0.0034 | +0.0000 | -0.0031 |
| hellaswag | 0.0039 | 1.0000 | +0.0022 | +0.0000 | -0.0016 |
| json-schema | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| paws-wiki | 0.0104 | 0.0000 | -0.0036 | +0.0000 | +0.0026 |
| qasc | 0.0312 | 0.7500 | +0.0023 | +0.0000 | -0.0018 |
| snli | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| truthfulqa | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| typed-decisions | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |

## 汇总

- 专家选择率：`0.0047`。
- 已选专家安全精度：`0.7143`。
- 逐项安全违规率：`0.0027`。
- 平均 proper-loss 收益：`+0.0005`。
- 准确率差：`+0.0000`。
- 交叉熵差：`-0.0004`。

## 判定

至少一个冻结门槛失败，不训练完整路由端点。

逐项门槛：`coverage.folds_seeds`=PASS, `coverage.no_outer_leakage`=PASS, `combined.accuracy`=PASS, `combined.cross_entropy`=PASS, `sources.accuracy`=PASS, `sources.cross_entropy`=PASS, `safety.violation_rate`=PASS, `safety.selection_precision`=FAIL, `coverage.expert_selection`=FAIL, `utility.proper_gain`=FAIL

机器证据：`docs/evidence/heterogeneous-semantic-consensus-route-v2.json`
