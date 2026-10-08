# 约束 margin 路由留一任务族结果

外层任务族完全留出；各约束保守界仅由训练来源内层留出预测确定。

| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |
|---|---:|---:|---:|---:|---:|
| clinc | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| json-schema | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| paws-wiki | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| qasc | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| snli | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| typed-decisions | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |

## 汇总

- 专家选择率：`0.0000`。
- 已选专家安全精度：`1.0000`。
- 逐项安全违规率：`0.0000`。
- 平均 proper-loss 收益：`+0.0000`。
- 准确率差：`+0.0000`。
- 交叉熵差：`+0.0000`。

## 判定

至少一个冻结门槛失败，不训练完整路由端点。

逐项门槛：`coverage.folds_seeds`=PASS, `coverage.no_outer_leakage`=PASS, `combined.accuracy`=PASS, `combined.cross_entropy`=PASS, `sources.accuracy`=PASS, `sources.cross_entropy`=PASS, `safety.violation_rate`=PASS, `safety.selection_precision`=PASS, `coverage.expert_selection`=FAIL, `utility.proper_gain`=FAIL

机器证据：`docs/evidence/constraint-margin-route-lofo-v1.json`
