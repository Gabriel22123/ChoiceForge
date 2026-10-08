# 七任务族语义一致性路由结果

TruthfulQA 加入专家库与元训练；三个种子仍须一致放行。

| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |
|---|---:|---:|---:|---:|---:|
| clinc | 0.0476 | 0.5000 | -0.0011 | +0.0000 | +0.0008 |
| json-schema | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| paws-wiki | 0.0938 | 0.2222 | -0.0068 | +0.0000 | +0.0056 |
| qasc | 0.0625 | 0.8750 | +0.0081 | +0.0000 | -0.0065 |
| snli | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| truthfulqa | 0.0234 | 0.3333 | +0.0034 | +0.0000 | -0.0026 |
| typed-decisions | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |

## 汇总

- 专家选择率：`0.0255`。
- 已选专家安全精度：`0.4800`。
- 逐项安全违规率：`0.0337`。
- 平均 proper-loss 收益：`+0.0005`。
- 准确率差：`+0.0000`。
- 交叉熵差：`-0.0004`。

## 判定

至少一个冻结门槛失败，不训练完整路由端点。

逐项门槛：`coverage.folds_seeds`=PASS, `coverage.no_outer_leakage`=PASS, `combined.accuracy`=PASS, `combined.cross_entropy`=PASS, `sources.accuracy`=PASS, `sources.cross_entropy`=PASS, `safety.violation_rate`=FAIL, `safety.selection_precision`=FAIL, `coverage.expert_selection`=FAIL, `utility.proper_gain`=FAIL

机器证据：`docs/evidence/extended-semantic-consensus-route-v1.json`
