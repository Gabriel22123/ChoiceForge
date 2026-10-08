# 分解式安全路由留一任务族结果

每个外层任务族完全留出；安全阈值只由训练来源的内层留出预测确定。

| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |
|---|---:|---:|---:|---:|---:|
| clinc | 0.0952 | 1.0000 | +0.0006 | +0.0000 | -0.0006 |
| json-schema | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| paws-wiki | 0.0208 | 0.6667 | +0.0006 | +0.0000 | -0.0004 |
| qasc | 0.0495 | 1.0000 | +0.0508 | +0.0078 | -0.0426 |
| snli | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| typed-decisions | 0.0573 | 0.2268 | -0.0133 | +0.0010 | +0.0131 |

## 汇总

- 专家选择率：`0.0424`。
- 已选专家安全精度：`0.5543`。
- 逐项安全违规率：`0.0382`。
- 平均 proper-loss 收益：`+0.0065`。
- 准确率差：`+0.0015`。
- 交叉熵差：`-0.0051`。

## 判定

至少一个冻结门槛失败，不训练完整路由端点。

逐项门槛：`coverage.folds_seeds`=PASS, `coverage.no_outer_leakage`=PASS, `combined.accuracy`=PASS, `combined.cross_entropy`=PASS, `sources.accuracy`=PASS, `sources.cross_entropy`=PASS, `safety.violation_rate`=FAIL, `safety.selection_precision`=FAIL, `coverage.expert_selection`=FAIL, `utility.proper_gain`=PASS

机器证据：`docs/evidence/factored-safe-route-lofo-v1.json`
