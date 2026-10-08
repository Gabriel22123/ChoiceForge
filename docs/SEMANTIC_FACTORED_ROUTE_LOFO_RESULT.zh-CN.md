# 语义增强分解式安全路由留一任务族结果

冻结 2B 语义摘要与行为特征拼接；外层任务族完全留出，阈值只由训练来源内层留出预测确定。

| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |
|---|---:|---:|---:|---:|---:|
| clinc | 0.0476 | 1.0000 | +0.0013 | +0.0000 | -0.0012 |
| json-schema | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| paws-wiki | 0.0660 | 0.4345 | -0.0031 | +0.0069 | +0.0025 |
| qasc | 0.2917 | 0.9187 | -0.0061 | +0.0000 | +0.0085 |
| snli | 0.0868 | 0.6768 | -0.0021 | -0.0069 | +0.0016 |
| typed-decisions | 0.0042 | 0.7500 | +0.0003 | +0.0000 | -0.0002 |

## 汇总

- 专家选择率：`0.0764`。
- 已选专家安全精度：`0.7711`。
- 逐项安全违规率：`0.0488`。
- 平均 proper-loss 收益：`-0.0016`。
- 准确率差：`+0.0000`。
- 交叉熵差：`+0.0019`。

## 判定

至少一个冻结门槛失败，不训练完整路由端点。

逐项门槛：`coverage.folds_seeds`=PASS, `coverage.no_outer_leakage`=PASS, `combined.accuracy`=PASS, `combined.cross_entropy`=FAIL, `sources.accuracy`=PASS, `sources.cross_entropy`=PASS, `safety.violation_rate`=FAIL, `safety.selection_precision`=FAIL, `coverage.expert_selection`=PASS, `utility.proper_gain`=FAIL

机器证据：`docs/evidence/semantic-factored-route-lofo-v1.json`
