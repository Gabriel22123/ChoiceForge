# 语义安全路由三种子一致性结果

只有三个种子都判定安全且收益为正的专家才被放行。

| 留出来源 | 专家选择率 | 安全精度 | proper 收益 | 准确率差 | 交叉熵差 |
|---|---:|---:|---:|---:|---:|
| clinc | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| json-schema | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| paws-wiki | 0.0417 | 0.5000 | -0.0011 | +0.0000 | +0.0008 |
| qasc | 0.1641 | 0.9048 | +0.0064 | +0.0000 | -0.0047 |
| snli | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| typed-decisions | 0.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |

## 汇总

- 专家选择率：`0.0345`。
- 已选专家安全精度：`0.8400`。
- 逐项安全违规率：`0.0166`。
- 平均 proper-loss 收益：`+0.0009`。
- 准确率差：`+0.0000`。
- 交叉熵差：`-0.0007`。

## 判定

至少一个冻结门槛失败，不训练完整路由端点。

逐项门槛：`coverage.folds_seeds`=PASS, `coverage.no_outer_leakage`=PASS, `combined.accuracy`=PASS, `combined.cross_entropy`=PASS, `sources.accuracy`=PASS, `sources.cross_entropy`=PASS, `safety.violation_rate`=PASS, `safety.selection_precision`=FAIL, `coverage.expert_selection`=FAIL, `utility.proper_gain`=FAIL

机器证据：`docs/evidence/semantic-consensus-route-lofo-v1.json`
