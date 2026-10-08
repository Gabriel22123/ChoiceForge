# 安全多专家路由留一任务族结果

路由只使用父模型与四个专家的候选顺序不变行为特征。六个来源逐一完全留出，每折三个随机种子。

| 留出来源 | 路径匹配率 | proper 收益 | 准确率差 | 交叉熵差 | 平均违规数 |
|---|---:|---:|---:|---:|---:|
| clinc | 0.5476 | +0.0043 | +0.0000 | -0.0041 | 8.33 |
| json-schema | 0.1587 | -0.0023 | +0.0000 | +0.0018 | 4.67 |
| paws-wiki | 0.4340 | -0.0082 | +0.0069 | +0.0064 | 13.67 |
| qasc | 0.6224 | +0.0696 | +0.0417 | -0.0455 | 20.67 |
| snli | 0.2431 | -0.1365 | +0.0000 | +0.1271 | 35.33 |
| typed-decisions | 0.1417 | -0.0901 | -0.0031 | +0.0866 | 110.67 |

## 汇总

- 路径匹配率：`0.3579`。
- 平均 proper-loss 收益：`-0.0272`。
- 准确率差：`+0.0076`。
- 交叉熵差：`+0.0287`。
- 安全违规率：`0.2670`。

## 判定

至少一个门槛失败，不训练新的路由端点。

逐项门槛：`coverage.folds_seeds`=PASS, `coverage.no_heldout_training`=PASS, `combined.accuracy`=PASS, `combined.cross_entropy`=FAIL, `sources.accuracy`=PASS, `sources.cross_entropy`=FAIL, `safety.violation_rate`=FAIL, `utility.proper_gain`=FAIL, `target.path_match`=FAIL

机器证据：`docs/evidence/safe-multi-expert-route-lofo-v1.json`
