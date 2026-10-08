# 二元路由留一任务族验证

7 个公开来源逐一留出，每折、每训练臂运行 3 个随机种子。

| 留出来源 | 普通 Brier | 族均衡 Brier | 族均衡平衡准确率 | 目标打开率 | 预测打开率 |
|---|---:|---:|---:|---:|---:|
| bandit | 0.1607 | 0.1738 | 0.5716 | 0.1562 | 0.1458 |
| clinc | 0.4162 | 0.4785 | 0.5018 | 0.0521 | 0.9965 |
| json-schema | 0.2493 | 0.2544 | 0.5000 | 0.3542 | 0.0000 |
| paws-wiki | 0.1595 | 0.1585 | 0.5000 | 0.1875 | 0.0000 |
| qasc | 0.5607 | 0.5614 | 0.5000 | 0.8340 | 0.0000 |
| snli | 0.1997 | 0.1976 | 0.5000 | 0.2500 | 0.0000 |
| typed-decisions | 0.2750 | 0.2666 | 0.5000 | 0.3854 | 0.0000 |

## 汇总

- 普通训练宏平均 Brier：`0.288707`。
- 任务族均衡宏平均 Brier：`0.298687`。
- 任务族均衡常数基线 Brier：`0.319516`。
- 任务族均衡宏平均平衡准确率：`0.510491`。

## 判定

至少一个门槛失败；不训练新的候选端点。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.expert_stage_isolated`=PASS, `coverage.sources_seeds_arms`=PASS, `coverage.no_heldout_training`=PASS, `balanced.brier_vs_row`=FAIL, `balanced.brier_vs_constant`=PASS, `balanced.balanced_accuracy`=FAIL, `balanced.each_source_noninferior`=FAIL

机器证据：`docs/evidence/binary-route-lofo-v1.json`
