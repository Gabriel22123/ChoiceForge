# 行为路由留一任务族验证

路由只读取父模型与冻结专家的推理行为特征；7 个公开来源逐一留出，每折每臂 3 个种子。

按宏平均 Brier 预注册选择：`behavior_source_balanced`。

| 留出来源 | 语义路由 Brier | 行为路由 Brier | 行为平衡准确率 | 目标打开率 | 预测打开率 |
|---|---:|---:|---:|---:|---:|
| bandit | 0.1607 | 0.2092 | 0.5148 | 0.1562 | 0.0417 |
| clinc | 0.4162 | 0.2119 | 0.4147 | 0.0521 | 0.2951 |
| json-schema | 0.2493 | 0.2447 | 0.4660 | 0.3542 | 0.1910 |
| paws-wiki | 0.1595 | 0.1943 | 0.4957 | 0.1875 | 0.0069 |
| qasc | 0.5607 | 0.3574 | 0.5320 | 0.8340 | 0.0534 |
| snli | 0.1997 | 0.2329 | 0.4954 | 0.2500 | 0.3542 |
| typed-decisions | 0.2750 | 0.2609 | 0.4391 | 0.3854 | 0.3542 |

## 汇总

- 语义路由宏平均 Brier：`0.288707`。
- 选中行为路由宏平均 Brier：`0.244483`。
- 选中行为路由常数基线 Brier：`0.319516`。
- 选中行为路由宏平均平衡准确率：`0.479664`。

## 判定

至少一个门槛失败；不训练新的候选端点。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.expert_stage_isolated`=PASS, `coverage.sources_seeds_arms`=PASS, `coverage.no_heldout_training`=PASS, `selected.brier_vs_semantic`=PASS, `selected.brier_vs_constant`=PASS, `selected.balanced_accuracy`=FAIL, `selected.each_source_noninferior`=FAIL

机器证据：`docs/evidence/behavior-route-lofo-v1.json`
