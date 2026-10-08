# 反事实效用路由留一任务族验证

路由回归父模型与专家的带符号反事实效用；7 个公开来源逐一留出，每折每臂 3 个种子。

按宏平均路由遗憾预注册选择：`utility_row_weighted`。

| 留出来源 | 路由遗憾 | 始终父模型遗憾 | 符号平衡准确率 | 变换 MAE | 恒零 MAE |
|---|---:|---:|---:|---:|---:|
| bandit | 0.0754 | 0.0059 | 0.5000 | 0.5341 | 0.0736 |
| clinc | 0.1079 | 0.0027 | 0.5000 | 0.5725 | 0.0989 |
| json-schema | 0.1835 | 0.0316 | 0.5000 | 0.5922 | 0.1966 |
| paws-wiki | 0.0639 | 0.0275 | 0.5000 | 0.5170 | 0.0837 |
| qasc | 0.0059 | 0.5874 | 0.5000 | 0.5170 | 0.4387 |
| snli | 0.1016 | 0.0593 | 0.5000 | 0.5147 | 0.1531 |
| typed-decisions | 0.2317 | 0.0752 | 0.5000 | 0.6485 | 0.2733 |

## 汇总

- 选中效用路由宏平均遗憾：`0.109980`。
- 始终父模型宏平均遗憾：`0.112788`。
- 选中效用路由变换后 MAE：`0.556584`。
- 恒预测零的变换后 MAE：`0.188279`。
- 选中效用路由宏平均平衡准确率：`0.500000`。

## 判定

至少一个门槛失败；不训练新的候选端点。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.expert_stage_isolated`=PASS, `coverage.sources_seeds_arms`=PASS, `coverage.no_heldout_training`=PASS, `selected.regret_vs_parent`=FAIL, `selected.mae_vs_zero`=FAIL, `selected.balanced_accuracy`=FAIL, `selected.each_source_noninferior`=FAIL

机器证据：`docs/evidence/utility-route-lofo-v1.json`
