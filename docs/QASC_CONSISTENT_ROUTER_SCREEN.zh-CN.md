# QASC 成对一致反事实路由筛选

反事实收益路由增加完整/空材料门一致性。全部评测任务均已消费。

| 指标 | 父模型 | 成对一致路由 | 差值 |
|---|---:|---:|---:|
| 已见准确率 | 0.7565 | 0.7536 | -0.0029 |
| 已见 NLL | 0.8245 | 0.8115 | -0.0129 |
| 开发准确率 | 0.6615 | 0.6615 | +0.0000 |
| QASC 准确率 | 0.8203 | 0.8984 | +0.0781 |
| QASC NLL | 0.6019 | 0.3052 | -0.2967 |

## 路由

- 学得门均值（回放/QASC）：`0.464366` / `0.787442`。
- 门差：`+0.323076`。
- 开发集完整/空材料门平均绝对差：`0.082191`。

## 判定

至少一个门槛失败；不进入实时集成。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.expert_stage_isolated`=PASS, `mechanism.router_stage_isolated`=PASS, `mechanism.target_coverage`=PASS, `mechanism.router_separates`=PASS, `seen.accuracy`=PASS, `seen.nll`=PASS, `development.accuracy`=PASS, `development.context_retention`=FAIL, `qasc.accuracy`=PASS, `qasc.nll`=PASS, `qasc.context`=PASS

机器证据：`docs/evidence/qasc-consistent-router-screen-v1.json`
