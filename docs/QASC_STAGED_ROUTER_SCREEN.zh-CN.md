# QASC 两阶段语义路由筛选

先训练非零专家，再冻结专家训练语义路由。全部任务均已消费，本结果只用于机制选择。

| 指标 | 父模型 | 两阶段路由 | 差值 |
|---|---:|---:|---:|
| 已见准确率 | 0.7565 | 0.7544 | -0.0022 |
| 已见 NLL | 0.8245 | 0.8024 | -0.0220 |
| 开发准确率 | 0.6615 | 0.6536 | -0.0078 |
| QASC 准确率 | 0.8203 | 0.9297 | +0.1094 |
| QASC NLL | 0.6019 | 0.2508 | -0.3511 |

## 路由

- 回放门均值：`0.997449`。
- QASC 门均值：`0.998690`。
- 门差：`+0.001242`。

## 判定

至少一个门槛失败；不进入实时集成。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.expert_stage_isolated`=PASS, `mechanism.router_stage_isolated`=PASS, `mechanism.router_separates`=FAIL, `seen.accuracy`=PASS, `seen.nll`=PASS, `development.accuracy`=FAIL, `development.context_retention`=FAIL, `qasc.accuracy`=PASS, `qasc.nll`=PASS, `qasc.context`=PASS

机器证据：`docs/evidence/qasc-staged-router-screen-v1.json`
