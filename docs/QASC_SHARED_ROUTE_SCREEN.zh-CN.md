# QASC 共享空材料路由筛选

路由只读取空材料任务/候选视图，并把同一个门用于完整与空材料残差。全部评测任务均已消费。

| 指标 | 父模型 | 共享路由 | 差值 |
|---|---:|---:|---:|
| 已见准确率 | 0.7565 | 0.7544 | -0.0022 |
| 已见 NLL | 0.8245 | 0.8139 | -0.0105 |
| 开发准确率 | 0.6615 | 0.6536 | -0.0078 |
| QASC 准确率 | 0.8203 | 0.9141 | +0.0938 |
| QASC NLL | 0.6019 | 0.2875 | -0.3144 |

## 路由

- 学得门均值（回放/QASC）：`0.367172` / `0.847127`。
- 门差：`+0.479955`。
- 完整/空材料门差：`0`（结构共享）。

## 判定

至少一个门槛失败；不进入实时集成。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.expert_stage_isolated`=PASS, `mechanism.router_stage_isolated`=PASS, `mechanism.target_coverage`=PASS, `mechanism.shared_gate_exact`=PASS, `mechanism.router_separates`=PASS, `seen.accuracy`=PASS, `seen.nll`=PASS, `development.accuracy`=FAIL, `development.context_retention`=PASS, `qasc.accuracy`=PASS, `qasc.nll`=PASS, `qasc.context`=PASS

机器证据：`docs/evidence/qasc-shared-route-screen-v1.json`
