# QASC 父决策稳定共享路由筛选

共享空材料路由的反事实目标增加父模型首选候选保持成本。全部评测任务均已消费。

| 指标 | 父模型 | 决策稳定路由 | 差值 |
|---|---:|---:|---:|
| 已见准确率 | 0.7565 | 0.7565 | +0.0000 |
| 已见 NLL | 0.8245 | 0.8154 | -0.0091 |
| 开发准确率 | 0.6615 | 0.6510 | -0.0104 |
| QASC 准确率 | 0.8203 | 0.9141 | +0.0938 |
| QASC NLL | 0.6019 | 0.2866 | -0.3153 |

## 路由

- 目标门均值（回放/QASC）：`0.250391` / `0.834766`。
- 学得门均值（回放/QASC）：`0.334298` / `0.850200`。
- 门差：`+0.515902`。
- 完整/空材料门差：`0`（结构共享）。

## 判定

至少一个门槛失败；不进入实时集成。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.expert_stage_isolated`=PASS, `mechanism.router_stage_isolated`=PASS, `mechanism.target_coverage`=PASS, `mechanism.shared_gate_exact`=PASS, `mechanism.router_separates`=PASS, `seen.accuracy`=PASS, `seen.nll`=PASS, `development.accuracy`=FAIL, `development.context_retention`=PASS, `qasc.accuracy`=PASS, `qasc.nll`=PASS, `qasc.context`=PASS

机器证据：`docs/evidence/qasc-decision-stable-route-screen-v1.json`
