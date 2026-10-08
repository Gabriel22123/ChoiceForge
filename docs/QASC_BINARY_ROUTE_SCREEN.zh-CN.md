# QASC 二元反事实路由筛选

路由在逐位保留父模型与完整使用冻结专家之间二选一。全部评测任务均已消费。

| 指标 | 父模型 | 二元路由 | 差值 |
|---|---:|---:|---:|
| 已见准确率 | 0.7565 | 0.7565 | +0.0000 |
| 已见 NLL | 0.8245 | 0.8245 | +0.0000 |
| 开发准确率 | 0.6615 | 0.6536 | -0.0078 |
| QASC 准确率 | 0.8203 | 0.9297 | +0.1094 |
| QASC NLL | 0.6019 | 0.2505 | -0.3514 |

## 路由

- 目标打开率（回放/QASC）：`0.240234` / `0.833984`。
- 推理打开率（回放/QASC）：`0.000000` / `1.000000`。
- 打开率差：`+1.000000`。
- 推理阈值：`0.5`。
- 完整/空材料路由逐样本一致；关闭路径与父模型逐位相同。

## 判定

至少一个门槛失败；不进入实时集成。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.expert_stage_isolated`=PASS, `mechanism.router_stage_isolated`=PASS, `mechanism.target_coverage`=PASS, `mechanism.closed_path_exact`=PASS, `mechanism.shared_gate_exact`=PASS, `mechanism.router_separates`=PASS, `seen.accuracy`=PASS, `seen.nll`=PASS, `development.accuracy`=FAIL, `development.context_retention`=PASS, `qasc.accuracy`=PASS, `qasc.nll`=PASS, `qasc.context`=PASS

机器证据：`docs/evidence/qasc-binary-route-screen-v1.json`
