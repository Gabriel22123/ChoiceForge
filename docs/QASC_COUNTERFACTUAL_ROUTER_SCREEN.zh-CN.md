# QASC 反事实收益路由筛选

路由目标由父函数与冻结专家的逐行反事实成本产生，不使用任务来源标签。全部评测任务均已消费。

| 指标 | 父模型 | 反事实路由 | 差值 |
|---|---:|---:|---:|
| 已见准确率 | 0.7565 | 0.7536 | -0.0029 |
| 已见 NLL | 0.8245 | 0.8116 | -0.0129 |
| 开发准确率 | 0.6615 | 0.6615 | +0.0000 |
| QASC 准确率 | 0.8203 | 0.8984 | +0.0781 |
| QASC NLL | 0.6019 | 0.3072 | -0.2947 |

## 路由

- 反事实目标均值（回放/QASC）：`0.266016` / `0.834766`。
- 学得门均值（回放/QASC）：`0.457000` / `0.783011`。
- 学得门差：`+0.326011`。

## 判定

至少一个门槛失败；不进入实时集成。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.expert_stage_isolated`=PASS, `mechanism.router_stage_isolated`=PASS, `mechanism.target_coverage`=PASS, `mechanism.router_separates`=PASS, `seen.accuracy`=PASS, `seen.nll`=PASS, `development.accuracy`=PASS, `development.context_retention`=FAIL, `qasc.accuracy`=PASS, `qasc.nll`=PASS, `qasc.context`=PASS

机器证据：`docs/evidence/qasc-counterfactual-router-screen-v1.json`
