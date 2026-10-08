# QASC 语义路由残差机制筛选

本实验使用冻结父模型特征，只检验请求级语义路由能否在学习 QASC 时保留旧函数。全部评测任务均已消费，结果不构成发布或零样本泛化结论。

| 指标 | 父模型 | 路由残差 | 差值 |
|---|---:|---:|---:|
| 已见准确率 | 0.7565 | 0.7565 | +0.0000 |
| 已见 NLL | 0.8245 | 0.8244 | -0.0001 |
| 消费开发集准确率 | 0.6615 | 0.6615 | +0.0000 |
| QASC 准确率 | 0.8203 | 0.8203 | +0.0000 |
| QASC NLL | 0.6019 | 0.6016 | -0.0003 |

## 路由行为

- 旧任务回放门均值：`0.003884`。
- QASC 训练门均值：`0.004993`。
- QASC − 回放门差：`+0.001109`。

## 判定

至少一个门槛未通过；当前路由机制不进入实时模型集成。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.router_separates`=FAIL, `seen.accuracy`=PASS, `seen.nll`=PASS, `development.accuracy`=PASS, `development.context_retention`=PASS, `qasc.accuracy`=FAIL, `qasc.nll`=FAIL, `qasc.context`=FAIL

机器证据：`docs/evidence/qasc-routed-residual-screen-v1.json`
