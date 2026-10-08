# QASC 冻结父函数残差专家筛选

父 LoRA 与原读出头在训练期间逐张量冻结；新增零初始化候选残差头。全部评测任务均已消费，本结果只回答机制问题。

| 已见准确率 | 已见 NLL | 开发准确率 | 因果材料优势 | QASC 准确率 | QASC NLL | QASC 材料优势 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7587 | 0.8056 | 0.6562 | -0.0247 | 0.8438 | 0.5418 | 1.7028 |

## 机制审计

- 训练前最大概率差：`0`。
- 训练后父参数最大绝对差：`0`。
- 残差探针绝对变化：`0.9428673387`。

## 结论

至少一个冻结门槛未通过；残差专家不进入新的任务族训练。

逐项门槛：`mechanism.initial_exact`=PASS, `mechanism.parent_parameters_exact`=PASS, `mechanism.residual_changed`=PASS, `seen.accuracy`=PASS, `seen.nll`=PASS, `development.accuracy`=FAIL, `development.context_retention`=FAIL, `qasc.accuracy`=FAIL, `qasc.nll`=PASS, `qasc.context`=PASS, `audit`=PASS

机器证据：`docs/evidence/qasc-residual-expert-screen-v1.json`
