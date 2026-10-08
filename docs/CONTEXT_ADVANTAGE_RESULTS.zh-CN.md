# 训练时上下文优势约束：三种子冻结研究

## 结论

**未通过全部预注册条件，不进入默认训练路线。**

本研究只改变一件事：训练时要求真实上下文相对空上下文提高正确候选的对数概率。
空上下文分支停止梯度并使用 eval 模式，因此不会改变对照组的 dropout 随机数序列。

## 已见任务

| 种子 / 任务 | 对照准确率 | 约束准确率 | 对照 NLL | 约束 NLL |
|---|---:|---:|---:|---:|
| seed42 / paws-wiki | 84.9% | 85.4% | 0.3188 | 0.3235 |
| seed42 / snli | 73.4% | 72.9% | 0.6141 | 0.6350 |
| seed42 / bandit | 100.0% | 100.0% | 0.3478 | 0.3470 |
| seed42 / clinc | 100.0% | 100.0% | 0.0637 | 0.0719 |
| seed42 / goemotions | 63.9% | 63.9% | 0.9925 | 1.0026 |
| seed42 / json-schema | 38.3% | 31.7% | 1.1181 | 1.1091 |
| seed43 / paws-wiki | 83.9% | 83.9% | 0.3429 | 0.3431 |
| seed43 / snli | 76.6% | 76.0% | 0.5480 | 0.5449 |
| seed43 / bandit | 100.0% | 100.0% | 0.4406 | 0.4512 |
| seed43 / clinc | 98.4% | 96.7% | 0.0795 | 0.0818 |
| seed43 / goemotions | 62.3% | 62.3% | 0.9690 | 0.9773 |
| seed43 / json-schema | 31.7% | 30.0% | 1.1423 | 1.1460 |
| seed44 / paws-wiki | 82.8% | 83.3% | 0.3360 | 0.3268 |
| seed44 / snli | 74.5% | 74.5% | 0.6585 | 0.6481 |
| seed44 / bandit | 100.0% | 100.0% | 0.2183 | 0.2423 |
| seed44 / clinc | 98.4% | 96.7% | 0.0828 | 0.0804 |
| seed44 / goemotions | 62.3% | 62.3% | 0.9643 | 0.9697 |
| seed44 / json-schema | 33.3% | 33.3% | 1.2552 | 1.2395 |

## 机制指标

`gold gain` 是真实上下文与空上下文下正确候选 log-probability 的差；越大说明模型越依赖材料。
`shortfall` 是该差距未达到预注册 0.2 margin 的剩余量；越小越好。

| 种子 | 对照 gold gain | 约束 gold gain | 对照 shortfall | 约束 shortfall |
|---|---:|---:|---:|---:|
| seed42 | 0.7488 | 0.7197 | 0.1348 | 0.1280 |
| seed43 | 0.6938 | 0.6759 | 0.1675 | 0.1594 |
| seed44 | 0.8028 | 0.8003 | 0.2838 | 0.2692 |

## 新任务盲测：BIG-bench disambiguation_qa

64 个答案不可见抽取的完整 m/f/t 三元组，共 192 题；所有训练端点和审计完成后才首次推理。

| 种子 | 对照准确率 | 约束准确率 | 对照 NLL | 约束 NLL | 对照 Ambiguous recall | 约束 Ambiguous recall | 对照完整组 | 约束完整组 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed42 | 56.2% | 56.2% | 0.9369 | 0.9235 | 58.5% | 56.6% | 29/64 | 30/64 |
| seed43 | 56.2% | 57.3% | 0.8524 | 0.8558 | 34.0% | 37.7% | 29/64 | 29/64 |
| seed44 | 58.3% | 59.9% | 0.9351 | 0.9044 | 15.1% | 22.6% | 32/64 | 33/64 |

## 训练成本与结构审计

| 种子 | 对照训练秒数 | 约束训练秒数 | 对照梯度 p95 | 约束梯度 p95 | 换序稳定 |
|---|---:|---:|---:|---:|---:|
| seed42 | 707.1 | 931.4 | 15.788 | 17.039 | 24/24 |
| seed43 | 698.2 | 924.8 | 23.202 | 25.096 | 24/24 |
| seed44 | 702.8 | 930.5 | 18.579 | 20.470 | 24/24 |

正式推理不运行空上下文分支，模型结构、参数量和一次请求的推理工作与 decoder 对照相同。

## 预注册晋级规则

- 未通过：`primary_combined_accuracy_wins_at_least_two_seeds`
- 通过：`primary_no_seed_task_accuracy_loss_below_minus_1pp`
- 未通过：`primary_mean_raw_nll_not_worse`
- 未通过：`retained_mean_accuracy_not_worse`
- 未通过：`retained_mean_raw_nll_not_worse`
- 未通过：`mechanism_gain_wins_at_least_two_seeds`
- 未通过：`mechanism_mean_gain_improves`
- 通过：`mechanism_mean_shortfall_decreases`
- 通过：`all_treatment_audits_stable`
- 通过：`blind_combined_accuracy_wins_at_least_two_seeds`
- 通过：`blind_mean_accuracy_improves`
- 通过：`blind_mean_raw_nll_not_worse`
- 通过：`blind_mean_ambiguous_recall_not_worse`
- 通过：`blind_mean_complete_group_accuracy_not_worse`
- 未通过：`eligible_as_default`

## 边界

- 本轮固定使用一个 weight 和 margin；smoke test 只验证实现可运行，没有用于选效果。
- BIG-bench 可能存在于底座预训练语料；盲测只排除本项目训练和调参泄漏。
- 机制指标证明这个约束是否改变材料依赖程度，不单独证明所有任务泛化。
- 机器可读证据：[context-advantage-v1.json](evidence/context-advantage-v1.json)。
