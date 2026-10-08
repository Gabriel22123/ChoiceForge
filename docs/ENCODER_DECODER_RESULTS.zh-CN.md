# 2B 级 encoder 与 decoder：同一候选决策接口的三种子比较

## 结论

**decoder 满足预注册默认路线条件。**

两条路线都不生成文本；模型只输出每个动态候选的标量分数，概率与固定 JSON 由程序构造。六个端点使用相同公开训练行、训练顺序、候选排列、更新数和适配层数。ARC 在六个端点和审计完成后才一次性启封，且全部端点都被评估。

## 已训练任务

| 种子 / 路线 | PAWS | SNLI | 总准确率 | PAWS NLL | SNLI NLL |
|---|---:|---:|---:|---:|---:|
| seed42 encoder | 49.0% | 33.3% | 46.2% | 0.6948 | 1.0958 |
| seed42 decoder | 84.9% | 73.4% | 75.9% | 0.3188 | 0.6141 |
| seed43 encoder | 49.0% | 34.9% | 47.0% | 0.6914 | 1.0927 |
| seed43 decoder | 83.9% | 76.6% | 75.5% | 0.3429 | 0.5480 |
| seed44 encoder | 54.7% | 40.6% | 50.3% | 0.6903 | 1.0870 |
| seed44 decoder | 82.8% | 74.5% | 74.7% | 0.3360 | 0.6585 |

## 保留任务与未见任务

GoEmotions 在训练中未出现；其余三类是训练回放任务。这里单列，避免总体分数掩盖迁移或遗忘。

| 种子 / 路线 | Bandit | CLINC | GoEmotions（未见） | JSON Schema |
|---|---:|---:|---:|---:|
| seed42 encoder | 70.0% | 93.4% | 39.3% | 33.3% |
| seed42 decoder | 100.0% | 100.0% | 63.9% | 38.3% |
| seed43 encoder | 50.0% | 95.1% | 39.3% | 38.3% |
| seed43 decoder | 100.0% | 98.4% | 62.3% | 31.7% |
| seed44 encoder | 80.0% | 93.4% | 41.0% | 28.3% |
| seed44 decoder | 100.0% | 98.4% | 62.3% | 33.3% |

## ARC 任务级盲测

| 种子 / 路线 | ARC-Challenge | ARC-Easy | 总准确率 |
|---|---:|---:|---:|
| seed42 encoder | 34.9% | 36.5% | 35.7% |
| seed42 decoder | 75.5% | 87.0% | 81.2% |
| seed43 encoder | 32.8% | 38.0% | 35.4% |
| seed43 decoder | 76.6% | 87.0% | 81.8% |
| seed44 encoder | 28.1% | 38.0% | 33.1% |
| seed44 decoder | 75.5% | 87.0% | 81.2% |

## 工程代价与输出稳定性

| 种子 / 路线 | 可训练参数 | 训练秒数 | 采样设备占用 | 推理中位数 | 推理 P95 | 梯度 P95 | 换序稳定 | 全顺序正确 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed42 encoder | 11,015,169 | 670.5 | 5.83 GiB | 267.3 ms | 506.0 ms | 22.63 | 24/24 | 13/24 |
| seed42 decoder | 10,097,153 | 707.1 | 5.41 GiB | 295.0 ms | 554.5 ms | 15.79 | 24/24 | 21/24 |
| seed43 encoder | 11,015,169 | 664.4 | 5.83 GiB | 267.5 ms | 502.0 ms | 20.27 | 24/24 | 13/24 |
| seed43 decoder | 10,097,153 | 698.2 | 5.40 GiB | 296.9 ms | 548.8 ms | 23.20 | 24/24 | 20/24 |
| seed44 encoder | 11,015,169 | 669.1 | 5.84 GiB | 271.7 ms | 505.4 ms | 21.55 | 24/24 | 12/24 |
| seed44 decoder | 10,097,153 | 702.8 | 5.42 GiB | 299.5 ms | 555.8 ms | 18.58 | 24/24 | 19/24 |

## decoder 默认路线规则

- 通过：`trained_combined_accuracy_wins_at_least_two_seeds`
- 通过：`no_trained_task_accuracy_loss_below_minus_1pp`
- 通过：`trained_mean_nll_not_worse`
- 通过：`generalization_mean_accuracy_not_worse`
- 通过：`generalization_mean_nll_not_worse`
- 通过：`all_decoder_order_stability_complete`
- 通过：`arc_combined_accuracy_wins_at_least_two_seeds`
- 通过：`arc_mean_accuracy_not_worse`
- 通过：`eligible_as_default`

## 边界

- 比较的是两个具体公开底座的端到端路线，不是注意力方向的纯消融。
- tokenizer、预训练语料、隐藏宽度和总参数不同；相同训练行与更新数不代表 FLOPs 完全相同。
- ARC 可能出现在公共底座的预训练语料中；本项目的封存协议只能排除本项目调参泄漏。
- 三个随机种子仍不足以估计完整模型分布。
- 机器可读证据：[evidence/architecture-comparison-v1.json](evidence/architecture-comparison-v1.json)。
