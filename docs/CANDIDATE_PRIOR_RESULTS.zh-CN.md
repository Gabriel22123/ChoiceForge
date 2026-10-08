# 候选先验修正：三种子冻结研究

## 结论

**修正方法未通过全部预注册条件，不进入默认推理路线。**

开发种子只用 validation NLL 选择出的共享强度为 `0.0`。
BIG-bench logical_deduction 在选择强度并写入哈希后才首次运行。

冻结的首版独立校验器存在一个仅影响复核的 JSON 键类型错误：内存中的浮点 λ 键与落盘后的
字符串键直接比较。修复版只在比较前做 JSON 规范化，没有改动协议、数据、模型、选择、
logits、预测、指标或晋级规则；修复后全部证据通过重算。哈希和错误信息见
[校验器勘误](evidence/candidate-prior-validator-errata.json)。

## 开发集强度选择

| λ | seed42 NLL | seed43 NLL | 等权平均 |
|---:|---:|---:|---:|
| 0 | 0.5915 | 0.5788 | 0.5851 |
| 0.25 | 0.5985 | 0.5889 | 0.5937 |
| 0.5 | 0.6221 | 0.6055 | 0.6138 |
| 0.75 | 0.6588 | 0.6276 | 0.6432 |
| 1 | 0.7013 | 0.6533 | 0.6773 |

## seed44 确认任务

| 任务 | raw 准确率 | 修正准确率 | raw NLL | 修正 NLL |
|---|---:|---:|---:|---:|
| paws-wiki | 82.8% | 82.8% | 0.3478 | 0.3478 |
| snli | 74.5% | 74.5% | 0.6022 | 0.6022 |
| bandit | 100.0% | 100.0% | 0.4588 | 0.4588 |
| clinc | 98.4% | 98.4% | 0.1586 | 0.1586 |
| goemotions | 62.3% | 62.3% | 1.1336 | 1.1336 |
| json-schema | 33.3% | 33.3% | 1.1495 | 1.1495 |

## 新任务盲测：BIG-bench logical_deduction

| 种子 / 子任务 | raw 准确率 | 修正准确率 | raw NLL | 修正 NLL |
|---|---:|---:|---:|---:|
| seed42 / three-objects | 79.2% | 79.2% | 0.5510 | 0.5510 |
| seed42 / five-objects | 46.9% | 46.9% | 1.2279 | 1.2279 |
| seed42 / seven-objects | 67.7% | 67.7% | 1.2104 | 1.2104 |
| seed43 / three-objects | 78.1% | 78.1% | 0.5821 | 0.5821 |
| seed43 / five-objects | 50.0% | 50.0% | 1.2077 | 1.2077 |
| seed43 / seven-objects | 67.7% | 67.7% | 1.1755 | 1.1755 |
| seed44 / three-objects | 79.2% | 79.2% | 0.5871 | 0.5871 |
| seed44 / five-objects | 50.0% | 50.0% | 1.2454 | 1.2454 |
| seed44 / seven-objects | 65.6% | 65.6% | 1.2416 | 1.2416 |

## 预注册晋级规则

- 未通过：`nonzero_selected_strength`
- 通过：`seed44_primary_no_accuracy_loss_below_minus_1pp`
- 通过：`seed44_primary_mean_accuracy_not_worse`
- 通过：`seed44_primary_mean_calibrated_nll_not_worse`
- 通过：`seed44_retained_mean_accuracy_not_worse`
- 通过：`seed44_retained_mean_calibrated_nll_not_worse`
- 未通过：`blind_combined_accuracy_wins_at_least_two_seeds`
- 未通过：`blind_mean_accuracy_improves`
- 通过：`blind_mean_calibrated_nll_not_worse`
- 通过：`all_real_model_audits_stable`
- 未通过：`eligible_as_default`

## 输出审计

| 种子 | 换序稳定 | ID 改名稳定 | 最大概率差 |
|---|---:|---:|---:|
| seed42 | 12/12 | 12/12 | 5.96e-08 |
| seed43 | 12/12 | 12/12 | 5.96e-08 |
| seed44 | 12/12 | 12/12 | 1.19e-07 |

## 边界

- 修正增加一次空上下文候选评分，推理工作量接近翻倍；本轮未重新测延迟。
- 空上下文分数可能包含有效常识，不保证都是偏差。
- BIG-bench 题目可能存在于底座预训练语料；盲测只排除本项目调参泄漏。
- 本报告比较固定端点上的推理变换，不证明训练阶段已经消除候选先验。
- 机器可读证据：[candidate-prior-v1.json](evidence/candidate-prior-v1.json)。
