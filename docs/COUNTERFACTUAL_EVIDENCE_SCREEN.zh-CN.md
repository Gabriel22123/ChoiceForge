# 反事实证据翻转筛选

本实验从函数保留端点续训。新样本由项目内确定性规则生成：每组固定任务、问题和 Yes/No 候选，只改变证据，使答案成对翻转。无教师模型标注，生成器和文本均在 Apache-2.0 源码边界内。

| 指标 | 父端点 | 反事实端点 | 变化 |
|---|---:|---:|---:|
| 已见准确率 | 0.7565 | 0.7544 | -0.0022 |
| 已见 NLL | 0.8245 | 0.8432 | +0.0187 |
| 已消费开发准确率 | 0.6615 | 0.6510 | -0.0104 |
| 生成验证准确率 | 0.8828 | 1.0000 | +0.1172 |
| 成对全对率 | 0.7656 | 1.0000 | +0.2344 |

## 冻结检查

| 条件 | 结果 |
|---|---|
| `development.accuracy` | FAIL |
| `development.context_advantage_each_source` | FAIL |
| `development.each_source_accuracy` | FAIL |
| `generated.validation_accuracy` | PASS |
| `generated.validation_complete_group_rate` | PASS |
| `seen.accuracy` | PASS |
| `seen.each_source_accuracy` | PASS |
| `seen.nll` | FAIL |

## 结论边界

- 生成验证集只能证明对这类规则翻转的学习，不是自然任务盲测。
- 已消费因果任务只用于验证原失败是否修复，不能形成发布结论。
- 即使全部通过，也只能晋级多种子；发布仍需从未读取的新任务族。

机器证据：`docs/evidence/counterfactual-evidence-screen-v1.json`
