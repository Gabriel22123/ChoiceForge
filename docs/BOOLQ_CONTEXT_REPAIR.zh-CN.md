# BoolQ 上下文修复筛选

本实验只使用公开数据，从函数保留端点继续训练。BoolQ 选择顺序不读取答案；此前已消费的 384 条 BoolQ 全部排除。生成的数据受 CC-BY-SA-3.0 约束，只保存在本地，不进入 Apache-2.0 源码树。

| 指标 | 父端点 | BoolQ 修复端点 | 变化 |
|---|---:|---:|---:|
| 已见准确率 | 0.7565 | 0.7558 | -0.0007 |
| 已见 NLL | 0.8245 | 0.8374 | +0.0128 |
| 已消费开发准确率 | 0.6615 | 0.6641 | +0.0026 |
| BoolQ validation 准确率 | 0.7266 | 0.7422 | +0.0156 |

## 冻结检查

| 条件 | 结果 |
|---|---|
| `boolq.validation_accuracy` | FAIL |
| `development.accuracy` | PASS |
| `development.context_advantage_each_source` | FAIL |
| `development.each_source_accuracy` | PASS |
| `seen.accuracy` | PASS |
| `seen.each_source_accuracy` | FAIL |
| `seen.nll` | FAIL |

## 结论边界

- BoolQ validation 参与开发筛选，不是盲测。
- 已消费因果任务只用于验证原先失败是否修复，不能形成发布结论。
- 即使全部通过，也只能晋级多种子；发布仍需从未读取的新任务族。

机器证据：`docs/evidence/boolq-context-repair-v1.json`
