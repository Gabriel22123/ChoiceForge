# 函数级保留与上下文更新筛选

本实验从同一个 Expanded 端点继续训练。两组使用相同公开样本、顺序、学习率和上下文约束；唯一差异是是否用父模型完整概率分布做 KL 锚定。已消费诊断集只用于开发。

| 方案 | 已见准确率 | 相对 Expanded | 已见最差源相对 Control | 开发集准确率 | 相对 Expanded | 开发集最差源相对 Control | 通过全部条件 | 晋级 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `warm-context` | 0.7565 | +0.0153 | -0.0164 | 0.6562 | +0.0130 | -0.0078 | 否 | 否 |
| `warm-functional-context` | 0.7565 | +0.0153 | -0.0164 | 0.6615 | +0.0182 | -0.0078 | 否 | 否 |

## 冻结检查

### `warm-context`

| 条件 | 结果 |
|---|---|
| `development.combined_accuracy` | PASS |
| `development.context_advantage_each_source` | FAIL |
| `development.each_source_accuracy` | PASS |
| `seen.combined_accuracy` | PASS |
| `seen.each_source_accuracy` | PASS |
| `seen.nll` | PASS |

### `warm-functional-context`

| 条件 | 结果 |
|---|---|
| `development.combined_accuracy` | PASS |
| `development.context_advantage_each_source` | FAIL |
| `development.each_source_accuracy` | PASS |
| `seen.combined_accuracy` | PASS |
| `seen.each_source_accuracy` | PASS |
| `seen.nll` | PASS |

## 结论边界

- 教师分布来自同一个公开数据父端点；公开真实标签仍保留，未被教师替换。
- 单种子开发筛选只能淘汰方案，不能证明稳定性或通用零样本能力。
- 只有函数级保留组有资格晋级；任何发布实验都必须另取从未读取的新任务族。

机器证据：`docs/evidence/functional-retention-screen-v1.json`
