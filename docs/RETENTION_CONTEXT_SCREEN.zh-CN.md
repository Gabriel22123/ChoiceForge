# 保留能力加权与上下文约束筛选

本实验只用于选择下一轮多种子方案。开发诊断集已经在 Release Candidate v1 中读取，不能再次作为新盲测或发布证据。

| 方案 | 已见准确率 | 相对 Expanded | 已见最差源相对 Control | 开发集准确率 | 相对 Expanded | 开发集最差源相对 Control | 晋级 |
|---|---:|---:|---:|---:|---:|---:|---|
| `retention` | 0.7166 | -0.0247 | -0.0417 | 0.6328 | -0.0104 | -0.0391 | 否 |
| `retention-context` | 0.7217 | -0.0196 | -0.0521 | 0.6354 | -0.0078 | -0.0391 | 否 |

## 冻结检查

### `retention`

| 条件 | 结果 |
|---|---|
| `development.combined_accuracy` | FAIL |
| `development.each_source_accuracy` | FAIL |
| `seen.combined_accuracy` | FAIL |
| `seen.each_source_accuracy` | FAIL |
| `seen.nll` | FAIL |

### `retention-context`

| 条件 | 结果 |
|---|---|
| `development.combined_accuracy` | PASS |
| `development.context_advantage_each_source` | FAIL |
| `development.each_source_accuracy` | FAIL |
| `seen.combined_accuracy` | FAIL |
| `seen.each_source_accuracy` | FAIL |
| `seen.nll` | PASS |

## 结论边界

- 训练数据全部来自本项目已审计的可再分发公开数据。
- 单种子筛选只能淘汰明显失败的方案，不能证明稳定性。
- 任何晋级方案都必须重新冻结三个种子，并使用从未读取的新任务族做一次性盲测。

机器证据：`docs/evidence/retention-context-screen-v1.json`
