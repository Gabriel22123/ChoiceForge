# QASC 八候选证据组合筛选

本实验从函数保留端点续训，加入 512 条 QASC 训练题；每题保留两条科学事实和全部八个候选。候选按不读取答案的文本哈希重排，原始 A–H 字母不进入模型，再按重排后的正确位置做等量抽样。QASC 官方以 CC BY 4.0 发布。

| 指标 | 父端点 | QASC 端点 | 变化 |
|---|---:|---:|---:|
| 已见准确率 | 0.7565 | 0.7544 | -0.0022 |
| 已见 NLL | 0.8245 | 0.8680 | +0.0435 |
| 已消费开发准确率 | 0.6615 | 0.6536 | -0.0078 |
| QASC validation 准确率 | 0.8203 | 0.9375 | +0.1172 |
| QASC validation NLL | 0.6013 | 0.1674 | -0.4339 |
| QASC 上下文优势 | 1.5401 | 2.0043 | +0.4643 |

## 冻结检查

| 条件 | 结果 |
|---|---|
| `development.accuracy` | FAIL |
| `development.context_advantage_each_source` | FAIL |
| `development.each_source_accuracy` | FAIL |
| `qasc.context_advantage_absolute` | PASS |
| `qasc.context_advantage_delta` | PASS |
| `qasc.validation_accuracy_absolute` | PASS |
| `qasc.validation_accuracy_delta` | PASS |
| `qasc.validation_nll` | PASS |
| `seen.accuracy` | PASS |
| `seen.each_source_accuracy` | FAIL |
| `seen.nll` | FAIL |

## 候选边际诊断

- 父端点预测与均衡真值的位置边际 TV：0.0547。
- QASC 端点预测与均衡真值的位置边际 TV：0.0391。
- 候选 ID 不进入模型；此统计用于发现答案文本打分坍缩，不用于宣称模型学习了位置。

## 结论边界

- QASC validation 在本轮报告生成后即成为已消费开发集，不是发布盲测。
- 已消费的三类开发任务只用于检查旧失败，不能形成新的发布结论。
- 即使全部通过，也只能晋级多种子；发布仍需从未读取的新任务族。
- BIG-bench 的 canary 明确要求任务数据不要进入训练语料，因此本轮没有使用新增 BIG-bench 数据。

机器证据：`docs/evidence/qasc-evidence-screen-v1.json`
