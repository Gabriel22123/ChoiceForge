# ChoiceForge Benchmark v0.1

这是一个与具体模型无关的候选决策评测入口。它接收一份公开数据集和一份预测文件，
生成可保存、可复核的审计报告。模型可以是 ChoiceForge，也可以是任何能够输出候选概率、
选中候选和弃权标记的引擎。

## 运行方式

```bash
decision-model benchmark \
  --data data/release-candidate-v1/expanded.jsonl \
  --predictions runs/my-engine/predictions.json \
  --manifest configs/my-benchmark-manifest.json \
  --output runs/my-engine/benchmark-v0.json
```

预测文件可以是 JSON 数组，也可以是包含 `predictions` 数组的 JSON 对象；每条记录至少包含：

```json
{
  "id": "case-001",
  "choice_id": "candidate_a",
  "probabilities": {
    "candidate_a": 0.82,
    "candidate_b": 0.18
  },
  "requires_review": false
}
```

`probabilities` 必须覆盖该样本的全部候选，数值必须归一化。评测会检查预测 ID 与数据集
完全一致，避免漏测或额外样本混入结果。

## 输出内容

报告包含：

- 选中候选的准确率和概率最大候选的准确率；
- NLL、Brier 和十桶 ECE；
- 明确输出 `requires_review` 的比例；
- 置信度阈值下的 coverage-risk 曲线；
- 按 `family`、`decision_type` 和 `evaluation_regime` 分组的结果；
- 数据文件、预测文件和可选 manifest 的 SHA-256；
- 可选的软目标交叉熵和 Brier。

默认置信度阈值为 `0.5/0.7/0.8/0.9/0.95`，可以重复传入 `--threshold` 覆盖：

```bash
decision-model benchmark \
  --data data.jsonl \
  --predictions predictions.json \
  --output report.json \
  --threshold 0.8 \
  --threshold 0.9
```

## Manifest 与预注册

manifest 至少可以记录评测名称、协议版本和数据 SHA-256：

```json
{
  "name": "public-knowledge-v1",
  "version": "0.1",
  "data_sha256": "<sha256 of data.jsonl>"
}
```

如果 manifest 提供 `data_sha256`，运行器会在计算指标前强制校验。正式评测还应在打开标签或
最终结果前登记模型 revision、候选变换、随机种子、环境和任务族留出规则。

## 解释边界

Benchmark v0.1 是审计协议，不是新的训练目标。它不会自动证明模型具备世界知识，也不会把
单一任务族上的提升解释成通用能力。正式结论必须同时查看任务族分组、校准结果和弃权风险，
并保留失败结果。

现有的 decoder/encoder 三种子研究可以通过
`scripts/benchmark_architecture_v1.py` 转换为这套协议的报告；对应的机器可读结果见
[`architecture-benchmark-v1.json`](evidence/architecture-benchmark-v1.json)。这一步是对已冻结
研究结果的统一审计，不会把旧结果伪装成新训练实验。
