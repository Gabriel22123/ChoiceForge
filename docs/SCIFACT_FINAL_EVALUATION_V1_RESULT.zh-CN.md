# SciFact 封存任务族最终评测

在未参与开发的科学主张判断任务族上，一次性检验冻结的三组专家组合。

| 种子 | 准确率差 | 交叉熵差 | Brier 差 | proper 收益 | 上下文收益差 | 平均组合步长 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.0000 | -0.0730 | -0.0149 | +0.0805 | +0.0809 | 0.9647 |
| 43 | +0.0000 | -0.0730 | -0.0149 | +0.0805 | +0.0809 | 0.9647 |
| 44 | +0.0000 | -0.0739 | -0.0150 | +0.0814 | +0.0818 | 0.9647 |

## 完整性

- 封存样本：`209` 行；原始与转换文本不进入发布包。
- 每一行、每个种子均通过严格 JSON 输出契约与候选顺序等变性检查。
- 决策稳定投影保证所有硬选择与冻结父模型一致。

## 判定

全部预注册门槛通过。

逐项门槛：`coverage.seeds`=PASS, `integrity.rows`=PASS, `integrity.cache`=PASS, `decision.exact_each_seed`=PASS, `calibration.cross_entropy_each_seed`=PASS, `calibration.brier_each_seed`=PASS, `utility.proper_each_seed`=PASS, `context.each_seed`=PASS, `contract.each_row_seed`=PASS, `permutation.each_row_seed`=PASS, `projection.material_each_seed`=PASS

机器证据：`docs/evidence/scifact-final-evaluation-v1.json`
