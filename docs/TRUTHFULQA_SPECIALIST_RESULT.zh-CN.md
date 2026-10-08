# TruthfulQA 多随机种子专家训练

在固定 2B 表示和父决策头上，为 TruthfulQA 训练独立残差专家。验证集已消费，结果只用于机制筛选。

| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.3135 | 0.5391 | 0.5312 | 0.5469 | +0.0156 | -0.2303 |
| 43 | +0.3355 | 0.5352 | 0.5312 | 0.5508 | +0.0195 | -0.2461 |
| 44 | +0.3077 | 0.5508 | 0.5312 | 0.5469 | +0.0156 | -0.2262 |

## 判定

全部门槛通过；TruthfulQA 可计为一个已消费的正效用专家族。

逐项门槛：`mechanism.initial_exact_each_seed`=PASS, `utility.combined_each_seed`=PASS, `utility.each_source_each_seed`=PASS, `accuracy.combined_each_seed`=PASS, `nll.combined_each_seed`=PASS, `nll.each_source_each_seed`=PASS

机器证据：`docs/evidence/truthfulqa-specialist-v1.json`
