# GSM8K capacity v2 多随机种子专家训练

在固定 2B 表示和父决策头上，为 GSM8K capacity v2 训练独立残差专家。验证集已消费，结果只用于机制筛选。

| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 | 最小上下文优势差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.3521 | 0.7188 | 0.3242 | 0.5625 | +0.2383 | -0.2727 | +0.1804 |
| 43 | +0.3476 | 0.7500 | 0.3242 | 0.5664 | +0.2422 | -0.2675 | +0.1742 |
| 44 | +0.3558 | 0.7383 | 0.3242 | 0.5430 | +0.2188 | -0.2760 | +0.1763 |

## 判定

全部门槛通过；GSM8K capacity v2 可计为一个已消费的正效用专家族。

逐项门槛：`mechanism.initial_exact_each_seed`=PASS, `utility.combined_each_seed`=PASS, `utility.each_source_each_seed`=PASS, `accuracy.combined_each_seed`=PASS, `nll.combined_each_seed`=PASS, `nll.each_source_each_seed`=PASS, `context.each_source_each_seed`=PASS

机器证据：`docs/evidence/gsm8k-specialist-capacity-v2.json`
