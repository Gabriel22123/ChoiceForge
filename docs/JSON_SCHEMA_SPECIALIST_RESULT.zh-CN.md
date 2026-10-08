# JSON Schema 多随机种子专家训练

在固定 2B 表示和父决策头上，为 JSON Schema 训练独立残差专家。验证集已消费，结果只用于机制筛选。

| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.0952 | 0.5000 | 0.4048 | 0.4524 | +0.0476 | +0.0737 |
| 43 | -0.0942 | 0.5000 | 0.4048 | 0.4524 | +0.0476 | +0.0739 |
| 44 | -0.1010 | 0.5238 | 0.4048 | 0.4524 | +0.0476 | +0.0775 |

## 判定

至少一个门槛失败；JSON Schema 暂不计为正效用专家族。

逐项门槛：`mechanism.initial_exact_each_seed`=PASS, `utility.combined_each_seed`=FAIL, `utility.each_source_each_seed`=FAIL, `accuracy.combined_each_seed`=PASS, `nll.combined_each_seed`=FAIL, `nll.each_source_each_seed`=FAIL

机器证据：`docs/evidence/json-schema-specialist-v1.json`
