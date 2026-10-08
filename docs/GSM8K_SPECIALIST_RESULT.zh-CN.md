# GSM8K 多随机种子专家训练

在固定 2B 表示和父决策头上，为 GSM8K 训练独立残差专家。验证集已消费，结果只用于机制筛选。

| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 | 最小上下文优势差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.0197 | 0.5273 | 0.3828 | 0.3945 | +0.0117 | -0.0114 | +0.0052 |
| 43 | +0.0233 | 0.5312 | 0.3828 | 0.4023 | +0.0195 | -0.0134 | +0.0063 |
| 44 | +0.0160 | 0.5469 | 0.3828 | 0.3906 | +0.0078 | -0.0095 | +0.0048 |

## 判定

至少一个门槛失败；GSM8K 暂不计为正效用专家族。

逐项门槛：`mechanism.initial_exact_each_seed`=PASS, `utility.combined_each_seed`=FAIL, `utility.each_source_each_seed`=FAIL, `accuracy.combined_each_seed`=FAIL, `nll.combined_each_seed`=FAIL, `nll.each_source_each_seed`=FAIL, `context.each_source_each_seed`=PASS

机器证据：`docs/evidence/gsm8k-specialist-v1.json`
