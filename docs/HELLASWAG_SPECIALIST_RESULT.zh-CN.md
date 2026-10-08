# HellaSwag 多随机种子专家训练

在固定 2B 表示和父决策头上，为 HellaSwag 训练独立残差专家。验证集已消费，结果只用于机制筛选。

| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 | 最小上下文优势差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.0142 | 0.7695 | 0.6758 | 0.6758 | +0.0000 | -0.0106 | +0.0009 |
| 43 | +0.0105 | 0.7070 | 0.6758 | 0.6758 | +0.0000 | -0.0078 | -0.0002 |
| 44 | +0.0121 | 0.6602 | 0.6758 | 0.6758 | +0.0000 | -0.0089 | -0.0008 |

## 判定

至少一个门槛失败；HellaSwag 暂不计为正效用专家族。

逐项门槛：`mechanism.initial_exact_each_seed`=PASS, `utility.combined_each_seed`=FAIL, `utility.each_source_each_seed`=FAIL, `accuracy.combined_each_seed`=FAIL, `nll.combined_each_seed`=FAIL, `nll.each_source_each_seed`=FAIL, `context.each_source_each_seed`=FAIL

机器证据：`docs/evidence/hellaswag-specialist-v1.json`
