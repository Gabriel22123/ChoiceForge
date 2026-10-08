# CLINC 多随机种子专家训练

在固定 2B 表示和父决策头上，为 CLINC 训练独立残差专家。验证集已消费，结果只用于机制筛选。

| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.0147 | 0.9048 | 0.9762 | 0.9762 | +0.0000 | -0.0179 |
| 43 | +0.0153 | 0.9048 | 0.9762 | 0.9762 | +0.0000 | -0.0182 |
| 44 | +0.0158 | 0.9048 | 0.9762 | 0.9762 | +0.0000 | -0.0186 |

## 判定

全部门槛通过；CLINC 可计为一个已消费的正效用专家族。

逐项门槛：`mechanism.initial_exact_each_seed`=PASS, `utility.combined_each_seed`=PASS, `utility.each_source_each_seed`=PASS, `accuracy.combined_each_seed`=PASS, `nll.combined_each_seed`=PASS, `nll.each_source_each_seed`=PASS

机器证据：`docs/evidence/clinc-specialist-v1.json`
