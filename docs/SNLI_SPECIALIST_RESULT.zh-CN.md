# SNLI 多随机种子专家训练

在固定 2B 表示和父决策头上，为 SNLI 训练独立残差专家。验证集已消费，结果只用于机制筛选。

| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.0693 | 0.4479 | 0.6667 | 0.6667 | +0.0000 | -0.0442 |
| 43 | +0.0721 | 0.4167 | 0.6667 | 0.6771 | +0.0104 | -0.0453 |
| 44 | +0.0720 | 0.4479 | 0.6667 | 0.6771 | +0.0104 | -0.0446 |

## 判定

全部门槛通过；SNLI 可计为一个已消费的正效用专家族。

逐项门槛：`mechanism.initial_exact_each_seed`=PASS, `utility.combined_each_seed`=PASS, `utility.each_source_each_seed`=PASS, `accuracy.combined_each_seed`=PASS, `nll.combined_each_seed`=PASS, `nll.each_source_each_seed`=PASS

机器证据：`docs/evidence/snli-specialist-v1.json`
