# ARC 多随机种子专家训练

在固定 2B 表示和父决策头上，为 ARC 训练独立残差专家。验证集已消费，结果只用于机制筛选。

| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.0207 | 0.8307 | 0.8073 | 0.8151 | +0.0078 | -0.0167 |
| 43 | +0.0224 | 0.8151 | 0.8073 | 0.8151 | +0.0078 | -0.0177 |
| 44 | +0.0196 | 0.8203 | 0.8073 | 0.8151 | +0.0078 | -0.0155 |

## 判定

至少一个门槛失败；ARC 暂不计为正效用专家族。

逐项门槛：`mechanism.initial_exact_each_seed`=PASS, `utility.combined_each_seed`=FAIL, `utility.each_source_each_seed`=PASS, `accuracy.combined_each_seed`=FAIL, `nll.combined_each_seed`=FAIL, `nll.each_source_each_seed`=PASS

机器证据：`docs/evidence/arc-specialist-v1.json`
