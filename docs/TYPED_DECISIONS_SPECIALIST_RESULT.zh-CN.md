# Typed Decisions 多随机种子专家训练

在固定 2B 表示和父决策头上，为 Typed Decisions 训练独立残差专家。验证集已消费，结果只用于机制筛选。

| 种子 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 准确率差 | NLL 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.0595 | 0.4813 | 0.6875 | 0.6406 | -0.0469 | -0.0024 |
| 43 | -0.0708 | 0.4813 | 0.6875 | 0.6500 | -0.0375 | +0.0093 |
| 44 | -0.0702 | 0.4938 | 0.6875 | 0.6406 | -0.0469 | +0.0140 |

## 判定

至少一个门槛失败；Typed Decisions 暂不计为正效用专家族。

逐项门槛：`mechanism.initial_exact_each_seed`=PASS, `utility.combined_each_seed`=FAIL, `utility.each_source_each_seed`=FAIL, `accuracy.combined_each_seed`=FAIL, `nll.combined_each_seed`=FAIL, `nll.each_source_each_seed`=FAIL

机器证据：`docs/evidence/typed-decisions-specialist-v1.json`
