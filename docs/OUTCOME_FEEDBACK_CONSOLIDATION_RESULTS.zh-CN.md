# 结果反馈汇总实验结果

这是在已经消耗的验证集上进行的开发跟进，不是新鲜确认试验。官方 MBPP 测试段仍未解析，实验没有新增环境查询。

A 臂只从已记录的选择动作反馈中汇总确定性结果；只有某行所有候选都被探索观测后，才对该行的全部并列最优候选进行学习。

| 端点 | 完整/不完整训练行 | R regret | D regret | A regret | A-R | A-D |
|---|---:|---:|---:|---:|---:|---:|
| seed42-decoder | 203/0 | 0.0444 | 0.0667 | 0.0444 | +0.0000 | -0.0222 |
| seed43-decoder | 202/1 | 0.0593 | 0.0593 | 0.0444 | -0.0148 | -0.0148 |
| seed44-decoder | 201/2 | 0.0815 | 0.0815 | 0.0815 | +0.0000 | +0.0000 |

| 端点 | R 期望奖励 | D 期望奖励 | A 期望奖励 | A-R | A-D | A 最优集命中 |
|---|---:|---:|---:|---:|---:|---:|
| seed42-decoder | 0.9370 | 0.9313 | 0.9313 | -0.0056 | +0.0000 | 0.9556 |
| seed43-decoder | 0.9216 | 0.9295 | 0.9408 | +0.0192 | +0.0112 | 0.9556 |
| seed44-decoder | 0.9153 | 0.9089 | 0.9112 | -0.0041 | +0.0022 | 0.9111 |

## 预注册判定

`advances_to_confirmatory_protocol = false`

- `A_vs_R.passes`: false
  - `mean_regret_lower`: true
  - `seed_regret_wins`: false
  - `maximum_single_seed_regret_regression`: true
  - `mean_expected_reward_does_not_decline`: true
- `A_vs_D.passes`: true
  - `mean_regret_lower`: true
  - `seed_regret_wins`: true
  - `maximum_single_seed_regret_regression`: true
  - `mean_expected_reward_does_not_decline`: true
- `integrity_passes`: true

## 解释

A 的三种子平均 regret 为 0.0568，R 为 0.0617，D 为 0.0691。A 相对 D 是 2 胜 1 平，但相对 R 只有 1 胜 2 平，没有达到“至少两个种子严格获胜”的规则。

A 的平均策略期望奖励为 0.9277，R 为 0.9246，D 为 0.9233。这说明汇总确定性反馈值得继续研究，但当前证据不足以触发官方测试或完整 LoRA 训练。

A 在 100 个优化 epoch 时已将 seed42/43 的验证 regret 降至 0.0444，seed44 在 100 epoch 为 0.0741、200 epoch 回到 0.0815。后续应预先冻结训练时长或只依赖训练信号的停止规则，不能继续用当前验证曲线调参后冒充新鲜证据。
