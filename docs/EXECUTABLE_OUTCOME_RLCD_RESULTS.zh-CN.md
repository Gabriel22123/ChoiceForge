# 可执行结果反馈筛选结果

本报告只比较相同日志反馈下的 IPS-LOO 与双重稳健方法。官方 MBPP 测试段仍未解析。

| 端点 | 初始 regret | H 硬标签 | O 全信息 | R IPS-LOO | D 双重稳健 |
|---|---:|---:|---:|---:|---:|
| seed42-decoder | 0.1259 | 0.1037 | 0.1037 | 0.1037 | 0.1037 |
| seed43-decoder | 0.1037 | 0.0815 | 0.1259 | 0.1259 | 0.1259 |
| seed44-decoder | 0.1259 | 0.1259 | 0.1259 | 0.1259 | 0.1259 |

| 端点 | R regret | D regret | Δ regret | R 期望奖励 | D 期望奖励 | Δ 期望奖励 |
|---|---:|---:|---:|---:|---:|---:|
| seed42-decoder | 0.1037 | 0.1037 | +0.0000 | 0.8560 | 0.8572 | +0.0012 |
| seed43-decoder | 0.1259 | 0.1259 | +0.0000 | 0.8470 | 0.8468 | -0.0002 |
| seed44-decoder | 0.1259 | 0.1259 | +0.0000 | 0.8366 | 0.8387 | +0.0021 |

| 端点 | R 权重 P95 / max / ESS | D 权重 P95 / max / ESS | D 奖励头 MSE |
|---|---:|---:|---:|
| seed42-decoder | 4.13 / 6.23 / 0.358 | 4.09 / 6.13 / 0.359 | 0.1989 |
| seed43-decoder | 4.07 / 6.34 / 0.368 | 4.02 / 6.44 / 0.367 | 0.1928 |
| seed44-decoder | 3.55 / 6.01 / 0.408 | 3.60 / 6.05 / 0.404 | 0.1806 |

## 晋级判断

`advances_to_lora_validation = false`

- `mean_regret_strictly_lower`: false
- `seed_regret_wins`: false
- `worst_seed_regret_within_limit`: true
- `mean_policy_expected_reward_does_not_decline`: true
- `no_integrity_violations`: true

## 解释

D 与 R 在三套端点上选择了完全相同的 argmax，因而 regret 没有任何改善。D 的策略期望奖励平均变化为 +0.0010，只改变了概率锐度；seed43 还小幅下降。
seed42 的 R/D 相对初始 regret 改善 0.0222，seed43 退化 0.0222，seed44 不变。有限反馈训练没有形成跨端点一致的选择收益。重要性权重保持有限，失败不能简单归因于权重爆炸。

全信息 O 和硬标签 H 是解释性参照，不参与同信息量晋级结论。H 的三种子平均 regret 更低，但它读取完整奖励表并把并列最优压成单标签，不能当作有限反馈胜出。
官方测试保持封存；没有通过开发集规则的方法不得用测试结果补救。
