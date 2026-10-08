# Release Candidate v1 结果

**结论：未通过全部冻结门槛，不产生发布候选。**

## 核心结果

| 范围 | Control 平均准确率 | Expanded 平均准确率 | 配对胜出种子 | 最差任务×种子退化 |
|---|---:|---:|---:|---:|
| 已见公共任务 | 0.6226 | 0.7243 | 3/3 | -0.0469 |
| 新任务族盲测 | 0.6415 | 0.6519 | 2/3 | -0.0312 |

## 冻结门槛

| 门槛 | 结果 |
|---|---|
| `blind.maximum_task_seed_regression` | FAIL |
| `blind.mean_accuracy_strictly_higher` | PASS |
| `blind.mean_probability_loss_not_worse` | PASS |
| `blind.paired_wins_at_least_two` | PASS |
| `contract.all_endpoints` | PASS |
| `prior.context_advantage.seed1701.bigbench-causal_judgment` | FAIL |
| `prior.context_advantage.seed1701.bigbench-formal_fallacies_syllogisms_negation` | FAIL |
| `prior.context_advantage.seed1701.bigbench-movie_recommendation` | PASS |
| `prior.context_advantage.seed1702.bigbench-causal_judgment` | PASS |
| `prior.context_advantage.seed1702.bigbench-formal_fallacies_syllogisms_negation` | PASS |
| `prior.context_advantage.seed1702.bigbench-movie_recommendation` | PASS |
| `prior.context_advantage.seed1703.bigbench-causal_judgment` | PASS |
| `prior.context_advantage.seed1703.bigbench-formal_fallacies_syllogisms_negation` | PASS |
| `prior.context_advantage.seed1703.bigbench-movie_recommendation` | PASS |
| `prior.cross_seed_null_js` | PASS |
| `seen.maximum_task_seed_regression` | FAIL |
| `seen.mean_probability_loss_not_worse` | PASS |
| `seen.paired_wins_at_least_two` | PASS |
| `seen.retained_task_mean_not_worse` | PASS |

## 解释边界

- 盲测任务只在六个端点和审计结果锁定后物化，未参与训练、校准、早停或选参。
- Typed Decisions 是公开教师标签，衡量的是教师一致性，不等同于真实业务行动成功率。
- 本协议只能控制本项目内的调参泄漏，无法证明基础模型预训练语料不含盲测任务。
- 即使复制种子表现更好，也不能替换预先指定的 seed1701。

机器可复核证据：`docs/evidence/release-candidate-v1.json`
