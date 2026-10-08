# Typed Decisions：固定 2B 表示的四折三种子目标筛选

本实验固定 MiniCPM5-2B-Base 候选表示，只训练共享候选头，从而先隔离监督目标与梯度估计。
通过只代表有资格进入端到端 LoRA 验证，不代表最终模型收益。

## 预注册晋级结果

| 对照 | 未见 CE 胜出端点 | 未见 CE 均值差 | 已见 CE 均值差 | 未见 Brier 均值差 | 晋级 LoRA |
|---|---:|---:|---:|---:|---|
| soft_minus_hard | 9/12 | -0.10806 | -0.05660 | -0.04109 | 否 |
| typed_rps_minus_soft | 8/12 | -0.00205 | +0.00141 | -0.00121 | 否 |
| rlcd_loo_minus_typed_rps | 1/12 | +0.01035 | -0.00061 | +0.00539 | 否 |

## 绝对结果

| 方法 | 未见 soft CE | 已见 soft CE | 未见 argmax agreement |
|---|---:|---:|---:|
| H-hard | 1.37215 | 1.08958 | 36.4% |
| S-soft | 1.26409 | 1.03298 | 38.1% |
| T-typed-rps | 1.26204 | 1.03439 | 38.0% |
| R-rlcd-loo | 1.27239 | 1.03378 | 37.7% |

## 判定细节

- `soft_minus_hard`：未通过 `no_workflow_mean_ce_regresses_over_0_01`；各工作流未见 CE 均值差为 agent_trace_observability -0.26021、customer_service -0.05167、invoice_processing -0.13242、security_incidents +0.01208。
- `typed_rps_minus_soft`：未通过 `heldout_ce_wins_at_least_9_of_12`；各工作流未见 CE 均值差为 agent_trace_observability -0.00664、customer_service -0.00012、invoice_processing +0.00216、security_incidents -0.00363。
- `rlcd_loo_minus_typed_rps`：未通过 `heldout_ce_wins_at_least_9_of_12, mean_heldout_ce_improves, no_workflow_mean_ce_regresses_over_0_01, mean_heldout_brier_nonregression`；各工作流未见 CE 均值差为 agent_trace_observability +0.00902、customer_service +0.00194、invoice_processing +0.02177、security_incidents +0.00867。

三种增量方法均未通过预注册规则，因此没有方法进入完整 LoRA。
本结果支持在可直接求导的教师概率监督上继续使用直接 proper loss；它不能否定 RLCD 在真实、不可微结果反馈上的价值。

## 训练梯度诊断

| 方法 | 端点平均裁剪前范数 | 端点最大范数 | 被裁剪更新 | 总更新 |
|---|---:|---:|---:|---:|
| H-hard | 0.46291 | 1.23148 | 10 | 2052 |
| S-soft | 0.27270 | 0.81276 | 0 | 2052 |
| T-typed-rps | 0.32579 | 1.14603 | 1 | 2052 |
| R-rlcd-loo | 0.50729 | 1.57170 | 21 | 2052 |

## 边界

四个工作流各自整族留出，种子固定为 42/43/44。48 个头和验证结果全部校验并锁定后，程序才首次解析密封 test 文件。
特征缓存来自关闭 LoRA 适配器的固定公开底座；因此本实验不能发现目标函数与表示学习之间的相互作用。
下一阶段只允许预注册通过的增量方法与其直接对照进入完整 LoRA；若没有增量方法通过，则保留更简单的直接监督基线。

机器证据：`docs/evidence/typed-decisions-objective-screen-v1.json`。
