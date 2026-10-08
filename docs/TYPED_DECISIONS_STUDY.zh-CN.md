# Typed Decisions 软目标与整工作流留出方案

## 结论边界

Typed Decisions 可以补齐本项目过去只有硬标签的缺口：每道 Boolean、Choice、Score
问题都带完整候选概率。它适合检查概率学习、校准和跨工作流迁移，但标签来自公开教师
模型的三次采样平均，衡量的是**对教师分布的一致性**，不是业务动作的客观成功率。

当前完成的是数据、训练和评测链路，以及一次 13 个案例的真实 2B 四组先导实验。
先导实验验证硬目标、软目标、有序 RPS 和独立 RLCD 风格估计都能经过同一 LoRA
训练图、验证集温度拟合和结果持久化；样本与种子太小，不构成模型效果证据。

## 固定公开来源

- 数据集：`LocalLLaMA/typed-decisions`
- 版本：`ea9306458d6e9563628369a3d1e72e362fb381d2`
- 许可证：Apache-2.0
- 上游训练 Parquet：598,824 bytes，SHA-256
  `46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5`
- 上游测试 Parquet：222,140 bytes，SHA-256
  `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`

下载器只接受上述固定文件。模型输入只包含 state、问题说明和候选语义；生成用 latent
factors、`label_agreement`、教师解释和其他上游列全部排除。

## 为什么做四折整工作流留出

数据共有四个工作流：agent trace、客服、发票、安全事件。每折选择一个完整工作流作为
未见任务：它的 300 个上游训练 case 全部丢弃，只保留 100 个官方测试 case；其余三个
工作流各用 240 个 case 训练、60 个 case 验证，再用各自 100 个官方测试 case 测量
已见任务的新样本表现。

每个 case 有 5 道题并共享 `group_id`，不会跨 split。每折最终为：

| split | case | 问题 |
|---|---:|---:|
| train | 720 | 3,600 |
| validation | 180 | 900 |
| test | 400 | 2,000 |

四个工作流轮流留出，最终必须报告所有折，不能在看分后挑一折作为“零样本结果”。

## 软目标训练与指标

每行保留 `target_probabilities`，按候选 ID 存储。每次候选乱序后，程序重新按 ID 对齐
概率，避免概率质量跟着位置移动。兼容旧硬标签：旧行继续走原整数目标路径，损失表达式
和历史行为不变；混合批次里的硬标签会转成 one-hot。

对目标分布 `p` 与模型分布 `q`，训练使用：

```text
soft CE = -Σ p_k log q_k
soft Brier = Σ (q_k - p_k)²
```

评测同时报告 soft CE、soft Brier、总变差距离、教师 argmax 一致率和教师目标熵。
温度只在 validation 的 soft CE 上选择；test 不参与温度或模型选择。ECE 将模型最大
概率与目标分布在该预测类别上的概率质量比较，不能解释成真实世界错误率校准。

## 已通过的链路验证

- 147 项单元测试通过，包括概率键集合、归一化、首个 argmax、候选乱序对齐、soft
  proper loss、有序 RPS、soft 校准、整工作流隔离和缺省 Boolean 语义。
- MiniCPM5-2B-Base CPU 先导实验的四组都完成 8 次真实优化更新；四组起始参数、
  样本/候选排列、encoder 工作量和更新数一致。
- `R-rlcd-loo` 在 logits 上采四个高斯动作，用独立留一基线且不做随机优势标准差
  归一化；真实风险与策略梯度 surrogate 分开记录。
- 完整说明见 [`TYPED_DECISIONS_RLCD_PILOT.zh-CN.md`](TYPED_DECISIONS_RLCD_PILOT.zh-CN.md)，
  机器证据见 [`typed-decisions-rlcd-pilot-v1.json`](evidence/typed-decisions-rlcd-pilot-v1.json)。

## 下一阶段比较

完整训练前先冻结三种子和四折协议。至少比较：硬 argmax 目标、教师软分布目标，以及在
同一软目标上增加的独立 RLCD 机制；所有方法使用相同底座、样本、更新预算和候选排列。
主要判断是未见工作流 soft CE/Brier 与 argmax 一致率是否多种子稳定改善，同时已见
工作流不能显著退化。当前先导实验不能用于选择方法、选择权重或宣称复现任何私有方案。
