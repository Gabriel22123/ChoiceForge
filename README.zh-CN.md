# ChoiceForge：可校准决策与可复现评测研究版

项目研究知识密集型决策需要多少世界知识，以及 decoder 与 encoder 在概率质量、弃权和跨任务表现上的差异。
同时提供冻结、可审计的第三方评测协议，用来复现和比较其他候选决策引擎。项目名为 **ChoiceForge**；
当前 Python 包和命令仍使用 `decision_model` / `decision-model`，以保持实验复现兼容。
项目不宣称已经做到所有场景零样本，也不把纯路由能力作为主要产品定位。

首次公开源码的纳入范围、排除项和审查门槛见
[`PUBLICATION_REVIEW.md`](PUBLICATION_REVIEW.md)。

实验与复现文档索引见 [`docs/README.md`](docs/README.md)。
项目定位、评测边界和后续路线见 [`docs/POSITIONING.zh-CN.md`](docs/POSITIONING.zh-CN.md)。

## 项目的研究亮点

**从公开 RLCD 的校准决策目标出发，拆解原理，再用可复现实验检验。**
重点包括：概率目标与优化算法的区别、策略梯度与直接求导的取舍、训练噪声与
推理概率是否一致，以及额外校准/一致性目标是否真的改善新任务判断。
这些是需要数据回答的问题，不能把已有方法的组合包装成新算法。

详见 [原理解构](docs/RESEARCH.zh-CN.md)与[受控实验协议](docs/RESEARCH.md)。
所有训练只使用许可明确的公开数据；后续使用者可以替换任务数据继续训练。

<!-- release-status:start -->
**ChoiceForge 源码与已验证参数包已公开发布。** 三个冻结种子在密封 SciFact 任务族上保持父模型的硬选择，同时将交叉熵改善 0.0730–0.0739、Brier 改善 0.0149–0.0150。公开的 55 MB 参数包位于 `models/choiceforge-local-bundle-v1/`；MiniCPM 2B 基础模型按固定 revision 从上游单独下载。详见[`本地模型包说明`](docs/LOCAL_MODEL_BUNDLE_V1.zh-CN.md)、[`最终结果`](docs/SCIFACT_FINAL_EVALUATION_V1_RESULT.zh-CN.md)和[`发布审查`](PUBLICATION_REVIEW.md)。
<!-- release-status:end -->

## 快速使用

```bash
git clone https://github.com/Gabriel22123/ChoiceForge.git
cd ChoiceForge
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model]'
choiceforge predict-merged \
  --bundle models/choiceforge-local-bundle-v1 \
  --input examples/request.json \
  --device cpu
```

模型输入是任务、上下文和任意数量候选项；输出由代码固定构造为选择、概率、复核标记、
分数语义和校准状态。模型不会生成思考文本或任意 JSON，且所有结果都要求人工复核。

## 作为 Skill 使用

仓库内置 `skills/choiceforge`，可以把自然语言请求路由到推理、训练、评测和审计流程。
将这个目录复制到 Codex 的 skills 目录后，即可用 `$choiceforge` 调用；具体输入格式和命令见
其中的 `references/workflow.md`。

## 如何训练

最小公开数据训练路径：

```bash
python -m pip install -e '.[data,model]'
decision-model train \
  --data data/release-candidate-v1/expanded.jsonl \
  --config configs/eurobert-public-pilot.json \
  --output runs/my-public-adapter \
  --device auto
```

完整研究路线依次执行公开数据准备与审计、各来源专家训练和特征缓存、决策稳定专家组合训练，
最后运行 `scripts/build_canonical_merged_adapter.py` 构建单一适配器。每次实验都会保存配置、
随机种子、样本 ID、哈希和指标；可用 `--init-from` 从已有适配器继续训练。

## 我们做了什么

- 用公开 MiniCPM5-2B decoder 与 EuroBERT-2.1B encoder 做同预算、多种子对照，选择 decoder 路线。
- 将 RLCD 思路拆成可检验组件：结果/概率监督、proper loss、全上下文与空上下文优势、来源专家、
  来源均衡组合，以及保持父模型硬选择的决策稳定投影。
- 将六个残差专家精确合并为单一适配器，在 5,765 条公共缓存样本上最大概率差为 `1.43e-6`。
- 在 209 条答案盲选的 SciFact 公共样本上，三种子均保持硬选择并改善概率质量和上下文利用。
- 增加与模型无关的 Benchmark v0.1，统一评测准确率、概率校准、弃权风险和审计哈希。

项目接口面向任务、上下文和候选项，不是固定的业务分类器；“通用接口”不等于已经验证了通用能力。

仓库只包含公开来源、许可证、代码和可复现实验证据。第三方基础模型权重不重复上传，使用者按公开
上游许可证自行下载。完整边界见[`发布审查`](PUBLICATION_REVIEW.md)和
[`第三方许可`](THIRD_PARTY.md)。

**最新保留能力筛选：**同一初始化和训练顺序下，分别测试了旧样本加权，以及
“旧样本加权 + 上下文优势约束”。后者优于纯加权端点（已见准确率 72.17% 对
71.66%，NLL 0.847 对 0.874），但仍比父 Expanded 低 1.96 个百分点；SNLI
相对 Control 退化 5.21 个百分点，形式谬误的上下文优势仍为负。两组单种子方案
均不晋级；已读取的诊断集只能用于开发，后续发布结论必须重新密封新任务。
见[`筛选报告`](docs/RETENTION_CONTEXT_SCREEN.zh-CN.md)。

**热启动函数保留通过六项冻结条件中的五项，但仍不晋级。** 两个匹配组的已见
准确率均为 75.65%，比父 Expanded 高 1.53 个百分点。加入 KL 后准确率保持不变，
已见 NLL 从 0.835 改善到 0.825，已消费开发诊断准确率从 65.63% 提升到 66.15%。
唯一失败项是因果判断的上下文优势仍为负（-0.0136），因此不启动多种子或发布实验。
见[`函数保留筛选报告`](docs/FUNCTIONAL_RETENTION_SCREEN.zh-CN.md)。
聚合诊断显示空材料时 128 条因果样本全部预测 Yes，失败集中在 Yes/No 候选先验与
证据翻转覆盖，见[`因果上下文诊断`](docs/CAUSAL_CONTEXT_DIAGNOSTIC.zh-CN.md)。

**BoolQ 上下文修复筛选不晋级。** 从函数保留端点续训，加入 512 条未使用的
公开 BoolQ 样本，并用类别权重平衡 Yes/No。保留集从 72.66% 提高到 74.22%，
预测分布也从全部 Yes 变为 Yes/No 各 64 条；但 +1.56 个百分点未达预设的
+2 门槛，已见 NLL 恶化 0.0128，因果判断上下文优势仍为负（-0.0197）。这说明
类别先验可修正，但单纯增加阅读理解数据不足以修复跨任务证据使用。见
[`BoolQ 上下文修复筛选`](docs/BOOLQ_CONTEXT_REPAIR.zh-CN.md)。

**反事实证据翻转可学，但仍未跨任务迁移。** 项目内确定性生成器构造了同一问题下
只改证据的 Yes/No 成对样本。候选模型在全新主体的生成验证集上从 88.28%
提高到 100%，成对全对率从 76.56% 提高到 100%；但已消费自然任务准确率下降
1.04 个百分点，因果上下文优势恶化到 -0.0628，已见 NLL 也恶化 0.0187。这是
格式内规则学习，不是通用证据判断，因此不晋级。见
[`反事实证据翻转筛选`](docs/COUNTERFACTUAL_EVIDENCE_SCREEN.zh-CN.md)。

**自然八候选科学证据取得目前最强的任务内提升，但仍未保住完整决策函数。** 冻结的
QASC 筛选加入 512 条 CC-BY-4.0 训练题，隐藏原始答案字母，并按不读取答案的候选位置
做等量抽样。保留集准确率从 82.03% 提高到 93.75%，NLL 从 0.601 降到 0.167，
上下文优势从 1.540 提高到 2.004，五项 QASC 门槛全部通过；但已见 NLL 恶化
0.0435，已消费开发集准确率下降 0.78 个百分点，因果判断上下文优势仍为负。冻结的
11 项门槛通过 6 项，因此单种子端点不晋级。见
[`QASC 八候选证据筛选`](docs/QASC_EVIDENCE_SCREEN.zh-CN.md)。

**缩放同一条 QASC 参数更新路径不能恢复保留能力。** 预先固定的 25%、50%、75%
插值都保留至少 3.91 个百分点的 QASC 收益，并通过换序和 ID 审计。25% 点勉强把已见
NLL 控制在 +0.01 内，但开发准确率仍超界 0.02 个百分点，因果材料优势下降 0.114；
更大步长让已见 NLL 继续恶化，并把因果下降扩大到 0.132/0.152。没有插值点被选为
训练复现目标；下一方案必须同时保留有材料函数与空材料函数。见
[`QASC 参数路径保留诊断`](docs/QASC_RETENTION_PATH.zh-CN.md)。

**蒸馏父模型的空材料函数可以减少总体漂移，但不能修复跨任务材料响应。** 两组匹配的
半权重 QASC 实验使用相同父模型、数据和候选顺序。增加空材料 KL 后，QASC 准确率
保持 90.63%，已见 NLL 从 0.8569 改善到 0.8474，已消费开发准确率从 65.36%
提高到 65.63%；但两项绝对门槛仍失败，因果材料优势也从 -0.03921 微降到
-0.03960，没有达到预设的 +0.01 机制改善。该方法只保留为部分有效的正则证据，不选为
下一路线。见[`QASC 双函数蒸馏筛选`](docs/QASC_DUAL_FUNCTION_SCREEN.zh-CN.md)。

发布训练数据已作为精确 JSONL 纳入源码范围，包含逐行公开来源、版本和许可证，
Control/Expanded 分别为 3,008/5,056 行；机器审计确认没有私有来源、本机路径或
盲测行。见[`数据再分发审计`](docs/RELEASE_DATA_AUDIT.zh-CN.md)。若最终全部门槛通过，
发布包将同时提供[独立使用、精确复训和继续微调指南](docs/RELEASE_QUICKSTART.md)；
盲测问题、标签、logits 和逐案例预测不会随包分发。

项目采用独立实现，只复现公开可描述的校准决策目标，不声称还原任何私有训练方案。

**公开软目标链路已打通：**训练器现在可直接学习完整候选概率，同时保持旧硬标签路径
不变。固定版本的 Typed Decisions 被整理成四折整工作流留出：每个工作流都会有一折
完全不进入训练和校准，只用于最终测试。真实 MiniCPM 2B 已从相同初始化跑通硬目标、
软目标、有序 RPS 和独立留一基线的 RLCD 风格估计四组对照。它只有 13 个案例、一个
种子，只验证链路和数值方向，不让任何方法晋级。见
[`Typed Decisions 方案`](docs/TYPED_DECISIONS_STUDY.zh-CN.md)与
[`四组先导报告`](docs/TYPED_DECISIONS_RLCD_PILOT.zh-CN.md)。
固定 2B 表示的四折三种子筛选现已完成；48 个头与 validation 全部锁定后才解析
密封 test。Soft 相对 hard 的未见 CE 均值改善 0.108、胜出 9/12，但 security
工作流平均回退 0.01208，超过预注册 0.01 上限；RPS 只胜出 8/12；RLCD 留一估计
只胜出 1/12，并使未见 CE 与 Brier 变差。因此三种增量均不晋级完整 LoRA，继续
保留更简单的直接监督基线。见[`正式筛选结果`](docs/TYPED_DECISIONS_OBJECTIVE_SCREEN.zh-CN.md)
与[`固定实验设计`](docs/TYPED_DECISIONS_OBJECTIVE_SCREEN_DESIGN.zh-CN.md)。

**真实结果反馈已与教师概率蒸馏分开验证。** 公开 MBPP 程序和确定性变异提供
真实通过/失败结果。预注册的 IPS 和双重稳健筛选未改变验证集决策后，新的固定日志开发
实验只汇总探索时实际观测到的候选结果，并对全部并列最优候选学习。它将三个种子的训练
regret 降为 0，验证 regret 在所有端点都不差于两条回放基线；但相对 IPS 只有一个种子
严格获胜，未达到两种子规则。因此不打开官方 MBPP 测试，也不进入完整 LoRA。见
[`实验设计`](docs/EXECUTABLE_OUTCOME_RLCD_DESIGN.zh-CN.md)、
[`首轮筛选`](docs/EXECUTABLE_OUTCOME_RLCD_RESULTS.zh-CN.md)和
[`反馈汇总结果`](docs/OUTCOME_FEEDBACK_CONSOLIDATION_RESULTS.zh-CN.md)。

**三种子上下文优势约束未晋级。** 该约束要求真实材料相对空材料提高正确候选
log-probability。它在三个种子中都降低 margin shortfall，新冻结的 BIG-bench
disambiguation_qa 也有小幅总体改善；但平均 gold gain 在三个种子中都下降，
已见任务的准确率/NLL 预注册条件也失败。因此不进入默认训练路线。见
[冻结实验报告](docs/CONTEXT_ADVANTAGE_RESULTS.zh-CN.md)。

**最新：同数据的成组与打散对照已完成 seed42/43 复核。** 768 条原始公开 SNLI 样本
组成 256 个同前提三类组，加相同的 256 条旧任务复习；每组均为两轮、256 次更新，
每次更新的三类数量和复习位置一致。seed42 成组/打散为 **146/192 对 131/192**，
seed43 为 **115/192 对 97/192**，分别高 7.8、9.4 个百分点；两种子描述性平均为
68.0% 对 59.4%。信息不足召回平均 66.4% 对 63.3%，精确率 55.7% 对 50.5%。
但 seed43 成组的“支持”仅 16/64，逐类判断仍不稳定。见[双种子报告](docs/RELATION_GROUP_MULTISEED.zh-CN.md)、
[seed42 详细报告](docs/RELATION_GROUP_RESULTS.zh-CN.md)与[冻结设计](docs/RELATION_GROUP_DESIGN.zh-CN.md)。
相对历史实验，数据、每次更新的标签平衡和轮数都改变了；
不能把历史涨分全部归因于分组。分组只影响梯度聚合，没有跨样本注意力或新增对比损失，
也没有证明所有场景零样本。

复现入口：`scripts/prepare_relation_groups.py`、
`scripts/run_relation_group_study.py --prepare-only`、同脚本 `--device mps`，
最后运行 `scripts/relation_group_report.py`。依赖已锁定的官方归档、此前公开数据和
D-independent 起点；seed43 使用 `--seed 43 --output runs/relation-group-seed43-v1`，
最后运行 `scripts/relation_group_multiseed_report.py`。四组权重均只保存在本地。
同一组证据翻转诊断在 seed42 偏向打散（14/18、2/6 对 15/18、4/6），在 seed43
反而偏向成组（13/18、1/6 对 11/18、0/6），方向没有复现。样例是已观察的自写诊断，
不是新外部基准。两个种子让成组方案值得继续研究，尚不能设为通用默认方案。

**新增：已分别隔离第二轮训练和更新内标签均衡。** 固定打散均衡批次时，第二轮把
seed42 从 76/192 提到 131/192（+28.6 个百分点），把 seed43 从 74/192 提到
97/192（+12.0 个百分点）。它同时增加 128 次更新和一次重复曝光，当前仍不能拆分
这两个作用。固定两轮、所有前提位置和旧任务位置后，强制每次更新三类各 2 条，相对
标签数量自然波动，在 seed42 为 131/192 对 69/192（+32.3 个百分点），seed43 为
97/192 对 74/192（+12.0 个百分点），两种子描述性平均 +22.1 个百分点。
自然波动组在两个种子分别塌向不同类别；均衡批次的逐类结果没有这么极端。自写证据翻转
中，自然波动/均衡为 7/18 对 15/18、5/18 对 11/18。见[第二轮报告](docs/SECOND_EPOCH_RESULTS.zh-CN.md)、
[标签均衡报告](docs/LABEL_BALANCE_RESULTS.zh-CN.md)与[冻结设计](docs/LABEL_BALANCE_DESIGN.zh-CN.md)。
这仍是已反复观察且参与训练的 SNLI，只支持当前协议下的条件结论。

复现入口为 `scripts/run_second_epoch_study.py`、`scripts/run_label_balance_study.py`
及各自报告脚本。六个固定终点可用 `scripts/export_study.py --kind relation-ablations`
打成本地复核包，不会上传远端。

**最新冻结迁移测试：SNLI 上的大幅收益没有迁移到 WinoGrande。** 首次推理前，先按
不含答案的输入哈希冻结 384 条 WinoGrande 1.1，并登记全部九个终点。均衡相对自然波动
在 seed42/43 为 +0.3、-0.3 个百分点，平均为 0；两个均衡终点相对公共起点也只有
+1.6、+0.3 个百分点，配对区间都跨 0。九个模型准确率只有 49.0%–52.1%，接近二选一
随机水平；均衡组的原始 NLL 在两个种子还都更差。模型选择官方 `option1` 内容的比例为
65.4%–77.9%，真实比例只有 50.3%。这说明当前方案能学稳训练内关系分类，但尚未形成
可证明的跨任务常识泛化。见[冻结盲测报告](docs/WINOGRANDE_BLIND_RESULTS.zh-CN.md)。
WinoGrande 不进入微调或校准，发布包也不包含其原文。

**最新公开任务扩展：PAWS 能学，但跨任务迁移不稳定。** 两个固定终点分别从
seed42/43 标签均衡关系模型出发，新增 768 条 PAWS-Wiki 和 256 条旧任务复习样本。
PAWS 测试从 51.0% 提到 74.5%、从 50.0% 提到 52.6%。新的 384 条 BoolQ 冻结集
按不含答案的哈希选取，准确率变化为 +2.6、-13.8 个百分点；原 WinoGrande 冻结集
则为 -0.8、-0.3。BoolQ 两个起点分别有 384/384、375/384 条预测 `yes`，PAWS 训练
又把两个种子推向不同方向。当前证据支持“能学习新增任务”，不支持稳定跨任务阅读
理解。见[任务扩展报告](docs/TASK_EXPANSION_RESULTS.zh-CN.md)。发布制品不包含
PAWS、BoolQ 或 WinoGrande 原文。

**共同起点任务配比实验：固定任务比例提高训练内任务准确率，但放大梯度长尾。** 四个
终点共享同一权重起点，使用 384 条 PAWS、384 条 SNLI、256 条旧任务复习样本，各曝光
一次并更新 128 次。固定每次更新 `3 PAWS + 3 SNLI + 2 复习`，相对随机混合让 SNLI
在 seed42/43 提高 +2.6/+17.7 个百分点，PAWS 提高 +1.0/+2.1。PAWS 原始 NLL 在两个
种子都略差，已观察的 BoolQ/WinoGrande 诊断方向也不一致。最大裁剪前梯度范数从
111.7 增到 8929.9、从 51.2 增到 604.4，四组所有更新都触发裁剪。见
[任务配比报告](docs/TASK_BALANCE_RESULTS.zh-CN.md)。

**最新梯度控制实验：尺度尖峰与方向冲突需要分别处理。** 六个终点从相同起点出发，
复用完全相同的行和种子内更新计划，对比原始聚合、中位数范数截断、确定性的对称冲突
投影。截断把裁剪前范数 p95 从 57.53/32.74 降到 29.67/29.25，但 PAWS 下降
3.6/0.5 个百分点；投影让 SNLI 提高 1.6/1.0、总体硬准确率提高 0.6/1.4 个百分点，
但 PAWS 为 +0.5/-3.1，最大范数仍达 154.59/371.32。PAWS、SNLI 的配对 bootstrap
区间全部跨 0，因此两个方法都不设为默认。见
[梯度控制报告](docs/TASK_GRADIENT_CONTROL_RESULTS.zh-CN.md)。下一轮将比较更柔和的
`2 × 更新内中位数` 截断与“先截断、再投影”；BoolQ/WinoGrande 仍是已观察诊断，
不是新的盲测。

**后续组合实验：准确率更稳定，但仍未完整通过晋级规则。** `2 × 中位数` 截断后再做
对称冲突投影，在 seed42/43 上让 PAWS 变化 +0.5/-0.5、SNLI 变化 +4.7/+4.2 个百分点，
四项平均提高 2.2；但四个配对区间都跨 0。合并梯度 p95 在 seed42 从 57.53 降到
31.41，在 seed43 却从 32.74 升到 37.50；WinoGrande 候选计数的跨种子差也从 10
扩大到 15。因此它是更有希望的研究候选，还不是默认方案。见
[梯度组合报告](docs/TASK_GRADIENT_COMPOSITION_RESULTS.zh-CN.md)。

**投影修正信赖域没有晋级。** 三个种子的 384 次更新都满足“修正后合并梯度不超过
同一步原始聚合范数”，证明几何约束实现正确；但 PAWS/SNLI 六项准确率相对 raw
平均下降 1.0 个百分点，最差下降 7.8 个百分点，且三个种子的梯度 p95 也没有一致
优于 raw。它约束了单步更新幅度，却没有稳定保留语义质量，因此不设为默认方法，也
不进入新的盲测。见[信赖域结果](docs/TASK_GRADIENT_TRUST_RESULTS.zh-CN.md)。

**正式三种子比较支持把 2B decoder 设为下一阶段默认路线。** MiniCPM5-2B-Base 与
EuroBERT-2.1B 使用相同 1,024 条公开训练行、种子内训练顺序、候选排列、128 次更新、
末 32 层 rank-8 LoRA 和相同损失。decoder 在 PAWS/SNLI、保留任务、未见 GoEmotions、
24 条换序审计，以及六个端点固定后才启封的 ARC-Challenge/Easy 上通过全部八项预注册
条件。ARC 总准确率三个种子为 81.2%/81.8%/81.2%，encoder 为
35.7%/35.4%/33.1%；decoder 的本机推理中位延迟约慢 10%–11%，采样设备占用略低。
见[结果报告](docs/ENCODER_DECODER_RESULTS.zh-CN.md)与
[公平比较设计](docs/ENCODER_DECODER_COMPARISON_DESIGN.zh-CN.md)。这是两个具体预训练
底座的端到端路线结论；预训练语料、tokenizer 和宽度不同，不能把差值只归因于注意力方向，
也不能据此宣称已经实现通用零样本能力。

**反事实候选先验减法没有晋级。** 在三个固定 decoder 端点上，开发种子只用 validation
NLL 从 `λ ∈ {0,.25,.5,.75,1}` 选择共享强度；seed42/43 都在 `λ=0` 最好，平均
NLL 随 λ 增大从 0.5851 单调恶化到 0.6773。因此预注册方法退化为不修正，不能设为
默认推理。选择哈希固定后首次运行的 288 条 BIG-bench `logical_deduction` 显示当前
decoder 本身具有稳定的新任务能力：三种子的 3/5/7 候选准确率范围分别为
78.1%–79.2%、46.9%–50.0%、65.6%–67.7%，三组 12/12 换序与 ID 改名审计全部稳定。
见[候选先验结果](docs/CANDIDATE_PRIOR_RESULTS.zh-CN.md)。该盲测不能被继续用于同一方法的
后验调参。

**此前：同更新预算、两个种子的预热对照已完成。** 每组均为 208 次更新、1,504 次
样本处理；前段比较“24 条推断样本反复训练”和“480 条混合样本各训练一次”，
后段训练相同。推断分别为 **90/192 对 64/192、103/192 对 66/192**，两个种子
均有收益；原始 NLL 也均改善。但平均意图、Schema、情绪成绩下降，第二个预热
种子的“信息不足”更是 0/64。值得保留为研究方向，尚不能设为通用默认方案。
详见[完整对照报告](docs/MATCHED_BUDGET_RESULTS.zh-CN.md)。更新数和样本曝光相等，
候选数量、输入长度及 FLOPs 不完全相等；一个预热组复用历史结果，其余三组新训练。
这是两个种子、已观察开发集上的结果，不是通用零样本证明。
[证据翻转诊断](docs/MATCHED_BUDGET_FLIPS.zh-CN.md)中，预热组为 11/18、12/18，
广样本组均为 6/18；四组仍都是 0/6 组完整通过，信息不足判断尚未解决。

复现入口：`scripts/run_matched_budget_study.py --prepare-only`、
`scripts/run_matched_budget_study.py --device mps`、`scripts/matched_budget_report.py`。
依赖此前公开数据和起点，详见[冻结设计](docs/MATCHED_BUDGET_DESIGN.zh-CN.md)。

**此前完成记忆诊断与预热后续训。** 固定 24 条公开训练样本可以学到 24/24；
仅预热时，测试推断仍只有 66/192，且错误更自信。再完成原来的 1,024 条混合训练后，
推断达到 90/192（46.9%），高于上一轮同种子的 64/192（33.3%）；意图和情绪各少答对
2 条，Schema 多答对 1 条。候选换序稳定 25/25。见[完整诊断与结果](docs/LEARNABILITY_RESULTS.zh-CN.md)。
本轮增加了 80 次预热更新，只有一个续训种子；这是局部进展，不能单独归因于训练顺序，
也不代表通用零样本能力。生产模型的读出位置没有改变。
[证据翻转诊断](docs/CURRICULUM_FLIPS.zh-CN.md)从上一轮同种子 6/18 到 11/18，
但完整三元组仍为 0/6，尤其容易把“信息不足”误判成“支持”。

复现实验依次运行 `scripts/overfit_sanity.py`、`scripts/frozen_readout_probe.py`、
`scripts/run_curriculum_probe.py`（均支持 `--device mps`），再运行
`scripts/learnability_report.py`。依赖前述公开数据和起点；须使用新的输出目录。
最终权重位于 `runs/curriculum-probe-v1/continued`，兼容原有推理和继续训练接口。

**此前完成两个种子的公开语义扩量训练。** 每组使用 768 条 SNLI 和 256 条
旧任务样本，共 128 次更新。固定测试集上，推断从起点 65/192 变为 64/192 和
70/192；意图从 54/61 变为 56/61 和 57/61。推断仍存在明显选项偏好，未证明稳定
的关系判断提升。见[完整报告](docs/SEMANTIC_SCALE_STUDY.zh-CN.md)和
[补充诊断](docs/SEMANTIC_COLLAPSE.zh-CN.md)。
另外，6 组自写的[证据翻转测试](docs/EVIDENCE_FLIPS.zh-CN.md)中，两份新权重都没有
整组答对；输出稳定与关系判断能力需要分别验证。

**已完成真实模型的策略梯度对照。** 常规损失、带噪直接求导、策略梯度
从同一权重各续训 24 次；后两组优化同一带噪目标，噪声计划相同。SNLI 准确率
分别为 15/45、11/45、15/45，均未超过起点；概率损失改善也没有自动变成更强判断。
见[实测报告](docs/GRADIENT_TRAINING_STUDY.zh-CN.md)与[算法实现说明](docs/GRADIENT_TRAINING_DESIGN.zh-CN.md)。

**最新架构对照：**按候选独立评分后，换序始终一致从 4/24 提高到 24/24，始终
答对从 4/24 提高到 14/24；本机请求中位延迟从 101.6 ms 增至 324.1 ms。
Schema 判断改善，意图与情绪各少答对 1 条，情绪概率质量变差。
见[完整架构报告](docs/ARCHITECTURE_STUDY.zh-CN.md)。稳定性改善不等于全面更强。

**受控对照已完成：**加 Brier/一致性后，意图排序从 55/61 到 58/61，但未参与
微调的情绪任务从 19/61 到 13/61；所有检查顺序都答对仍是 4/24。
本轮未证明复杂目标带来全面提升。见[研究结论](docs/FINDINGS.zh-CN.md)和
[完整对照报告](docs/STUDY.zh-CN.md)。这是单种子、小样本开发诊断。

首轮公开数据实测：意图候选排序 58/61、任务零样本情绪识别 13/61、Schema 判断
16/60；换序始终一致仅 4/24。当前是可运行和继续训练的研究版本，性能仍需迭代。
详见 [完整实测](docs/RESULTS.zh-CN.md)。

公开 SNLI 已准备 4,224 条，其中训练 3,072 条；梯度实验使用 48 条，语义扩量
实验的两个种子使用相同的 768 条续训，不能合计成 1,536 条不同数据。
更早的目标函数和架构实验不包含 SNLI。见[数据接入与重建说明](docs/SNLI_PREPARATION.md)。

## 最快入门：不下载模型也能复现实验

```bash
python -m pip install -e . torch numpy
python scripts/gradient_study.py --output runs/gradient-estimators-v1
python scripts/noise_optimum_study.py --output runs/noise-optimum-v1
python scripts/normalization_study.py --output runs/normalization-v1
```

三个 CPU 小实验分别检查：梯度方差、训练噪声与推理概率、组内标准化是否保留
原目标最优点。它们不需要模型权重或业务数据，也不用于证明模型准确率。
每次运行使用新输出目录。

## 接口

输入：`task`（任务说明）、`context`（材料）、`choices`（候选项 ID 与含义）。
输出：选中的 ID 与各候选概率。程序生成固定 JSON，模型不生成思考文本。
同一模型支持不同任务、不同候选名称，以及 2–32 个候选项。

调用方可以定义“信息不足”或“以上都不是”选项；这些业务选项不等于模型自身
没把握。没有兜底选项时，模型只能在所提供的候选项中排序。

## 模型与训练

- 从公开 EuroBERT-2.1B 原始底座开始，重新初始化适配器与决策头。
- 32 层 LoRA，共享候选评分头，不为每个业务固定训练一个类别输出。
- 支持所有候选联合编码、按候选独立评分两种方式；参数集合相同，计算成本不同。
- 交叉熵 + Brier 概率评分，并对不同候选顺序施加一致性训练。
- 已接入真实模型的带噪直接求导与策略梯度；记录实际成本、替代损失和裁剪情况。
- 只复现公开可描述的校准决策目标，不声称还原任何私有训练方案。
- 保留本地继续训练、推理、独立评测和换序检查入口。

## 数据与“零样本”的检验

数据只从公开、许可明确、版本固定的来源构建；每条记录保留来源和标签依据。
已有 JSON Schema 规则判断、Bandit 局部代码策略判断、CLINC 用户意图排序。
GoEmotions 情绪识别整类任务只用于测试，不进入训练和温度校准。

“训练过的任务中换一条测试数据”和“整个任务没参加微调”分别报告。
基础预训练是否见过公开基准无法排除，因此后者只称**微调阶段的任务零样本**。

## 使用

完整安装、数据重建、训练、推理命令见 [README](README.md)。

```bash
python -m pip install -e '.[data,model]'
decision-model prepare --output data/public-multitask-v1
decision-model train --data data/public-multitask-v1/cases.jsonl \
  --config configs/eurobert-public-pilot.json --output runs/public-multitask-v1
decision-model predict --checkpoint runs/public-multitask-v1 --input examples/request.json
```

小规模配置只用 384 条训练样本、2 轮，用来验证流程；完整公开数据不等于都已
参与训练。新数据可以按 [数据格式](docs/DATA.md) 组织，再用 `--init-from` 继续
微调已有适配器（优化器重新初始化）。目前的权重属于实验版本。

固定 JSON 的字段、概率和人工复核约束见
[输出合同](docs/OUTPUT_CONTRACT.zh-CN.md)及随包分发的 JSON Schema。

同一个 `predict` 命令已支持严格 Boolean、Choice 和有序 Score 输入：

```bash
.venv/bin/decision-model predict --checkpoint runs/architecture-comparison-v1/seed42-decoder \
  --base-path models/minicpm5-2b-base --device cpu --input examples/boolean-request.json

.venv/bin/decision-model predict --checkpoint runs/architecture-comparison-v1/seed42-decoder \
  --base-path models/minicpm5-2b-base --device cpu --input examples/score-request.json
```

Boolean 额外返回真值概率；Score 返回最优等级和概率加权期望分数。
三种输出都保留完整候选概率与 `requires_review: true`，且不允许自由文本字段。

架构对照入口：

```bash
python scripts/run_architecture_study.py --device mps
python scripts/architecture_report.py
```

单独训练独立评分版可使用 `configs/eurobert-independent-pilot.json`。
该模式下每个候选应能独立理解；“以上都不是”需要结合其他候选，不能直接假设
与联合编码语义等价。两种模式禁止直接互相热启动。详见[架构设计与取舍](docs/CANDIDATE_ENCODING.zh-CN.md)。

语义扩量入口（需先准备公开混合数据 v2 和固定独立评分起点）：

```bash
python scripts/run_semantic_scale_study.py --device mps
python scripts/semantic_scale_report.py
python scripts/semantic_collapse_diagnostic.py
python scripts/evidence_flip_probe.py --device mps
```

每次使用新目录。最后两项是观察到问题后的补充诊断，不参加训练或权重选择。
可以对 `runs/semantic-scale-v1/seed-42` 或 `seed-43` 使用原有推理/续训入口；
两份都是研究权重，没有选定可部署版本。

本机已准备独立环境 `.venv` 和原始公开底座 `models/eurobert-2.1b`，可直接运行：

```bash
.venv/bin/decision-model predict --checkpoint runs/public-multitask-v1 \
  --base-path models/eurobert-2.1b --input examples/request.json
```

上面的首轮权重在工具路由示例中选错了候选，详见实测报告。试用本轮独立评分权重：

```bash
.venv/bin/decision-model predict --checkpoint runs/architecture-study-v1/D-independent \
  --base-path models/eurobert-2.1b --input examples/request.json
```

本轮权重在这个提醒示例中选择了 `calendar`（约 0.668）。这是以前检查过的展示
样例，不是新增盲测成绩；返回值仍标记需要人工复核。

后续需要扩展任务类型、困难负例、指令改写、多语言和新的盲测任务集，再评价
通用能力。候选顺序稳定性、置信度和错误决策率同样要持续验证。
