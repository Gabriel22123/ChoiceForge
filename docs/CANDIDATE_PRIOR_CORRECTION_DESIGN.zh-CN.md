# 候选先验修正：冻结实验设计

状态：实验已完成；共享强度选择为 `λ=0`，方法未晋级。结果见
[三种子报告](CANDIDATE_PRIOR_RESULTS.zh-CN.md)。

## 问题

独立候选评分消除了候选位置与其他候选文本的直接干扰，但固定端点仍可能偏爱某种候选
语义。softmax 只能消除所有候选共有的加性偏移，不能消除每个候选自身的语言先验。

## 方法

对每个请求做两类不生成文本的评分：

1. `s(task, context, candidate)`：真实证据分数；
2. `b(task, "No case-specific context is provided.", candidate)`：保留任务与候选描述、
   移除案例证据后的反事实先验分数。

修正分数为：

```text
adjusted = s - λ × (b - mean(b))
```

中心化不改变 softmax 结果，只便于审计。候选 ID、标签和其他候选仍不进入单个候选的
编码；最终 JSON 继续由程序构造。该方法通常增加接近一倍的推理工作量。

## 冻结协议

- 固定端点：正式架构比较中的 MiniCPM5-2B seed42/43/44 decoder 权重，不再训练。
- 强度网格：`λ ∈ {0, .25, .5, .75, 1}`。
- 选择：只用 seed42/43 validation，在每个 λ 上重新拟合各自温度，以两个种子的等权
  平均 validation NLL 选择一个共享 λ；平局选更小值。
- 确认：seed44 的 PAWS、SNLI 与四类保留/未见开发任务。测试标签不参与 λ 选择。
- 新盲测：BIG-bench `logical_deduction` 固定提交 `092b196…`，Apache-2.0；对 3/5/7
  个对象三个子任务，分别按不含答案的哈希冻结 96 条，共 288 条。共享 λ 写入文件并
  固定哈希后，三个 decoder 端点才允许首次运行这些题目。
- ARC 已在架构比较中启封，不能用于本轮后验方法选择或“新盲测”声明。

## 晋级条件

所有条件必须同时满足：

1. 选择出的 λ 大于 0；否则只能说明“不修正”最好。
2. seed44 的 PAWS、SNLI 均不得下降超过 1 个百分点，二者平均准确率不下降，平均校准
   NLL 不变差。
3. seed44 的 Bandit、CLINC、GoEmotions、JSON Schema 平均准确率和平均校准 NLL
   均不变差。
4. 新盲测的综合准确率至少在两个种子上优于 raw；九个“种子 × 子任务”比较的平均
   准确率必须严格提高，平均校准 NLL 不变差。
5. 三个真实端点各抽取 12 条盲测样例，候选换序与候选 ID 改名后语义选择必须全部稳定。

候选边际分布与真实边际分布的总变差距离只作诊断，不能挽救主终点失败。若任一条件失败，
该方法保留为负结果，不进入默认路线。

## 可复现入口

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_logical_deduction_blind.py --download-only
PYTHONPATH=src .venv/bin/python scripts/prepare_logical_deduction_blind.py --seal-selection
PYTHONPATH=src .venv/bin/python scripts/run_candidate_prior_study.py --prepare-only --device mps
PYTHONPATH=src .venv/bin/python scripts/run_candidate_prior_study.py --run-prepared --device mps
PYTHONPATH=src .venv/bin/python scripts/candidate_prior_report.py
```

运行器会冻结源码、数据、底座和端点哈希；报告脚本独立重算 λ 选择、全部指标和晋级条件。
原始盲测题目、标签、logits 和逐例预测只保存在本地，不进入发布包。

## 解释边界

- 空上下文分数不一定全是有害偏差；候选的语言自然度和常识合理性有时是有效信号。
- 共享 λ 刻意放弃按任务调参，避免把零样本接口变成隐藏的任务特定分类器。
- BIG-bench 可能存在于底座预训练语料；封存协议只能排除本项目的调参泄漏。
- 本轮研究的是固定权重上的推理变换，不代表训练过程已经学会消除候选先验。
