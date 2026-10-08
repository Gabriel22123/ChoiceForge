# 公开任务族扩展与许可证筛选

## 数据准入规则

新任务族只有同时满足以下条件才进入训练候选池：

1. 官方来源明确允许再分发，许可证可固定到具体修订；
2. 原始文件、修订号和 SHA-256 可复核；
3. 选择规则不读取答案，训练/验证用途提前声明；
4. 与现有请求做输入级去重；
5. 先用冻结父模型与冻结专家做效用筛选，不因结果修改样本；
6. 只有形成至少三个独立正效用任务族，才恢复路由训练。

## 当前候选

| 数据集 | 许可证证据 | 当前决定 |
|---|---|---|
| ARC | 项目已固定 `allenai/ai2_arc` 修订 `210d026...`；官方数据元信息为 CC BY-SA 4.0 | 本地已有 384 条答案盲选样本。先做冻结专家效用筛选；它已被消费，不能再承担新颖发布评测。 |
| TruthfulQA | 官方 `sylinrl/TruthfulQA` 仓库标注 Apache-2.0，包含公开多选数据 | 下一优先独立来源；下载前固定官方提交、许可证和原始文件哈希。 |
| OpenBookQA | 官方代码仓库标注 Apache-2.0，但当前 Hugging Face 数据卡把数据许可证标为 `unknown` | 暂缓纳入可发布数据。只有确认许可证覆盖数据文件后再使用。 |
| HellaSwag | 官方仓库通常标注 MIT，但样本源自 ActivityNet/WikiHow，数据权利链更复杂 | 仅作后备；先完成来源与再分发条款审计。 |

## ARC 首轮筛选

使用已经固定、已经消费的 ARC-Challenge 和 ARC-Easy 各 192 条样本。缓存冻结父模型的
完整/空材料候选表示，将现有 QASC 冻结专家应用于两组样本，比较父模型与完整专家的
准确率、NLL 和带符号反事实效用。该筛选只回答“是否值得把 ARC 作为正效用训练族”，
不产生新发布结论，也不改变此前 ARC 评测记录。

筛选结果：ARC 合并平均效用仅 `+0.0009`，ARC-Easy 为负，NLL 也出现回退。ARC 不计
为 QASC 专家的正效用任务族。后续如果使用 ARC，应固定公开训练拆分并训练 ARC 专家或
任务族均衡联合专家，而不是复用 QASC 专家。

来源：

- ARC：<https://huggingface.co/datasets/allenai/ai2_arc>
- TruthfulQA：<https://github.com/sylinrl/TruthfulQA>
- OpenBookQA：<https://github.com/allenai/OpenBookQA>
- HellaSwag：<https://github.com/rowanz/hellaswag>
