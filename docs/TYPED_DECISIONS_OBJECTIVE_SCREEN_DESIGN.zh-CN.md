# Typed Decisions：固定 2B 表示的四折三种子目标筛选设计

## 为什么增加这一层

先导实验已经证明四条目标路径能在真实 MiniCPM5-2B 训练图上运行，但本机没有
CUDA/MPS。直接执行 `4 个工作流 × 3 个种子 × 4 个方法` 的完整 LoRA，会重复数百次
相同的 2B 主干计算，也会把“表示学到了什么”和“目标怎样更新概率头”混在一起。

正式路线因此分两级：

1. 关闭 LoRA 适配器，用固定公开底座一次性缓存每个候选最后 token 的隐藏状态；
2. 在完整四折和三个种子上只训练同一个共享标量头，筛选监督目标与梯度估计；
3. 只有通过预注册条件的增量方法，才与它的直接对照进入端到端 32 层 LoRA 验证。

第一层是算法筛选，不是最终模型实验。固定表示可能隐藏目标与表示学习的相互作用，
所以通过只代表“值得付出完整 LoRA 算力”，不能作为开源模型的最终收益声明。

## 数据与密封规则

- 数据：固定版本 `LocalLLaMA/typed-decisions`，四个整工作流轮流留出。
- 每折 fit：3,600 条 train + 900 条 validation。
- 每折密封 test：三个已见工作流与一个未见工作流各 500 题，共 2,000 题。
- 种子：42、43、44。
- 端点：4 折 × 3 种子 × 4 方法 = 48 个头。
- 程序先完成并校验全部 48 个头、训练计划、权重和 validation 结果，写入
  `test-lock.json`，之后才解析任何 `test.jsonl`。

候选特征缓存只包含模型输入产生的向量、候选 ID/顺序和请求哈希，不包含标签或教师
概率。标签仍从分离后的 fit/test 文件读取。特征以 safetensors 分片保存，每片及完整
manifest 都有 SHA-256；中断后只允许从已校验分片继续。
训练开始前还必须通过固定的 16 行数值门槛：缓存与串行直接前向的相对 L2、余弦、
固定头 logits、概率和 argmax 全部合格，而且证据中的 manifest 哈希必须与本次缓存一致。
这 16 行只从 fit 文件按 ID 哈希选择。正式 runner 在全局锁定前也只按 ID 懒加载 fit
张量；锁定前只校验缓存分片的原始字节哈希，密封 test 行不会解析、对应向量也不会
解码成模型张量。两者都在 48 个头锁定后才进入评测路径。

## 四个方法

| arm | 监督与优化 |
|---|---|
| `H-hard` | 教师分布压成 argmax，精确 CE + 0.5 Brier |
| `S-soft` | 完整教师分布，精确 soft CE + 0.5 Brier |
| `T-typed-rps` | S，并只对有序 Score 增加归一化 RPS |
| `R-rlcd-loo` | T 的高斯平滑风险，32 个动作、独立留一基线、无优势标准差归一化，并加 1.0 clean soft-CE 锚点 |

所有 arm 使用相同的初始头、逐行顺序、候选排列、epoch、batch 和更新数。R 的高斯噪声
来自独立的固定生成器，不消耗候选排列或模型初始化的随机数。
CE 与 Brier 对候选排列不敏感；有序 Score 的 RPS 会先把 logits、目标和噪声动作恢复到
原始等级顺序后再累计，避免在随机展示顺序上计算出无意义的距离。
32 个动作取自本项目的计算预算设计，也高于先导实验的 4 个；本项目此前的
梯度研究已显示 score-function 的方差远高于 pathwise，因此正式筛选不以过小采样数
制造一个容易失败的 RLCD 对照。噪声标准差仍固定为本项目已单独验证的 `0.4`。

## 预注册晋级规则

分别判断 `S-H`、`T-S`、`R-T`。每个增量方法必须同时满足：

1. 12 个未见工作流端点中至少 9 个 soft CE 更低；
2. 未见工作流 soft CE 的 12 端点均值更低；
3. 任一工作流的三种子平均 soft CE 恶化不超过 0.01；
4. 已见工作流 soft CE 均值恶化不超过 0.005；
5. 未见工作流 soft Brier 均值不恶化。

测试指标不反向修改损失权重、训练轮数或学习率。没有增量方法通过时，保留更简单的
直接监督基线；不会为了“必须有创新”而从失败结果中挑一个赢家。

## 固定训练配置

- 共享头：`LayerNorm(2048) → Linear(2048,256) → GELU → Linear(256,1)`。
- Dropout：0，避免把 dropout 抽样混入梯度估计比较。
- 三轮，batch 64，AdamW，学习率 `3e-4`，weight decay `0.01`。
- 梯度范数裁剪：1.0，逐更新记录裁剪前范数。
- PyTorch 确定性算法开启，CPU 线程固定为 4；运行时版本写入协议。
- validation 只用于独立温度拟合；晋级使用 raw test 概率。

固定协议：`configs/typed-decisions-objective-screen-v1.json`。
准备入口：`scripts/prepare_typed_decisions_objective_screen.py`。
特征入口：`scripts/cache_typed_decisions_decoder_features.py`。
训练入口：`scripts/run_typed_decisions_objective_screen.py`。
报告入口：`scripts/typed_decisions_objective_screen_report.py`。
