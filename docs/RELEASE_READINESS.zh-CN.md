# Decoder 发布就绪审计

<!-- release-status:start -->
Release Candidate v1 只通过 15/19 项冻结门槛，不产生最终候选，也不允许从复制种子中事后挑选。完整结论见 [`RELEASE_CANDIDATE_RESULTS.zh-CN.md`](RELEASE_CANDIDATE_RESULTS.zh-CN.md)。
<!-- release-status:end -->

## 结论

当前已经选定 **MiniCPM5-2B-Base decoder 路线**，还没有选定可作为最终发布模型的
单一 checkpoint。三个 decoder 端点可以组成带完整来源信息的研究复核包，不能从中
按已观察成绩挑一个并改称最终模型。

## 已验证制品

三个端点都使用固定底座版本、末 32 层 rank-8 LoRA、独立候选编码和同一标量头；
每个适配器加决策头约 40.5 MB。所有权重、配置、温度和研究 manifest 的 SHA-256
均重新核对通过。

| 端点 | 训练行 | 更新 | 权重大小 | 温度 | 已观察测试准确率 |
|---|---:|---:|---:|---:|---:|
| seed42 decoder | 1,024 | 128 | 40,452,596 B | 1.1747 | 75.87% |
| seed43 decoder | 1,024 | 128 | 40,452,596 B | 1.5841 | 75.52% |
| seed44 decoder | 1,024 | 128 | 40,452,596 B | 1.8608 | 74.65% |

表内成绩只用于说明端点身份。原协议预注册的是“decoder 是否晋级为默认架构”，并将
六个模型全部固定在最后一步；它没有预注册三种子中选一个发布的规则。因此不能按表中
准确率、NLL 或 Brier 事后选择 seed42、43 或 44。

## 最终模型仍需完成

1. 在训练前冻结一个最终公开数据协议和固定种子，不引用三个已观察端点的成绩选种子。
2. 用声明的公开训练语料训练最终适配器；封存测试不得参与 checkpoint 选择。
3. 对不可变权重只运行一次新的整任务族盲测。ARC、logical deduction、
   disambiguation、BoolQ 和 WinoGrande 已经用于诊断，不能再充当新确认集。
4. 输出单一适配器包、温度、checksum manifest、代码与数据来源、安装信息、精确复现命令。

机器可读审计：[`evidence/release-readiness-v1.json`](evidence/release-readiness-v1.json)。
重跑命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_release_readiness.py
```
