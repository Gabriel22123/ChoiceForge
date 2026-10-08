# 证据翻转：材料改变后，答案会正确改变吗？

观察到第一组推断失败后增加的诊断。18 条项目自写的英文逻辑样例，分为 6 组三元组；不是训练数据，也不是独立外部基准。
每组保持任务说明、假设和候选含义不变，只改变前提，使正确关系依次为支持、矛盾、无法判断。三种材料都答对才算整组通过。

| 模型 | 单题答对 | 三种材料全部答对的组 | 原始 NLL |
|---|---:|---:|---:|
| small-42 | 11/18 | 0/6 | 1.0883 |
| mixed-42 | 6/18 | 0/6 | 1.0986 |
| small-43 | 12/18 | 0/6 | 0.9920 |
| mixed-43 | 6/18 | 0/6 | 1.1010 |

## 逐组决策

下表每格按“支持材料 / 矛盾材料 / 信息不足材料”的顺序展示模型决策。

| 关系 | small-42 | mixed-42 | small-43 | mixed-43 |
|---|---|---|---|---|
| count | entailment / contradiction / entailment | neutral / neutral / neutral | entailment / contradiction / contradiction | neutral / neutral / neutral |
| color | entailment / contradiction / entailment | neutral / neutral / neutral | entailment / contradiction / entailment | entailment / neutral / neutral |
| transfer | entailment / contradiction / entailment | neutral / neutral / neutral | entailment / contradiction / entailment | contradiction / neutral / neutral |
| location | entailment / contradiction / entailment | neutral / neutral / neutral | entailment / contradiction / entailment | neutral / contradiction / contradiction |
| state | entailment / contradiction / entailment | neutral / neutral / neutral | entailment / contradiction / contradiction | neutral / neutral / entailment |
| quantifier | entailment / entailment / contradiction | neutral / neutral / neutral | entailment / contradiction / contradiction | neutral / neutral / neutral |

## 全部输入与标签依据

### count

固定假设：Exactly two people are in the boat.

- 支持：Exactly two people are in the boat.
- 矛盾：Exactly three people are in the boat.
- 无法判断：Some people are in the boat.

### color

固定假设：The only ball in the box is red.

- 支持：The box contains exactly one ball, and that ball is red.
- 矛盾：The box contains exactly one ball, and that ball is not red.
- 无法判断：The box contains exactly one ball.

### transfer

固定假设：Alice handed the book to Bob.

- 支持：Alice handed the book to Bob.
- 矛盾：Alice did not hand the book to Bob.
- 无法判断：Alice and Bob discussed a book.

### location

固定假设：The cup is on the table.

- 支持：The cup is on the table.
- 矛盾：The cup is not on the table.
- 无法判断：There is a cup in the room.

### state

固定假设：The door is open.

- 支持：The door is open.
- 矛盾：The door is not open.
- 无法判断：The door is wooden.

### quantifier

固定假设：Every box in the room is empty.

- 支持：All boxes in the room are empty.
- 矛盾：At least one box in the room contains a book.
- 无法判断：There are three boxes in the room.

这些样例可用于定位具体关系判断失败，不能当作所有语义任务的代表。概率变化与硬决策变化应分别看；即使没有正确翻转，也不能据此断言模型完全不读取材料。

完整概率、权重指纹和各组结果见[机器可读记录](evidence/matched-budget-flips-v1.json)。
