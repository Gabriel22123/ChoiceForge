# 完整目标分布安全 oracle 设计

v1 只保护硬 `label`，无法约束带 `target_probabilities` 行的完整概率质量。v2 保持相同
专家库、724 行验证集、逐行可行性过滤和 frozen gates，只替换监督统计的定义：

- 目标分布 `q` 优先读取公开 `target_probabilities`；不存在时使用 `label` 的 one-hot。
- 交叉熵：`-Σ qᵢ log pᵢ`。
- Brier：`Σ(pᵢ-qᵢ)²`。
- 上下文优势：`Σ qᵢ(log p_full,i-log p_null,i)`。
- 正确性：模型 argmax 必须等于 `q` 的第一个 argmax，与项目输出契约一致。

专家只有在正确性保护、交叉熵不退化、Brier 不退化、期望上下文优势不退化且
`CE + 0.5 × Brier` 严格改善时才可被选择。安全集合内仍按 proper-loss 改善排序；
无可行专家时保留父路径。

通过门槛与 v1 相同：逐行四类安全违规为 0，每个来源准确率和完整目标交叉熵不退化，
合并准确率提高至少 1 个百分点、交叉熵下降至少 0.02、平均 proper-loss 改善至少 0.02。
本结果仍是读取金标的已消费 oracle，只用于冻结后续路由监督规则。
