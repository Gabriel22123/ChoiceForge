# 完整目标分布安全专家 oracle

先逐行过滤会破坏父模型正确性、完整目标交叉熵、Brier 或期望上下文优势的专家，再在安全集合内选择 proper-loss 改善最大的路径。

## 汇总

- 平均 proper-loss 改善：`+0.1996`。
- 准确率：`0.7293` → `0.7721` (`+0.0428`)。
- NLL：`0.8218` → `0.6565` (`-0.1653`)。
- 路径选择：`clinc`=202, `parent`=146, `paws-wiki`=134, `qasc`=133, `snli`=109。
- 安全违规：`brier_regressed`=0, `context_regressed`=0, `nll_regressed`=0, `parent_correct_lost`=0。

## 判定

全部门槛通过，可进入安全多专家路由设计。

逐项门槛：`safety.parent_correct`=PASS, `safety.nll`=PASS, `safety.brier`=PASS, `safety.context`=PASS, `sources.accuracy`=PASS, `sources.nll`=PASS, `combined.accuracy`=PASS, `combined.nll`=PASS, `combined.proper_improvement`=PASS

机器证据：`docs/evidence/distribution-safe-specialist-oracle-v1.json`
