# 约束优先的安全专家 oracle

先逐行过滤会破坏父模型正确性、NLL、Brier 或上下文优势的专家，再在安全集合内选择 proper-loss 改善最大的路径。

## 汇总

- 平均 proper-loss 改善：`+0.2293`。
- 准确率：`0.7293` → `0.7707` (`+0.0414`)。
- NLL：`0.8218` → `0.7690` (`-0.0528`)。
- 路径选择：`clinc`=321, `parent`=152, `paws-wiki`=68, `qasc`=114, `snli`=69。
- 安全违规：`brier_regressed`=0, `context_regressed`=0, `nll_regressed`=0, `parent_correct_lost`=0。

## 判定

至少一个门槛失败，暂不训练安全多专家路由。

逐项门槛：`safety.parent_correct`=PASS, `safety.nll`=PASS, `safety.brier`=PASS, `safety.context`=PASS, `sources.accuracy`=PASS, `sources.nll`=FAIL, `combined.accuracy`=PASS, `combined.nll`=PASS, `combined.proper_improvement`=PASS

机器证据：`docs/evidence/safe-specialist-oracle-v1.json`
