# ARC 冻结专家效用筛选

把现有 QASC 冻结专家直接应用于已消费的 ARC-Challenge 与 ARC-Easy。

| 来源 | 行数 | 平均效用 | 应打开率 | 父准确率 | 专家准确率 | 父 NLL | 专家 NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| arc-challenge | 192 | +0.0032 | 0.2500 | 0.7500 | 0.7552 | 0.6757 | 0.6775 |
| arc-easy | 192 | -0.0014 | 0.2031 | 0.8646 | 0.8750 | 0.4126 | 0.4178 |

## 汇总

- 合并平均效用：`+0.000886`。
- 合并应打开率：`0.226562`。
- 合并准确率：父模型 `0.807292`，专家 `0.815104`。
- 合并 NLL：父模型 `0.544152`，专家 `0.547616`。

## 判定

至少一个门槛失败；ARC 不计为正效用训练族。

逐项门槛：`coverage.rows`=PASS, `coverage.sources`=PASS, `mechanism.weights_loaded`=PASS, `utility.combined`=FAIL, `utility.each_source`=FAIL, `accuracy.combined`=FAIL, `nll.each_source`=FAIL

机器证据：`docs/evidence/arc-expert-utility-screen-v1.json`
