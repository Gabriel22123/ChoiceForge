# 发布训练数据再分发审计

**结论：通过。** 两个固定训练文件只包含七个登记的公开来源，逐行保留许可证、
固定版本和外部 HTTPS 来源；未发现公司内网标识、本机路径或盲测任务行。

| 数据 | 行数 | 来源数 | provenance 条目 | SHA-256 |
|---|---:|---:|---:|---|
| control | 3008 | 7 | 3010 | `65a242bfadcdeac608281938f4817bb8129ae4b0b12805deb8b12a35c78e32f9` |
| expanded | 5056 | 7 | 5058 | `b0079df73c00d45de69a8197c719ff16b398b11bc1f631fed440db39bc82de2f` |

## 边界

- 组合数据集不重新声明为 Apache-2.0；每个来源继续服从其上游许可证。
- SNLI 的 CC-BY-SA、CLINC/GoEmotions 的署名要求和 PAWS 数据许可均随包说明。
- 新任务族盲测案例、标签、logits 和逐案例预测不进入源码或发布包。
- 该审计证明来源和本项目文件边界，不替代各上游作者的法律解释。

机器可读证据：`docs/evidence/release-data-redistribution-v1.json`
