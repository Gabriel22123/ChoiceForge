# 本地模型包使用说明

公开模型包由两个部分组成：

- 冻结父模型决策参数，约 39 MB；
- 已验证的单文件合并适配器，约 14 MB。

它不包含训练数据、缓存或基础模型。基础模型使用公开的 `openbmb/MiniCPM5-2B-Base` 固定版本，并由仓库内 lock 文件逐文件校验。发布包位于 `models/choiceforge-local-bundle-v1/`；`runs/choiceforge-local-bundle-v1/` 只是本地构建输出。

## 目录结构

```text
choiceforge-local-bundle-v1/
├── bundle.json
├── adapter.json
├── adapter.safetensors
└── parent/
    ├── model.json
    ├── decision.safetensors
    └── calibration.json
```

## 推理

```bash
choiceforge predict-merged \
  --bundle models/choiceforge-local-bundle-v1 \
  --base-path models/minicpm5-2b-base \
  --input examples/request.json \
  --device cpu
```

返回值固定为五个字段：`choice_id`、`probabilities`、`requires_review`、`score_kind`、`calibration`。模型不会生成思考过程或任意文本；`requires_review` 恒为 `true`。

## 完整性

加载器会校验模型包内全部文件，也会按固定 lock 校验外部基础模型。任何文件变化都会拒绝加载。模型包中的所有参数都来自公开数据训练，发布内容限定为公开来源和可复现实验产物。
