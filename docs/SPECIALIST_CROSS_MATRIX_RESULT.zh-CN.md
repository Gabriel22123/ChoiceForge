# 多专家交叉效用矩阵

数值为固定专家相对父路径的平均反事实效用；正值表示完整打开专家更优。

| 专家 | clinc | json-schema | paws-wiki | qasc | snli | typed-decisions |
|---|---:|---:|---:|---:|---:|---:|
| qasc | +0.0001 | -0.0241 | +0.0434 | +0.4561 | +0.0448 | +0.0088 |
| snli | -0.0083 | +0.0163 | +0.0378 | -0.0589 | +0.0693 | +0.0146 |
| paws-wiki | -0.0072 | +0.0431 | +0.0604 | -0.0063 | +0.0394 | +0.0265 |
| clinc | +0.0147 | -0.3597 | -0.4258 | -0.2777 | -0.4741 | -0.3977 |

## 金标成本 oracle

- 合并平均效用：`+0.2773`。
- 准确率：`0.7293` → `0.7790` (`+0.0497`)。
- NLL：`0.8218` → `0.8016` (`-0.0202`)。
- 路径选择：`clinc`=484, `parent`=6, `paws-wiki`=64, `qasc`=101, `snli`=69。

## 判定

全部门槛通过，可进入多专家路由协议设计。

逐项门槛：`experts.own_family_utility`=PASS, `oracle.mean_utility`=PASS, `oracle.accuracy`=PASS, `oracle.nll`=PASS, `routing.needed`=PASS

机器证据：`docs/evidence/specialist-cross-matrix-v1.json`
