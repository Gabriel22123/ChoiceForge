# 固定输出合同

模型不会生成 JSON 文本。它只给调用方传入的每个候选项计算一个标量分数；程序负责
归一化概率并构造固定对象：

```json
{
  "choice_id": "approve",
  "probabilities": {"approve": 0.73, "reject": 0.27},
  "requires_review": true,
  "score_kind": "candidate_probability_not_a_guarantee",
  "calibration": "validation_temperature"
}
```

可机读 Schema 随包分发在 `decision_model/schemas/prediction.schema.json`。运行时还会调用
`decision_model.output_contract.validate_prediction` 检查 JSON Schema 无法表达的关联约束：

- 只能出现五个公开字段，不能追加自由文本、思考过程或模型生成的其他字段；
- 概率键必须与请求中的候选 ID 完全一致；JSON 对象的键顺序不影响语义；
- 概率必须有限、位于 `[0, 1]`，且总和为一；
- `choice_id` 必须是按请求顺序遇到的第一个最高概率候选；
- 研究预览固定返回 `requires_review: true`；
- 分数类型和校准方式只能使用已登记枚举值。

任务、上下文和候选描述会进入模型；候选 ID 只是路由标识，不进入模型。输出不包含
自由解释、思维链、工具调用或模型自行发明的 JSON 字段。

概率只针对本次请求给出的候选集合。增加、删除或改写候选项都会改变决策问题；高概率
不等于正确性保证，也不授权自动执行。

## Boolean / Choice / Score 视图

普通 `predict` 命令现在可以直接接收三种严格请求，底层仍是同一个通用候选模型：

- `choice`：2–32 个动态候选；
- `boolean`：显式定义 `true` 和 `false` 的判定语义；
- `score`：2–10 个按低到高排列的等级 ID 和描述。

类型化输出由程序构造，并由
`decision_model/schemas/typed-prediction.schema.json` 校验。每种输出都保留完整
概率、`requires_review`、分数语义和校准方式，另增加以下有界字段：

- 所有类型：最高概率 `top_probability` 和前两名差值 `margin`；
- Boolean：JSON 布尔值 `value` 和 `probability_true`；
- Score：最高概率等级的 `level_index`，以及按从 0 开始的等级序号计算的
  概率加权 `expected_score`。

这些字段都从已验证的概率向量确定计算；运行时会再检查 margin、真值概率、
等级和期望分数是否与概率完全一致。模型无法追加解释、思考过程或任意 JSON 键。
