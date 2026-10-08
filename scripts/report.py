"""Render a local results note from saved measured artifacts."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
run_dir=root/"runs/public-multitask-v1"
read=lambda p:json.loads(p.read_text())
run=read(run_dir/"run.json"); evaluation=read(run_dir/"evaluation.json")
calibration=read(run_dir/"calibration.json")
audit=read(run_dir/"audit.json")
manifest=read(root/"data/public-multitask-v1/manifest.json")
names={"bandit":"局部代码策略判断","json-schema":"Schema 规则判断","clinc":"用户意图候选排序","goemotions":"情绪识别（整类任务未参与微调）"}
lines=["# 本地通用决策模型：公开数据首轮实测","",
       "这是通用候选决策架构的第一轮公开数据实验。质量规则只是其中一个任务。尚未发布 GitHub。", "",
       "## 训练与数据", "",
       f"- 公开数据共 {manifest['rows']:,} 条，来自四个有明确许可的固定版本来源。",
       f"- 实际使用训练 {len(run['selected_ids']['train'])} 条、校准 {len(run['selected_ids']['validation'])} 条、测试 {len(run['selected_ids']['test'])} 条。",
       f"- 原始 EuroBERT-2.1B 底座，32 层 LoRA；可训练参数 {run['trainable_parameters']:,}。",
       f"- 完成 {run['updates']} 次更新，训练耗时 {run['training_seconds']/60:.1f} 分钟。",
       "- 目标：CE + 0.5 Brier + 0.05 候选换序一致性；不是任何私有训练方案复刻。",
       "- 训练不包含情绪识别；温度校准也不包含该任务。固定最终轮次，不根据测试成绩挑选权重。",
       f"- 数据 SHA-256：`{run['dataset_sha256']}`。", "",
       "## 按来源测试结果", "",
       "| 任务 | 正确数 | 正确率 | 按候选数均匀随机选择的期望正确率 |",
       "|---|---:|---:|---:|"]
for source,m in evaluation["test"]["raw"]["by_source"].items():
    lines.append(f"| {names[source]} | {m['correct']}/{m['n']} | {m['accuracy']:.1%} | {m['uniform_random_accuracy']:.1%} |")
lines += ["", "CLINC 是抽样 2/4/8 个候选的排序任务，不是原始 150 类分类基准。情绪任务是七类、单标签子集，也不能与原始 GoEmotions 多标签基准直接比较。", "",
          "## 保存与换序验证", "",
          f"- 抽取 {audit['cases']} 条测试样本，每条检查全部或最多六种候选顺序。",
          f"- 所有检查顺序答案一致：{audit['stable_all_orders']}/{audit['cases']}。",
          f"- 所有检查顺序均正确：{audit['correct_all_orders']}/{audit['cases']}。",
          f"- 权重重载后的最大概率差：{audit['max_reload_probability_difference']:.8g}。",
          f"- 仅修改候选 ID 后最大概率差：{audit['max_id_rename_probability_difference']:.8g}。", "",
          "## 概率指标", "",
          f"校准温度 T={calibration['temperature']:.4f}，仅用 {calibration['fit_rows']} 条验证样本拟合。", "",
          "| 测试指标 | 原始概率 | 温度校准后 |", "|---|---:|---:|"]
for key in ("nll","brier","ece_10"):
    lines.append(f"| {key} | {evaluation['test']['raw'][key]:.4f} | {evaluation['test']['temperature_scaled'][key]:.4f} |")
lines += ["", "混合任务的概率指标只作诊断，各来源详细指标保存在 evaluation.json。新任务的概率可靠性仍需单独验证。", "",
          "## 结果边界", "",
          "这轮验证了动态候选架构、公开数据处理和训练流程。384 条训练样本不足以训练通用可用的决策模型，不能据此宣称一切场景零样本。", "",
          "零样本指该任务未参加本次微调；不能保证公开基准从未出现在底座预训练中。反复根据这套测试调参后，应另留最终盲测任务。", "",
          "## 本地产物", "",
          "- `runs/public-multitask-v1/decision.safetensors`：公开数据适配器与评分头。",
          "- `runs/public-multitask-v1/run.json`：配置、数据、源码哈希和样本清单。",
          "- `runs/public-multitask-v1/evaluation.json`：分来源评测。",
          "- `runs/public-multitask-v1/audit.json`：换序与重载检查。",
          "- `README.md`：独立安装、继续微调、推理与评测命令。", ""]
lines += ["## 本地流程验收", "",
          "- 独立虚拟环境安装、源码包及 wheel 构建通过；17 项单元与编码器集成检查通过。",
          "- 公开数据重新构建后的文件哈希一致。",
          "- 从主实验权重继续微调，完成一次参数更新；主实验权重保持独立。",
          "- 独立 wheel 加载权重后输出可解析、符合接口约束的 JSON。",
          "- 工具路由示例要求安排提醒，模型却选了 search（概率约 0.465），正确候选应为 calendar。这是语义判断失败，不能把格式验收当作功能准确率。", ""]
(root/"docs/RESULTS.zh-CN.md").write_text("\n".join(lines))
print(root/"docs/RESULTS.zh-CN.md")
