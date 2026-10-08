"""Audit both seeds and compare complete recipes without choosing a winner."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from decision_model.core import digest, load_rows, write_json
from budget_warmup import make_plan
from run_matched_budget_study import verify_checkpoint
from semantic_scale_report import paired_diagnostics, NAMES


def read(path): return json.loads(path.read_text())


def planned_work(root, rows, plan, config):
    from transformers import AutoTokenizer
    from decision_model.model import Judge
    judge=Judge.__new__(Judge);judge.config=config
    judge.tokenizer=AutoTokenizer.from_pretrained(str(root/'models/eurobert-2.1b'),local_files_only=True,trust_remote_code=False)
    by_id={r['id']:r for r in rows};lengths={}
    for key in {key for point in plan for key in point['rows']}:
        request=by_id[key]['request'];encoded=judge.encode(request)
        lengths[key]={c['id']:len(pair[0]) for c,pair in zip(request['choices'],encoded)}
    work=dict(encoder_calls=0,encoder_sequences=0,unpadded_tokens=0,padded_token_slots=0,attention_token_pairs=0)
    for point in plan:
        for begin in range(0,6,2):
            flat=[lengths[key][candidate] for key,order in zip(point['rows'][begin:begin+2],point['choices'][begin:begin+2]) for candidate in order]
            for offset in range(0,len(flat),config['candidate_chunk_size']):
                chunk=flat[offset:offset+config['candidate_chunk_size']];width=max(chunk)
                work['encoder_calls']+=1;work['encoder_sequences']+=len(chunk)
                work['unpadded_tokens']+=sum(chunk);work['padded_token_slots']+=len(chunk)*width
                work['attention_token_pairs']+=len(chunk)*width*width
    return work


def load_study(root,study):
    if read(study/'status.json')['state']!='complete': raise ValueError('All arms and audits must finish')
    protocol=read(study/'protocol.json')
    for name,expected in read(study/'protocol-checksums.json').items():
        if digest((study/name).read_bytes())!=expected: raise ValueError('Frozen input changed: '+name)
    rows=load_rows(study/'cases.jsonl');test=[r for r in rows if r['split']=='test']
    selected={s:sorted(r['id'] for r in rows if r['split']==s) for s in ('train','validation','test')}
    arms={};initial_hash=None;audit_ids=None
    for name,arm in protocol['arms'].items():
        warm=root/arm['phase1'];final=root/arm['final']
        verify_checkpoint(warm);verify_checkpoint(final)
        config=read(study/(name+'-phase1.json'))
        if read(warm/'model.json')!=config: raise ValueError('Phase1 configuration changed')
        plan=[json.loads(line) for line in (study/(name+'-plan.jsonl')).read_text().splitlines()]
        if plan!=make_plan(rows,arm['mode'],arm['seed']): raise ValueError('Phase1 plan changed')
        exposure=Counter(k for p in plan for k in p['rows'])
        if sum(exposure.values())!=480 or not set(exposure)<=set(selected['train']): raise ValueError('Phase1 exposure differs')
        work=planned_work(root,rows,plan,config)
        if arm['reused_historical']:
            if digest((warm/'decision.safetensors').read_bytes())!=protocol['historical_warmup_sha256'] or digest((final/'decision.safetensors').read_bytes())!=protocol['historical_final_sha256']:
                raise ValueError('Historical weights changed')
            steps=[json.loads(line) for line in (warm/'steps.jsonl').read_text().splitlines()]
            if [p['rows'] for p in plan]!=[p['rows'] for p in steps]: raise ValueError('Historical exposure changed')
            warm_run=read(warm/'result.json')
            if warm_run['updates']!=80 or warm_run['repetitions_per_row']!=20: raise ValueError('Historical budget differs')
        else:
            warm_run=read(warm/'run.json')
            steps=[json.loads(line) for line in (warm/'steps.jsonl').read_text().splitlines()]
            if [p['step'] for p in steps]!=list(range(1,81)): raise ValueError('Missing or duplicated optimizer updates')
            for source,expected in protocol['source_files'].items():
                if source.startswith('src/decision_model/') and digest((warm/'source'/Path(source).name).read_bytes())!=expected:
                    raise ValueError('Warm-up executable source changed')
            if warm_run['encoder_work']!=work: raise ValueError('Observed vs reconstructed compute differs')
            if warm_run['plan_sha256']!=digest((study/(name+'-plan.jsonl')).read_bytes()): raise ValueError('Wrong executed plan')
            if warm_run['starting_weights_sha256']!=protocol['initial_weight_sha256']: raise ValueError('Wrong initial weights')
            if initial_hash is None: initial_hash=warm_run['initial_trainable_sha256']
            if warm_run['initial_trainable_sha256']!=initial_hash: raise ValueError('Unequal initial trainable tensors')
            if warm_run['updates']!=80 or warm_run['example_exposures']!=480: raise ValueError('Warm-up budget differs')
            if digest((warm/'plan.jsonl').read_bytes())!=warm_run['plan_sha256']: raise ValueError('Saved executed plan differs')
        record=read(final/'run.json');previous=read(root/f"runs/semantic-scale-v1/seed-{arm['seed']}/run.json")
        for key in ('selected_ids_sha256','training_plan_sha256','updates','encoder_work'):
            if record[key]!=previous[key]: raise ValueError('Phase2 mismatch: '+key)
        if read(final/'model.json')!=read(study/(name+'-phase2.json')): raise ValueError('Phase2 config differs')
        if record['dataset_sha256']!=protocol['dataset_sha256'] or record['selected_ids']!=selected: raise ValueError('Wrong data selection')
        if record['warm_start_weights_sha256']!=digest((warm/'decision.safetensors').read_bytes()): raise ValueError('Wrong warm start')
        if digest((final/'training-plan.jsonl').read_bytes())!=record['training_plan_sha256']: raise ValueError('Phase2 plan changed')
        for source,expected in record['source_files'].items():
            if digest((final/'source'/source).read_bytes())!=expected: raise ValueError('Source snapshot changed')
            if source!='__init__.py' and protocol['source_files']['src/decision_model/'+source]!=expected: raise ValueError('Core code differs')
        for split in ('validation','test'):
            if record['evaluation_ids'][split]!=selected[split]: raise ValueError('Evaluation cases differ')
        audit=read(final/'audit.json');ids=[r['id'] for r in audit['details']]
        if audit_ids is None: audit_ids=ids
        if ids!=audit_ids: raise ValueError('Audit examples differ')
        predictions=read(final/'test-predictions.json');evaluation=read(final/'evaluation.json')
        for source,metric in evaluation['test']['raw']['by_source'].items():
            source_ids={r['id'] for r in test if r['source']==source}
            ps=[p for p in predictions if p['id'] in source_ids]
            if len(ps)!=metric['n'] or sum(max(p['probabilities'],key=p['probabilities'].get)==p['label'] for p in ps)!=metric['correct']:
                raise ValueError('Metrics differ from predictions')
        phase1_sources=Counter(next(r['source'] for r in rows if r['id']==key) for key in exposure.elements())
        arms[name]={'arm':arm,'warm_run':warm_run,'run':record,'evaluation':evaluation,'audit':audit,
                    'phase1_encoder_work':work,'phase1_source_exposures':dict(phase1_sources),
                    'phase1_distinct_rows':len(exposure),'weights_sha256':digest((final/'decision.safetensors').read_bytes())}
    paired={str(seed):paired_diagnostics(test,read(root/protocol['arms'][f'mixed-{seed}']['final']/'test-predictions.json'),
                                      read(root/protocol['arms'][f'small-{seed}']['final']/'test-predictions.json')) for seed in protocol['seeds']}
    averages={source:{mode:statistics.mean(arms[f'{mode}-{seed}']['evaluation']['test']['raw']['by_source'][source]['accuracy'] for seed in protocol['seeds'])
                      for mode in ('small','mixed')} for source in NAMES}
    return {'protocol':protocol,'arms':arms,'paired_small_minus_mixed':paired,'seed_mean_accuracy':averages}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',default='runs/matched-budget-v1');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];result=load_study(root,root/args.study)
    write_json(root/'docs/evidence/matched-budget-v1.json',result)
    names=list(result['arms'])
    lines=['# 相同更新预算下：小样本预热是否值得保留？','','## 1. 比较口径','',
           'small：先将 24 条推断训练样本各训练 20 次。mixed：先使用原混合训练集里按种子打乱抽取的 480 条不同样本。',
           '随后两组均重置优化器，用完全相同的 1,024 条样本再训练一轮。每组共 208 次更新、1,504 次样本处理。',
           '前段 dropout 都为 0，后段都为 0.05；相同种子的后段样本和候选顺序核对一致。',
           'small-42 复用已完成的历史实验，其余三组新训练；两个种子、四个终点均完整报告。',
           '两组候选数和长度不同，不声称 FLOPs 相等。SNLI 已参与训练，测试是已多轮查看的开发诊断。','',
           '## 2. 同一测试集的准确率','','| 任务 | '+' | '.join(names)+' |','|'+'---|'*(len(names)+1)]
    for source,title in NAMES.items():
        values=[result['arms'][name]['evaluation']['test']['raw']['by_source'][source] for name in names]
        lines.append('| '+title+' | '+' | '.join(f"{v['correct']}/{v['n']}（{v['accuracy']:.1%}）" for v in values)+' |')
    lines+=['','### 推断各类答对数','','每类恰好 64 条。该表用于判断是否仍偏向固定候选。','',
            '| 组 | 矛盾 | 信息不足 | 支持 |','|---|---:|---:|---:|']
    for name,arm in result['arms'].items():
        side='trained' if arm['arm']['mode']=='small' else 'initial'
        matrix=result['paired_small_minus_mixed'][str(arm['arm']['seed'])]['snli']['confusion'][side]
        lines.append('| '+name+' | '+' | '.join(f"{matrix[label].get(label,0)}/64" for label in ('contradiction','neutral','entailment'))+' |')
    zero_neutral=[]
    for name,arm in result['arms'].items():
        side='trained' if arm['arm']['mode']=='small' else 'initial'
        matrix=result['paired_small_minus_mixed'][str(arm['arm']['seed'])]['snli']['confusion'][side]
        if matrix['neutral'].get('neutral',0)==0: zero_neutral.append(name)
    if zero_neutral:
        lines+=['', '**关键能力缺口：'+ '、'.join(zero_neutral)+' 的 64 条“信息不足”全部答错。总准确率提高不等于三类关系判断全面改善。**']
    lines+=['','### small 减 mixed：逐种子配对变化','','| 任务 | 种子 | 净提升 | 修正 / 新增错误 | 案例分组重采样 95% 区间 |','|---|---:|---:|---:|---:|']
    for source,title in NAMES.items():
        for seed,paired in result['paired_small_minus_mixed'].items():
            p=paired[source];lo,hi=p['paired_group_bootstrap_95_percentile']
            lines.append(f"| {title} | {seed} | {100*p['accuracy_change']:+.1f} 个百分点 | {p['fixed']} / {p['regressed']} | [{100*lo:+.1f}, {100*hi:+.1f}] |")
    lines+=['','区间仅反映这批案例的不确定性，不包含种子总体差异或开发调参效应；没有多重比较校正。','',
            '### 两个种子的均值（不是两份独立测试集）','','| 任务 | small | mixed | 差值 |','|---|---:|---:|---:|']
    for source,title in NAMES.items():
        v=result['seed_mean_accuracy'][source]
        lines.append(f"| {title} | {v['small']:.1%} | {v['mixed']:.1%} | {100*(v['small']-v['mixed']):+.1f} 个百分点 |")
    lines+=['','## 3. 概率质量与旧任务保留','','| 任务 | 组 | 原始 NLL | 校准 NLL | 原始 Brier |','|---|---|---:|---:|---:|']
    for source,title in NAMES.items():
        for name,arm in result['arms'].items():
            raw=arm['evaluation']['test']['raw']['by_source'][source];scaled=arm['evaluation']['test']['temperature_scaled']['by_source'][source]
            lines.append(f"| {title} | {name} | {raw['nll']:.4f} | {scaled['nll']:.4f} | {raw['brier']:.4f} |")
    lines+=['','## 4. 训练预算核对','','| 组 | 前段不同样本数 | 前段候选序列数 | 前段有效 token | 前段 padding 后 token | 前段注意力位置对数 |','|---|---:|---:|---:|---:|---:|']
    for name,arm in result['arms'].items():
        w=arm['phase1_encoder_work']
        lines.append(f"| {name} | {arm['phase1_distinct_rows']} | {w['encoder_sequences']} | {w['unpadded_tokens']} | {w['padded_token_slots']} | {w['attention_token_pairs']} |")
    lines+=['','上表从固定训练计划和生产 tokenizer 重建；新训练组同时核对运行时计数。历史 small-42 没有保存训练工作量，使用重建值。',
            '这是输入形状统计，不是硬件 FLOPs；不含反向传播、梯度重计算或评测。后段同种子两组的编码工作量完全一致。','',
            '## 5. 顺序稳定性','','| 组 | 换序稳定 | 所有顺序都正确 |','|---|---:|---:|']
    for name,arm in result['arms'].items():
        a=arm['audit'];lines.append(f"| {name} | {a['stable_all_orders']}/{a['cases']} | {a['correct_all_orders']}/{a['cases']} |")
    deltas=[p['snli']['accuracy_change'] for p in result['paired_small_minus_mixed'].values()]
    lines+=['','## 6. 解释边界','',f"两个种子中，small 的推断准确率胜过 mixed 的次数为 {sum(d>0 for d in deltas)}/2。",
            '这比较的是重复小样本与更广混合曝光两套数据策略；标签平衡、任务比例、样本多样性共同变化，不能分别归因。',
            '仅两个种子、同一开发集，不能声称通用零样本能力或生产可靠性。结构稳定不代表关系判断正确。',
            '完整原始记录见 [evidence/matched-budget-v1.json](evidence/matched-budget-v1.json)，冻结设计见 [MATCHED_BUDGET_DESIGN.zh-CN.md](MATCHED_BUDGET_DESIGN.zh-CN.md)。','',
            '```bash','python scripts/run_matched_budget_study.py --prepare-only','python scripts/run_matched_budget_study.py --device mps','python scripts/matched_budget_report.py','```','',
            '依赖既有公开数据和历史起点；重建时先运行此前实验。部分已完成阶段仅在检查通过后复用，不自动续跑半完成的优化器状态。','']
    (root/'docs/MATCHED_BUDGET_RESULTS.zh-CN.md').write_text('\n'.join(lines))
    print(canonical_summary(result))


def canonical_summary(result):
    return json.dumps({'snli':{name:arm['evaluation']['test']['raw']['by_source']['snli'] for name,arm in result['arms'].items()}},ensure_ascii=False)


if __name__=='__main__': main()
