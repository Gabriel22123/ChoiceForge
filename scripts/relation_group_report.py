"""Verify source rows, executed plans and the fixed same-data sampling comparison."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from decision_model.core import digest,load_rows,write_json
from prepare_snli import normalized
from relation_group_plan import LABELS,make_orders
from semantic_scale_report import NAMES,paired_diagnostics


def read(path): return json.loads(path.read_text())


def verify_public_rows(root,rows,expected_sha):
    archive=root/'data/.public-downloads/snli_1.0.zip'
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=expected_sha: raise ValueError('Source archive changed')
    selected=[r for r in rows if r['source']=='snli' and r['split']=='train']
    targets={r['provenance'][0]['index']:r for r in selected}
    if len(targets)!=768: raise ValueError('Repeated or missing source row indexes')
    seen=set()
    with zipfile.ZipFile(archive) as z:
        with z.open('snli_1.0/snli_1.0_train.jsonl') as stream:
            for index,line in enumerate(stream):
                if index not in targets: continue
                raw=json.loads(line);row=targets[index];provenance=row['provenance'][0]
                if provenance['member']!='snli_1.0/snli_1.0_train.jsonl' or provenance['original_split']!='train': raise ValueError('Wrong official split')
                if provenance['file_sha256']!=expected_sha or row['label']!=raw['gold_label']: raise ValueError('Wrong source identity/label')
                if json.loads(row['request']['context'])!={'premise':raw['sentence1'],'hypothesis':raw['sentence2']}: raise ValueError('Source text changed')
                semantic=digest([normalized(raw['sentence1']),normalized(raw['sentence2'])])
                if row['id']!=digest(['snli-1.0',semantic]) or row['group_id']!='snli:'+digest(normalized(raw['sentence1'])): raise ValueError('Wrong content identity')
                seen.add(index)
    if seen!=set(targets): raise ValueError('Missing upstream rows')
    return {'official_training_rows_verified':len(seen),'archive_sha256':expected_sha,'original_text_and_gold_labels_match':True}


def load_study(root,study):
    if read(study/'status.json')['state']!='complete': raise ValueError('Both arms and audits must finish')
    protocol=read(study/'protocol.json')
    for name,expected in read(study/'protocol-checksums.json').items():
        if digest((study/name).read_bytes())!=expected: raise ValueError('Frozen input changed: '+name)
    rows=load_rows(study/'cases.jsonl');by_id={r['id']:r for r in rows}
    source_audit=verify_public_rows(root,rows,protocol['data_manifest']['archive_sha256'])
    old=load_rows(root/'runs/semantic-scale-v1/cases.jsonl')
    if [r for r in rows if r['split']!='train']!=[r for r in old if r['split']!='train']: raise ValueError('Evaluation changed')
    initial=read(root/'runs/semantic-scale-v1/initial.json')
    if initial['weight_sha256']!=protocol['initial_weight_sha256']: raise ValueError('Wrong reference weights')
    if any(initial['selected_ids'][s]!=protocol['selected_ids'][s] for s in ('validation','test')): raise ValueError('Reference evaluation differs')
    test=[r for r in rows if r['split']=='test'];arms={};fingerprint=None;audit_ids=None
    plans={}
    for arm in protocol['arms']:
        path=study/arm;run=read(path/'run.json');config=read(study/(arm+'.json'))
        orders=make_orders(rows,arm,protocol['seed'],protocol['epochs']);plans[arm]=orders
        if config['epoch_row_orders']!=orders: raise ValueError('Frozen orders differ')
        if read(path/'model.json')!=config or run['config_sha256']!=digest((study/(arm+'.json')).read_bytes()): raise ValueError('Wrong model config')
        if run['updates']!=256 or run['selected_ids']!=protocol['selected_ids'] or run['dataset_sha256']!=protocol['dataset_sha256']: raise ValueError('Wrong budget/data')
        if run['warm_start_weights_sha256']!=protocol['initial_weight_sha256']: raise ValueError('Wrong starting weights')
        if fingerprint is None: fingerprint=run['initial_trainable_sha256']
        if fingerprint!=run['initial_trainable_sha256']: raise ValueError('Initial trainable tensors differ')
        for name,expected in read(path/'checksums.json').items():
            if Path(name).name!=name or digest((path/name).read_bytes())!=expected: raise ValueError('Checkpoint changed')
        for name,expected in run['source_files'].items():
            if digest((path/'source'/name).read_bytes())!=expected or protocol['source_files']['src/decision_model/'+name]!=expected: raise ValueError('Source snapshot differs')
        raw=(path/'training-plan.jsonl').read_bytes()
        if digest(raw)!=run['training_plan_sha256']: raise ValueError('Executed plan changed')
        executed=[json.loads(line) for line in raw.splitlines()]
        for epoch,order in enumerate(orders,1):
            selected=[p for p in executed if p['epoch']==epoch]
            if [k for p in selected for k in p['rows']]!=order: raise ValueError('Did not execute frozen row order')
            for point in selected:
                if 'second' in point: raise ValueError('Unexpected second training view')
                for key,choices in zip(point['rows'],point['first']):
                    if Counter(choices)!=Counter(c['id'] for c in by_id[key]['request']['choices']): raise ValueError('Candidate identity changed')
        if len(executed)!=1024 or len(read(path/'curve.json'))!=2: raise ValueError('Wrong epoch count')
        steps=[json.loads(line) for line in (path/'optimizer-steps.jsonl').read_text().splitlines()]
        if [p['update'] for p in steps]!=list(range(1,257)): raise ValueError('Missing optimizer update')
        if any(run['evaluation_ids'][s]!=protocol['selected_ids'][s] for s in ('validation','test')): raise ValueError('Evaluation ID mismatch')
        audit=read(path/'audit.json');ids=[p['id'] for p in audit['details']]
        if audit_ids is None: audit_ids=ids
        if audit_ids!=ids: raise ValueError('Different audit cases')
        predictions=read(path/'test-predictions.json');evaluation=read(path/'evaluation.json')
        before_after=paired_diagnostics(test,initial['predictions'],predictions)
        for source,counts in before_after.items():
            if evaluation['test']['raw']['by_source'][source]['correct']!=counts['both_correct']+counts['fixed']: raise ValueError('Metrics disagree with predictions')
        arms[arm]={'run':run,'evaluation':evaluation,'audit':audit,'paired_vs_initial':before_after,
                   'weights_sha256':digest((path/'decision.safetensors').read_bytes())}
    for grouped,dispersed in zip(plans['grouped'],plans['dispersed']):
        for a,b in zip(grouped,dispersed):
            if by_id[a]['source']!=by_id[b]['source'] or by_id[a]['label']!=by_id[b]['label']: raise ValueError('Label/source position differs')
            if by_id[a]['source']!='snli' and a!=b: raise ValueError('Replay position differs')
        for start in range(0,1024,8):
            for arm,order,expected in (('grouped',grouped,[3,3]),('dispersed',dispersed,[1]*6)):
                batch=[by_id[k] for k in order[start:start+8]];snli=[r for r in batch if r['source']=='snli']
                if Counter(r['label'] for r in snli)!=dict.fromkeys(LABELS,2): raise ValueError('Per-update label balance differs')
                if sorted(Counter(r['group_id'] for r in snli).values())!=expected: raise ValueError('Wrong premise co-occurrence')
    for key in ('encoder_sequences','unpadded_tokens'):
        if arms['grouped']['run']['encoder_work'][key]!=arms['dispersed']['run']['encoder_work'][key]: raise ValueError('Input work budget differs')
    paired=paired_diagnostics(test,read(study/'dispersed/test-predictions.json'),read(study/'grouped/test-predictions.json'))
    neutral=[r for r in test if r['source']=='snli' and r['label']=='neutral'];neutral_ids={r['id'] for r in neutral}
    neutral_paired=paired_diagnostics(neutral,[p for p in read(study/'dispersed/test-predictions.json') if p['id'] in neutral_ids],
                                     [p for p in read(study/'grouped/test-predictions.json') if p['id'] in neutral_ids])['snli']
    return {'protocol':protocol,'source_audit':source_audit,'initial':initial,'arms':arms,'paired_grouped_minus_dispersed':paired,
            'neutral_recall_paired':neutral_paired}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',default='runs/relation-group-v1')
    parser.add_argument('--report',default='docs/RELATION_GROUP_RESULTS.zh-CN.md')
    parser.add_argument('--evidence',default='docs/evidence/relation-group-v1.json')
    parser.add_argument('--flip-evidence',default='docs/evidence/relation-group-flips-v1.json')
    parser.add_argument('--flip-report',default='docs/RELATION_GROUP_FLIPS.zh-CN.md')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];result=load_study(root,root/args.study)
    flip_path=root/args.flip_evidence
    if flip_path.exists():
        flip=read(flip_path)
        for arm,value in result['arms'].items():
            if flip['models'][arm]['weight_sha256']!=value['weights_sha256']: raise ValueError('Flip probe used different weights')
        result['evidence_flip']={arm:{'correct':v['metrics']['correct'],'n':v['metrics']['n'],
            'complete_groups':sum(g['all_three_correct'] for g in v['groups'].values()),'groups':len(v['groups']),
            'weight_sha256':v['weight_sha256']} for arm,v in flip['models'].items()}
    write_json(root/args.evidence,result)
    grouped=result['arms']['grouped']['evaluation']['test']['raw']['by_source']['snli']
    dispersed=result['arms']['dispersed']['evaluation']['test']['raw']['by_source']['snli']
    delta=result['paired_grouped_minus_dispersed']['snli']['accuracy_change']
    lines=['# 同前提成组训练：信息不足判断是否改善？','','## 本轮结果','',
           f"相同数据与更新预算下，成组答对 {grouped['correct']}/192（{grouped['accuracy']:.1%}），打散答对 {dispersed['correct']}/192（{dispersed['accuracy']:.1%}）；成组减打散为 {100*delta:+.1f} 个百分点。",
           '这是一组预先固定种子的开发实验，差值不等于已经建立跨种子的方法优势。信息不足的召回率、精确率以及其他任务变化见下表。','',
           '## 1. 设计与可复现性','',
           '两组使用完全相同的公开数据：256 个前提各一组三类样本，共 768 条推断样本，加原 256 条旧任务复习。',
           f"同一 D-independent 起点、seed{result['protocol']['seed']}、两轮、256 次更新、2,048 次样本处理，固定最终终点。",
           '每次更新都为支持/矛盾/信息不足各 2 条，加相同位置的 2 条旧任务样本；仅改变同前提样本是否同次更新。',
           'grouped 每次包含两个完整前提组；dispersed 包含六个不同前提。模型仍逐条读输入，没有跨样本注意力或额外对比损失。',
           '768 条推断的原文、标签、官方 train 索引已逐条核对归档；训练顺序与日志、参数起点、标签和复习位置核对通过。',
           '原 192 条验证、384 条开发测试保持不变；只有验证集用于温度拟合。当前仅一个种子。','',
           '## 2. 三类推断分别答对多少','','| 版本 | 矛盾 | 信息不足 | 支持 | 总计 |','|---|---:|---:|---:|---:|']
    baseline_matrix=result['arms']['grouped']['paired_vs_initial']['snli']['confusion']['initial']
    matrices={'起点':baseline_matrix,**{arm:value['paired_vs_initial']['snli']['confusion']['trained'] for arm,value in result['arms'].items()}}
    for name,matrix in matrices.items():
        correct=[matrix[label].get(label,0) for label in LABELS]
        lines.append('| '+name+' | '+' | '.join(f'{v}/64' for v in correct)+f' | {sum(correct)}/192（{sum(correct)/192:.1%}） |')
    lines+=['','### 防止一律回答“信息不足”获得虚假进步','','| 版本 | 选为信息不足的数量 | 其中答对数量 | 判信息不足时的正确率 |','|---|---:|---:|---:|']
    for name,matrix in matrices.items():
        predicted=sum(v.get('neutral',0) for v in matrix.values());correct=matrix['neutral'].get('neutral',0)
        precision=f'{correct/predicted:.1%}' if predicted else '未选择'
        lines.append(f'| {name} | {predicted}/192 | {correct} | {precision} |')
    p=result['neutral_recall_paired'];lo,hi=p['paired_group_bootstrap_95_percentile']
    lines+=['',f"在 64 条真实信息不足样本上，grouped 相对 dispersed 的召回率差值为 {100*p['accuracy_change']:+.1f} 个百分点，案例分组重采样 95% 区间为 [{100*lo:+.1f}, {100*hi:+.1f}]。",
            '信息不足是材料与命题的关系标签，不等同于模型自身置信度低。既看找回多少信息不足，也看是否把支持/矛盾误判成信息不足。','',
            '## 3. 旧任务与整体推断','','| 任务 | 起点 | grouped | dispersed |','|---|---:|---:|---:|']
    for source,title in NAMES.items():
        metrics=[result['initial']['evaluation']['test']['raw']['by_source'][source]]+[result['arms'][arm]['evaluation']['test']['raw']['by_source'][source] for arm in ('grouped','dispersed')]
        lines.append('| '+title+' | '+' | '.join(f"{m['correct']}/{m['n']}（{m['accuracy']:.1%}）" for m in metrics)+' |')
    lines+=['','### grouped 减 dispersed：配对变化','','| 任务 | 准确率变化 | 修正 / 引入错误 | 案例分组重采样 95% 区间 |','|---|---:|---:|---:|']
    for source,title in NAMES.items():
        p=result['paired_grouped_minus_dispersed'][source];lo,hi=p['paired_group_bootstrap_95_percentile']
        lines.append(f"| {title} | {100*p['accuracy_change']:+.1f} 个百分点 | {p['fixed']} / {p['regressed']} | [{100*lo:+.1f}, {100*hi:+.1f}] |")
    lines+=['','区间仅反映本批开发案例，不包含跨种子或调参不确定性，也未做多重比较校正。','',
            '## 4. 概率质量','','| 任务 | 组 | 原始 NLL | 校准 NLL | 原始 Brier |','|---|---|---:|---:|---:|']
    for source,title in NAMES.items():
        for arm,value in result['arms'].items():
            raw=value['evaluation']['test']['raw']['by_source'][source];scaled=value['evaluation']['test']['temperature_scaled']['by_source'][source]
            lines.append(f"| {title} | {arm} | {raw['nll']:.4f} | {scaled['nll']:.4f} | {raw['brier']:.4f} |")
    lines+=['','## 5. 输入工作量和顺序稳定性','','| 组 | 候选序列 | 有效 token | padding 后 token | 注意力位置对 | 换序稳定 | 所有顺序正确 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,value in result['arms'].items():
        w=value['run']['encoder_work'];a=value['audit']
        lines.append(f"| {arm} | {w['encoder_sequences']} | {w['unpadded_tokens']} | {w['padded_token_slots']} | {w['attention_token_pairs']} | {a['stable_all_orders']}/{a['cases']} | {a['correct_all_orders']}/{a['cases']} |")
    lines+=['','输入统计不包含反向传播、梯度重计算或评估，不是硬件 FLOPs；批组合变化也会改变 dropout 掩码在样本上的对应关系。','',
            '## 6. 解释边界','',
            '- 只用一个预先固定的种子，不能把一次差异当作稳定方法优势。',
            '- 与历史实验相比，训练样本、轮数和每次更新的标签平衡都改变了；只有本轮两组之间是同数据、同预算对照。',
            '- 原始标签全部保留，但 768 条中有 704 条只具备一个上游标注，未人工重标或假称多标注共识。',
            '- 测试已被多轮开发观察；SNLI 是训练任务，不是通用零样本或独立盲测证明。',
            '- 本轮固定 clean CE+0.5 Brier，不比较策略梯度和直接求导。',
            '- 保留两个固定终点，不按测试分数挑选部署模型。','',
            f"设计见 [RELATION_GROUP_DESIGN.zh-CN.md](RELATION_GROUP_DESIGN.zh-CN.md)，完整记录见 [evidence/{Path(args.evidence).name}](evidence/{Path(args.evidence).name})。",'',
            '```bash','python scripts/prepare_relation_groups.py','python scripts/run_relation_group_study.py --prepare-only','python scripts/run_relation_group_study.py --device mps','python scripts/relation_group_report.py','```','',
            '需要已校验的官方归档、此前公开数据及 D-independent 起点；新实验使用新目录。所有产物仅保存在本地。','']
    if 'evidence_flip' in result:
        flip_lines=['## 6. 证据翻转诊断','','| 组 | 单题正确 | 完整三元组 |','|---|---:|---:|']
        for arm,value in result['evidence_flip'].items():
            flip_lines.append(f"| {arm} | {value['correct']}/{value['n']} | {value['complete_groups']}/{value['groups']} |")
        flip_name=Path(args.flip_report).name
        flip_lines+=['','六组题为项目自写、已观察的开发诊断，未进入训练，不是新外部基准。',
                f'详见 [{flip_name}]({flip_name})，权重指纹已与本轮模型核对。','']
        index=lines.index('## 6. 解释边界');lines[index]='## 7. 解释边界';lines[index:index]=flip_lines
    (root/args.report).write_text('\n'.join(lines))
    print(json.dumps({arm:value['evaluation']['test']['raw']['by_source'] for arm,value in result['arms'].items()},ensure_ascii=False))


if __name__=='__main__': main()
