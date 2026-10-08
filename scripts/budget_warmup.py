"""Fixed 80-update training-only phase for the matched-budget experiment."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from decision_model.core import canonical, digest, load_rows, write_json
from decision_model.model import Judge, verify_local_base
from decision_model.train import balanced_subset
from decision_model.training_objective import TrainingObjective
from overfit_sanity import balanced_batches


def make_plan(rows, mode, seed):
    training = sorted((r for r in rows if r['split'] == 'train'), key=lambda r:r['id'])
    if len(training) != 1024:
        raise ValueError('Expected the frozen 1024-row training selection')
    if mode == 'small':
        chosen = balanced_subset([r for r in training if r['source'] == 'snli'], 24)
        batches = list(balanced_batches(chosen, seed, 80))
    elif mode == 'mixed':
        random.Random(seed).shuffle(training)
        chosen = training[:480]
        batches = [chosen[i:i+6] for i in range(0,480,6)]
    else:
        raise ValueError('Unknown warm-up mode')
    rng = random.Random(seed+871)
    plan = []
    for step, batch in enumerate(batches,1):
        choices = []
        for row in batch:
            ids = [c['id'] for c in row['request']['choices']]
            rng.shuffle(ids); choices.append(ids)
        plan.append({'step':step,'rows':[r['id'] for r in batch], 'choices':choices})
    return plan


def main():
    parser=argparse.ArgumentParser()
    for key in ('data','plan','config','initial','output','base'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='mps')
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=Path(args.output)
    if output.exists(): raise ValueError('Use a new output directory')
    rows={r['id']:r for r in load_rows(args.data)}
    plan=[json.loads(line) for line in Path(args.plan).read_text().splitlines()]
    config=json.loads(Path(args.config).read_text())
    if len(plan)!=80 or any(len(p['rows'])!=6 for p in plan):
        raise ValueError('Budget must be 80 updates of six examples')
    if any(rows[key]['split']!='train' for p in plan for key in p['rows']):
        raise ValueError('Only training rows are permitted')
    if config['dropout']!=0.: raise ValueError('Both first phases disable dropout')
    verify_local_base(args.base)
    judge=Judge(config,base_path=args.base,checkpoint=args.initial,device=args.device)
    for key in {key for p in plan for key in p['rows']}: judge.encode(rows[key]['request'])
    output.mkdir(parents=True)
    params=list(judge.named_trainable())
    fingerprint=hashlib.sha256()
    for name,p in params:
        fingerprint.update(name.encode());fingerprint.update(p.detach().float().cpu().numpy().tobytes())
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in params if n.startswith('encoder.')],'lr':config['encoder_lr']},
        {'params':[p for n,p in params if n.startswith('head.')],'lr':config['head_lr']},
    ],weight_decay=.01,eps=1e-6)
    objective=TrainingObjective(config)
    record={'dataset_sha256':digest(Path(args.data).read_bytes()),'plan_sha256':digest(Path(args.plan).read_bytes()),
            'starting_weights_sha256':digest((Path(args.initial)/'decision.safetensors').read_bytes()),
            'initial_trainable_sha256':fingerprint.hexdigest(),'config':config,
            'updates':80,'example_exposures':480,'distinct_rows':len({k for p in plan for k in p['rows']}),
            'selection':'fixed final update; no training/validation/test performance gate'}
    (output/'source').mkdir()
    for path in Path(__file__).resolve().parents[1].joinpath('src/decision_model').glob('*.py'):
        (output/'source'/path.name).write_bytes(path.read_bytes())
    (output/'plan.jsonl').write_bytes(Path(args.plan).read_bytes())
    write_json(output/'status.json',{'state':'train'})
    judge.train(True);start=time.monotonic();total=0.
    for point in plan:
        optimizer.zero_grad(set_to_none=True);risk_total=0.
        for begin in range(0,6,2):
            batch=[rows[k] for k in point['rows'][begin:begin+2]]
            requests=[]
            for row,order in zip(batch,point['choices'][begin:begin+2]):
                request=copy.deepcopy(row['request']); by_id={c['id']:c for c in request['choices']}
                if set(order)!=set(by_id) or len(order)!=len(by_id): raise ValueError('Candidate plan differs')
                request['choices']=[by_id[key] for key in order];requests.append(request)
            target=torch.tensor([[c['id'] for c in v['choices']].index(r['label']) for r,v in zip(batch,requests)],device=judge.device)
            loss,risk=objective(judge.logits([judge.encode(v) for v in requests]),target,[len(v['choices']) for v in requests])
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            (loss/3).backward();risk_total+=risk.item()/3
        norm=torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True).item()
        optimizer.step();total+=risk_total
        with (output/'steps.jsonl').open('a') as stream:
            stream.write(canonical({'step':point['step'],'risk':risk_total,'pre_clip_norm':norm})+'\n')
        if point['step']==1 or point['step']%8==0:
            print(canonical({'step':point['step'],'mean_risk':total/point['step'],'seconds':time.monotonic()-start}),flush=True)
    judge.save(output)
    record.update(encoder_work=dict(judge.work),training_seconds=time.monotonic()-start)
    write_json(output/'run.json',record)
    write_json(output/'checksums.json',{name:digest((output/name).read_bytes()) for name in ('model.json','decision.safetensors')})
    write_json(output/'status.json',{'state':'complete'})


if __name__=='__main__': main()
