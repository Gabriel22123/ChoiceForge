"""Two seeds, matched updates/exposures, fixed endpoints and no test gating."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

from decision_model.core import canonical, digest, load_rows, write_json
from budget_warmup import make_plan


def read(path): return json.loads(path.read_text())


def verify_checkpoint(path):
    for name,expected in read(path/'checksums.json').items():
        if Path(name).name!=name or digest((path/name).read_bytes())!=expected:
            raise ValueError('Checkpoint changed: '+str(path))


def prepare(root,out):
    if out.exists(): raise ValueError('Use a fresh study directory')
    data=root/'runs/semantic-scale-v1/cases.jsonl'
    if digest(data.read_bytes())!='dc952a6d731dbcc273790ccc1c52dd4b6c33f0a14d92ad3ff09b6fd09f84638e':
        raise ValueError('Frozen public dataset changed')
    rows=load_rows(data)
    initial=root/'runs/architecture-study-v1/D-independent'
    if digest((initial/'decision.safetensors').read_bytes())!='8d5475f12df673591ce10a189058e26ae805ce78425a0b3f7df2fc08ceadf10e':
        raise ValueError('Wrong common start')
    legacy=root/'runs/curriculum-probe-v1/continued'
    verify_checkpoint(legacy)
    # Historical reuse must have identical executable model/training code.
    for name,expected in read(legacy/'run.json')['source_files'].items():
        if name!='__init__.py' and digest((root/'src/decision_model'/name).read_bytes())!=expected:
            raise ValueError('Cannot reuse historical arm after core code change')
    oldwarm=root/'runs/overfit-sanity-v1'
    verify_checkpoint(oldwarm)
    small_plan=make_plan(rows,'small',42)
    oldsteps=[json.loads(line) for line in (oldwarm/'steps.jsonl').read_text().splitlines()]
    if [p['rows'] for p in small_plan]!=[p['rows'] for p in oldsteps]: raise ValueError('Historical small-example exposure mismatch')
    out.mkdir(parents=True)
    (out/'cases.jsonl').write_bytes(data.read_bytes())
    arms={}
    for seed in (42,43):
        for mode in ('small','mixed'):
            name=f'{mode}-{seed}'
            config=read(root/f'runs/semantic-scale-v1/seed-{seed}.json')
            phase1=read(oldwarm/'model.json');phase1['seed']=seed
            write_json(out/(name+'-phase1.json'),phase1)
            write_json(out/(name+'-phase2.json'),config)
            plan=make_plan(rows,mode,seed)
            (out/(name+'-plan.jsonl')).write_text(''.join(canonical(p)+'\n' for p in plan))
            reused=(seed==42 and mode=='small')
            arms[name]={'seed':seed,'mode':mode,'reused_historical':reused,
                        'phase1':str((oldwarm if reused else out/(name+'-phase1')).relative_to(root)),
                        'final':str((legacy if reused else out/name).relative_to(root))}
    sources=[*sorted((root/'src/decision_model').glob('*.py')),
             root/'scripts/budget_warmup.py',root/'scripts/overfit_sanity.py',Path(__file__).resolve()]
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'question':'Does repeated small-example warm-up beat broader mixed exposure at equal update/example budgets?',
              'arms':arms,'initial_checkpoint':str(initial.relative_to(root)),
              'initial_weight_sha256':digest((initial/'decision.safetensors').read_bytes()),
              'dataset_sha256':digest(data.read_bytes()),'seeds':[42,43],
              'budget':{'phase1_updates':80,'phase1_examples_per_update':6,'phase2_updates':128,'phase2_examples_per_update':8,'total_updates':208,'total_example_exposures':1504},
              'controlled':'Same start, objective, learning rates, phase1 dropout0, phase2 dropout.05, optimizer reset after80, fixed final endpoints; phase2 identical within each seed',
              'treatment':'Phase1 repeats balanced24 SNLI rows20times vs first480 of a seeded shuffle of the same1024 mixed training rows; phase2 uses all1024 once',
              'evaluation':'Same192 validation and384 development test. Temperature fits validation only; no test gating, early stopping or winner selection.',
              'historical_reuse':'small-42 fixed result from curriculum-probe-v1; identical core code checked except version string. This contrast is retrospective; seed43 is a fresh replication.',
              'limitations':['Two seeds and previously inspected development tasks; no universal zero-shot claim',
                             'Matched optimizer updates and example exposures, not FLOPs or wall time; candidate counts and token lengths differ',
                             'Compares complete data-selection recipes, not repetition vs label balance vs task proportions separately',
                             'Both groups reset optimizer after80; this is not a continuously trained optimizer baseline'],
              'source_files':{p.relative_to(root).as_posix():digest(p.read_bytes()) for p in sources},
              'historical_final_sha256':digest((legacy/'decision.safetensors').read_bytes()),
              'historical_warmup_sha256':digest((oldwarm/'decision.safetensors').read_bytes())}
    write_json(out/'protocol.json',protocol)
    write_json(out/'protocol-checksums.json',{p.name:digest(p.read_bytes()) for p in out.iterdir() if p.is_file()})
    write_json(out/'status.json',{'state':'prepared'})
    print(canonical({'prepared':str(out),'arms':list(arms)}),flush=True)


def run(root,out,device):
    protocol=read(out/'protocol.json');data=out/'cases.jsonl';base=root/'models/eurobert-2.1b'
    initial=root/protocol['initial_checkpoint']
    def check():
        for name,expected in protocol['source_files'].items():
            if digest((root/name).read_bytes())!=expected: raise ValueError('Frozen source changed: '+name)
        for name,expected in read(out/'protocol-checksums.json').items():
            if digest((out/name).read_bytes())!=expected: raise ValueError('Frozen protocol/input changed: '+name)
        if digest((initial/'decision.safetensors').read_bytes())!=protocol['initial_weight_sha256']: raise ValueError('Initial weights changed')
    def execute(name,stage,command):
        check();write_json(out/'status.json',{'state':stage,'arm':name})
        print(canonical({'arm':name,'stage':stage}),flush=True)
        try:
            with (out/(name+'-'+stage+'.log')).open('w') as stream:
                subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError:
            write_json(out/'status.json',{'state':'failed','arm':name,'stage':stage});raise
    for name,arm in protocol['arms'].items():
        check();warm=root/arm['phase1'];final=root/arm['final']
        if arm['reused_historical']:
            verify_checkpoint(warm);verify_checkpoint(final)
            if digest((warm/'decision.safetensors').read_bytes())!=protocol['historical_warmup_sha256'] or digest((final/'decision.safetensors').read_bytes())!=protocol['historical_final_sha256']:
                raise ValueError('Historical result changed')
        else:
            if not warm.exists():
                execute(name,'phase1',[sys.executable,'scripts/budget_warmup.py','--data',str(data),'--plan',str(out/(name+'-plan.jsonl')),
                        '--config',str(out/(name+'-phase1.json')),'--initial',str(initial),'--output',str(warm),'--base',str(base),'--device',device])
            if read(warm/'status.json')['state']!='complete': raise ValueError('Incomplete phase1 requires explicit recovery')
            verify_checkpoint(warm)
            if not final.exists():
                execute(name,'phase2',[sys.executable,'-m','decision_model.cli','train','--data',str(data),'--config',str(out/(name+'-phase2.json')),
                        '--init-from',str(warm),'--output',str(final),'--base-path',str(base),'--device',device])
            verify_checkpoint(final)
        record=read(final/'run.json'); previous=read(root/f"runs/semantic-scale-v1/seed-{arm['seed']}/run.json")
        for key in ('selected_ids_sha256','training_plan_sha256','updates','encoder_work'):
            if record[key]!=previous[key]: raise ValueError('Phase2 controls differ: '+key)
        if record['warm_start_weights_sha256']!=digest((warm/'decision.safetensors').read_bytes()): raise ValueError('Wrong phase1 weights')
        if not (final/'audit.json').exists():
            execute(name,'audit',[sys.executable,'-m','decision_model.cli','audit','--checkpoint',str(final),'--data',str(data),
                    '--output',str(final/'audit.json'),'--max-cases','25','--base-path',str(base),'--device',device])
    write_json(out/'status.json',{'state':'complete'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/matched-budget-v1')
    parser.add_argument('--device',default='mps');parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=root/args.output
    if not out.exists(): prepare(root,out)
    if not args.prepare_only: run(root,out,args.device)


if __name__=='__main__': main()
