"""Fixed data/label-balanced sampling comparison; only premise grouping changes."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

from decision_model.core import digest,load_rows,write_json
from relation_group_plan import make_orders


def read(path): return json.loads(path.read_text())


def prepare(root,out,seed):
    if out.exists(): raise ValueError('Use a fresh output directory')
    source=root/'data/public-relation-groups-v1';manifest=read(source/'manifest.json')
    data=source/'cases.jsonl'
    if digest(data.read_bytes())!=manifest['dataset_sha256']: raise ValueError('Dataset checksum differs')
    rows=load_rows(data);old=load_rows(root/'runs/semantic-scale-v1/cases.jsonl')
    if [r for r in rows if r['split']!='train']!=[r for r in old if r['split']!='train']:
        raise ValueError('Evaluation cases changed')
    initial=root/'runs/architecture-study-v1/D-independent'
    expected='8d5475f12df673591ce10a189058e26ae805ce78425a0b3f7df2fc08ceadf10e'
    if digest((initial/'decision.safetensors').read_bytes())!=expected: raise ValueError('Starting weights changed')
    config=read(root/'runs/semantic-scale-v1/seed-42.json');config.update(epochs=2,seed=seed,max_train_evaluation_rows=128)
    out.mkdir(parents=True);(out/'cases.jsonl').write_bytes(data.read_bytes())
    write_json(out/'data-manifest.json',manifest)
    for arm in ('grouped','dispersed'):
        write_json(out/(arm+'.json'),dict(config,epoch_row_orders=make_orders(rows,arm,seed,2)))
    files=[*sorted((root/'src/decision_model').glob('*.py')),
           root/'scripts/prepare_relation_groups.py',root/'scripts/relation_group_plan.py',
           root/'scripts/evaluate_reference.py',Path(__file__).resolve()]
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arms':['grouped','dispersed'],'seed':seed,
              'question':'At equal rows, per-update labels and replay positions, does shared-premise co-occurrence improve overall relation judgment and the recall/precision tradeoff for neutral?',
              'initial_checkpoint':str(initial.relative_to(root)),'initial_weight_sha256':expected,
              'dataset_sha256':digest(data.read_bytes()),'data_manifest':manifest,
              'source_files':{p.relative_to(root).as_posix():digest(p.read_bytes()) for p in files},
              'updates':256,'epochs':2,'example_exposures':2048,
              'controls':'Same initial weights, clean CE+.5Brier,32-layer rank8 LoRA, learning rates, dropout.05, optimizer,2epochs,1024rows, labels and old replay IDs at every position in every update',
              'treatment':'Grouped update:2premises x3labels +2old rows. Dispersed update:6different premises, still2rows per NLI label +the same2old rows. Only within-update premise co-occurrence changes.',
              'model_input':'Single row only; no cross-example attention or new contrastive loss; row/group IDs and labels excluded from input text',
              'checkpoint_selection':'Fixed final second epoch only, no validation/test early stopping or winner selection',
              'primary_endpoints':['Raw neutral recall on64 fixed SNLI test rows','Overall SNLI accuracy and raw NLL'],
              'secondary_endpoints':['Per-class recall','Old-task retention','Calibration','Fixed evidence-flip diagnostic','Candidate order stability'],
              'evaluation':'Same192 validation and384 previously inspected development test; fit temperature on validation only',
              'limitations':[f'One predeclared conventional seed{seed}; interpret together with independently run seeds',
                             'Changed training selection vs earlier studies; historical score comparisons are not data-matched causal estimates',
                             'Same total candidate sequences/unpadded tokens; padded work can differ because batch composition changes',
                             'SNLI is a trained task, not zero-shot; no pristine external test'],
              'selected_ids':{s:[r['id'] for r in rows if r['split']==s] for s in ('train','validation','test')}}
    write_json(out/'protocol.json',protocol)
    write_json(out/'protocol-checksums.json',{p.name:digest(p.read_bytes()) for p in out.iterdir() if p.is_file()})
    write_json(out/'status.json',{'state':'prepared'})
    print(json.dumps({'prepared':str(out),'training_rows':1024,'epochs':2,'updates_per_arm':256}),flush=True)


def run(root,out,device):
    protocol=read(out/'protocol.json');initial=root/protocol['initial_checkpoint'];base=root/'models/eurobert-2.1b';data=out/'cases.jsonl'
    def check():
        for name,expected in protocol['source_files'].items():
            if digest((root/name).read_bytes())!=expected: raise ValueError('Frozen source changed: '+name)
        for name,expected in read(out/'protocol-checksums.json').items():
            if digest((out/name).read_bytes())!=expected: raise ValueError('Frozen input changed: '+name)
        if digest((initial/'decision.safetensors').read_bytes())!=protocol['initial_weight_sha256']: raise ValueError('Starting weights changed')
    def execute(arm,stage,command):
        check();write_json(out/'status.json',{'state':stage,'arm':arm});print(json.dumps({'arm':arm,'stage':stage}),flush=True)
        try:
            with (out/(arm+'-'+stage+'.log')).open('w') as stream:
                subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError:
            write_json(out/'status.json',{'state':'failed','arm':arm,'stage':stage});raise
    for arm in protocol['arms']:
        target=out/arm
        if not target.exists():
            execute(arm,'train',[sys.executable,'-m','decision_model.cli','train','--data',str(data),'--config',str(out/(arm+'.json')),
                               '--output',str(target),'--init-from',str(initial),'--base-path',str(base),'--device',device])
        if not (target/'checksums.json').exists(): raise ValueError('Incomplete training requires explicit recovery')
        for name,expected in read(target/'checksums.json').items():
            if digest((target/name).read_bytes())!=expected: raise ValueError('Checkpoint changed')
        record=read(target/'run.json')
        if record['updates']!=256 or record['selected_ids']!=protocol['selected_ids']: raise ValueError('Budget or selection differs')
        if not (target/'audit.json').exists():
            execute(arm,'audit',[sys.executable,'-m','decision_model.cli','audit','--checkpoint',str(target),'--data',str(data),
                               '--output',str(target/'audit.json'),'--max-cases','25','--base-path',str(base),'--device',device])
    write_json(out/'status.json',{'state':'complete'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/relation-group-v1')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--device',default='mps');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/args.output
    if not out.exists(): prepare(root,out,args.seed)
    elif read(out/'protocol.json')['seed']!=args.seed: raise ValueError('Existing study uses a different seed')
    if not args.prepare_only: run(root,out,args.device)


if __name__=='__main__': main()
