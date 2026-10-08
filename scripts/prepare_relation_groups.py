"""Public SNLI training triplets; no relabeling, synthetic training or held-out reuse."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

from decision_model.core import canonical, digest, load_rows, write_json
from prepare_snli import DESCRIPTIONS, SOURCE_URL, normalized, records


def select_triples(archive, expected_sha, cap, accept=lambda request:True, excluded_groups=()):
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=expected_sha:
            raise ValueError('SNLI archive checksum mismatch')
    heldout=set(excluded_groups);labels=defaultdict(set);excluded=Counter()
    # All official held-out premises, including rows without consensus labels.
    with zipfile.ZipFile(archive) as z:
        for split in ('dev','test'):
            with z.open(f'snli_1.0/snli_1.0_{split}.jsonl') as stream:
                for line in io.TextIOWrapper(stream,encoding='utf-8'):
                    heldout.add(digest(normalized(json.loads(line)['sentence1'])))
    for _,_,_,_,row,_,semantic in records(archive,excluded):
        labels[semantic].add(row['gold_label'])
    conflicts={key for key,value in labels.items() if len(value)>1};del labels
    groups=defaultdict(dict);seen=set()
    for split,original,member,index,raw,group,semantic in records(archive):
        if split!='train': continue
        if group in heldout: excluded['heldout_premise_rows']+=1;continue
        if semantic in conflicts: excluded['conflicting_pair_rows']+=1;continue
        if semantic in seen: excluded['duplicate_pair_rows']+=1;continue
        seen.add(semantic);identity=digest(['snli-1.0',semantic]);label=raw['gold_label']
        previous=groups[group].get(label)
        if previous and previous['id']<identity: continue
        choices=[{'id':key,'description':value} for key,value in DESCRIPTIONS.items()]
        choices.sort(key=lambda c:digest([identity,c['id']]))
        groups[group][label]={
            'id':identity,'group_id':'snli:'+group,'source':'snli','family':'natural_language_inference',
            'split':'train','evaluation_regime':'seen_task_new_examples','label':label,
            'request':{'task':'Determine how the premise relates to the hypothesis. Use only the information stated in the premise.',
                       'context':canonical({'premise':raw['sentence1'],'hypothesis':raw['sentence2']}),'choices':choices},
            'provenance':[{'url':SOURCE_URL,'revision':'SNLI-1.0','file_sha256':expected_sha,'member':member,'index':index,
                           'pair_id':raw.get('pairID'),'original_split':original,'license':'CC-BY-SA-4.0',
                           'label_method':'upstream_gold_label','annotator_labels':raw.get('annotator_labels',[]),
                           'transformation':'one minimum-hash pair per gold label per normalized training premise; complete three-label groups; deterministic premise selection; held-out premise/conflict/duplicate exclusion'}]}
    complete=sorted(key for key,value in groups.items() if set(value)==set(DESCRIPTIONS))
    selected=[];over_budget=0
    for key in complete:
        triple=[groups[key][label] for label in sorted(DESCRIPTIONS)]
        if not all(accept(row['request']) for row in triple): over_budget+=1;continue
        selected.extend(triple)
        if len(selected)==cap*3: break
    if len(selected)!=cap*3: raise ValueError('Not enough complete admissible training groups')
    selected.sort(key=lambda row:row['id'])
    return selected,{'archive_sha256':expected_sha,'complete_eligible_groups_before_token_admission':len(complete),
                     'selected_groups':cap,'selected_rows':len(selected),'excluded':dict(excluded),
                     'conflicting_pairs':len(conflicts),'forbidden_premise_groups':len(heldout),
                     'token_rejected_groups_before_cap':over_budget,
                     'annotation_vote_counts':dict(Counter(len(r['provenance'][0]['annotator_labels']) for r in selected))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='data/public-relation-groups-v1')
    parser.add_argument('--archive',default='data/.public-downloads/snli_1.0.zip');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/args.output
    if out.exists(): raise ValueError('Use a new dataset directory')
    old=root/'runs/semantic-scale-v1/cases.jsonl'
    if digest(old.read_bytes())!='dc952a6d731dbcc273790ccc1c52dd4b6c33f0a14d92ad3ff09b6fd09f84638e': raise ValueError('Reference data changed')
    old_rows=load_rows(old)
    from transformers import AutoTokenizer
    from decision_model.model import Judge
    judge=Judge.__new__(Judge);judge.config=json.loads((root/'runs/semantic-scale-v1/seed-42.json').read_text())
    judge.tokenizer=AutoTokenizer.from_pretrained(str(root/'models/eurobert-2.1b'),local_files_only=True,trust_remote_code=False)
    def accept(request):
        try: judge.encode(request);return True
        except ValueError as error:
            if 'Input too long' not in str(error): raise
            return False
    expected=json.loads((root/'configs/snli-source.json').read_text())['sha256']
    triples,manifest=select_triples(root/args.archive,expected,256,accept)
    replay=[r for r in old_rows if r['split']=='train' and r['source']!='snli']
    if len(replay)!=256: raise ValueError('Expected fixed256 old-task replay rows')
    evaluation=[r for r in old_rows if r['split']!='train']
    rows=sorted(triples+replay+evaluation,key=lambda r:r['id'])
    for row in rows: judge.encode(row['request'])
    out.mkdir(parents=True);target=out/'cases.jsonl'
    target.write_text(''.join(canonical(r)+'\n' for r in rows));load_rows(target)
    old_train_ids={r['id'] for r in old_rows if r['source']=='snli' and r['split']=='train'}
    manifest.update(dataset_sha256=digest(target.read_bytes()),reference_dataset_sha256=digest(old.read_bytes()),
                    reference_snli_training_id_overlap=len(old_train_ids & {r['id'] for r in triples}),
                    split_source_counts={s:dict(Counter(r['source'] for r in rows if r['split']==s)) for s in ('train','validation','test')},
                    evaluation_policy='Byte-identical row objects from existing192 validation and384 development test; no synthetic test cases in training',
                    script_sha256=digest(Path(__file__).read_bytes()),
                    limitations=['Upstream gold labels retained; most training rows have only one annotation',
                                 'Normalized exact grouping does not exclude all semantic paraphrases',
                                 'New training selection and previously inspected evaluation; not a blind benchmark or zero-shot task'])
    write_json(out/'manifest.json',manifest)
    with zipfile.ZipFile(root/args.archive) as z: (out/'UPSTREAM_README.txt').write_bytes(z.read('snli_1.0/README.txt'))
    (out/'ATTRIBUTION.md').write_text('# Public relation-group training selection\n\nSNLI1.0: Bowman, Angeli, Potts and Manning (2015), Stanford NLP Group.\nSource: https://nlp.stanford.edu/projects/snli/\nSNLI-derived data: CC BY-SA4.0, https://creativecommons.org/licenses/by-sa/4.0/ .\nOriginal text and gold labels retained; complete premise groups selected and reformatted.\nUnderlying caption attribution is preserved in UPSTREAM_README.txt.\nOther replay/evaluation sources retain their per-row licenses; see project THIRD_PARTY.md.\nThe Apache2.0 code license does not replace dataset licenses.\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
