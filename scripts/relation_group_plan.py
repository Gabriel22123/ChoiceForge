"""Same rows, labels and replay positions; vary premise co-occurrence per update."""
from collections import defaultdict
import random


LABELS=('contradiction','neutral','entailment')


def make_orders(rows,mode,seed,epochs=2):
    if mode not in ('grouped','dispersed'): raise ValueError('Unknown grouping mode')
    groups=defaultdict(dict);replay=[]
    for row in rows:
        if row['split']!='train': continue
        if row['source']=='snli':
            if row['label'] in groups[row['group_id']]: raise ValueError('Duplicate label in premise group')
            groups[row['group_id']][row['label']]=row['id']
        else: replay.append(row['id'])
    if len(groups)!=256 or len(replay)!=256 or any(set(v)!=set(LABELS) for v in groups.values()):
        raise ValueError('Expected256 complete premise groups and256 replay rows')
    orders=[]
    for epoch in range(epochs):
        rng=random.Random(seed+1009*epoch);keys=sorted(groups);old=sorted(replay)
        rng.shuffle(keys);rng.shuffle(old);order=[]
        for update in range(128):
            batch=[]
            for label_index,label in enumerate(LABELS):
                offset=0 if mode=='grouped' else label_index*85
                batch += [groups[keys[(2*update+i+offset)%256]][label] for i in range(2)]
            batch += old[2*update:2*update+2]
            rng.shuffle(batch);order+=batch
        orders.append(order)
    return orders


def make_label_unbalanced_orders(rows,seed,epochs=2):
    """Keep dispersed premise/replay positions, but vary relation counts per update."""
    by_id={row['id']:row for row in rows}
    groups=defaultdict(dict)
    for row in rows:
        if row['split']=='train' and row['source']=='snli':
            groups[row['group_id']][row['label']]=row['id']
    if len(groups)!=256 or any(set(value)!=set(LABELS) for value in groups.values()):
        raise ValueError('Expected256 complete premise groups')
    orders=[]
    for epoch,balanced in enumerate(make_orders(rows,'dispersed',seed,epochs)):
        rng=random.Random(seed+700001+1009*epoch)
        remap={}
        for group_id in sorted(groups):
            shuffled=list(LABELS);rng.shuffle(shuffled)
            remap[group_id]=dict(zip(LABELS,shuffled))
        order=[]
        for identity in balanced:
            row=by_id[identity]
            if row['source']=='snli':
                identity=groups[row['group_id']][remap[row['group_id']][row['label']]]
            order.append(identity)
        orders.append(order)
    return orders
