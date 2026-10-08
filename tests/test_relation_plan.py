from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from relation_group_plan import make_orders,make_label_unbalanced_orders,LABELS


class RelationPlanTests(unittest.TestCase):
    def test_equal_rows_label_counts_replay_positions_and_premise_cooccurrence(self):
        rows=[{'id':f'{i}-{label}','group_id':str(i),'source':'snli','label':label,'split':'train'}
              for i in range(256) for label in LABELS]
        rows += [{'id':f'replay-{i}','source':'old','label':'x','split':'train'} for i in range(256)]
        by_id={r['id']:r for r in rows};plans={mode:make_orders(rows,mode,42) for mode in ('grouped','dispersed')}
        for mode,orders in plans.items():
            self.assertEqual(orders,make_orders(rows,mode,42))
            self.assertNotEqual(orders[0],orders[1])
            for order in orders:
                self.assertEqual(Counter(order),Counter(by_id.keys()))
                for start in range(0,1024,8):
                    batch=[by_id[k] for k in order[start:start+8]];snli=[r for r in batch if r['source']=='snli']
                    self.assertEqual(Counter(r['label'] for r in snli),dict.fromkeys(LABELS,2))
                    self.assertEqual(len(batch)-len(snli),2)
                    counts=Counter(r['group_id'] for r in snli)
                    self.assertEqual(sorted(counts.values()),[3,3] if mode=='grouped' else [1]*6)
        for grouped,dispersed in zip(plans['grouped'],plans['dispersed']):
            for a,b in zip(grouped,dispersed):
                self.assertEqual(by_id[a]['source'],by_id[b]['source'])
                self.assertEqual(by_id[a]['label'],by_id[b]['label'])
                if by_id[a]['source']=='old': self.assertEqual(a,b)
        self.assertNotEqual(plans['grouped'],make_orders(rows,'grouped',43))

    def test_unbalanced_labels_preserve_every_premise_and_replay_position(self):
        rows=[{'id':f'{i}-{label}','group_id':str(i),'source':'snli','label':label,'split':'train'}
              for i in range(256) for label in LABELS]
        rows += [{'id':f'replay-{i}','group_id':f'replay-{i}','source':'old','label':'x','split':'train'}
                 for i in range(256)]
        by_id={row['id']:row for row in rows}
        balanced=make_orders(rows,'dispersed',42)
        unbalanced=make_label_unbalanced_orders(rows,42)
        self.assertEqual(unbalanced,make_label_unbalanced_orders(rows,42))
        self.assertNotEqual(unbalanced,make_label_unbalanced_orders(rows,43))
        varying_batches=0
        for balanced_epoch,unbalanced_epoch in zip(balanced,unbalanced):
            self.assertEqual(Counter(unbalanced_epoch),Counter(by_id.keys()))
            for old,new in zip(balanced_epoch,unbalanced_epoch):
                self.assertEqual(by_id[old]['source'],by_id[new]['source'])
                self.assertEqual(by_id[old]['group_id'],by_id[new]['group_id'])
                if by_id[old]['source']!='snli': self.assertEqual(old,new)
            for start in range(0,1024,8):
                labels=Counter(by_id[key]['label'] for key in unbalanced_epoch[start:start+8]
                               if by_id[key]['source']=='snli')
                varying_batches += labels != dict.fromkeys(LABELS,2)
        self.assertGreater(varying_batches,200)


if __name__=='__main__': unittest.main()
