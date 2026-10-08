from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from budget_warmup import make_plan


class BudgetPlanTests(unittest.TestCase):
    def rows(self):
        return [{'id':f'{i:04d}','split':'train','source':'snli' if i<768 else 'other',
                 'family':'inference' if i<768 else 'routing','label':str(i%3),
                 'request':{'choices':[{'id':str(j),'description':str(j)} for j in range(3)]}}
                for i in range(1024)]

    def test_equal_exposures_distinct_sampling_and_candidate_integrity(self):
        rows=self.rows()
        for seed in (42,43):
            for mode in ('small','mixed'):
                plan=make_plan(rows,mode,seed)
                self.assertEqual(len(plan),80)
                self.assertTrue(all(len(p['rows'])==6 for p in plan))
                counts=Counter(k for p in plan for k in p['rows'])
                self.assertEqual(sum(counts.values()),480)
                self.assertEqual(len(counts),24 if mode=='small' else 480)
                self.assertEqual(set(counts.values()),{20} if mode=='small' else {1})
                self.assertTrue(all(sorted(order)==['0','1','2'] for p in plan for order in p['choices']))
                self.assertEqual(plan,make_plan(rows,mode,seed))
                if mode=='small':
                    self.assertTrue(all(Counter(rows[int(k)]['label'] for k in p['rows'])=={'0':2,'1':2,'2':2} for p in plan))

    def test_heldout_rows_never_selected_and_seeds_change_plan(self):
        rows=self.rows()
        rows += [dict(rows[0],id='heldout',split='test')]
        for mode in ('small','mixed'):
            a=make_plan(rows,mode,42); b=make_plan(rows,mode,43)
            self.assertNotEqual(a,b)
            self.assertFalse(any('heldout' in p['rows'] for p in a+b))


if __name__=='__main__': unittest.main()
