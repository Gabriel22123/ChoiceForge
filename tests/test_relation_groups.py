from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from prepare_relation_groups import select_triples
from decision_model.core import digest
from decision_model.train import validate_epoch_orders


class RelationPreparationTests(unittest.TestCase):
    def test_complete_groups_official_heldout_conflicts_and_determinism(self):
        train=[]
        for premise in ('safe1','safe2','shared','conflict'):
            for label in ('entailment','neutral','contradiction'):
                train.append({'sentence1':premise,'sentence2':premise+' '+label,'gold_label':label})
        train.append(dict(train[0]))
        train.append({'sentence1':'conflict','sentence2':'conflict neutral','gold_label':'contradiction'})
        heldout=[{'sentence1':' SHARED ','sentence2':'unknown','gold_label':'-'}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'snli.zip'
            with zipfile.ZipFile(path,'w') as z:
                for split,rows in (('train',train),('dev',heldout),('test',[])):
                    z.writestr(f'snli_1.0/snli_1.0_{split}.jsonl','\n'.join(json.dumps(r) for r in rows))
            sha=digest(path.read_bytes());a,manifest=select_triples(path,sha,2)
            b,_=select_triples(path,sha,2)
            self.assertEqual(a,b)
            self.assertEqual(len(a),6)
            self.assertEqual(manifest['complete_eligible_groups_before_token_admission'],2)
            self.assertEqual(Counter(r['label'] for r in a),{'entailment':2,'neutral':2,'contradiction':2})
            self.assertTrue(all(json.loads(r['request']['context'])['premise'] in ('safe1','safe2') for r in a))
            self.assertTrue(all(r['split']=='train' and r['provenance'][0]['label_method']=='upstream_gold_label' for r in a))
            admitted,_=select_triples(path,sha,1,lambda request:json.loads(request['context'])['premise']=='safe2')
            self.assertTrue(all(json.loads(r['request']['context'])['premise']=='safe2' for r in admitted))
            with self.assertRaisesRegex(ValueError,'checksum'): select_triples(path,'0'*64,1)

    def test_frozen_orders_cannot_duplicate_omit_or_leak_heldout_rows(self):
        rows=[{'id':'a'},{'id':'b'}]
        self.assertIsNone(validate_epoch_orders(rows,{'epochs':1}))
        result=validate_epoch_orders(rows,{'epochs':2,'epoch_row_orders':[['b','a'],['a','b']]})
        self.assertEqual([[r['id'] for r in order] for order in result],[['b','a'],['a','b']])
        for orders in ([['a','a']],[['a','test']],[['a']],[['a','b'],['a','b']],[[{}]]):
            with self.assertRaises(ValueError): validate_epoch_orders(rows,{'epochs':1,'epoch_row_orders':orders})


if __name__=='__main__': unittest.main()
