import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from decision_model.core import digest
from decision_model.data import schema_rows, unsupported, get_sources


class DataTests(unittest.TestCase):
    def test_public_dialect_and_remote_reference(self):
        self.assertFalse(unsupported({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"integer"}))
        self.assertTrue(unsupported({"$ref":"https://example.org/private"}))

    def test_possible_worlds_all_three_outcomes(self):
        with tempfile.TemporaryDirectory() as d:
            source={"id":"s","repository":"https://github.com/example/public","revision":"a"*40,"license":"MIT"}
            p=Path(d)/"s"/"test.json";p.parent.mkdir()
            p.write_text(json.dumps([{"schema":{"type":"integer"},"tests":[
                {"data":1,"valid":True},{"data":2,"valid":True},
                {"data":"a","valid":False},{"data":"b","valid":False}]}]))
            source["files"]=[{"path":"test.json","sha256":digest(p.read_bytes())}]
            rows=list(schema_rows(source,d,Counter()))
            pairs=[r for r in rows if len(json.loads(r["request"]["evidence"])["possible_instances"])==2]
            self.assertEqual({r["label"] for r in pairs},{"satisfied","violated","insufficient"})
            self.assertEqual(len({r["group_id"] for r in rows}),1)

    def test_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"s"/"file.txt";p.parent.mkdir();p.write_text("tampered")
            lock=Path(d)/"lock.json"
            lock.write_text(json.dumps({"sources":[{"id":"s","repository":"https://github.com/example/public","revision":"a"*40,
                "license":"MIT","files":[{"path":"file.txt","sha256":digest(b"original")}]}]}))
            with self.assertRaisesRegex(ValueError,"modified"):get_sources(lock,d,True)


if __name__=="__main__":unittest.main()
