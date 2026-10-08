import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from decision_model.core import digest, load_rows

spec = importlib.util.spec_from_file_location("prepare_snli", Path(__file__).resolve().parents[1] / "scripts/prepare_snli.py")
snli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snli)


class SnliPrepareTests(unittest.TestCase):
    def fixture(self, directory, extra=None):
        archive = directory / "source.zip"
        content = {}
        for split in ("train", "dev", "test"):
            content[split] = [dict(sentence1=f"Unique premise {split} {label}",
                                   sentence2=f"Hypothesis {label}", gold_label=label,
                                   pairID=f"{split}-{label}") for label in snli.DESCRIPTIONS]
        if extra:
            extra(content)
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("snli_1.0/README.txt", "Synthetic test fixture; not upstream data.")
            for split, rows in content.items():
                output.writestr(f"snli_1.0/snli_1.0_{split}.jsonl", "\n".join(json.dumps(row) for row in rows))
        return archive

    def test_rebuild_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.fixture(root)
            caps = dict.fromkeys(("train", "validation", "test"), 1)
            first = snli.transform(archive, root / "one", digest(archive.read_bytes()), caps)
            second = snli.transform(archive, root / "two", digest(archive.read_bytes()), caps)
            self.assertEqual(first["data_sha256"], second["data_sha256"])
            rows = load_rows(root / "one/cases.jsonl")
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(len(row["request"]["choices"]) == 3 for row in rows))
            self.assertTrue(all(row["provenance"][0]["license"] == "CC-BY-SA-4.0" for row in rows))

    def test_leakage_conflict_duplicate_and_unlabeled_filtering(self):
        def extra(content):
            content["train"] += [dict(sentence1=" Shared premise ", sentence2="First claim", gold_label="entailment"),
                                 dict(sentence1="Conflicted premise", sentence2="Same claim", gold_label="entailment"),
                                 dict(sentence1="Conflicted premise", sentence2="Same claim", gold_label="contradiction"),
                                 dict(sentence1="Unlabeled", sentence2="Claim", gold_label="-")]
            content["test"].append(dict(sentence1="shared PREMISE", sentence2="Another claim", gold_label="neutral"))
            content["train"].append(content["train"][0].copy())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.fixture(root, extra)
            result = snli.transform(archive, root / "out", digest(archive.read_bytes()), dict.fromkeys(("train", "validation", "test"), 10))
            self.assertEqual(result["summary"]["rows"], 9)
            self.assertEqual(result["excluded"], {"unknown_or_no_consensus_label_rows": 1, "cross_split_premise_rows": 2, "conflicting_label_rows": 2, "duplicate_pair_rows": 1})

    def test_reject_changed_archive_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "checksum"):
                snli.transform(archive, root / "out", "0" * 64, dict.fromkeys(("train", "validation", "test"), 1))
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
