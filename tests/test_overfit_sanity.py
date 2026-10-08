from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from overfit_sanity import balanced_batches


class BalancedSanityTests(unittest.TestCase):
    def test_every_update_is_balanced_and_every_example_has_equal_exposure(self):
        rows = [{"id": f"{label}-{i}", "label": label} for label in ("a", "b", "c") for i in range(8)]
        batches = list(balanced_batches(rows, 42, 80))
        self.assertTrue(all(Counter(r["label"] for r in batch) == {"a":2,"b":2,"c":2} for batch in batches))
        counts = Counter(r["id"] for batch in batches for r in batch)
        self.assertEqual(set(counts.values()), {20})
        self.assertEqual(len(counts), 24)
        self.assertEqual(batches, list(balanced_batches(rows, 42, 80)))
