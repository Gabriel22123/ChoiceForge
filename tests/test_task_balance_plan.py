import unittest
from collections import Counter

from scripts.task_balance_plan import make_orders, task_name


def row(identity, source, label, group=None):
    return {"id": identity, "source": source, "label": label, "group_id": group or identity, "split": "train"}


class TaskBalancePlanTests(unittest.TestCase):
    def rows(self):
        rows = []
        for label in ("different", "paraphrase"):
            rows += [row(f"p-{label}-{i}", "paws-wiki", label) for i in range(192)]
        for i in range(128):
            for label in ("contradiction", "neutral", "entailment"):
                rows.append(row(f"s-{i}-{label}", "snli", label, f"premise-{i}"))
        rows += [row(f"r-{i}", "clinc", f"intent-{i}") for i in range(256)]
        return rows

    def test_stratified_updates_and_matched_exposure(self):
        rows = self.rows(); by_id = {value["id"]: value for value in rows}
        mixed = make_orders(rows, "mixed", 42)[0]
        stratified = make_orders(rows, "stratified", 42)[0]
        self.assertEqual(Counter(mixed), Counter(stratified))
        self.assertNotEqual(mixed, stratified)
        patterns = set()
        for start in range(0, 1024, 8):
            block = [by_id[key] for key in stratified[start:start + 8]]
            self.assertEqual(Counter(task_name(value) for value in block),
                             Counter({"paws": 3, "snli": 3, "replay": 2}))
            self.assertEqual(Counter(value["label"] for value in block if value["source"] == "snli"),
                             Counter({"contradiction": 1, "neutral": 1, "entailment": 1}))
            self.assertEqual(len({value["group_id"] for value in block if value["source"] == "snli"}), 3)
            patterns.add(tuple(sorted(value["label"] for value in block if value["source"] == "paws-wiki")))
        self.assertEqual(patterns, {("different", "different", "paraphrase"),
                                    ("different", "paraphrase", "paraphrase")})

    def test_seed_changes_both_orders(self):
        rows = self.rows()
        for mode in ("mixed", "stratified"):
            self.assertNotEqual(make_orders(rows, mode, 42), make_orders(rows, mode, 43))


if __name__ == "__main__":
    unittest.main()
