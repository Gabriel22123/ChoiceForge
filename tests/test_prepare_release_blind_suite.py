import unittest

from scripts.prepare_release_blind_suite import deduplicate_eligible, model_input_key


class ReleaseBlindSelectionTest(unittest.TestCase):
    def request(self, order=("a", "b"), context="same"):
        descriptions = {"a": "first meaning", "b": "second meaning"}
        return {
            "task": "Choose the supported answer.",
            "context": context,
            "choices": [{"id": value, "description": descriptions[value]} for value in order],
        }

    def test_semantic_key_ignores_candidate_order_and_ids(self):
        left = self.request(("a", "b"))
        right = {
            "task": left["task"],
            "context": left["context"],
            "choices": [
                {"id": "renamed-b", "description": "second meaning"},
                {"id": "renamed-a", "description": "first meaning"},
            ],
        }
        self.assertEqual(model_input_key(left), model_input_key(right))

    def test_deduplication_is_input_only_and_deterministic(self):
        duplicate_late = ("z-key", 2, self.request(("a", "b")), 0)
        duplicate_early = ("a-key", 9, self.request(("b", "a")), 1)
        distinct = ("m-key", 1, self.request(context="different"), 0)
        selected, removed = deduplicate_eligible(
            [duplicate_late, distinct, duplicate_early])
        self.assertEqual(removed, 1)
        self.assertEqual([item[0] for item in selected], ["a-key", "m-key"])
        self.assertEqual(selected[0][3], 1)  # The answer is carried, never consulted.


if __name__ == "__main__":
    unittest.main()
