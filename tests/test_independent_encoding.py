import types
import unittest

from decision_model.core import render
from decision_model.model import Judge


class IndependentEncodingTests(unittest.TestCase):
    def setUp(self):
        self.request = {"task": "Choose a tool.", "context": "Set a reminder.", "choices": [
            {"id": "id_one", "description": "Calendar schedules reminders."},
            {"id": "id_two", "description": "Calculator evaluates arithmetic."}]}

    def test_single_render_has_only_selected_candidate_and_no_ids(self):
        text = render(self.request, candidate_index=0)
        self.assertIn("Set a reminder.", text)
        self.assertIn("Calendar schedules reminders.", text)
        self.assertNotIn("Calculator", text)
        self.assertNotIn("id_one", text)
        self.assertEqual(text.count("<|mask|>"), 1)

    def test_both_modes_keep_full_request_token_budget(self):
        tokenizer = types.SimpleNamespace(mask_token="<|mask|>", mask_token_id=31,
            encode=lambda text, **kwargs: [1] + [31 if c == "§" else 4 + ord(c) % 20
                                                 for c in text.replace("<|mask|>", "§")] + [2])
        full_length = len(tokenizer.encode(render(self.request)))
        pair_length = len(tokenizer.encode(render(self.request, candidate_index=0)))
        self.assertLess(pair_length, full_length - 1)
        judge = Judge.__new__(Judge)
        judge.tokenizer = tokenizer
        for mode in ("joint", "independent"):
            judge.config = {"candidate_encoding": mode, "max_tokens": full_length - 1}
            with self.assertRaisesRegex(ValueError, "Input too long"):
                judge.encode(self.request)
        judge.config = {"candidate_encoding": "independent", "max_tokens": full_length}
        pairs = judge.encode(self.request)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(len(markers) == 1 for _, markers in pairs))


if __name__ == "__main__":
    unittest.main()
