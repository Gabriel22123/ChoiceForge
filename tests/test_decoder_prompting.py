import unittest

from decision_model.prompting import render_decoder_admission, render_decoder_candidate


class DecoderPromptingTest(unittest.TestCase):
    def setUp(self):
        self.request = {
            "task": "Choose the supported claim.",
            "context": "The light is red.",
            "choices": [
                {"id": "arbitrary-secret-id", "description": "The light is red."},
                {"id": "other-secret-id", "description": "The light is green."},
            ],
        }

    def test_only_selected_semantics_enter_prompt(self):
        value = render_decoder_candidate(self.request, 0)
        self.assertIn("The light is red.", value)
        self.assertNotIn("The light is green.", value)
        self.assertNotIn("arbitrary-secret-id", value)
        self.assertNotIn("other-secret-id", value)

    def test_candidate_permutation_does_not_change_semantic_prompt(self):
        original = render_decoder_candidate(self.request, 0)
        permuted = dict(self.request, choices=list(reversed(self.request["choices"])))
        self.assertEqual(original, render_decoder_candidate(permuted, 1))

    def test_rejects_bad_index(self):
        with self.assertRaisesRegex(ValueError, "Invalid candidate index"):
            render_decoder_candidate(self.request, 2)

    def test_admission_contains_all_semantics_but_no_ids(self):
        value = render_decoder_admission(self.request)
        self.assertIn("The light is red.", value)
        self.assertIn("The light is green.", value)
        self.assertNotIn("secret-id", value)


if __name__ == "__main__":
    unittest.main()
