import unittest

from decision_model.prompting import render_decoder_candidate
from scripts.prepare_arc_blind import make_request


class ArcBlindPreparationTest(unittest.TestCase):
    def test_decoder_candidate_prompt_excludes_ids_and_other_choices(self):
        request = make_request("Why?", ["first answer", "second answer"])
        text = render_decoder_candidate(request, 1)
        self.assertIn("second answer", text)
        self.assertNotIn("first answer", text)
        self.assertNotIn("option_1", text)
        self.assertTrue(text.endswith("Decision score:"))

    def test_request_uses_semantic_candidate_descriptions(self):
        request = make_request("Question", ["A", "B", "C"])
        self.assertEqual([choice["id"] for choice in request["choices"]],
                         ["option_0", "option_1", "option_2"])
        self.assertEqual([choice["description"] for choice in request["choices"]], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
