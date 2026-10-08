import unittest

from decision_model.decoder_model import DecoderJudge
from decision_model.judge_factory import judge_class, verify_base_for_config
from decision_model.model import Judge


class JudgeFactoryTest(unittest.TestCase):
    def test_legacy_config_defaults_to_encoder(self):
        self.assertIs(judge_class({}), Judge)

    def test_decoder_config_selects_decoder(self):
        self.assertIs(judge_class({"architecture": "causal_decoder"}), DecoderJudge)

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown decision backbone"):
            judge_class({"architecture": "sequence_to_sequence"})

    def test_unknown_architecture_verifier_is_rejected_before_io(self):
        with self.assertRaisesRegex(ValueError, "Unknown decision backbone"):
            verify_base_for_config({"architecture": "sequence_to_sequence"}, "/does/not/matter")


if __name__ == "__main__":
    unittest.main()
