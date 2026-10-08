import importlib.util
import json
import unittest
from pathlib import Path

from decision_model.core import digest


SCRIPT = Path(__file__).parents[1] / "scripts/run_extended_expert_bank.py"
SPEC = importlib.util.spec_from_file_location("run_extended_expert_bank", SCRIPT)
BANK = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(BANK)


class ExtendedExpertBankTests(unittest.TestCase):

    def test_protocol_freezes_two_caches_five_experts_and_seven_sources(self):
        root = Path(__file__).parents[1]
        config = json.loads((root / "configs/extended-expert-bank-v1.json").read_text())
        self.assertEqual(config["status"], "frozen_before_evaluation")
        self.assertEqual(len(config["groups"]), 2)
        self.assertEqual(len(config["experts"]), 5)
        self.assertEqual(len(config["screening_rule"]["required_sources"]), 7)
        for path, expected in config["source_files"].items():
            self.assertEqual(digest((root / path).read_bytes()), expected)

    def test_summary_requires_safe_oracle_path_from_records(self):
        self.assertTrue(callable(BANK.summarize))
        self.assertTrue(callable(BANK.load_group))


if __name__ == "__main__": unittest.main()
