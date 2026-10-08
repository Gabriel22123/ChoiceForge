import importlib.util
from pathlib import Path
import unittest

from decision_model.typed_api import typed_to_request


path = Path(__file__).parents[1] / "scripts/evaluate_release_endpoint.py"
spec = importlib.util.spec_from_file_location("evaluate_release_endpoint", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseEndpointContractTest(unittest.TestCase):
    def test_audit_payloads_cover_public_types_and_candidate_range(self):
        payloads = module.contract_payloads()
        self.assertEqual({payload["type"] for payload in payloads}, {"boolean", "choice", "score"})
        counts = [len(typed_to_request(payload)["choices"]) for payload in payloads]
        self.assertEqual(counts, [2, 4, 10, 32])


if __name__ == "__main__":
    unittest.main()
