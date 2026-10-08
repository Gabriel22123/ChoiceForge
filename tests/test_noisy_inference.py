from pathlib import Path
import sys
import unittest

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from noisy_inference_diagnostic import reconstruct


class NoisyInferenceTests(unittest.TestCase):
    def test_temperature_inversion_preserves_raw_distribution(self):
        logits = torch.tensor([-2.1, .3, 1.4], dtype=torch.float64)
        calibrated = (logits / 1.8).softmax(-1).tolist()
        saved = [{"label": "b", "probabilities": dict(zip(("a", "b", "c"), calibrated))}]
        recovered, label = reconstruct(saved, 1.8)[0]
        self.assertEqual(label, 1)
        self.assertTrue(torch.allclose(recovered.softmax(-1), logits.softmax(-1), atol=1e-12, rtol=0))

    def test_refuses_underflowed_probability_information(self):
        with self.assertRaisesRegex(ValueError, "Cannot recover"):
            reconstruct([{"label": "a", "probabilities": {"a": 1., "b": 0.}}], 1.)


if __name__ == "__main__":
    unittest.main()
