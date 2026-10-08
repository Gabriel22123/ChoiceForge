import math
import unittest

from scripts.task_gradient_control_report import percentile, recalculate


class TaskGradientControlReportTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([0.0, 10.0, 20.0], .5), 10.0)
        self.assertAlmostEqual(percentile([0.0, 10.0, 20.0], .95), 19.0)

    def test_saved_temperature_probabilities_can_recover_raw_metrics(self):
        raw = [.8, .2]
        temperature = 2.0
        scaled_weights = [value ** (1 / temperature) for value in raw]
        total = sum(scaled_weights)
        scaled = [value / total for value in scaled_weights]
        rows = [{
            "id": "case", "label": "yes", "source": "example",
            "request": {"task": "choose", "context": "data", "choices": [
                {"id": "yes", "description": "yes"}, {"id": "no", "description": "no"},
            ]},
        }]
        predictions = [{"id": "case", "label": "yes",
                        "probabilities": {"yes": scaled[0], "no": scaled[1]}}]
        result = recalculate(rows, predictions, temperature)
        self.assertAlmostEqual(result["nll"], -math.log(.8), places=7)
        self.assertAlmostEqual(result["brier"], .08, places=7)


if __name__ == "__main__":
    unittest.main()
