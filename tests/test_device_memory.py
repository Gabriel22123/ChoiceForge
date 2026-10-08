import unittest

from decision_model.device_memory import AllocationSampler, allocated_bytes


class Device:
    def __init__(self, kind):
        self.type = kind


class Memory:
    def __init__(self):
        self.values = iter((10, 25))

    def driver_allocated_memory(self):
        return next(self.values)


class Torch:
    def __init__(self):
        self.mps = Memory()


class DeviceMemoryTest(unittest.TestCase):
    def test_cpu_is_explicitly_unavailable(self):
        self.assertIsNone(allocated_bytes(Torch(), Device("cpu")))

    def test_sampler_records_fixed_checkpoints_and_maximum(self):
        sampler = AllocationSampler(Torch(), Device("mps"))
        sampler.sample("model_loaded")
        sampler.sample("after_backward", 1)
        self.assertEqual(sampler.record(), {
            "kind": "fixed_checkpoint_samples_not_exact_peak",
            "backend": "mps",
            "samples": [
                {"stage": "model_loaded", "update": None, "allocated_bytes": 10},
                {"stage": "after_backward", "update": 1, "allocated_bytes": 25},
            ],
            "maximum_sampled_bytes": 25,
        })


if __name__ == "__main__":
    unittest.main()
