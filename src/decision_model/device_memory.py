"""Portable, explicitly sampled device-allocation diagnostics."""
from __future__ import annotations


def allocated_bytes(torch_module, device):
    """Return current device allocation when the backend exposes it."""
    kind = getattr(device, "type", str(device))
    if kind == "cuda":
        return int(torch_module.cuda.memory_allocated(device))
    if kind == "mps":
        return int(torch_module.mps.driver_allocated_memory())
    return None


class AllocationSampler:
    """Track maxima only at named checkpoints; this is not a peak profiler."""

    def __init__(self, torch_module, device):
        self.torch = torch_module
        self.device = device
        self.samples = []

    def sample(self, stage, update=None):
        value = allocated_bytes(self.torch, self.device)
        if value is not None:
            self.samples.append({"stage": stage, "update": update, "allocated_bytes": value})
        return value

    def record(self):
        values = [sample["allocated_bytes"] for sample in self.samples]
        return {
            "kind": "fixed_checkpoint_samples_not_exact_peak",
            "backend": getattr(self.device, "type", str(self.device)),
            "samples": self.samples,
            "maximum_sampled_bytes": max(values) if values else None,
        }
