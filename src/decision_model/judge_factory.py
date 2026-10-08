"""Select the non-generative decision implementation from a frozen config."""
from __future__ import annotations


def judge_class(config):
    architecture = config.get("architecture", "bidirectional_encoder")
    if architecture == "bidirectional_encoder":
        from .model import Judge
        return Judge
    if architecture == "causal_decoder":
        from .decoder_model import DecoderJudge
        return DecoderJudge
    raise ValueError("Unknown decision backbone architecture: " + str(architecture))


def make_judge(config, *, base_path=None, checkpoint=None, device="auto"):
    return judge_class(config)(config, base_path=base_path, checkpoint=checkpoint, device=device)


def verify_base_for_config(config, base_path):
    """Verify the local base with the lock belonging to the selected architecture."""
    architecture = config.get("architecture", "bidirectional_encoder")
    if architecture == "bidirectional_encoder":
        from .model import verify_local_base
        return verify_local_base(base_path)
    if architecture == "causal_decoder":
        from .decoder_model import verify_local_decoder_base
        return verify_local_decoder_base(base_path)
    raise ValueError("Unknown decision backbone architecture: " + str(architecture))
