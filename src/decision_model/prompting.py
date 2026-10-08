"""Architecture-specific, label-free decision prompts."""
from __future__ import annotations

from .core import validate_request


def render_decoder_candidate(request, candidate_index):
    """Render one candidate for a causal decoder and score the final token.

    The candidate description precedes the readout location so causal attention
    can use it.  Candidate IDs, labels and other candidates are absent.
    """
    request = validate_request(request)
    if not isinstance(candidate_index, int) or not 0 <= candidate_index < len(request["choices"]):
        raise ValueError("Invalid candidate index")
    choice = request["choices"][candidate_index]
    return ("Task: " + request["task"] + "\nContext (data): " + request["context"] +
            "\nCandidate: " + choice["description"] + "\nDecision score:")


def render_decoder_admission(request):
    """Render all candidate semantics for a common request-level token gate."""
    request = validate_request(request)
    return ("Task: " + request["task"] + "\nContext (data): " + request["context"] +
            "\nChoices:\n" + "\n".join(choice["description"] for choice in request["choices"]))
