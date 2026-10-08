"""Auditable consolidation of chosen-action outcome logs."""
from __future__ import annotations

import math
from collections import Counter, defaultdict


def consolidate_feedback(rows, logs, tolerance=1e-12):
    """Recover candidate rewards only where logged exploration observed them.

    ``rows`` needs only IDs and candidate counts.  No complete reward table is
    accepted, which makes accidental access to unchosen outcomes impossible.
    Repeated deterministic observations must agree.
    """
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    counts = {row["id"]: len(row["request"]["choices"]) for row in rows}
    if len(counts) != len(rows):
        raise ValueError("duplicate feedback row ID")
    observed = defaultdict(dict); observation_counts = Counter()
    for record in logs:
        if not isinstance(record, dict) or record.get("row") not in counts:
            raise ValueError("feedback log references an unknown row")
        actions, rewards = record.get("actions"), record.get("rewards")
        if not isinstance(actions, list) or not isinstance(rewards, list) or len(actions) != len(rewards):
            raise ValueError("feedback actions and rewards differ")
        row_id = record["row"]
        for action, reward in zip(actions, rewards):
            if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < counts[row_id]:
                raise ValueError("feedback action is outside the candidate set")
            if (not isinstance(reward, (int, float)) or isinstance(reward, bool) or
                    not math.isfinite(reward) or not 0 <= reward <= 1):
                raise ValueError("feedback reward must be finite and in [0,1]")
            if action in observed[row_id] and abs(observed[row_id][action] - reward) > tolerance:
                raise ValueError("repeated deterministic feedback disagrees")
            observed[row_id][action] = float(reward); observation_counts[(row_id, action)] += 1
    complete, incomplete = {}, {}
    for row_id, count in counts.items():
        missing = [action for action in range(count) if action not in observed[row_id]]
        if missing:
            incomplete[row_id] = missing
        else:
            complete[row_id] = [observed[row_id][action] for action in range(count)]
    repetitions = list(observation_counts.values())
    report = {
        "rows": len(rows), "complete_rows": len(complete), "incomplete_rows": len(incomplete),
        "observed_candidate_pairs": len(observation_counts),
        "missing_candidate_pairs": sum(map(len, incomplete.values())),
        "minimum_observations_per_seen_candidate": min(repetitions) if repetitions else 0,
        "maximum_observations_per_seen_candidate": max(repetitions) if repetitions else 0,
    }
    return complete, incomplete, report
