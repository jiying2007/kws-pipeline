"""Text-only triage of saved ASR discrepancies; never an acceptance gate.

Inputs must already be normalized by the original comparator. This helper
performs no normalization, alignment, model call, audio read, or acceptance
decision. Readings are ordinary dictionary citation forms, not observed audio.
Only the reviewed differing characters in this six-clip panel are covered.
"""

READINGS = {
    "窝": ("wo", 1), "沃": ("wo", 4), "我": ("wo", 3),
    "屋": ("wu", 1), "小": ("xiao", 3), "叫": ("jiao", 4),
    "老": ("lao", 3), "今": ("jin", 1), "明": ("ming", 2),
}


def classify(intended, normalized_asr, *, status="success", completeness="complete"):
    """Return one conservative text relation, with no implied acoustic truth."""
    if status != "success" or completeness != "complete":
        return "unknown"
    if not isinstance(intended, str) or not isinstance(normalized_asr, str):
        return "unknown"
    if not intended or not normalized_asr:
        return "unknown"
    if intended == normalized_asr:
        return "lexical_match"
    if len(intended) != len(normalized_asr):
        return "unknown"  # No invented insert/delete alignment.
    changes = [(a, b) for a, b in zip(intended, normalized_asr) if a != b]
    if any(a not in READINGS or b not in READINGS for a, b in changes):
        return "unknown"  # No guessed readings for unreviewed characters.
    if any(READINGS[a][0] != READINGS[b][0] for a, b in changes):
        return "segmental_mismatch"
    if any(READINGS[a][1] != READINGS[b][1] for a, b in changes):
        return "segmental_agreement_tone_conflict"
    return "unknown"  # Different words with the same reading need separate review.
