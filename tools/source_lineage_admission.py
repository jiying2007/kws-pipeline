"""Current source-alias checks layered on the existing speech receipt contract.

The original speech_label_admission bytes remain pinned by the historical
offline ASR reader. Current consumers apply these additional lineage checks
without repinning that reader or weakening its receipt/source predicates.
"""
from __future__ import annotations

from source_identity import LINEAGE_FIELDS, SOURCE_ID_FIELDS, validate_source_identity_values
from speech_label_admission import (
    REAL_MODE, latest_review, validate_admission as _validate_admission,
    validate_review_record as _validate_review_record,
)


def validate_review_record(row: dict) -> None:
    _validate_review_record(row)
    validate_source_identity_values(row, "review")


def validate_admission(admission: object, *, mode: str, provenance: dict, **kwargs) -> None:
    _validate_admission(admission, mode=mode, provenance=provenance, **kwargs)
    if mode != REAL_MODE:
        return
    # The original validator has already verified the exact receipt and, for
    # machine labels, the live source resolver and complete trust-root binding.
    from speech_asr_consensus import has_machine_consensus
    latest = (admission["consensus_record"] if has_machine_consensus(admission)
              else latest_review(admission["review_history"]))
    validate_source_identity_values(latest, "receipt")
    validate_source_identity_values(provenance, "provenance")
    for field in SOURCE_ID_FIELDS:
        if latest.get(field) != provenance.get(field):
            raise ValueError(f"audio admission {field} mismatch")
