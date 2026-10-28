"""Recording/source-family identity namespace shared by split consumers.

These are lineage edges, not alternate spellings that must agree within a row:
a recording can name both its own source ID and one or more ancestor families.
Only explicit recording lineage is included. Provider/engine metadata is not a
person, session, or recording identity.
"""
from __future__ import annotations

SOURCE_ID_FIELDS = (
    "source_id", "source_family", "source_family_id", "family_id",
    "derivation_family", "derivation_family_id",
)
SOURCE_FAMILY_FIELDS = SOURCE_ID_FIELDS[1:]
LINEAGE_FIELDS = (*SOURCE_FAMILY_FIELDS, "speaker_id", "reference_audio_sha256", "session_id")


def source_identity_namespace(field: str) -> str:
    return "source" if field in SOURCE_ID_FIELDS else field


def validate_source_identity_values(row: dict, label: str) -> None:
    """Validate supplied recording lineage without treating unknowns as proof."""
    if not isinstance(row, dict):
        raise ValueError(f"{label} must be an object")
    for field in SOURCE_ID_FIELDS:
        if field in row:
            value = row[field]
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{label}.{field} must be canonical non-empty text")
