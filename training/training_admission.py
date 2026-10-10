"""Consume reviewed source receipts before optimization, independent of Torch.

These checks bind supplied review records; they do not authenticate a listener.
The diagnostic lane is deliberately non-promotable, including its descendants.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from speech_label_admission import LINEAGE_FIELDS, REAL_MODE, validate_admission
from audit_dataset import verify_manifest_lineage, lineage_path

POLICY = "reviewed-source-training-consumption-v1"
DIAGNOSTIC = "synthetic-contract-test-only"


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_training_manifests(paths: list[pathlib.Path], token_map: dict[str, int],
                              *, diagnostic: bool = False) -> dict:
    if not paths:
        raise ValueError("training admission requires manifests")
    receipts = []
    count = 0
    for path in paths:
        entry = {"name": path.name, "sha256": digest(path)}
        if diagnostic:
            # Explicit local algorithm fixtures are never product evidence.
            entry["lineage_sha256"] = None
        else:
            rows = verify_manifest_lineage(path)
            if not rows:
                raise ValueError("reviewed training manifest must not be empty")
            for row in rows:
                provenance = row.get("speech_like_provenance")
                if not isinstance(provenance, dict):
                    raise ValueError("reviewed source provenance missing before CTC training")
                tokens = row.get("tokens")
                if not isinstance(tokens, list) or any(
                    not isinstance(token, str) or token not in token_map for token in tokens
                ):
                    raise ValueError("reviewed transcript contains unknown token")
                targets = row.get("target_ids")
                if not isinstance(targets, list) or any(type(item) is not int for item in targets):
                    raise ValueError("reviewed training target_ids must be canonical integers")
                if [token_map[token] for token in tokens] != targets or any(item <= 0 for item in targets):
                    raise ValueError("training targets differ from reviewed transcript")
                for field in LINEAGE_FIELDS:
                    if row.get(field) != provenance.get(field):
                        raise ValueError(f"training lineage {field} differs from reviewed source")
                if "source_id" in row and row["source_id"] != provenance.get("source_id"):
                    raise ValueError("training lineage source_id differs from reviewed source")
                validate_admission(
                    row.get("admission"), mode=REAL_MODE,
                    source_id=provenance.get("source_id"),
                    file_sha256=row.get("source_wav_sha256"),
                    pcm_sha256=row.get("source_pcm_sha256"),
                    text=row.get("actual_text"), tokens=tokens,
                    kind=row.get("kind"), keyword_id=row.get("keyword_id"),
                    provenance=provenance,
                )
                count += 1
            entry["lineage_sha256"] = digest(path if path.suffix.lower() == ".jsonl" else lineage_path(path))
        receipts.append(entry)
    return {"policy": POLICY, "purpose": DIAGNOSTIC if diagnostic else "reviewed-ctc-training",
            "promotion_allowed": not diagnostic, "reviewed_rows": count,
            "manifests": receipts, "listener_authenticity_verified": False}


def require_promotable_admission(value: object) -> dict:
    if not isinstance(value, dict) or value.get("policy") != POLICY:
        raise ValueError("new model promotion requires reviewed training admission")
    if value.get("purpose") != "reviewed-ctc-training" or value.get("promotion_allowed") is not True:
        raise ValueError("diagnostic or unreviewed training cannot be promoted")
    if "ancestor_admission" in value:
        raise ValueError("diagnostic or unverified ancestor cannot be promoted")
    if value.get("listener_authenticity_verified") is not False:
        raise ValueError("training receipt validation cannot authenticate a listener")
    if type(value.get("reviewed_rows")) is not int or value["reviewed_rows"] < 1:
        raise ValueError("new model promotion requires reviewed rows")
    manifests = value.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("training admission manifest bindings missing")
    for row in manifests:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError("invalid training admission manifest")
        for key in ("sha256", "lineage_sha256"):
            item = row.get(key)
            if not isinstance(item, str) or len(item) != 64 or any(c not in "0123456789abcdef" for c in item):
                raise ValueError("training admission hash binding invalid")
    return value


def inherit_warm_start_admission(admission: dict, ancestor: object) -> None:
    """Keep any diagnostic/unverified ancestor taint through all descendants."""
    try:
        require_promotable_admission(ancestor)
    except ValueError:
        admission["promotion_allowed"] = False
        admission["ancestor_admission"] = "diagnostic-or-unverified"
