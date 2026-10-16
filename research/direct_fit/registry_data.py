"""Narrow exposed-fit consumer of the existing pinned sixteen-candidate registry.

Never modifies production admission or source roots. The historical thirteen
remain excluded: an ID join is not a same-PCM recognizer receipt. This consumer
reads no ZIP until all declared archive and source-rights paths exist.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types

from .data import TOKEN_MAP, WAKE, whole_source_features

PURPOSE = "same-pcm-registry-exposed-positive-fit-v1"
REGISTRY_PATH = "research/kws-candidate-registry-v1/candidates.json"
VERIFIER_PATH = "research/kws-candidate-registry-v1/build_registry.py"
REGISTRY_SHA256 = "28033edbbc1e444eae27c33af2e060d1654c4b7a072e65401b055fc27fc3b1b0"
VERIFIER_SHA256 = "9d37fb852cdd5048a4d5c88f9e21efe2b3c1e18f25244babb2d811675ea9fbae"
ARCHIVES = (
    "asr-11899008382.zip", "qwen06-11895552988.zip",
    "qwen17-11895324049.zip", "kw1-supplement-11942325284.zip",
)


def select_actual_rows(registry):
    """Metadata projection only, not byte verification or training permission."""
    if any(registry.get(key) is not False for key in
           ("training_admitted", "human_gold", "source_trust_root_admitted")):
        raise ValueError("registry must retain its original non-admission status")
    selected, excluded = [], []
    for row in registry["records"]:
        if (any(row.get(key) is not False for key in ("training_admitted", "human_gold",
                 "source_trust_root_admitted", "evaluation_gold", "history_complete"))
                or row.get("speaker_id") is not None
                or any(row.get("role_eligibility", {}).get(key) is not False
                       for key in ("calibration", "selection", "sealed_test"))):
            raise ValueError("candidate evidence or eligibility was upgraded")
        asr = row.get("asr", [])
        pcm = row["derived_pcm"]["sha256"]
        if (row.get("same_pcm_asr_binding_verified") is not True or len(asr) != 3
                or {a.get("family") for a in asr} != {"qwen_asr", "sensevoice", "whisper_small"}
                or any(a.get("binding_status") != "three_recognizer_pcm_receipts_verified"
                       or a.get("bound_pcm_sha256") != pcm for a in asr)):
            excluded.append({"record_id": row["record_id"],
                             "reason": "no-complete-same-PCM-three-recognizer-chain"})
            continue
        actual = [a.get("normalized_text") for a in asr]
        if (any(type(a.get("raw_text")) is not str or not a["raw_text"] for a in asr)
                or any(type(text) is not str for text in actual) or len(set(actual)) != 1 or actual[0] not in ("你好小窝", "小窝小窝")):
            excluded.append({"record_id": row["record_id"], "reason": "actual-transcripts-not-unanimous-canonical-keyword"})
            continue
        # Do not read target_text or target_class: these are generation intent.
        target = 1 if actual[0] == "你好小窝" else 2
        selected.append((row, target))
    if not selected:
        raise ValueError("no same-PCM actual-label candidates; historical ID joins stay excluded")
    return selected, excluded


def validate_inputs(value):
    if type(value) is not dict or set(value) != {"artifacts", "rights_bindings"}:
        raise ValueError("registry inputs require artifacts and explicit source rights_bindings")
    if type(value["artifacts"]) is not str or not value["artifacts"]:
        raise ValueError("registry artifact directory required")
    bindings = value["rights_bindings"]
    if type(bindings) is not list or not 1 <= len(bindings) <= 16:
        raise ValueError("source rights_bindings are missing")
    for binding in bindings:
        if (type(binding) is not dict or set(binding) != {"record_id", "path", "sha256"}
                or any(type(binding[key]) is not str or not binding[key] for key in binding)
                or len(binding["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in binding["sha256"])):
            raise ValueError("invalid source rights path/hash binding")
    if len({b["record_id"] for b in bindings}) != len(bindings):
        raise ValueError("duplicate source rights binding")


def read_rights(bindings, selected):
    """Bind supplied rights bytes using the existing retained-source schema.

    Hash/type checks are not legal adjudication or a production trust root.
    A real retained source-rights document is an input, never generated here.
    """
    by_id = {binding["record_id"]: binding for binding in bindings}
    if set(by_id) != {row["record_id"] for row, _ in selected}:
        raise ValueError("rights bindings must exactly cover the selected actual-label sources")
    result = []
    for row, _ in selected:
        binding = by_id[row["record_id"]]
        path = Path(binding["path"])
        if path.stat().st_size > 65536:
            raise ValueError("source rights document exceeds 64KiB")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError("source rights bytes differ from the declared binding: " + str(path))
        def unique(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate source rights key: " + key)
                result[key] = item
            return result
        def invalid(value):
            raise ValueError("non-finite source rights JSON: " + value)
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
        if type(value) is not dict:
            raise ValueError("source rights must be an object")
        source_ids = value.get("source_ids")
        if source_ids is not None and (type(source_ids) is not list or not source_ids
                or any(type(item) is not str or not item for item in source_ids)
                or len(set(source_ids)) != len(source_ids)):
            raise ValueError("invalid shared source-rights membership")
        covers_source = (value.get("source_id") == row["slot_id"] or
                         source_ids is not None and row["slot_id"] in source_ids)
        if (value.get("schema") != "retained-asr-source-rights-v1"
                or value.get("training_use_allowed") is not True
                or ("training_use_prohibited" in value and value["training_use_prohibited"] is not False)
                or not covers_source
                or value.get("source_family_id") != row["source"]["campaign_id"]
                or type(value.get("license_id")) is not str or not value["license_id"].strip()):
            raise ValueError("retained source rights do not declare training permission for " + row["record_id"])
        result.append({**binding, "rights": value})
    return result


def load_registry_development(config, rt, repo):
    from kws_vocab import load_tokens
    from corpus_identity import canonical_hash
    inputs = config["registry_inputs"]
    validate_inputs(inputs)
    if config["label_reuse_purpose"] != PURPOSE:
        raise ValueError("explicit same-PCM exposed-fit purpose required")
    if config["coverage_mode"] != "incomplete-coverage-fit-only":
        raise ValueError("this registry path is explicitly incomplete-coverage positive fit only")
    registry_path, verifier_path = repo / REGISTRY_PATH, repo / VERIFIER_PATH
    needed = [registry_path, verifier_path, Path(config["tokens"])]
    needed += [Path(inputs["artifacts"]) / name for name in ARCHIVES]
    needed += [Path(b["path"]) for b in inputs["rights_bindings"]]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise ValueError("missing exposed-fit input files: " + "; ".join(missing))
    registry_raw, verifier_raw = registry_path.read_bytes(), verifier_path.read_bytes()
    if (hashlib.sha256(registry_raw).hexdigest() != REGISTRY_SHA256 or
            hashlib.sha256(verifier_raw).hexdigest() != VERIFIER_SHA256):
        raise ValueError("fixed registry/verifier source SHA256 mismatch")
    registry = json.loads(registry_raw)
    selected, excluded = select_actual_rows(registry)
    rights = read_rights(inputs["rights_bindings"], selected)
    if load_tokens(Path(config["tokens"])) != TOKEN_MAP:
        raise ValueError("exact five-token vocabulary required")
    verifier = types.ModuleType("direct_fit_verified_registry")
    verifier.__file__ = str(verifier_path)
    # Execute the single fixed public source snapshot, never an arbitrary path.
    exec(compile(verifier_raw, str(verifier_path), "exec"), verifier.__dict__)
    consumed = verifier.verify_inputs(registry, {"repo": repo, "artifacts": Path(inputs["artifacts"])})
    pcm_by_id = {row["record_id"]: row for row in consumed}
    if len(pcm_by_id) != len(consumed) or set(pcm_by_id) != {r["record_id"] for r in registry["records"]}:
        raise ValueError("registry verifier PCM membership mismatch")
    torch, _, _, features, frontend, _, _ = rt
    items, recordings = [], []
    for row, target in selected:
        consumed_row = pcm_by_id[row["record_id"]]
        raw = consumed_row["pcm_bytes"]
        if (type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != row["derived_pcm"]["sha256"]
                or len(raw) != row["derived_pcm"]["bytes"]):
            raise ValueError("selected source PCM differs from the regenerated registry")
        acoustic, window = whole_source_features(raw, config["window_frames"],
            torch=torch, features=features, frontend=frontend)
        ids = [TOKEN_MAP[t] for t in WAKE[target]]
        items.append((acoustic, torch.tensor(ids, dtype=torch.long), target))
        recordings.append({"recording": row["record_id"], "class_target": target,
                           "actual_text": row["asr"][0]["normalized_text"],
                           "target_ids": ids, "registry_record": row, **window})
    receipt = {"schema": "same-pcm-registry-exposed-positive-fit-data-v1",
        "promotion_allowed": False, "training_admitted": False, "source_trust_root_admitted": False,
        "human_gold": False, "independent_evaluation": False, "history_complete": False,
        "purpose": PURPOSE, "coverage": {str(k): sum(item[2] == k for item in items) for k in range(3)},
        "window_frames": config["window_frames"], "registry_sha256": REGISTRY_SHA256,
        "verifier_sha256": VERIFIER_SHA256, "input_archives": registry["input_archives"],
        "source_rights_bindings": rights, "excluded_records": excluded, "recordings": recordings,
        "token_sha256": hashlib.sha256(Path(config["tokens"]).read_bytes()).hexdigest(),
        "label_basis": "same-PCM-three-ASR-actual-transcript; not human truth",
        "rights_scope": "bound supplied source-rights declarations; not production admission or legal adjudication"}
    receipt["data_sha256"] = canonical_hash(receipt)
    return items, receipt
