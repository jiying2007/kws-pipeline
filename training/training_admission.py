"""Consume reviewed source receipts before optimization, independent of Torch.

These checks bind supplied review records; they do not authenticate a listener.
The diagnostic lane is deliberately non-promotable, including its descendants.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from speech_label_admission import LINEAGE_FIELDS, REAL_MODE, validate_admission, require_sha
from audit_dataset import verify_manifest_lineage, lineage_path
from corpus_identity import (IDENTITY_FIELDS, audio_identity, canonical_hash, corpus_digest,
                             training_corpus_identity, training_manifest_audio_bindings,
                             require_audited_audio, validate_audio_identity)
from model_provenance import normalize_corpus

POLICY = "reviewed-source-training-consumption-v1"
DIAGNOSTIC = "synthetic-contract-test-only"
ANCESTRY_SCHEMA = 1
ANCESTRY_CORPUS_POLICY = "hashed-path-and-metadata-v1"


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
    result = {"policy": POLICY, "purpose": DIAGNOSTIC if diagnostic else "reviewed-ctc-training",
              "promotion_allowed": not diagnostic, "reviewed_rows": count,
              "manifests": receipts, "listener_authenticity_verified": False}
    if not diagnostic:
        result["dataset_ancestry"] = initial_dataset_ancestry(
            receipts, training_corpus_identity(paths), count,
        )
    return result


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
    validate_dataset_ancestry(value)
    return value


def _manifest_bindings(value: object, label: str, *, lineage: bool = False,
                       allow_missing_lineage: bool = False) -> Counter:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = []
    for row in value:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                or not row["name"].strip() or pathlib.Path(row["name"]).name != row["name"]):
            raise ValueError(f"{label} requires manifest basenames")
        digest = require_sha(row.get("sha256"), f"{label}.sha256")
        binding = (row["name"], digest)
        if lineage:
            lineage_hash = row.get("lineage_sha256")
            if lineage_hash is not None or not allow_missing_lineage:
                lineage_hash = require_sha(lineage_hash, f"{label}.lineage_sha256")
            if pathlib.Path(row["name"]).suffix.lower() == ".jsonl" and lineage_hash != digest:
                raise ValueError(f"{label} JSONL lineage must match its manifest hash")
            binding += (lineage_hash,)
        result.append(binding)
    return Counter(result)


def _ancestry_receipt(stages: list[dict]) -> dict:
    payload = {"schema_version": ANCESTRY_SCHEMA, "stages": copy.deepcopy(stages)}
    return {**payload, "sha256": canonical_hash(payload)}


def _private_identity(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ancestry_corpus(corpus: dict) -> dict:
    """Keep cumulative evidence without publishing ancestor private paths/IDs."""
    normalized = normalize_corpus(corpus, require_sha(corpus.get("corpus_sha256"), "source corpus hash"))
    if normalized != corpus:
        raise ValueError("source training corpus must use canonical identity fields")
    for row in normalized["recordings"]:
        for key in ("path", *IDENTITY_FIELDS):
            if key in row:
                row[key] = _private_identity(row[key])
    normalized["corpus_sha256"] = corpus_digest(normalized["recordings"])
    return normalized


def _ancestry_audit(dataset_audit: dict) -> dict:
    projected = copy.deepcopy(dataset_audit)
    for binding in projected["manifest_bindings"]:
        identity = validate_audio_identity(binding.get("audio_identity"), "dataset audit audio identity")
        rows = copy.deepcopy(identity["recordings"])
        for row in rows:
            row["path"] = _private_identity(row["path"])
        binding["audio_identity"] = audio_identity(rows)
    return projected


def initial_dataset_ancestry(manifests: list[dict], corpus: dict, reviewed_rows: int) -> dict:
    """Create a cold-start data receipt after reviewed manifest consumption.

    Dataset stages deliberately remain separate from the current optimization
    corpus. Reusing the same dataset does not erase earlier checkpoint links.
    """
    return _ancestry_receipt([{
        "manifests": copy.deepcopy(manifests), "corpus_identity": _ancestry_corpus(corpus),
        "source_corpus_sha256": corpus["corpus_sha256"],
        "corpus_identity_policy": ANCESTRY_CORPUS_POLICY, "reviewed_rows": reviewed_rows, "parent_checkpoint_sha256": None,
        "parent_ancestry_sha256": None,
    }])


def validate_dataset_ancestry(admission: dict) -> list[dict]:
    ancestry = admission.get("dataset_ancestry")
    if (not isinstance(ancestry, dict) or set(ancestry) != {"schema_version", "stages", "sha256"}
            or type(ancestry.get("schema_version")) is not int
            or ancestry["schema_version"] != ANCESTRY_SCHEMA):
        raise ValueError("reviewed training requires complete dataset ancestry")
    stages = ancestry.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("training dataset ancestry stages are missing")
    expected_fields = {"manifests", "corpus_identity", "source_corpus_sha256",
                       "corpus_identity_policy", "reviewed_rows",
                       "parent_checkpoint_sha256", "parent_ancestry_sha256"}
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or set(stage) != expected_fields:
            raise ValueError("training dataset ancestry stage fields are invalid")
        _manifest_bindings(stage["manifests"], "ancestor training manifests", lineage=True)
        if stage["corpus_identity_policy"] != ANCESTRY_CORPUS_POLICY:
            raise ValueError("ancestor corpus identity privacy policy mismatch")
        require_sha(stage["source_corpus_sha256"], "ancestor source corpus hash")
        corpus = stage["corpus_identity"]
        if not isinstance(corpus, dict):
            raise ValueError("ancestor training corpus identity is missing")
        normalized = normalize_corpus(corpus, require_sha(corpus.get("corpus_sha256"), "ancestor corpus hash"))
        if normalized != corpus:
            raise ValueError("ancestor training corpus must use canonical identity fields")
        for row in corpus["recordings"]:
            for key in ("path", *IDENTITY_FIELDS):
                if key in row:
                    value = row[key]
                    if not value.startswith("sha256:"):
                        raise ValueError("ancestor corpus must not expose private paths or identities")
                    require_sha(value[7:], "ancestor private identity hash")
        if (type(stage["reviewed_rows"]) is not int
                or stage["reviewed_rows"] != len(corpus["recordings"])):
            raise ValueError("ancestor reviewed rows differ from consumed corpus")
        # Check membership, row ordering, and exact recording coverage without
        # relying on an ancestor's manifest basename being globally unique.
        training_manifest_audio_bindings(stage["manifests"], corpus)
        if index == 0:
            if stage["parent_checkpoint_sha256"] is not None or stage["parent_ancestry_sha256"] is not None:
                raise ValueError("cold-start ancestry must not omit an earlier stage")
        else:
            require_sha(stage["parent_checkpoint_sha256"], "ancestor checkpoint hash")
            if stage["parent_ancestry_sha256"] != _ancestry_receipt(stages[:index])["sha256"]:
                raise ValueError("training dataset ancestry chain is incomplete or changed")
    if ancestry["sha256"] != _ancestry_receipt(stages)["sha256"]:
        raise ValueError("training dataset ancestry digest mismatch")
    if (stages[-1]["manifests"] != admission.get("manifests")
            or stages[-1]["reviewed_rows"] != admission.get("reviewed_rows")):
        raise ValueError("current training admission differs from final ancestry stage")
    return stages


def require_current_dataset(admission: dict, manifests: object, corpus: object,
                            *, warm_start_binding: object = None) -> list[dict]:
    """Bind ancestry to the dataset actually consumed by this checkpoint."""
    stages = validate_dataset_ancestry(admission)
    recorded = _manifest_bindings(manifests, "checkpoint training manifests")
    admitted = _manifest_bindings(admission["manifests"], "training admission manifests")
    if recorded != admitted:
        raise ValueError("checkpoint manifests differ from reviewed training admission")
    if (not isinstance(corpus, dict)
            or corpus.get("corpus_sha256") != stages[-1]["source_corpus_sha256"]
            or _ancestry_corpus(corpus) != stages[-1]["corpus_identity"]):
        raise ValueError("checkpoint corpus differs from reviewed training ancestry")
    parent = stages[-1]["parent_checkpoint_sha256"]
    if parent is None:
        if warm_start_binding is not None:
            raise ValueError("warm-start checkpoint is missing inherited dataset ancestry")
    elif (not isinstance(warm_start_binding, dict)
          or warm_start_binding.get("source_checkpoint_sha256") != parent):
        raise ValueError("warm-start checkpoint differs from dataset ancestry")
    return stages


def require_qualification_admission(training: object, selected_manifests: object,
                                     dataset_audit: object, reference_sha256: str) -> dict:
    """Bind current certification to the reviewed training and audited lineage.

    This validates metadata receipts, not listener authenticity. Historical
    provenance readers may omit admission; current certification may not.
    """
    if not isinstance(training, dict):
        raise ValueError("qualification training metadata is missing")
    admission = require_promotable_admission(training.get("admission"))
    recorded = _manifest_bindings(training.get("manifests"), "training.manifests")
    selected = _manifest_bindings(selected_manifests, "selected training manifests")
    admitted = _manifest_bindings(admission["manifests"], "training admission manifests")
    if admitted != recorded or admitted != selected:
        raise ValueError("qualification manifest identities differ from training admission")
    stages = require_current_dataset(
        admission, training.get("manifests"), training.get("corpus_identity"),
        warm_start_binding=training.get("warm_start_binding"),
    )
    admitted_lineage = _manifest_bindings(admission["manifests"], "training admission", lineage=True)
    selected_lineage = _manifest_bindings(selected_manifests, "selected training", lineage=True)
    if admitted_lineage != selected_lineage:
        raise ValueError("selected training lineage differs from training admission")
    if (not isinstance(dataset_audit, dict) or type(dataset_audit.get("schema_version")) is not int
            or dataset_audit["schema_version"] != 3):
        raise ValueError("qualification dataset audit schema_version must be 3")
    audited = _manifest_bindings(dataset_audit.get("manifest_bindings"), "dataset audit bindings",
                                 lineage=True, allow_missing_lineage=True)
    required_lineage = Counter()
    for stage in stages:
        required_lineage |= _manifest_bindings(stage["manifests"], "ancestor training", lineage=True)
    if required_lineage - audited:
        raise ValueError("dataset audit does not cover reviewed training lineage and all ancestors")
    audited_hashes = dataset_audit.get("audited_manifest_sha256s")
    if not isinstance(audited_hashes, list):
        raise ValueError("dataset audit manifest hashes are missing")
    hashes = Counter(require_sha(item, "dataset audit manifest hash") for item in audited_hashes)
    if hashes != Counter(binding[1] for binding in audited.elements()):
        raise ValueError("dataset audit manifest hashes differ from its lineage bindings")
    required = Counter(binding[1] for binding in selected.elements())
    required[require_sha(reference_sha256, "qualification reference hash")] += 1
    if required - hashes:
        raise ValueError("dataset audit does not cover every selected training/qualification manifest")
    # Deduplicate cumulative exposure while retaining every stage's provenance.
    # Path hashes are recomputed from the audit instead of publishing ancestors'
    # private path text. Within-stage manifest multiplicity is checked above.
    required_audio = {}
    for stage in stages:
        for binding in training_manifest_audio_bindings(stage["manifests"], stage["corpus_identity"]):
            required_audio[canonical_hash(binding)] = binding
    require_audited_audio(_ancestry_audit(dataset_audit), list(required_audio.values()))
    return admission


def inherit_warm_start_admission(admission: dict, ancestor: object, *,
                                 source_checkpoint_sha256: str | None = None,
                                 ancestor_manifests: object = None,
                                 ancestor_corpus_identity: object = None,
                                 ancestor_warm_start_binding: object = None) -> None:
    """Carry every reviewed ancestor dataset, or permanently taint promotion.

    A legacy receipt alone cannot prove which audio produced inherited weights.
    Missing checkpoint/corpus/lineage evidence remains explicitly unverified.
    """
    try:
        require_promotable_admission(ancestor)
        source_hash = require_sha(source_checkpoint_sha256, "warm-start source checkpoint hash")
        parents = require_current_dataset(
            ancestor, ancestor_manifests, ancestor_corpus_identity,
            warm_start_binding=ancestor_warm_start_binding,
        )
        current = validate_dataset_ancestry(admission)
        if len(current) != 1:
            raise ValueError("warm-start ancestry was already inherited")
        stage = copy.deepcopy(current[0])
        stage["parent_checkpoint_sha256"] = source_hash
        stage["parent_ancestry_sha256"] = ancestor["dataset_ancestry"]["sha256"]
        admission["dataset_ancestry"] = _ancestry_receipt([*parents, stage])
    except ValueError:
        admission["promotion_allowed"] = False
        admission["ancestor_admission"] = "diagnostic-or-unverified"
