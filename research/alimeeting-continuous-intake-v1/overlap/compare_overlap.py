#!/usr/bin/env python3
"""Extract existing metadata identities; compare a later AliMeeting audit.

Standard library only. Never opens audio, imports repository code, downloads,
trains, runs inference, or modifies source receipts. A successful comparison is
NOT a declaration of speaker, session, source ancestry, or corpus independence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

HEX = re.compile(r"[0-9a-f]{64}\Z")
NAME_PATTERN = re.compile(r"alimeeting|ali[_ -]meeting|eval_ali_(?:far|near)|R\d{4}_M\d{4}", re.I)
INDEX_SCHEMA = "historical-audio-identity-index-v1"
HASH_NAMESPACE = {
    "file_sha256": "file_bytes", "wav_sha256": "file_bytes",
    "source_wav_sha256": "file_bytes", "pcm_sha256": "mono_16khz_pcm16_bytes",
    "source_pcm_sha256": "mono_16khz_pcm16_bytes",
}
PROVENANCE = "models/registry/model-749187ec1d66/xiaowo-model-provenance.json"
INVENTORY = "research/effect-data-inventory-v2/report.json"
DOMAIN = "research/known-far-diagnostic-v1/original-domain-index.jsonl.gz"
LIMITS = [
    "Only exact declared whole-file and mono 16-kHz PCM16 hash identities are compared.",
    "Historical metadata is retained evidence, not live reverification of historical waveform bodies.",
    "No match does not exclude excerpts, other channels, gain changes, resampling, remixes, or transformed derivatives.",
    "Historical speaker/session identities are missing; no exact match cannot establish speaker/session independence.",
    "Local current training manifests are incomplete; this is not an exhaustive audit of all training corpora or pretrained models.",
    "AliMeeting transcript/negative-label review and release/license decisions are outside this comparison.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def binding(path, root=None):
    path = Path(path)
    return {"path": str(path.relative_to(root) if root else path),
            "sha256": sha(path), "bytes": path.stat().st_size}


def hash_value(value, field):
    require(isinstance(value, str) and HEX.fullmatch(value), "invalid SHA256 in " + field)
    return value


def entry(row, record_id, fields=("file_sha256", "pcm_sha256")):
    result = {"record_id": record_id, "hashes": {field: hash_value(row[field], field)
              for field in fields if field in row}}
    for key in ("path", "source_path", "audio_root_id", "manifest", "frames", "duration_s",
                "source_family_id", "family_id", "source_id", "speaker_id", "session_id",
                "derivation_family_id", "source_collection", "split", "planned_split"):
        if key in row:
            result[key] = row[key]
    return result


def dataset(name, records, sources, evidence):
    counts = {field: len({r["hashes"][field] for r in records if field in r["hashes"]})
              for field in sorted(HASH_NAMESPACE)
              if any(field in r["hashes"] for r in records)}
    return {"name": name, "record_count": len(records), "unique_hash_counts": counts,
            "sources": sources, "evidence": evidence, "records": records}


def extract(repository, nearby=None):
    root = Path(repository).resolve()
    inputs = {}
    def load(rel, jsonl=False):
        inputs[rel] = binding(root / rel, root)
        return read_jsonl(root / rel) if jsonl else read_json(root / rel)

    provenance = load(PROVENANCE)
    train = provenance["training"]["corpus_identity"]["recordings"]
    report = load(INVENTORY)
    measured = report["rows"]
    development_rel = "research/effect-data-inventory-v2/development.jsonl"
    development = load(development_rel, True)
    require(measured == development, "inventory rows differ from development manifest")
    require(len(train) == 1856, "historical training record count changed")
    require(len(measured) == 385, "measured inventory record count changed")
    require(report["historical_training_metadata"]["source_sha256"] == inputs[PROVENANCE]["sha256"],
            "inventory model-provenance binding mismatch")
    for label, rows in (("historical training", train), ("measured inventory", measured)):
        for field in ("file_sha256", "pcm_sha256"):
            require(len({hash_value(x[field], field) for x in rows}) == len(rows), label + " has duplicate hashes")
    # Reproduce the retained corpus-identity digest without importing repository code.
    normalized = [{key: row.get(key) for key in ("recording", "path", "file_sha256", "pcm_sha256", "frames")}
                  | {key: row[key] for key in ("speaker_id", "session_id", "source_id", "room_id", "device_id") if key in row}
                  for row in train]
    corpus_digest = hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(corpus_digest == provenance["training"]["corpus_identity"]["corpus_sha256"], "corpus digest mismatch")
    domain = load(DOMAIN, True)
    require(len(domain) == 240, "FAR domain count changed")
    pools, pool_paths = {}, []
    for seed in (1103, 3301):
        rel = f"research/known-far-diagnostic-v1/original-injections-{seed}.jsonl.gz"
        pool_paths.append(rel)
        pools[seed] = load(rel, True)
    sets = [{hash_value(row["source_sha256"], "source_sha256") for row in pools[seed]} for seed in (1103, 3301)]
    require(sets[0] == sets[1] and len(sets[0]) == 224, "FAR schedule source identity set changed")
    by_measured = {row["file_sha256"]: row for row in measured}
    by_domain = {row["wav_sha256"]: row for row in domain}
    require(sets[0] <= by_measured.keys() and sets[0] <= by_domain.keys(), "FAR source lacks measured/domain hash bridge")
    injected = []
    for digest in sorted(sets[0]):
        row = by_measured[digest]
        original = by_domain[digest]
        record = entry(row, row["record_id"], ("file_sha256", "pcm_sha256", "source_pcm_sha256"))
        record["hashes"]["source_wav_sha256"] = hash_value(original["source_wav_sha256"], "source_wav_sha256")
        require(row["source_pcm_sha256"] == original["source_pcm_sha256"], "FAR source PCM lineage mismatch")
        record["original_schedule_source_path"] = original["path"]
        record["source_family_id"] = original["family_id"]
        record["pool_occurrences"] = {str(seed): sum(x["source_sha256"] == digest for x in pools[seed]) for seed in (1103, 3301)}
        injected.append(record)
    source_rows = {}
    for row in domain:
        digest = row["source_wav_sha256"]
        current = {"path": row["source_path"], "file_sha256": digest,
                   "pcm_sha256": row["source_pcm_sha256"], "family_id": row["family_id"], "split": row["split"]}
        require(digest not in source_rows or source_rows[digest] == current, "conflicting original source identity")
        source_rows[digest] = current
    require(len(source_rows) == 120, "original FAR source count changed")
    native = report["inventory"]["noncanonical_source_files"]
    require(len(native) == 22, "native source count changed")
    datasets = [
        dataset("historical_training_1856", [entry(row, row["recording"]) for row in train], [PROVENANCE],
                "retained training provenance; no waveform bodies reread in this task"),
        dataset("far_injected_224", injected, pool_paths + [DOMAIN, INVENTORY],
                "224 shared injected domain WAV identities; PCM bridged by exact file SHA256 to retained measured inventory"),
        dataset("far_original_sources_120", [entry(row, row["family_id"]) for _, row in sorted(source_rows.items())], [DOMAIN],
                "120 original source files/families in frozen domain metadata"),
        dataset("measured_inventory_385", [entry(row, row["record_id"], tuple(HASH_NAMESPACE)) for row in measured], [INVENTORY, development_rel],
                "retained inventory previously measured actual mono 16-kHz PCM16 bytes; not reread by this extraction"),
        dataset("noncanonical_native_sources_22", [entry(row, row["audio_root_id"] + ":" + row["path"], ("file_sha256",)) for row in native], [INVENTORY],
                "native non-PCM16 source files; file-byte SHA256 only, no canonical PCM identity"),
    ]
    manifests = []
    for filename in ("train.jsonl", "development.jsonl", "sealed_acceptance.jsonl", "quarantine.jsonl"):
        rel = "research/effect-data-inventory-v2/" + filename
        rows = load(rel, True)
        manifests.append(inputs[rel] | {"rows": len(rows), "role": filename.removesuffix(".jsonl")})
    near_manifest_checks = []
    if nearby:
        for path in sorted(Path(nearby).glob("cli-*/train.jsonl")):
            near_manifest_checks.append(binding(path) | {"rows": len(read_jsonl(path))})
    # Search only explicit relevant receipt trees; include compressed JSONL metadata.
    searched, hits = [], []
    for subdir in ("models", "research", "training", "configs/training"):
        for path in sorted((root / subdir).rglob("*")):
            if not path.is_file() or not (path.suffix in (".json", ".jsonl", ".tsv", ".csv", ".md") or path.name.endswith(".jsonl.gz")):
                continue
            text = gzip.decompress(path.read_bytes()).decode("utf-8") if path.name.endswith(".gz") else path.read_text(encoding="utf-8")
            searched.append(path.relative_to(root).as_posix())
            matches = sorted({match.group() for match in NAME_PATTERN.finditer(text)})
            if matches:
                hits.append({"path": path.relative_to(root).as_posix(), "terms": matches})
    for rel, receipt in inputs.items():
        require(sha(root / rel) == receipt["sha256"], "input changed during extraction: " + rel)
    return {"schema": INDEX_SCHEMA, "repository_root": str(root), "sources": inputs,
            "hash_namespaces": HASH_NAMESPACE, "datasets": datasets,
            "training_receipt": {"manifest_counts": dict(Counter(row["manifest"] for row in train)),
                "declared_manifest_digests": provenance["training"]["manifests"],
                "training_manifest_bodies_present_in_candidate": sorted(str(path.relative_to(root)) for path in root.rglob("*.tsv") if path.name in ("train.tsv", "hard-negatives.tsv")),
                "corpus_digest_verified": corpus_digest,
                "known_speaker_ids": sum(bool(row.get("speaker_id")) for row in train),
                "known_session_ids": sum(bool(row.get("session_id")) for row in train)},
            "far_schedule_summary": {"entries": {str(seed): len(rows) for seed, rows in pools.items()},
                "shared_injected_file_hashes": 224, "shared_injected_ancestor_families": len({row["source_family_id"] for row in injected}),
                "complete_domain_rows": len(domain), "complete_original_source_families": len(source_rows)},
            "current_split_manifests": manifests, "nearby_training_manifest_checks": near_manifest_checks,
            "remote_metadata_only": report["remote_declared_not_materialized"],
            "name_scan": {"pattern": NAME_PATTERN.pattern, "files_scanned": len(searched), "scope": searched, "matches": hits,
                "meaning": "name-string scan only; renamed/missing provenance cannot be ruled out"},
            "independence_verified": False, "sealed_qualification_allowed": False, "limitations": LIMITS}


def compare(index, audit):
    require(index.get("schema") == INDEX_SCHEMA, "unsupported historical index schema")
    require(audit.get("schema") == "alimeeting-eval-audit-v1", "unsupported AliMeeting audit schema")
    rows = audit.get("rows")
    require(isinstance(rows, list) and rows, "audit requires nonempty rows")
    require(audit.get("unique_session_count") == len(rows), "audit session count mismatch")
    require(len({row.get("recording") for row in rows}) == len(rows), "duplicate audit recording")
    require(len({row.get("session_id") for row in rows}) == len(rows), "duplicate audit session")
    lookup = defaultdict(list)
    names = []
    for data in index["datasets"]:
        require(data["record_count"] == len(data["records"]), "index record count mismatch")
        for record in data["records"]:
            identity = {"dataset": data["name"], "record_id": record["record_id"]}
            for field, value in record["hashes"].items():
                require(field in HASH_NAMESPACE, "unknown historical hash field")
                lookup[(HASH_NAMESPACE[field], hash_value(value, field))].append(identity | {"historical_hash_field": field})
            for field in ("record_id", "path", "source_path", "original_schedule_source_path", "session_id", "source_id", "source_family_id", "family_id"):
                value = record.get(field)
                if isinstance(value, str) and value:
                    names.append((value, identity | {"historical_name_field": field}))
    results = []
    all_hits, all_name_hits = 0, 0
    for row in rows:
        require(isinstance(row.get("session_id"), str) and row["session_id"], "missing candidate session identity")
        require(isinstance(row.get("recording"), str) and row["recording"], "missing candidate recording identity")
        require(row.get("sample_rate_hz") == 16000 and row.get("channel_index") == 0, "unexpected mono hash semantics")
        hits = []
        for field in ("source_wav_sha256", "wav_sha256", "pcm_sha256"):
            digest = hash_value(row.get(field), field)
            hits.extend({"candidate_hash_field": field, "sha256": digest} | item
                        for item in lookup.get((HASH_NAMESPACE[field], digest), []))
        require(isinstance(row.get("member"), str) and row["member"], "missing candidate source member")
        tokens = {row["session_id"], Path(row["member"]).stem}
        name_hits = []
        for token in sorted(tokens):
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])", re.I)
            for value, identity in names:
                if pattern.search(value):
                    name_hits.append({"candidate_token": token, "historical_value": value} | identity)
        results.append({"recording": row["recording"], "session_id": row["session_id"],
                        "exact_hash_matches": hits, "source_session_name_matches": name_hits})
        all_hits += len(hits)
        all_name_hits += len(name_hits)
    return {"schema": "alimeeting-historical-overlap-comparison-v1", "candidate_recordings": len(rows),
            "exact_hash_match_count": all_hits, "source_session_name_match_count": all_name_hits,
            "comparison_status": "exact_overlap_found" if all_hits else "no_exact_hash_overlap_in_examined_metadata",
            "name_status": "name_matches_require_review" if all_name_hits else "no_source_session_name_matches_in_index",
            "dataset_counts": {data["name"]: data["record_count"] for data in index["datasets"]},
            "rows": results, "independence_verified": False, "sealed_qualification_allowed": False,
            "accepted_negative_duration_s_added": 0, "limitations": LIMITS}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--repository", type=Path, required=True)
    ex.add_argument("--nearby-effect-data", type=Path)
    ex.add_argument("--out", type=Path, required=True)
    co = sub.add_parser("compare")
    co.add_argument("--index", type=Path, required=True)
    co.add_argument("--audit", type=Path, required=True)
    co.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "extract":
        result = extract(args.repository, args.nearby_effect_data)
    else:
        result = compare(read_json(args.index), read_json(args.audit))
        result["inputs"] = {"index": binding(args.index), "audit": binding(args.audit)}
    write_new(args.out, result)
    print(json.dumps({"output": str(args.out), "sha256": sha(args.out),
                      "schema": result["schema"], "independence_verified": False}, sort_keys=True))


if __name__ == "__main__":
    main()
