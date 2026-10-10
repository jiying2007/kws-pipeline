#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import wave
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import audio_identity, canonical_audio_path, resolve_audio_path  # noqa: E402

SAMPLE_RATE_HZ = 16000
PCM_IDENTITY_FIELDS = ("pcm_sha256", "source_pcm_sha256", "original_pcm_sha256")
LINEAGE_IDENTITY_FIELDS = (
    "source_pcm_sha256", "original_pcm_sha256", "source_family_id", "family_id",
    "reference_audio_sha256", "derivation_family_id",
)
IDENTITY_FIELDS = (
    "speaker_id", "session_id", "source_id", "room_id", "device_id",
    *LINEAGE_IDENTITY_FIELDS,
)
HARD_IDENTITY_FIELDS = {"speaker_id", "session_id", "source_id", *LINEAGE_IDENTITY_FIELDS}
# These dimensions can be unknown even when all observed identities are disjoint.
# A provider, synthesis engine or voice setting is not a person/recording identity.
COVERAGE_FIELDS = (
    "source_pcm_sha256", "source_family_id", "speaker_id", "session_id",
    "reference_audio_sha256", "derivation_family_id",
)
SHA_RE = re.compile(r"[0-9a-f]{64}")


def split_assignment(text: str, label: str) -> tuple[str, pathlib.Path]:
    if "=" not in text:
        raise ValueError(f"{label} must use NAME=PATH")
    name, raw_path = text.split("=", 1)
    name = name.strip()
    if not name or not raw_path.strip():
        raise ValueError(f"{label} must use non-empty NAME=PATH")
    return name, pathlib.Path(raw_path)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: pathlib.Path) -> tuple[int, float, str]:
    try:
        with wave.open(str(path), "rb") as wf:
            if (
                wf.getnchannels() != 1
                or wf.getframerate() != SAMPLE_RATE_HZ
                or wf.getsampwidth() != 2
                or wf.getcomptype() != "NONE"
            ):
                raise ValueError(f"{path}: expected mono 16-kHz PCM16 WAV")
            frames = wf.getnframes()
            if frames <= 0:
                raise ValueError(f"{path}: empty PCM payload")
            pcm = wf.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError(f"{path}: truncated PCM payload")
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"{path}: invalid WAV: {exc}") from exc
    return frames, frames / SAMPLE_RATE_HZ, hashlib.sha256(pcm).hexdigest()


def lineage_path(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(path) + ".lineage.json")


def write_lineage_sidecar(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    sidecar = lineage_path(path)
    sidecar.write_text(
        json.dumps({"schema_version": 1, "manifest_sha256": sha256_file(path),
                    "rows": rows}, ensure_ascii=False, sort_keys=True, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8",
    )
    return sidecar


def parse_tsv(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    targets: list[list[int]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" not in raw:
            raise ValueError(f"{path}:{line_no}: expected WAV<TAB>...")
        wav_path, target_text = raw.split("\t", 1)
        wav_path = wav_path.strip()
        if not wav_path:
            raise ValueError(f"{path}:{line_no}: empty WAV path")
        rows.append({"path": wav_path, "metadata": {}, "lineage": {}})
        targets.append([int(value) for value in target_text.split()])
    sidecar = lineage_path(path)
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
                or payload.get("schema_version") != 1
                or payload.get("manifest_sha256") != sha256_file(path)):
            raise ValueError(f"{sidecar}: lineage sidecar manifest binding mismatch")
        records = payload.get("rows")
        if not isinstance(records, list) or len(records) != len(rows):
            raise ValueError(f"{sidecar}: lineage sidecar row count mismatch")
        for index, (row, record) in enumerate(zip(rows, records), 1):
            if (not isinstance(record, dict) or record.get("path") != row["path"]
                    or record.get("target_ids") != targets[index - 1]):
                raise ValueError(f"{sidecar}:{index}: lineage sidecar row binding mismatch")
            rows[index - 1] = parse_record(record, sidecar, index)
    return rows


def _metadata_value(row: dict, field: str, path: pathlib.Path, line_no: int) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_no}: {field} must be non-empty text when present")
    return value.strip()


def parse_record(row: dict, path: pathlib.Path, line_no: int) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{path}:{line_no}: expected JSON object")
    path_value = canonical_audio_path(row, f"{path}:{line_no}")
    if "target_ids" in row and (not isinstance(row["target_ids"], list)
                                or any(type(value) is not int for value in row["target_ids"])):
        raise ValueError(f"{path}:{line_no}: target_ids must be an integer list")
    metadata = {
        field: value for field in IDENTITY_FIELDS
        if (value := _metadata_value(row, field, path, line_no)) is not None
    }
    for field, value in metadata.items():
        if field.endswith("_sha256") and SHA_RE.fullmatch(value) is None:
            raise ValueError(f"{path}:{line_no}: {field} must be lowercase sha256")
    lineage = {
        field: value for field in ("source_path", "source_wav_sha256", "wav_sha256")
        if (value := _metadata_value(row, field, path, line_no)) is not None
    }
    for field in ("source_wav_sha256", "wav_sha256"):
        if field in lineage and SHA_RE.fullmatch(lineage[field]) is None:
            raise ValueError(f"{path}:{line_no}: {field} must be lowercase sha256")
    source_fields = ("source_path" in lineage, "source_wav_sha256" in lineage,
                     "source_pcm_sha256" in metadata)
    if any(source_fields) and not all(source_fields):
        raise ValueError(f"{path}:{line_no}: source lineage requires path, WAV and PCM hashes")
    return {"path": path_value.strip(), "metadata": metadata, "lineage": lineage, "record": row}


def parse_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        rows.append(parse_record(row, path, line_no))
    return rows


def manifest_rows(path: pathlib.Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return parse_jsonl(path)
    return parse_tsv(path)


def metadata_leaks(by_identity: dict[str, dict[str, list[dict]]]) -> list[dict]:
    result: list[dict] = []
    for field in IDENTITY_FIELDS:
        for value, entries in sorted(by_identity[field].items()):
            splits = sorted({entry["split"] for entry in entries})
            if len(splits) <= 1:
                continue
            result.append(
                {
                    "field": field,
                    "value": value,
                    "splits": splits,
                    "paths": sorted({entry["path"] for entry in entries}),
                }
            )
    return result


def audit_splits(
    split_specs: list[tuple[str, pathlib.Path]], *,
    roots: dict[str, pathlib.Path] | None = None,
    fail_within_split: bool = False,
    fail_room_overlap: bool = False,
    fail_device_overlap: bool = False,
    require_metadata: tuple[str, ...] | list[str] = (),
    require_lineage: bool = False,
    _validated_rows: list[dict] | None = None,
) -> dict:
    roots = roots or {}
    names = [name for name, _ in split_specs]
    if not names or len(set(names)) != len(names):
        raise ValueError("audit split names must be non-empty and unique")
    if set(roots) - set(names):
        raise ValueError("audio roots reference unknown splits")
    by_pcm_hash: dict[str, list[dict]] = defaultdict(list)
    by_identity: dict[str, dict[str, list[dict]]] = {
        field: defaultdict(list) for field in IDENTITY_FIELDS
    }
    split_summaries: dict[str, dict] = {}
    within_duplicates: list[dict] = []
    missing_metadata: list[dict] = []
    inspection_cache: dict[pathlib.Path, tuple[int, float, str, str]] = {}

    def inspect(path: pathlib.Path) -> tuple[int, float, str, str]:
        if path not in inspection_cache:
            frames, duration, pcm = inspect_wav(path)
            inspection_cache[path] = (frames, duration, pcm, sha256_file(path))
        return inspection_cache[path]

    for name, manifest in split_specs:
        if not manifest.is_file():
            raise ValueError(f"{manifest}: split manifest does not exist")
        root = roots.get(name, manifest.parent)
        rows = manifest_rows(manifest)
        if not rows:
            raise ValueError(f"{manifest}: split contains no audio rows")
        local_pcm_hashes: dict[str, list[str]] = defaultdict(list)
        total_frames = 0
        resolved_rows: list[dict] = []
        audio_rows: list[dict] = []
        metadata_coverage = {field: 0 for field in IDENTITY_FIELDS}
        verified_source_rows = 0

        for row_index, row in enumerate(rows, 1):
            raw_path = str(row["path"])
            metadata = dict(row["metadata"])
            resolved = resolve_audio_path(row, root, f"{manifest}:{row_index}")
            frames, duration_s, pcm_sha256, file_sha256 = inspect(resolved)
            lineage = row.get("lineage", {})
            if "wav_sha256" in lineage and lineage["wav_sha256"] != file_sha256:
                raise ValueError(f"{manifest}:{row_index}: rendered WAV hash mismatch")
            if "source_path" in lineage:
                source = pathlib.Path(lineage["source_path"])
                if not source.is_absolute():
                    source = root / source
                source = source.resolve(strict=True)
                _, _, source_pcm, source_wav = inspect(source)
                if (lineage["source_wav_sha256"] != source_wav
                        or metadata["source_pcm_sha256"] != source_pcm):
                    raise ValueError(f"{manifest}:{row_index}: source WAV/PCM identity mismatch")
                verified_source_rows += 1
            elif require_lineage:
                missing_metadata.append({"split": name, "row": row_index,
                                         "path": str(resolved), "field": "source_lineage"})
            if require_lineage and "wav_sha256" not in lineage:
                missing_metadata.append({"split": name, "row": row_index,
                                         "path": str(resolved), "field": "wav_sha256"})
            if _validated_rows is not None and "record" in row:
                _validated_rows.append(row["record"])
            total_frames += frames
            entry = {
                "split": name,
                "path": str(resolved),
                "pcm_sha256": pcm_sha256,
                "file_sha256": file_sha256,
                "frames": frames,
                "duration_s": duration_s,
                "metadata": metadata,
            }
            resolved_rows.append(entry)
            audio_rows.append({"path": raw_path, "file_sha256": file_sha256,
                               "pcm_sha256": pcm_sha256, "frames": frames})
            local_pcm_hashes[pcm_sha256].append(str(resolved))
            # Final and ancestor PCM hashes name the same content namespace.
            # Keep the field and verification basis so a declared original hash
            # is never presented as a decoded/verified source observation.
            pcm_observations = [("pcm_sha256", pcm_sha256, "decoded-final-wav", str(resolved))]
            if "source_pcm_sha256" in metadata:
                pcm_observations.append(("source_pcm_sha256", metadata["source_pcm_sha256"],
                                         "verified-source-wav", str(source)))
            if "original_pcm_sha256" in metadata:
                pcm_observations.append(("original_pcm_sha256", metadata["original_pcm_sha256"],
                                         "declared-original-pcm", None))
            for field, digest, basis, observed_path in pcm_observations:
                by_pcm_hash[digest].append(dict(entry, identity_field=field,
                                               evidence_basis=basis, observed_path=observed_path))
            for field, value in metadata.items():
                metadata_coverage[field] += 1
                by_identity[field][value].append(entry)
            for field in require_metadata:
                if field not in metadata:
                    missing_metadata.append(
                        {
                            "split": name,
                            "row": row_index,
                            "path": str(resolved),
                            "field": field,
                        }
                    )

        for digest, paths in sorted(local_pcm_hashes.items()):
            if len(paths) > 1:
                within_duplicates.append(
                    {"split": name, "pcm_sha256": digest, "paths": sorted(paths)}
                )
        split_summaries[name] = {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "audio_identity": audio_identity(audio_rows),
            "examples": len(resolved_rows),
            "unique_pcm": len(local_pcm_hashes),
            "audio_hours": total_frames / SAMPLE_RATE_HZ / 3600.0,
            "metadata_coverage": metadata_coverage,
            "source_lineage_verified_rows": verified_source_rows,
            "identity_coverage": {
                field: {"known": metadata_coverage[field],
                        "unknown": len(resolved_rows) - metadata_coverage[field]}
                for field in COVERAGE_FIELDS
            },
            **({"lineage_sidecar_sha256": sha256_file(lineage_path(manifest))}
               if manifest.suffix.lower() != ".jsonl" and lineage_path(manifest).is_file() else {}),
        }

    cross_split_leaks: list[dict] = []
    for digest, entries in sorted(by_pcm_hash.items()):
        splits = sorted({entry["split"] for entry in entries})
        if len(splits) > 1:
            cross_split_leaks.append(
                {
                    "pcm_sha256": digest,
                    "splits": splits,
                    "paths": sorted({entry["path"] for entry in entries}),
                    "file_sha256": sorted({entry["file_sha256"] for entry in entries}),
                    "observations": sorted(
                        ({"split": entry["split"], "path": entry["path"],
                          "field": entry["identity_field"], "evidence_basis": entry["evidence_basis"],
                          "observed_path": entry["observed_path"]} for entry in entries),
                        key=lambda item: (item["split"], item["path"], item["field"])),
                }
            )

    identity_leaks = metadata_leaks(by_identity)
    identity_violations = [
        leak
        for leak in identity_leaks
        if leak["field"] in HARD_IDENTITY_FIELDS
        or (leak["field"] == "room_id" and fail_room_overlap)
        or (leak["field"] == "device_id" and fail_device_overlap)
    ]
    clean = (
        not cross_split_leaks
        and not identity_violations
        and not missing_metadata
        and (not fail_within_split or not within_duplicates)
    )
    report = {
        "schema_version": 3,
        "audio_identity": "decoded-mono-16khz-pcm16-sha256",
        "identity_policy": {
            "hard_cross_split_fields": sorted(HARD_IDENTITY_FIELDS),
            "shared_pcm_identity_fields": list(PCM_IDENTITY_FIELDS),
            "fail_room_overlap": bool(fail_room_overlap),
            "fail_device_overlap": bool(fail_device_overlap),
            "required_metadata": sorted(set(require_metadata)),
            "require_source_lineage": bool(require_lineage),
            "provider_engine_identity_is_recording_identity": False,
        },
        "splits": split_summaries,
        "cross_split_leaks": cross_split_leaks,
        "identity_leaks": identity_leaks,
        "identity_violations": identity_violations,
        "missing_metadata": missing_metadata,
        "within_split_duplicates": within_duplicates,
        "clean": clean,
        "clean_scope": "observed-identities-only",
        "identity_disjointness": {
            field: ("violated" if any(leak["field"] == field for leak in identity_leaks)
                    or any(observation["field"] == field for leak in cross_split_leaks
                           for observation in leak["observations"])
                    else "unknown" if any(split["identity_coverage"][field]["unknown"]
                                          for split in split_summaries.values())
                    else "verified" if len(split_specs) > 1 else "not-cross-split-tested")
            for field in COVERAGE_FIELDS
        },
    }
    return report


def verify_manifest_lineage(
    manifest: pathlib.Path, *, audio_root: pathlib.Path | None = None,
) -> list[dict]:
    """Return full records after exact manifest and live WAV/source verification.

    TSV records come from <manifest>.lineage.json and are bound to the manifest
    SHA, row order, path and target IDs. Admission receipts are retained verbatim;
    callers must independently validate their review history/label semantics.
    """
    records: list[dict] = []
    report = audit_splits(
        [("training", manifest)],
        roots={"training": audio_root} if audio_root is not None else None,
        require_lineage=True, _validated_rows=records,
    )
    if not report["clean"]:
        raise ValueError(f"{manifest}: verified source lineage is required on every row")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect decoded-PCM and identity leakage across training/calibration/"
            "evaluation splits. JSONL manifests may use audio/audio_path/path plus speaker_id, "
            "session_id, source_id and source lineage. TSV lineage sidecars bind exact "
            "manifest rows and original WAV/PCM. Known identity overlap is a hard failure; "
            "room/device overlap is policy-driven. Missing lineage identities remain unknown."
        )
    )
    parser.add_argument(
        "--split",
        required=True,
        action="append",
        help="split manifest as NAME=PATH; TSV uses first column, JSONL uses audio/audio_path/path",
    )
    parser.add_argument(
        "--audio-root",
        action="append",
        default=[],
        help="optional split-specific audio root as NAME=DIR",
    )
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--fail-within-split", action="store_true")
    parser.add_argument("--require-lineage", action="store_true",
                        help="require bound rendered WAV and verified original source WAV/PCM")
    parser.add_argument("--fail-room-overlap", action="store_true")
    parser.add_argument("--fail-device-overlap", action="store_true")
    parser.add_argument(
        "--require-metadata",
        action="append",
        choices=IDENTITY_FIELDS,
        default=[],
        help="require this identity field on every row; repeat for multiple fields",
    )
    args = parser.parse_args()

    split_specs: list[tuple[str, pathlib.Path]] = []
    seen_names: set[str] = set()
    for text in args.split:
        name, manifest = split_assignment(text, "--split")
        if name in seen_names:
            raise ValueError(f"duplicate split name: {name}")
        seen_names.add(name)
        split_specs.append((name, manifest))

    roots: dict[str, pathlib.Path] = {}
    for text in args.audio_root:
        name, root = split_assignment(text, "--audio-root")
        if name in roots:
            raise ValueError(f"duplicate audio root for split: {name}")
        roots[name] = root
    unknown_roots = sorted(set(roots) - seen_names)
    if unknown_roots:
        raise ValueError(f"audio roots reference unknown split(s): {', '.join(unknown_roots)}")

    report = audit_splits(
        split_specs, roots=roots, fail_within_split=args.fail_within_split,
        fail_room_overlap=args.fail_room_overlap, fail_device_overlap=args.fail_device_overlap,
        require_metadata=args.require_metadata, require_lineage=args.require_lineage,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")

    if not report["clean"]:
        print(
            "dataset audit failed: "
            f"pcm_leaks={len(report['cross_split_leaks'])} "
            f"identity_violations={len(report['identity_violations'])} "
            f"missing_metadata={len(report['missing_metadata'])} "
            f"within_duplicates={len(report['within_split_duplicates']) if args.fail_within_split else 0}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
