#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import wave

SAMPLE_RATE_HZ = 16000
IDENTITY_FIELDS = ("speaker_id", "session_id", "source_id", "room_id", "device_id")
AUDIO_PATH_FIELDS = ("audio", "audio_path", "path")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def canonical_audio_path(row: dict, label: str = "audio row") -> str:
    """One alias contract for audit, training, identity and execution.

    Aliases name a manifest-relative path. All present aliases must be valid
    and agree after whitespace stripping; no consumer-specific precedence or
    filesystem-dependent alias equivalence is permitted.
    """
    if not isinstance(row, dict):
        raise ValueError(f"{label}: expected JSON object")
    values = []
    for field in AUDIO_PATH_FIELDS:
        if field in row:
            value = row[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}: {field} must be non-empty text")
            values.append(value.strip())
    if not values:
        raise ValueError(f"{label}: expected non-empty audio/audio_path/path")
    if len(set(values)) != 1:
        raise ValueError(f"{label}: audio/audio_path/path aliases disagree")
    return values[0]


def resolve_audio_path(row: dict, root: pathlib.Path, label: str = "audio row") -> pathlib.Path:
    path = pathlib.Path(canonical_audio_path(row, label))
    try:
        return (path if path.is_absolute() else root / path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{label}: audio file does not exist: {path}") from exc


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pcm16_wav(path: pathlib.Path) -> dict:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"{path}: expected mono 16-kHz PCM16 WAV")
        frames = reader.getnframes()
        if frames <= 0:
            raise ValueError(f"{path}: empty PCM payload")
        pcm = reader.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError(f"{path}: truncated PCM payload")
    return {
        "file_sha256": sha256_file(path),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "frames": frames,
        "duration_s": frames / float(SAMPLE_RATE_HZ),
    }


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def corpus_digest(rows: list[dict]) -> str:
    normalized = []
    for row in rows:
        item = {
            "recording": row.get("recording"),
            "path": row.get("path"),
            "file_sha256": row["file_sha256"],
            "pcm_sha256": row["pcm_sha256"],
            "frames": row["frames"],
        }
        for field in IDENTITY_FIELDS:
            if field in row:
                item[field] = row[field]
        normalized.append(item)
    return canonical_hash(normalized)


def identity_bundle(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("corpus identity requires at least one recording")
    return {
        "schema_version": 1,
        "corpus_sha256": corpus_digest(rows),
        "recordings": rows,
    }


def _metadata(row: dict, label: str) -> dict:
    result = {}
    for field in IDENTITY_FIELDS:
        value = row.get(field)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}: {field} must be non-empty text")
            result[field] = value.strip()
    return result


def training_manifest_rows(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if path.suffix.lower() == ".jsonl":
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            audio = canonical_audio_path(value, f"{path}:{line_no}")
            rows.append({"audio": audio.strip(), "metadata": _metadata(value, f"{path}:{line_no}")})
    else:
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            if "\t" not in raw:
                raise ValueError(f"{path}:{line_no}: expected WAV<TAB>token_ids")
            audio = raw.split("\t", 1)[0].strip()
            if not audio:
                raise ValueError(f"{path}:{line_no}: empty WAV path")
            rows.append({"audio": audio, "metadata": {}})
    return rows


def training_corpus_identity(manifests: list[pathlib.Path]) -> dict:
    identities: list[dict] = []
    for manifest_index, manifest in enumerate(manifests):
        root = manifest.parent
        for row_index, row in enumerate(training_manifest_rows(manifest), 1):
            raw_path = row["audio"]
            resolved = resolve_audio_path(row, root, f"{manifest}:{row_index}")
            identities.append(
                {
                    "recording": f"manifest-{manifest_index}:{row_index}",
                    "manifest": manifest.name,
                    "path": raw_path,
                    **inspect_pcm16_wav(resolved),
                    **row["metadata"],
                }
            )
    return identity_bundle(identities)


def evaluation_corpus_identity(references: pathlib.Path, audio_root: pathlib.Path) -> dict:
    identities: list[dict] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(references.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{references}:{line_no}: expected JSON object")
        recording = row.get("recording")
        raw_path = canonical_audio_path(row, f"{references}:{line_no}")
        if not isinstance(recording, str) or not recording or recording in seen:
            raise ValueError(f"{references}:{line_no}: recording must be unique non-empty text")
        resolved = resolve_audio_path(row, audio_root, f"{references}:{line_no}")
        measured = inspect_pcm16_wav(resolved)
        declared_duration = row.get("duration_s")
        if isinstance(declared_duration, bool) or not isinstance(declared_duration, (int, float)):
            raise ValueError(f"{references}:{line_no}: duration_s must be numeric")
        if not math.isfinite(float(declared_duration)) or not math.isclose(
            float(declared_duration), measured["duration_s"], rel_tol=0.0, abs_tol=1.0 / SAMPLE_RATE_HZ
        ):
            raise ValueError(
                f"{references}:{line_no}: duration_s={declared_duration} does not match WAV duration {measured['duration_s']}"
            )
        identities.append(
            {
                "recording": recording,
                "path": raw_path.strip(),
                **measured,
                **_metadata(row, f"{references}:{line_no}"),
            }
        )
        seen.add(recording)
    return identity_bundle(identities)


def audio_identity(rows: list[dict]) -> dict:
    """Bind ordered manifest paths to measured WAV bytes and decoded PCM.

    A root relocation is safe only when the selected bytes are unchanged.
    Recording labels and metadata are bound by manifest/lineage hashes, not
    repeated here, so the same binding applies to JSONL and TSV manifests.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("audio identity requires non-empty recordings")
    recordings = []
    for index, row in enumerate(rows):
        label = f"audio identity[{index}]"
        path = canonical_audio_path(row, label)
        item = {"path": path}
        for field in ("file_sha256", "pcm_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{label}: {field} must be lowercase SHA256")
            item[field] = value
        frames = row.get("frames")
        if type(frames) is not int or frames <= 0:
            raise ValueError(f"{label}: frames must be a positive integer")
        item["frames"] = frames
        recordings.append(item)
    return {"schema_version": 1, "audio_sha256": canonical_hash(recordings),
            "recordings": recordings}


def validate_audio_identity(value: object, label: str = "audio identity") -> dict:
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError(f"{label}: audio identity schema_version must be 1")
    normalized = audio_identity(value.get("recordings"))
    if value.get("audio_sha256") != normalized["audio_sha256"]:
        raise ValueError(f"{label}: audio identity digest mismatch")
    return normalized


def training_manifest_audio_bindings(manifests: list[dict], corpus: dict) -> list[dict]:
    """Split a training corpus by its explicit manifest index, never basename."""
    if (not isinstance(manifests, list) or not manifests
            or any(not isinstance(manifest, dict) for manifest in manifests)
            or not isinstance(corpus, dict)):
        raise ValueError("training audio bindings require manifests and corpus")
    rows = corpus.get("recordings")
    if not isinstance(rows, list) or not rows or corpus_digest(rows) != corpus.get("corpus_sha256"):
        raise ValueError("training audio bindings require a valid corpus identity")
    groups: list[list[dict]] = [[] for _ in manifests]
    for row in rows:
        match = re.fullmatch(r"manifest-([0-9]+):([1-9][0-9]*)", str(row.get("recording", "")))
        if match is None:
            raise ValueError("training corpus recording is missing its manifest/row index")
        manifest_index, row_index = map(int, match.groups())
        if manifest_index >= len(manifests) or row_index != len(groups[manifest_index]) + 1:
            raise ValueError("training corpus manifest/row index is inconsistent")
        if row.get("manifest") != manifests[manifest_index].get("name"):
            raise ValueError("training corpus manifest name does not match selected manifest")
        groups[manifest_index].append(row)
    return [{**manifest, "audio_identity": audio_identity(group)}
            for manifest, group in zip(manifests, groups)]


def require_audited_audio(dataset_audit: dict, required_bindings: list[dict]) -> None:
    """Compare actual/current or inherited audio against every matching audit.

    Consumers retain these bindings rather than trusting clean=true or a
    manifest hash to identify the external WAV files. Also independently check
    the audited decoded PCM sets, including ancestor splits, for overlap.
    """
    bindings = dataset_audit.get("manifest_bindings") if isinstance(dataset_audit, dict) else None
    if (not isinstance(bindings, list) or not bindings
            or not isinstance(required_bindings, list) or not required_bindings):
        raise ValueError("dataset audit audio bindings are missing")
    for binding in [*bindings, *required_bindings]:
        if (not isinstance(binding, dict) or not isinstance(binding.get("sha256"), str)
                or SHA256_RE.fullmatch(binding["sha256"]) is None):
            raise ValueError("audio manifest binding requires a lowercase SHA256")
    normalized = []
    seen_pcm: dict[str, int] = {}
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            raise ValueError("dataset audit manifest binding must be an object")
        identity = validate_audio_identity(binding.get("audio_identity"), f"dataset audit binding[{index}]")
        for row in identity["recordings"]:
            pcm = row["pcm_sha256"]
            if pcm in seen_pcm and seen_pcm[pcm] != index:
                raise ValueError("dataset audit audio bindings contain cross-split PCM overlap")
            seen_pcm[pcm] = index
        normalized.append((binding, identity))
    for required in required_bindings:
        identity = validate_audio_identity(required.get("audio_identity"), "selected audio")
        matches = [audio for binding, audio in normalized
                   if binding.get("sha256") == required.get("sha256")
                   and all(binding.get(key) == required[key]
                           for key in ("name", "lineage_sha256") if key in required)]
        if not matches:
            raise ValueError("dataset audit does not cover selected audio manifest/lineage")
        if any(match != identity for match in matches):
            raise ValueError("dataset audit audio identity does not match selected WAV/PCM")


def require_audited_corpora(dataset_audit: dict, training_corpus: dict,
                            training_manifests: list[dict], evaluation_rows: list[dict],
                            references_sha256: str) -> None:
    required = training_manifest_audio_bindings(training_manifests, training_corpus)
    required.append({"sha256": references_sha256, "audio_identity": audio_identity(evaluation_rows)})
    require_audited_audio(dataset_audit, required)
