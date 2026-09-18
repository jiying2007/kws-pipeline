#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

SPLITS = ("train", "calibration", "test", "qualification")
SUMMARY_CLASS = "speech-like-base-dataset-v1"
SHA_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"external base index is empty: {path}")
    return rows


def resolve(root: pathlib.Path, raw: object, label: str) -> pathlib.Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = pathlib.Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def expected_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase sha256")
    return value


def _identity(row: dict, split: str, index: int) -> tuple[str, str, str]:
    if row.get("split") != split:
        raise ValueError(f"{split} row {index}: split mismatch")
    wav_sha = row.get("wav_sha256")
    if not isinstance(wav_sha, str) or SHA_RE.fullmatch(wav_sha) is None:
        raise ValueError(f"{split} row {index}: wav_sha256 is required")
    provenance = row.get("speech_like_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{split} row {index}: speech_like_provenance is required")
    voice = provenance.get("voice_id")
    source = provenance.get("source_id")
    provider = provenance.get("provider_kind")
    license_id = provenance.get("license_id")
    if provider == "tone":
        raise ValueError(f"{split} row {index}: tone provenance is forbidden")
    for value, label in ((voice, "voice_id"), (source, "source_id"), (provider, "provider_kind"), (license_id, "license_id")):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{split} row {index}: {label} is required")
    return str(wav_sha), str(voice), str(source)


def load_external_base_bundle(config_path: pathlib.Path, config: dict) -> tuple[list[dict], dict] | None:
    generator = config.get("generator")
    if not isinstance(generator, dict):
        return None
    spec = generator.get("external_base_dataset")
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("generator.external_base_dataset must be an object")
    if set(spec) != set(SPLITS):
        missing = sorted(set(SPLITS) - set(spec))
        extra = sorted(set(spec) - set(SPLITS))
        raise ValueError(f"external base dataset must define exactly {SPLITS}; missing={missing} extra={extra}")

    root = config_path.parent.resolve()
    rows: list[dict] = []
    sources: dict[str, dict] = {}
    identity_sources: dict[str, dict] = {}
    wav_owner: dict[str, str] = {}
    voice_owner: dict[str, str] = {}
    source_owner: dict[str, str] = {}

    for split in SPLITS:
        item = spec[split]
        if not isinstance(item, dict):
            raise ValueError(f"external base {split} spec must be an object")
        index_path = resolve(root, item.get("index"), f"external base {split}.index")
        summary_path = resolve(root, item.get("summary"), f"external base {split}.summary")
        index_sha = expected_sha(item.get("index_sha256"), f"external base {split}.index_sha256")
        summary_sha = expected_sha(item.get("summary_sha256"), f"external base {split}.summary_sha256")
        if not index_path.is_file() or not summary_path.is_file():
            raise ValueError(f"external base {split}: index/summary missing")
        if sha256_file(index_path) != index_sha:
            raise ValueError(f"external base {split}: index sha256 mismatch")
        if sha256_file(summary_path) != summary_sha:
            raise ValueError(f"external base {split}: summary sha256 mismatch")
        summary = load_json(summary_path)
        if int(summary.get("schema_version", 0)) != 1 or summary.get("evidence_class") != SUMMARY_CLASS:
            raise ValueError(f"external base {split}: summary identity mismatch")
        if summary.get("split") != split or summary.get("tone_backend_used") is not False:
            raise ValueError(f"external base {split}: summary split/tone contract mismatch")
        split_rows = load_jsonl(index_path)
        if int(summary.get("recordings", -1)) != len(split_rows):
            raise ValueError(f"external base {split}: recording count mismatch")
        path_contract = str(summary.get("audio_path_contract", "legacy-v0"))
        if path_contract not in {"index-relative-v1", "absolute-v1", "legacy-v0"}:
            raise ValueError(f"external base {split}: unsupported audio_path_contract={path_contract}")
        normalized_rows: list[dict] = []
        index_root = index_path.parent.resolve()
        for idx, row in enumerate(split_rows):
            wav_sha, voice, source = _identity(row, split, idx)
            raw_audio = row.get("path")
            if not isinstance(raw_audio, str) or not raw_audio.strip():
                raise ValueError(f"external base {split} row {idx}: audio path is required")
            audio_ref = pathlib.Path(raw_audio)
            if path_contract == "index-relative-v1":
                if audio_ref.is_absolute() or any(part in {"", ".", ".."} for part in audio_ref.parts):
                    raise ValueError(f"external base {split} row {idx}: portable audio path is unsafe")
                audio_path = (index_root / audio_ref).resolve()
                try:
                    audio_path.relative_to(index_root)
                except ValueError as exc:
                    raise ValueError(
                        f"external base {split} row {idx}: portable audio path escaped index directory"
                    ) from exc
            else:
                audio_path = (
                    audio_ref.resolve()
                    if audio_ref.is_absolute()
                    else (index_root / audio_ref).resolve()
                )
            if not audio_path.is_file():
                raise ValueError(f"external base {split} row {idx}: audio file missing: {audio_path}")
            if sha256_file(audio_path) != wav_sha:
                raise ValueError(f"external base {split} row {idx}: audio sha256 mismatch")
            normalized = dict(row)
            normalized["path"] = str(audio_path)
            normalized_rows.append(normalized)
            for identity, owners, label in ((wav_sha, wav_owner, "wav"), (voice, voice_owner, "voice"), (source, source_owner, "source")):
                previous = owners.get(identity)
                if previous is not None and previous != split:
                    raise ValueError(f"external base cross-split {label} overlap: {identity} in {previous}/{split}")
                owners[identity] = split
        rows.extend(normalized_rows)
        sources[split] = {
            "index": str(index_path),
            "summary": str(summary_path),
            "index_sha256": index_sha,
            "summary_sha256": summary_sha,
            "corpus_sha256": summary.get("corpus_sha256"),
            "recordings": len(normalized_rows),
            "audio_path_contract": path_contract,
        }
        identity_sources[split] = {
            "index_sha256": index_sha,
            "summary_sha256": summary_sha,
            "corpus_sha256": summary.get("corpus_sha256"),
            "recordings": len(normalized_rows),
            "audio_path_contract": path_contract,
        }

    aggregate = {
        "schema_version": 1,
        "evidence_class": "external-speech-like-base-bundle-v1",
        "tone_backend_used": False,
        "bundle_identity_policy": "content-addressed-v2",
        "splits": sources,
        "recordings": len(rows),
    }
    identity = {
        "schema_version": 1,
        "evidence_class": "external-speech-like-base-bundle-v1",
        "tone_backend_used": False,
        "bundle_identity_policy": "content-addressed-v2",
        "splits": identity_sources,
        "recordings": len(rows),
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    aggregate["bundle_sha256"] = hashlib.sha256(raw).hexdigest()
    return rows, aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    result = load_external_base_bundle(config_path, config)
    if result is None:
        raise ValueError("config does not declare generator.external_base_dataset")
    rows, summary = result
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"external base rows={len(rows)} bundle={summary['bundle_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
