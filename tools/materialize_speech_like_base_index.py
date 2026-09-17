#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "training"))

from kws_vocab import load_tokens  # noqa: E402
from synthetic_audio import SPLITS, activity_bounds, parse_keywords, sha256_file  # noqa: E402

MANIFEST_CLASS = "speech-like-synthetic-recording-v1"
ALLOWED_KINDS = {"positive", "confusable", "negative", "background"}


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def inspect_pcm16(path: pathlib.Path) -> tuple[list[int], str]:
    if not path.is_file():
        raise ValueError(f"audio file missing: {path}")
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2 or reader.getframerate() != 16000 or reader.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected mono PCM16 16-kHz WAV")
        raw = reader.readframes(reader.getnframes())
    if not raw:
        raise ValueError(f"{path}: empty WAV")
    import struct
    samples = list(struct.unpack("<" + "h" * (len(raw) // 2), raw))
    return samples, hashlib.sha256(raw).hexdigest()


def corpus_sha(rows: list[dict]) -> str:
    value = [
        {
            "wav_sha256": row["wav_sha256"],
            "kind": row["kind"],
            "keyword_id": row["keyword_id"],
            "tokens": row["tokens"],
            "voice_id": row["voice_id"],
            "source_id": row["source_id"],
            "generation_config_sha256": row["generation_config_sha256"],
            "label_provenance_sha256": row["label_provenance_sha256"],
        }
        for row in rows
    ]
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def materialize(*, manifest: pathlib.Path, split: str, tokens_path: pathlib.Path, keywords_path: pathlib.Path) -> tuple[list[dict], dict]:
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    token_map = load_tokens(tokens_path)
    keywords = parse_keywords(keywords_path, token_map)
    keyword_by_id = {int(item["id"]): item for item in keywords}
    rows: list[dict] = []
    seen_audio: set[str] = set()
    for index, source in enumerate(load_jsonl(manifest), 1):
        label = f"{manifest}:{index}"
        if int(source.get("schema_version", 0)) != 1 or source.get("evidence_class") != MANIFEST_CLASS:
            raise ValueError(f"{label}: speech-like manifest identity mismatch")
        if source.get("provider_kind") == "tone":
            raise ValueError(f"{label}: tone provider is forbidden for speech-like ingestion")
        kind = str(source.get("kind", ""))
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"{label}: missing/invalid immutable training kind")
        label_sha = source.get("label_provenance_sha256")
        if not isinstance(label_sha, str) or len(label_sha) != 64:
            raise ValueError(f"{label}: label provenance is required")
        audio_raw = source.get("audio")
        if not isinstance(audio_raw, str) or not audio_raw:
            raise ValueError(f"{label}: audio is required")
        audio = pathlib.Path(audio_raw)
        audio = audio.resolve() if audio.is_absolute() else (manifest.parent / audio).resolve()
        file_sha = sha256_file(audio)
        if file_sha != source.get("file_sha256"):
            raise ValueError(f"{label}: WAV sha256 mismatch")
        if file_sha in seen_audio:
            raise ValueError(f"{label}: duplicate WAV identity")
        seen_audio.add(file_sha)
        samples, pcm_sha = inspect_pcm16(audio)
        if pcm_sha != source.get("pcm_sha256"):
            raise ValueError(f"{label}: PCM sha256 mismatch")
        tokens = source.get("tokens")
        if not isinstance(tokens, list) or any(not isinstance(item, str) or item not in token_map for item in tokens):
            raise ValueError(f"{label}: tokens are invalid for canonical vocabulary")
        token_ids = [int(token_map[item]) for item in tokens]
        if any(value == 0 for value in token_ids):
            raise ValueError(f"{label}: blank token cannot appear in labeled speech")
        keyword_id = source.get("keyword_id")
        event_start = event_end = None
        if kind == "positive":
            if isinstance(keyword_id, bool) or not isinstance(keyword_id, int) or keyword_id not in keyword_by_id:
                raise ValueError(f"{label}: positive keyword_id is invalid")
            expected_tokens = list(keyword_by_id[keyword_id]["tokens"])
            if tokens != expected_tokens:
                raise ValueError(f"{label}: positive tokens do not match canonical keyword")
            event_start, event_end = activity_bounds(samples)
        elif keyword_id is not None:
            raise ValueError(f"{label}: non-positive keyword_id must be null")
        if kind == "background" and tokens:
            raise ValueError(f"{label}: background must not carry token targets")
        rows.append({
            "split": split,
            "kind": kind,
            "family_id": f"speech-like-{split}-{index-1:06d}",
            "variant": 0,
            "keyword_id": keyword_id,
            "tokens": list(tokens),
            "target_ids": token_ids,
            "wav_sha256": file_sha,
            "frames": len(samples),
            "event_start_frame": event_start,
            "event_end_frame": event_end,
            "path": str(audio),
            "speech_like_provenance": {
                "provider_kind": source.get("provider_kind"),
                "provider_name": source.get("provider_name"),
                "provider_version": source.get("provider_version"),
                "license_id": source.get("license_id"),
                "voice_id": source.get("voice_id"),
                "source_id": source.get("source_id"),
                "generation_config_sha256": source.get("generation_config_sha256"),
                "label_provenance_sha256": label_sha,
                "pcm_sha256": pcm_sha,
            },
        })
    summary = {
        "schema_version": 1,
        "evidence_class": "speech-like-base-dataset-v1",
        "split": split,
        "manifest_sha256": sha256_file(manifest),
        "tokens_sha256": sha256_file(tokens_path),
        "keywords_sha256": sha256_file(keywords_path),
        "recordings": len(rows),
        "positive_recordings": sum(1 for row in rows if row["kind"] == "positive"),
        "negative_recordings": sum(1 for row in rows if row["kind"] != "positive"),
        "corpus_sha256": corpus_sha(rows),
        "tone_backend_used": False,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize immutable speech-like recordings into the canonical base dataset-index schema.")
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--output-index", required=True, type=pathlib.Path)
    parser.add_argument("--output-summary", required=True, type=pathlib.Path)
    args = parser.parse_args()
    rows, summary = materialize(
        manifest=args.manifest.resolve(), split=args.split,
        tokens_path=args.tokens.resolve(), keywords_path=args.keywords.resolve(),
    )
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    args.output_index.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"speech-like base index: split={args.split} recordings={summary['recordings']} corpus={summary['corpus_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
