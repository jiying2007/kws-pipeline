#!/usr/bin/env python3
"""Check the negative-only ASR alias boundary with real WAV identity checks."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.codex_assets.phonetic_negative_materialize import materialize


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        audio = root / "sample.wav"
        with wave.open(str(audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 16000)
        wav_sha = hashlib.sha256(audio.read_bytes()).hexdigest()
        policy = root / "policy.json"
        tokens = root / "tokens.txt"
        candidates = root / "candidates.jsonl"
        asr = root / "asr.jsonl"
        split = root / "split.json"
        existing = root / "existing.jsonl"
        rule = {
            "schema_version": 1, "policy": "phonetic-negative-equivalence-v1",
            "scope": "development-only-negative-competitors",
            "positive_acceptance": "exact-normalized-text-v1-only",
            "asr_model_revision": "fixed-revision",
            "asr_model_sha256": "a" * 64,
            "cases": [{"official_text": "你好亚", "accepted_asr_text": ["你好呀"],
                       "tokens": ["ni3", "hao3", "ya"], "minimum_rows": 1}],
        }
        write_json(policy, rule)
        tokens.write_text("<blk> 0\nni3 1\nhao3 2\nxiao3 3\nwo1 4\nya 5\n", encoding="utf-8")
        source = {
            "recording": "sample-1", "audio_path": str(audio),
            "text": "你好亚", "kind": "confusable", "keyword_id": None,
            "speaker_id": "train-1", "session_id": "s1",
            "source_id": "source-1", "file_sha256": wav_sha,
            "usage_class": "research-only",
        }
        receipt = {
            "schema_version": 1,
            "evidence_class": "qwen3-asr-himia-competitor-screen-v1",
            "recording": "sample-1", "source_id": "source-1",
            "file_sha256": wav_sha, "intended_text": "你好亚",
            "kind": "confusable", "keyword_id": None,
            "asr_text": "你好呀。", "verdict": "rejected",
            "asr_model_revision": "fixed-revision",
            "asr_model_sha256": "a" * 64,
        }
        write_jsonl(candidates, [source])
        write_jsonl(asr, [receipt])
        write_json(split, {"train_speakers": ["train-1"],
                           "holdout_speakers": ["holdout-1"]})
        write_jsonl(existing, [{"file_sha256": "b" * 64}])
        paths = dict(policy_path=policy, tokens_path=tokens,
                     candidates_path=candidates, asr_path=asr,
                     speaker_split_path=split, existing_manifest_path=existing)
        rows, summary = materialize(**paths)
        assert summary["accepted"] == 1
        assert rows[0]["tokens"] == [1, 2, 5]
        assert rows[0]["asr_exact_verdict"] == "rejected"
        assert rows[0]["keyword_id"] is None

        source["kind"] = "positive"
        source["keyword_id"] = 1
        write_jsonl(candidates, [source])
        try:
            materialize(**paths)
        except ValueError as exc:
            assert "identity mismatch" in str(exc)
        else:
            raise AssertionError("positive was accepted through the negative alias")

        source["kind"] = "confusable"
        source["keyword_id"] = None
        source["file_sha256"] = "c" * 64
        write_jsonl(candidates, [source])
        try:
            materialize(**paths)
        except ValueError as exc:
            assert "identity mismatch" in str(exc)
        else:
            raise AssertionError("stale WAV identity was accepted")

        source["file_sha256"] = wav_sha
        write_jsonl(candidates, [source])
        rule["cases"][0]["accepted_asr_text"] = ["你好小窝"]
        write_json(policy, rule)
        try:
            materialize(**paths)
        except ValueError as exc:
            assert "negative-only" in str(exc) or "aliases" in str(exc)
        else:
            raise AssertionError("target wake text was allowed as a negative alias")

        rule["cases"][0]["accepted_asr_text"] = ["你好呀"]
        rule["cases"][0]["tokens"] = ["ni3", "hao3", "xiao3", "wo1"]
        write_json(policy, rule)
        try:
            materialize(**paths)
        except ValueError as exc:
            assert "full wake target" in str(exc)
        else:
            raise AssertionError("full wake token sequence was labeled negative")
    print("phonetic negative materialization: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
