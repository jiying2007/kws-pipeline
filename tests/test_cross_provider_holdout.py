from __future__ import annotations

import pathlib
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from speech_like_corpus_plan import (  # noqa: E402
    validate_audio_review,
    validate_asr_review,
    validate_cross_provider_holdout,
    validate_mixed_reviews,
    merge_generated_manifests,
    normalize_asr_text,
)
import generate_speech_like_command_provider as command_provider  # noqa: E402
import speech_like_corpus_plan as corpus_plan  # noqa: E402


def rows() -> dict[str, list[dict]]:
    return {
        "train": [{"provider_name": "vits-aishell3", "provider_identity_sha256": "a" * 64}],
        "calibration": [{"provider_name": "matcha-zh", "provider_identity_sha256": "b" * 64}],
        "test": [{"provider_name": "matcha-zh", "provider_identity_sha256": "b" * 64}],
        "qualification": [{"provider_name": "cosyvoice2", "provider_identity_sha256": "c" * 64}],
    }


def reject(value: dict[str, list[dict]], reason: str) -> None:
    try:
        validate_cross_provider_holdout(value)
    except ValueError as exc:
        assert reason in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted invalid provider holdout: {reason}")


def main() -> int:
    assert len(validate_cross_provider_holdout(rows())) == 4

    mixed_train = rows()
    mixed_train["train"].append(
        {"provider_name": "melo-zh", "provider_identity_sha256": "d" * 64}
    )
    assert len(validate_cross_provider_holdout(mixed_train)["train"]["providers"]) == 2

    drifting_train = rows()
    drifting_train["train"].append(
        {"provider_name": "vits-aishell3", "provider_identity_sha256": "d" * 64}
    )
    reject(drifting_train, "multiple asset identities")

    same_generator = rows()
    same_generator["qualification"][0]["provider_name"] = "vits-aishell3"
    reject(same_generator, "reuses a TTS provider")

    same_weights = rows()
    same_weights["qualification"][0]["provider_identity_sha256"] = "a" * 64
    reject(same_weights, "reuses a TTS provider")

    missing_identity = rows()
    del missing_identity["test"][0]["provider_identity_sha256"]
    reject(missing_identity, "provider_identity_sha256 is required")

    mixed_search = rows()
    mixed_search["test"][0]["provider_identity_sha256"] = "d" * 64
    reject(mixed_search, "calibration/test must share")

    with tempfile.TemporaryDirectory() as directory:
        review_path = pathlib.Path(directory) / "review.jsonl"
        intent_map = {("train", "voice", "source", "小窝小窝", ("xiao3", "wo1")):
                      {"text": "小窝小窝", "kind": "positive", "keyword_id": 2}}
        generated = {key: {"source_id": "source", "file_sha256": "a" * 64}
                     for key in intent_map}
        review = {"schema_version": 1, "evidence_class": "speech-like-audio-review-v1",
                  "source_id": "source", "file_sha256": "a" * 64,
                  "intended_text": "小窝小窝", "kind": "positive", "keyword_id": 2,
                  "reviewer_id": "reviewer-1", "verdict": "accepted"}

        def write_review(value):
            review_path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")

        write_review(review)
        assert validate_audio_review(review_path, intent_map, generated)["recordings"] == 1
        for mutation, reason in (({"verdict": "uncertain"}, "not accepted"),
                                 ({"file_sha256": "b" * 64}, "missing"),
                                 ({"intended_text": "小窝"}, "differs from plan")):
            write_review({**review, **mutation})
            try:
                validate_audio_review(review_path, intent_map, generated)
            except ValueError as exc:
                assert reason in str(exc), str(exc)
            else:
                raise AssertionError(f"accepted invalid audio review: {mutation}")

        assert normalize_asr_text(" 小窝，小窝。") == "小窝小窝"
        clean_audio = pathlib.Path(directory) / "continuous.wav"
        with wave.open(str(clean_audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\xe8\x03" * 3200)
        clean_sha = hashlib.sha256(clean_audio.read_bytes()).hexdigest()
        asr_generated = {key: {"source_id": "source", "file_sha256": clean_sha,
                               "audio": str(clean_audio)} for key in intent_map}
        asr_review = {
            "schema_version": 1, "evidence_class": "speech-like-asr-review-v1",
            "asr_standard": "exact-normalized-text-v1",
            "source_id": "source", "file_sha256": clean_sha,
            "intended_text": "小窝小窝", "kind": "positive", "keyword_id": 2,
            "asr_text": "小窝，小窝。", "verdict": "accepted",
            "asr_binary_sha256": "b" * 64, "asr_model_sha256": "c" * 64,
            "asr_tokens_sha256": "d" * 64,
        }
        write_review(asr_review)
        assert validate_asr_review(review_path, intent_map, asr_generated)["recordings"] == 1
        for mutation, reason in (({"asr_text": "小沃小窝"}, "verdict differs"),
                                 ({"asr_text": "小沃小窝", "verdict": "rejected"}, "not accepted"),
                                 ({"file_sha256": "e" * 64}, "missing"),
                                 ({"asr_model_sha256": "not-a-hash"}, "engine SHA-256")):
            write_review({**asr_review, **mutation})
            try:
                validate_asr_review(review_path, intent_map, asr_generated)
            except ValueError as exc:
                assert reason in str(exc), str(exc)
            else:
                raise AssertionError(f"accepted invalid ASR review: {mutation}")

        pause_audio = pathlib.Path(directory) / "paused.wav"
        with wave.open(str(pause_audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\xe8\x03" * 3200 + b"\0\0" * 3200 + b"\xe8\x03" * 3200)
        pause_sha = hashlib.sha256(pause_audio.read_bytes()).hexdigest()
        write_review({**asr_review, "file_sha256": pause_sha})
        paused_generated = {key: {"source_id": "source", "file_sha256": pause_sha,
                                  "audio": str(pause_audio)} for key in intent_map}
        try:
            validate_asr_review(review_path, intent_map, paused_generated)
        except ValueError as exc:
            assert "long internal silence" in str(exc), str(exc)
        else:
            raise AssertionError("accepted a paused continuous keyword")

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        human_path = root / "human.jsonl"
        machine_path = root / "machine.jsonl"
        key_h = ("train", "human-voice", "human-source", "你好小窝", ("ni3", "hao3"))
        key_m = ("train", "machine-voice", "machine-source", "小窝小窝", ("xiao3", "wo1"))
        mixed_intents = {
            key_h: {"text": "你好小窝", "kind": "positive", "keyword_id": 1},
            key_m: {"text": "小窝小窝", "kind": "positive", "keyword_id": 2},
        }
        machine_audio = root / "machine.wav"
        with wave.open(str(machine_audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\xe8\x03" * 3200)
        machine_sha = hashlib.sha256(machine_audio.read_bytes()).hexdigest()
        mixed_generated = {
            key_h: {"source_id": "human-source", "file_sha256": "a" * 64},
            key_m: {"source_id": "machine-source", "file_sha256": machine_sha,
                    "audio": str(machine_audio)},
        }
        human_path.write_text(json.dumps({
            "schema_version": 1, "evidence_class": "speech-like-audio-review-v1",
            "source_id": "human-source", "file_sha256": "a" * 64,
            "intended_text": "你好小窝", "kind": "positive", "keyword_id": 1,
            "reviewer_id": "reviewer-1", "verdict": "accepted"}, ensure_ascii=False) + "\n")
        machine_path.write_text(json.dumps({
            "schema_version": 1, "evidence_class": "speech-like-asr-review-v1",
            "asr_standard": "exact-normalized-text-v1",
            "source_id": "machine-source", "file_sha256": machine_sha,
            "intended_text": "小窝小窝", "kind": "positive", "keyword_id": 2,
            "asr_text": "小窝小窝", "verdict": "accepted",
            "asr_binary_sha256": "c" * 64, "asr_model_sha256": "d" * 64,
            "asr_tokens_sha256": "e" * 64}, ensure_ascii=False) + "\n")
        human_summary, machine_summary = validate_mixed_reviews(
            human_path, machine_path, mixed_intents, mixed_generated)
        assert human_summary["recordings"] == machine_summary["recordings"] == 1
        machine_path.write_text(human_path.read_text())
        try:
            validate_mixed_reviews(human_path, machine_path, mixed_intents, mixed_generated)
        except ValueError as exc:
            assert "overlap" in str(exc), str(exc)
        else:
            raise AssertionError("accepted overlapping human and ASR reviews")

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        audio = root / "voice.wav"
        with wave.open(str(audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 1600)
        manifest = root / "manifest.jsonl"
        manifest.write_text(json.dumps({"source_id": "source", "file_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                                        "audio": audio.name, "text": "小窝小窝"}, ensure_ascii=False) + "\n")
        intents = root / "intents.jsonl"
        intents.write_text(json.dumps({"schema_version": 1,
                                       "evidence_class": "speech-like-request-label-intent-v1",
                                       "source_id": "source", "text": "小窝小窝",
                                       "kind": "positive", "keyword_id": 2}, ensure_ascii=False) + "\n")
        engine = root / "engine"
        model = root / "model"
        tokens = root / "tokens"
        for path in (engine, model, tokens):
            path.write_bytes(b"fixed-asr-fixture")
        original_run = corpus_plan.subprocess.run
        corpus_plan.subprocess.run = lambda *_args, **_kwargs: SimpleNamespace(stdout='{"text":"小窝小窝"}\n')
        try:
            result = corpus_plan.generate_asr_review([manifest], None, engine, model, tokens,
                                                     root / "asr-review.jsonl", intents=intents)
        finally:
            corpus_plan.subprocess.run = original_run
        assert result["accepted"] == 1
        assert result["rejected"] == 0

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        inputs = []
        for ordinal, provider_name in enumerate(("vits-aishell3", "melo-zh")):
            source_root = root / provider_name
            source_root.mkdir()
            audio = source_root / "clip.wav"
            audio.write_bytes(f"wav-fixture-{ordinal}".encode())
            row = {"schema_version": 1,
                   "evidence_class": "speech-like-synthetic-recording-v1",
                   "provider_name": provider_name,
                   "provider_identity_sha256": chr(ord("a") + ordinal) * 64,
                   "source_id": f"source-{ordinal}", "audio": "clip.wav",
                   "file_sha256": hashlib.sha256(audio.read_bytes()).hexdigest()}
            manifest = source_root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            inputs.append(manifest)
        output = root / "merged" / "manifest.jsonl"
        assert merge_generated_manifests(inputs, output)["providers"] == 2
        merged = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(merged) == 2
        assert all(pathlib.Path(row["audio"]).is_absolute() for row in merged)
        try:
            merge_generated_manifests(inputs, output)
        except ValueError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("provider merge overwrote an existing manifest")

    # Existing manifests stay byte-compatible unless the new provenance flag is used.
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        provider = {
            "identity": {"model": "pinned"},
            "provider_name": "fixture-tts",
            "provider_version": "1",
            "license_id": "MIT",
            "locale": "zh-CN",
            "executable": pathlib.Path("/bin/true").resolve(),
            "assets": {},
            "argv_template": ["{executable}", "{output}"],
            "timeout_seconds": 5,
        }
        request = {
            "group": "train", "text": "小窝小窝", "tokens": ["xiao3", "wo1"],
            "voice_id": "voice-1", "source_id": "source-1", "parameters": {},
        }
        original_run = command_provider.subprocess.run

        def synthesize(argv, **_kwargs):
            with wave.open(argv[1], "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 1600)

        command_provider.subprocess.run = synthesize
        try:
            common = {"policy": {"allowed_request_groups": ["train"]},
                      "provider": provider, "requests": [request]}
            command_provider.generate(**common, output_root=root / "legacy")
            command_provider.generate(**common, output_root=root / "new", emit_provider_identity=True)
        finally:
            command_provider.subprocess.run = original_run
        legacy = json.loads((root / "legacy/train/manifest.jsonl").read_text())
        new = json.loads((root / "new/train/manifest.jsonl").read_text())
        assert "provider_identity_sha256" not in legacy
        assert new.pop("provider_identity_sha256") == command_provider.canonical_sha256(provider["identity"])
        assert new == legacy

    print("cross-provider synthetic holdout: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
