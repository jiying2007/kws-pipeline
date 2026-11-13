"""Invented-only regressions for exact English-63 audio reuse; no real audio."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

from english_research_label_cases import invented_report
import english_research_label_bridge as labels
import english_research_audio_intake as intake


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def invented_sources(folder: Path):
    report = invented_report()
    archive = folder / "synthetic-original.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for idx, row in enumerate(report["rows"], 1):
            pcm = struct.pack("<400h", *([idx * 7] * 400))
            buf = BytesIO()
            with wave.open(buf, "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(pcm)
            raw = buf.getvalue()
            row["wav_sha256"] = sha(raw)
            row["pcm_sha256"] = sha(pcm)
            prefix = ("english-seven-voice-actual/capture/cells/" +
                      row["cell_id"] + "/")
            z.writestr(prefix + "audio-16k.wav", raw)
            # Pinned real source contains a second 24-kHz FLOAT WAV
            # for each cell. The PCM16 intake must ignore these bytes.
            z.writestr(prefix + "native-f32.wav", b"invented-native-float")
        z.writestr("original/metadata.json", "{}")
    review = folder / "invented-review.json"
    review.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True),
                      encoding="utf-8")
    return archive, review, report


class EnglishOriginalAudioIntakeTests(unittest.TestCase):
    def run_frozen(self, archive: Path, review: Path, dest: Path):
        with patch.object(intake, "SOURCE_TTS_ZIP_SHA256", sha(archive.read_bytes())):
            with patch.object(labels, "SOURCE_BYTES_SHA256", sha(review.read_bytes())):
                return intake.materialize(archive, review, dest)

    def test_reuses_all_63_exact_original_bytes_with_disjoint_buckets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, review, original = invented_sources(root)
            dest = root / "private-source-reuse"
            out = self.run_frozen(archive, review, dest)
            self.assertEqual(out["status"],
                             "EXACT_AUDIO_REUSED_FOR_WEAK_DIAGNOSTIC_ONLY")
            self.assertEqual(out["counts"], {
                "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY": 30,
                "EXPOSED_DEVELOPMENT_NO_TUNE": 12,
                "QUARANTINE_AMBIGUOUS_NEARPHONE": 21})
            self.assertEqual(len(out["rows"]), 63)
            self.assertEqual(len({r["pcm_sha256"] for r in out["rows"]}), 63)
            self.assertTrue(all(r["training_admitted"] is False and
                                r["formal_training_eligible"] is False and
                                r["ctc_target_ids"] is None for r in out["rows"]))
            self.assertEqual(sum(1 for r in out["rows"]
                                 if "nearphone_quarantine/" in
                                 r["local_relative_wav"]), 21)
            for r in out["rows"]:
                file = dest / r["local_relative_wav"]
                self.assertTrue(file.is_file())
                self.assertEqual(sha(file.read_bytes()), r["wav_sha256"])
            self.assertEqual(len(original["rows"]), 63)
            self.assertTrue((dest / "MANIFEST.json").is_file())
            with self.assertRaisesRegex(ValueError, "EN_AUDIO_DESTINATION_CREATE_ONLY"):
                self.run_frozen(archive, review, dest)

    def test_original_tts_zip_changed_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, review, _ = invented_sources(root)
            digest = sha(archive.read_bytes())
            with archive.open("ab") as stream:
                stream.write(b"tamper")
            with patch.object(intake, "SOURCE_TTS_ZIP_SHA256", digest):
                with patch.object(labels, "SOURCE_BYTES_SHA256", sha(review.read_bytes())):
                    with self.assertRaisesRegex(
                            ValueError, "EN_AUDIO_ORIGINAL_TTS_DIGEST_DRIFT"):
                        intake.prepare(archive, review)

    def test_metadata_pcm_drift_rejected_before_any_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, review, record = invented_sources(root)
            record["rows"][0]["pcm_sha256"] = sha(b"forged")
            review.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True),
                              encoding="utf-8")
            dest = root / "should-not-exist"
            with self.assertRaisesRegex(ValueError,
                                        "EN_AUDIO_PCM_OR_ADMISSION_DRIFT"):
                self.run_frozen(archive, review, dest)
            self.assertFalse(dest.exists())

    def test_unexpected_additional_audio_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, review, _ = invented_sources(root)
            with zipfile.ZipFile(archive, "a") as z:
                z.writestr("original/extra.wav",
                           (root / "not-a-source").as_posix().encode())
            with self.assertRaisesRegex(ValueError,
                                        "EN_AUDIO_UNEXPECTED_WAV_MEMBER"):
                self.run_frozen(archive, review, root / "blocked")
            self.assertFalse((root / "blocked").exists())

    def test_path_escape_is_rejected_even_with_rebased_fixture_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, review, _ = invented_sources(root)
            with zipfile.ZipFile(archive, "a") as z:
                z.writestr("../escape.txt", "no")
            with self.assertRaisesRegex(ValueError,
                                        "EN_AUDIO_ZIP_TRAVERSAL_OR_ALIAS"):
                self.run_frozen(archive, review, root / "blocked")


    def test_missing_native_float_companion_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, review, _ = invented_sources(root)
            rebuilt = root / "missing-companion.zip"
            with zipfile.ZipFile(archive) as source:
                with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_STORED) as sink:
                    lost = False
                    for item in source.infolist():
                        if item.filename.endswith("/native-f32.wav") and not lost:
                            lost = True
                            continue
                        sink.writestr(item, source.read(item.filename))
            self.assertTrue(lost)
            with self.assertRaisesRegex(
                    ValueError,
                    "EN_AUDIO_EXPECTED_63_PCM16_PLUS_63_NATIVE_FLOAT_WAVS"):
                self.run_frozen(rebuilt, review, root / "blocked")
            self.assertFalse((root / "blocked").exists())

    def test_source_cell_cannot_be_reassigned_even_if_wav_hashes_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, review, record = invented_sources(root)
            left, right = record["rows"][:2]
            for field in ("wav_sha256", "pcm_sha256"):
                left[field], right[field] = right[field], left[field]
            review.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True),
                              encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "EN_AUDIO_PCM_OR_ADMISSION_DRIFT"):
                self.run_frozen(archive, review, root / "blocked")
            self.assertFalse((root / "blocked").exists())

    def test_strict_original_pin_rejects_invented_zip_without_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, review, _ = invented_sources(root)
            with self.assertRaisesRegex(ValueError,
                                        "EN_AUDIO_ORIGINAL_TTS_DIGEST_DRIFT"):
                with patch.object(labels, "SOURCE_BYTES_SHA256",
                                  sha(review.read_bytes())):
                    intake.prepare(archive, review)


if __name__ == "__main__":
    unittest.main()
