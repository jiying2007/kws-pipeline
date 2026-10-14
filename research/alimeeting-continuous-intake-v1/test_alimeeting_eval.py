#!/usr/bin/env python3
"""Synthetic stdlib tests. No public audio, network, Torch, or KWS inference."""
import contextlib
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest import mock
import wave

_spec = importlib.util.spec_from_file_location("alimeeting_eval", Path(__file__).with_name("alimeeting_eval.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def wav_bytes(channels=8, frames=16, sample_rate=16000, seed=1):
    result = io.BytesIO()
    with wave.open(result, "wb") as output:
        output.setparams((channels, 2, sample_rate, 0, "NONE", "not compressed"))
        output.writeframes(b"".join(struct.pack("<h", seed + frame + channel * 100) for frame in range(frames) for channel in range(channels)))
    return result.getvalue()


def extensible_wav_bytes(subtype=1, valid_bits=16, channel_mask=0x63F):
    original = wav_bytes()
    base_fmt = bytearray(original[20:36])
    struct.pack_into("<H", base_fmt, 0, 0xFFFE)
    guid = struct.pack("<I", subtype) + bytes.fromhex("00001000800000aa00389b71")
    fmt = bytes(base_fmt) + struct.pack("<HHI", 22, valid_bits, channel_mask) + guid
    raw = bytearray(original[:12] + b"fmt " + struct.pack("<I", 40) + fmt +
                    b"fact" + struct.pack("<II", 4, 16) + original[36:])
    struct.pack_into("<I", raw, 4, len(raw)-8)
    return bytes(raw)


def grid(marks=("正常讨论",), duration=0.001, names=("speaker_1",)):
    text = f'''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = {duration}
tiers? <exists>
size = {len(names)}
item []:
'''
    for i, name in enumerate(names, 1):
        text += f'''    item [{i}]:
        class = "IntervalTier"
        name = "{name}"
        xmin = 0
        xmax = {duration}
        intervals: size = {len(marks)}
'''
        for j, mark in enumerate(marks, 1):
            mark = mark.replace('"', '""')
            text += f'''        intervals [{j}]:
            xmin = {duration * (j-1) / len(marks)}
            xmax = {duration * j / len(marks)}
            text = "{mark}"
'''
    return text.encode()


def archive(path, extras=(), sessions=1, same_audio=False, missing_text=False, text=None):
    with tarfile.open(path, "w:gz") as output:
        for i in range(sessions):
            pairs = [(f"Eval_Ali_far/audio_dir/R{i}_M{i}_MS.wav", wav_bytes(seed=1 if same_audio else i+1))]
            if not missing_text:
                pairs.append((f"Eval_Ali_far/textgrid_dir/R{i}_M{i}.TextGrid", text if text is not None else grid()))
            # Near-field copy must never contribute to duration or selection.
            pairs.append((f"Eval_Ali_near/audio_dir/R{i}_M{i}.wav", wav_bytes(channels=1)))
            for name, data in pairs:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))
        for member, data in extras:
            output.addfile(member, io.BytesIO(data) if data is not None else None)


class FakeResponse(io.BytesIO):
    status = 200
    def __init__(self, data, declared_length=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if declared_length is None else declared_length)}
    def geturl(self):
        return m.SOURCE_URL


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "fixture.tar.gz"
        self.out = self.root / "out"

    def run_audit(self, **kwargs):
        return m.audit(self.bundle, m.digest_file(self.bundle), self.out, expected_sessions=1, **kwargs)

    def test_default_plan_is_offline_and_has_no_publisher_hash(self):
        with mock.patch.object(m.urllib.request, "build_opener", side_effect=AssertionError("network forbidden")):
            with contextlib.redirect_stdout(io.StringIO()) as result:
                self.assertEqual(m.main([]), 0)
        plan = json.loads(result.getvalue())
        self.assertFalse(plan["network_default"])
        self.assertIsNone(plan["publisher_sha256"])
        self.assertFalse(plan["sealed_qualification"])

    def test_acquire_requires_explicit_flag_before_creating_output(self):
        with self.assertRaisesRegex(ValueError, "network disabled"):
            m.acquire(self.out)
        self.assertFalse(self.out.exists())

    def test_acquire_hashes_mock_response_and_does_not_claim_trusted_lock(self):
        payload = b"synthetic archive bytes"
        result = m.acquire(self.out, True, opener=lambda *a, **k: FakeResponse(payload), expected_bytes=len(payload))
        self.assertEqual(result["archive_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result["hash_status"], "first-acquisition-observation-only")
        self.assertIsNone(result["publisher_sha256"])

    def test_acquire_rejects_changed_or_truncated_body(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            m.acquire(self.out, True, opener=lambda *a, **k: FakeResponse(b"short", 10), expected_bytes=10)
        self.assertFalse((self.out / "Eval_Ali.tar.gz").exists())
        self.assertFalse((self.out / "acquisition.json").exists())

    def test_failed_acquisition_retains_partial_bytes_and_receipt(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            m.acquire(self.out, True, opener=lambda *a, **k: FakeResponse(b"short", 10), expected_bytes=10)
        partial = self.out / "Eval_Ali.tar.gz.partial"
        self.assertEqual(partial.read_bytes(), b"short")
        result = json.loads((self.out / "acquisition-failed.json").read_text())
        self.assertEqual(result["retained_sha256"], hashlib.sha256(b"short").hexdigest())
        self.assertEqual(result["retained_bytes"], 5)
        self.assertEqual(result["attempt_count"], 1)
        self.assertFalse(result["acquisition_complete"])

    def test_acquire_rejects_wrong_expected_hash(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            m.acquire(self.out, True, "0"*64, opener=lambda *a, **k: FakeResponse(b"abc"), expected_bytes=3)
        self.assertFalse((self.out / "Eval_Ali.tar.gz").exists())

    def test_acquire_rejects_redirect_and_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "redirect"):
            m.NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid")
        with self.assertRaisesRegex(ValueError, "source length"):
            m.acquire(self.out, True, opener=lambda *a, **k: FakeResponse(b"abc"), expected_bytes=4)

    def test_single_channel_identity_and_unique_duration(self):
        archive(self.bundle)
        result = self.run_audit()
        row = result["rows"][0]
        self.assertEqual(result["candidate_duration_s"], 16 / 16000)
        self.assertEqual(result["accepted_negative_duration_s"], 0)
        self.assertEqual(row["channel_index"], 0)
        self.assertEqual(row["sample_count"], 16)
        with wave.open(str(self.out / row["wav"]), "rb") as source:
            self.assertEqual(source.getnchannels(), 1)
            raw = source.readframes(100)
        self.assertEqual(raw, struct.pack("<16h", *range(1, 17)))
        self.assertEqual(row["pcm_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(len(list((self.out / "mono").glob("*.wav"))), 1)
        candidates = json.loads((self.out / "reference-candidates.jsonl").read_text())
        self.assertIsNone(candidates["expected"])
        self.assertEqual(candidates["path"], row["wav"])
        self.assertFalse(result["independence_verified"])

    def test_all_eight_synthetic_sessions_default_count(self):
        archive(self.bundle, sessions=8)
        result = m.audit(self.bundle, m.digest_file(self.bundle), self.out)
        self.assertEqual(result["unique_session_count"], 8)
        self.assertEqual(result["candidate_duration_s"], 8 * 16 / 16000)

    def test_no_silent_partial_selection(self):
        archive(self.bundle)
        with self.assertRaisesRegex(ValueError, "WAV count"):
            m.audit(self.bundle, m.digest_file(self.bundle), self.out)
        self.assertFalse((self.out / "audit.json").exists())

    def test_wrong_archive_hash_rejected_before_outputs(self):
        archive(self.bundle)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            m.audit(self.bundle, "0"*64, self.out)
        self.assertFalse(self.out.exists())

    def test_duplicate_pcm_rejected(self):
        archive(self.bundle, sessions=2, same_audio=True)
        with self.assertRaisesRegex(ValueError, "duplicate decoded"):
            m.audit(self.bundle, m.digest_file(self.bundle), self.out, expected_sessions=2)

    def test_missing_textgrid_rejected(self):
        archive(self.bundle, missing_text=True)
        with self.assertRaisesRegex(ValueError, "WAV/TextGrid mapping"):
            self.run_audit()

    def test_textgrid_extent_mismatch_rejected(self):
        archive(self.bundle, text=grid(duration=1))
        with self.assertRaisesRegex(ValueError, "extent mismatch"):
            self.run_audit()

    def test_links_and_traversal_rejected_even_if_unselected(self):
        for index, (name, kind) in enumerate((("../escape", tarfile.REGTYPE), ("unused/link", tarfile.SYMTYPE), ("unused/hard", tarfile.LNKTYPE), ("/absolute", tarfile.REGTYPE))):
            with self.subTest(name=name):
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = "outside"
                path = self.root / f"bad{index}.tar.gz"
                archive(path, extras=((member, None),))
                out = self.root / f"bad{index}"
                with self.assertRaises(ValueError):
                    m.audit(path, m.digest_file(path), out, expected_sessions=1)
                self.assertFalse((out / "audit.json").exists())
        self.assertFalse((self.root / "escape").exists())

    def test_duplicate_canonical_member_rejected(self):
        data = grid()
        member = tarfile.TarInfo("./Eval_Ali_far/textgrid_dir/R0_M0.TextGrid")
        member.size = len(data)
        archive(self.bundle, extras=((member, data),))
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            self.run_audit()

    def test_nonstandard_audio_and_truncation_rejected(self):
        for index, data in enumerate((wav_bytes(channels=1), wav_bytes(sample_rate=8000), wav_bytes()[:-2])):
            source = self.root / f"bad{index}.wav"
            source.write_bytes(data)
            with self.assertRaises(ValueError):
                m.extract_mono(source, self.root / f"result{index}.wav")

    def test_archive_symlink_rejected(self):
        archive(self.bundle)
        link = self.root / "link.tar.gz"
        link.symlink_to(self.bundle)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            m.audit(link, m.digest_file(self.bundle), self.out)

    def test_output_directory_not_overwritten(self):
        archive(self.bundle)
        self.out.mkdir()
        marker = self.out / "existing"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_audit()
        self.assertEqual(marker.read_text(), "keep")

    def test_member_and_expansion_bounds(self):
        member = tarfile.TarInfo("normal")
        member.size = m.MAX_MEMBER_BYTES + 1
        with self.assertRaisesRegex(ValueError, "size exceeds"):
            m.safe_member(member)
        archive(self.bundle)
        with mock.patch.object(m, "MAX_EXPANDED_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "expanded archive"):
                self.run_audit()

    def test_cross_interval_and_all_tier_target_screening(self):
        result = m.textgrid_audit(grid(marks=("你好", "(SIL)小窝"), names=("speaker_1", "speaker_2")))
        flags = [r for r in result["flags"] if r["phrase"] == "你好小窝"]
        self.assertEqual({r["tier_index"] for r in flags}, {0, 1})
        self.assertFalse(result["screening_complete"])

    def test_official_sparse_annotations_keep_gaps_and_tail_explicit(self):
        raw = grid(duration=1).replace(b"xmin = 0.0", b"xmin = 0.25").replace(b"xmax = 1.0", b"xmax = 0.75")
        result = m.textgrid_audit(raw)
        self.assertEqual(result["annotation_gap_count"], 2)
        self.assertTrue(result["text_scan_complete"])
        self.assertFalse(result["acoustic_target_absence_verified"])

    def test_shorter_textgrid_extent_does_not_trim_or_admit_audio(self):
        archive(self.bundle, text=grid(duration=0.0005))
        result = self.run_audit()
        row = result["rows"][0]
        self.assertEqual(row["sample_count"], 16)
        self.assertAlmostEqual(row["audio_after_textgrid_end_s"], 0.0005)
        self.assertFalse(row["annotation_extent_matches_audio"])
        self.assertEqual(result["accepted_negative_duration_s"], 0)

    def test_overlapping_full_targets_are_not_silently_skipped(self):
        result = m.textgrid_audit(grid(marks=("小窝", "小窝", "小窝"), duration=3))
        full = [r for r in result["flags"] if r["phrase"] == "小窝小窝"]
        self.assertEqual([(r["first_interval_index"],r["last_interval_index"]) for r in full], [(0,1),(1,2)])
        self.assertEqual([(r["start_s"],r["end_s"]) for r in full], [(0,2),(1,3)])

    def test_phrase_hits_have_real_interval_span_and_all_occurrences(self):
        result = m.textgrid_audit(grid(marks=("小窝小窝",)))
        partials = [r for r in result["flags"] if r["phrase"] == "小窝"]
        self.assertEqual(len(partials), 2)
        self.assertEqual(partials[0]["start_s"], 0)
        self.assertEqual(partials[0]["end_s"], 0.001)
        self.assertFalse(partials[0]["cross_interval"])

    def test_nearphone_not_relabelled_positive(self):
        result = m.textgrid_audit(grid(marks=("你好，小屋",)))
        self.assertTrue(any(r["phrase"] == "小屋" for r in result["flags"]))
        self.assertFalse(any(r["kind"] == "possible-full-target" for r in result["flags"]))

    def test_quoted_unicode_and_utf16(self):
        raw = grid(marks=('他说"你好小窝"。',))
        self.assertEqual(m.textgrid_audit(raw), m.textgrid_audit(raw.decode().encode("utf-16")))

    def test_extended_headers_rejected_before_body_parser(self):
        for i, kind in enumerate((tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE,
                                  tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK,
                                  tarfile.GNUTYPE_SPARSE)):
            header = m.PlainTarInfo("metadata")
            header.type = kind
            header.size = 1000000000
            with self.subTest(kind=kind):
                with mock.patch.object(tarfile.TarInfo, "_proc_member", side_effect=AssertionError("body parser must not run")):
                    with self.assertRaisesRegex(ValueError, "metadata or sparse"):
                        header._proc_member(None)

    def test_pax_expansion_bypass_regression(self):
        member = tarfile.TarInfo("unselected")
        member.pax_headers = {"comment": "x" * 65536}
        archive(self.bundle, extras=((member, b""),))
        with mock.patch.object(m, "MAX_MEMBER_BYTES", 4096), mock.patch.object(m, "MAX_EXPANDED_BYTES", 4096):
            with self.assertRaisesRegex(ValueError, "metadata or sparse"):
                self.run_audit()
        self.assertFalse((self.out / "audit.json").exists())

    def test_partial_frame_rejected_even_with_consistent_riff_lengths(self):
        raw = bytearray(wav_bytes())
        raw += b"\x01\x00"  # one extra data byte and its RIFF pad
        struct.pack_into("<I", raw, 4, len(raw) - 8)
        struct.pack_into("<I", raw, 40, 257)
        source = self.root / "partial.wav"
        source.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "partial PCM frame"):
            m.extract_mono(source, self.root / "partial-mono.wav")

    def test_real_layout_extensible_pcm16_preserves_first_channel(self):
        source = self.root / "extensible.wav"
        source.write_bytes(extensible_wav_bytes())
        dest = self.root / "extensible-mono.wav"
        result = m.extract_mono(source, dest)
        self.assertEqual(result["source_format_tag"], 0xFFFE)
        self.assertEqual(result["source_channel_mask"], 0x63F)
        with wave.open(str(dest), "rb") as src:
            self.assertEqual(src.readframes(16), struct.pack("<16h", *range(1, 17)))

    def test_non_pcm_or_invalid_extensible_formats_rejected(self):
        for i, kwargs in enumerate(({"subtype": 3}, {"valid_bits": 15}, {"channel_mask": 3})):
            source = self.root / f"bad-extensible{i}.wav"
            source.write_bytes(extensible_wav_bytes(**kwargs))
            with self.assertRaises(ValueError):
                m.extract_mono(source, self.root / f"bad-extensible{i}-mono.wav")

    def test_wrong_block_alignment_and_bits_rejected(self):
        for i, (offset, value) in enumerate(((32, 15), (34, 15))):
            raw = bytearray(wav_bytes())
            struct.pack_into("<H", raw, offset, value)
            source = self.root / f"fmt{i}.wav"
            source.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "PCM16 RIFF format"):
                m.extract_mono(source, self.root / f"fmt{i}-mono.wav")

    def test_stage_receipts_do_not_claim_acquisition_before_download(self):
        self.assertEqual(m.plan()["evidence_class"], "proposed-development-evaluation-not-acquired")
        payload = b"fake"
        receipt = m.acquire(self.root / "download", True, hashlib.sha256(payload).hexdigest(),
                            opener=lambda *a, **k: FakeResponse(payload), expected_bytes=4)
        self.assertEqual(receipt["sha256_lock_status"], "supplied-lock-verified")
        archive(self.bundle)
        result = self.run_audit()
        self.assertEqual(result["sha256_lock_status"], "local-archive-verified-against-supplied-sha256")
        self.assertEqual(result["evidence_class"], "audited-development-candidates-pending-review")

    def test_archive_mutation_during_audit_rejected(self):
        archive(self.bundle)
        original = m.textgrid_audit
        def changing(data):
            before = self.bundle.stat()
            os.utime(self.bundle, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
            return original(data)
        with mock.patch.object(m, "textgrid_audit", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "archive changed"):
                self.run_audit()
        self.assertFalse((self.out / "audit.json").exists())

    def test_short_malformed_nonfinite_and_incomplete_textgrid_rejected(self):
        for raw in (b'"ooTextFile short"', grid().replace(b"size = 1", b"size = 2", 1), grid().replace(b"xmax = 0.001", b"xmax = NaN", 1), grid().replace(b"xmin = 0.0", b"xmin = -0.0001")):
            with self.subTest(raw=raw[:50]):
                with self.assertRaises(ValueError):
                    m.textgrid_audit(raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
