#!/usr/bin/env python3
"""Source-only controls; no Torch, project audio, training or KWS qualification."""
from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_contract as c


def fixture_rows():
    rows = []
    families = sorted(c.TRAIN_FAMILIES | c.DEV_FAMILIES)
    for family in families:
        for label in (1, 1, 2, 2, 0, 0, None, None, None):
            train = family in c.TRAIN_FAMILIES
            rows.append({"cell_id": "en-"+format(len(rows), "024x"),
                         "source_family_id": family,
                         "role": "train-candidate" if train else "development-only",
                         "action": c.QUARANTINE_ACTION if label is None else
                                   c.TRAIN_ACTION if train else c.DEV_ACTION,
                         "intent_requested": "near_miss" if label is None else
                                             "ordinary_negative" if label == 0 else "wake",
                         "weak_wholeclip_class": label, "ctc_target_ids": None,
                         "human_gold": False, "training_admitted": False,
                         "trusted_negative": False})
    return rows


def fixture_summary():
    return {"schema": c.POLICY, "status": c.STATUS, "config": c.fixed_config(),
            "source_audio_clips": 63, "source_pcm_samples": 2101760,
            "role_counts": dict(c.COUNTS), "formal_training_admitted": False,
            "human_gold": False, "model_promotion_allowed": False,
            "model_selection_made": False, "new_tts_asr_ipa_calls": 0,
            "threshold_selection": False,
            **{key: "NOT_RUN" for key in c.UNMEASURED},
            "models": [{"kind": kind, "optimizer_updates": 48, "train_rows": 30,
                        "train_nearphone_rows": 0, "train_development_rows": 0,
                        "weight_file": kind+"-weights.npz", "weights_sha256": "a"*64,
                        "warm_start_sha256": "b"*64, "model_promotion_allowed": False}
                       for kind in c.KINDS]}


class EnglishOnlinePilotSourceCases(unittest.TestCase):
    def test_exact_weak_training_roster_excludes_exposed_and_nearphones(self):
        rows = fixture_rows()
        selected = c.select_training_rows(rows)
        self.assertEqual(len(selected), 30)
        self.assertEqual({r["source_family_id"] for r in selected}, c.TRAIN_FAMILIES)
        self.assertTrue(all(r["weak_wholeclip_class"] is not None for r in selected))
        for action in (c.DEV_ACTION, c.QUARANTINE_ACTION):
            changed = copy.deepcopy(rows)
            row = next(r for r in changed if r["action"] == action)
            row["action"] = c.TRAIN_ACTION
            row["weak_wholeclip_class"] = 0
            with self.assertRaises(ValueError):
                c.select_training_rows(changed)

    def test_rejects_forged_gold_admission_source_roles_and_duplicates(self):
        for field, value in (("human_gold", True), ("training_admitted", True),
                             ("trusted_negative", True), ("ctc_target_ids", [1]),
                             ("source_family_id", "invented-person")):
            rows = fixture_rows()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                c.select_training_rows(rows)
        rows = fixture_rows()
        rows[-1]["cell_id"] = rows[0]["cell_id"]
        with self.assertRaises(ValueError):
            c.select_training_rows(rows)

    def test_existing_source_auditor_accepts_synthetic_manifest_schema(self):
        # Exercise the real auditor/schema with tiny unique generated fixtures,
        # never the 63 project recordings and never an acoustic/model execution.
        from online_pilot import source_audit
        rows = fixture_rows()
        with tempfile.TemporaryDirectory(prefix="online-source-test-") as temp:
            archive_path = Path(temp) / "synthetic.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index, row in enumerate(rows):
                    payload = struct.pack("<64h", *[index * 64 + i for i in range(64)])
                    out = io.BytesIO()
                    with wave.open(out, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(16000)
                        wav.writeframes(payload)
                    raw = out.getvalue()
                    row["wav_sha256"] = hashlib.sha256(raw).hexdigest()
                    row["pcm_sha256"] = hashlib.sha256(payload).hexdigest()
                    archive.writestr("english-seven-voice-actual/capture/cells/" +
                                     row["cell_id"] + "/audio-16k.wav", raw)
            result = source_audit(archive_path, rows)
            self.assertTrue(result["inventory_clean"])
            self.assertTrue(result["train_development_clean"])
            self.assertEqual(result["weak_train_rows"], 30)
            self.assertEqual(result["identity_disjointness"]["speaker_id"], "unknown")
            self.assertEqual(result["identity_disjointness"]["source_family_id"], "verified")
            rows[0]["pcm_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "source WAV/PCM identity mismatch"):
                source_audit(archive_path, rows)

    def test_arbitrary_pcm_chunking_has_same_real_frames_no_padding(self):
        source = [((i % 101)-50)/100 for i in range(10317)]
        expected = [source[i:i+c.FRAME] for i in range(0, len(source)-c.FRAME+1, c.HOP)]
        randomizer = random.Random(7331)
        for sizes in ((1,), (160,), (3199,), tuple(randomizer.randint(1, 3199) for _ in range(29))):
            framer = c.BoundedFrames()
            result, offset, index = [], 0, 0
            while offset < len(source):
                end = min(len(source), offset+sizes[index % len(sizes)])
                result.extend(window for _, window in framer.push(source[offset:end]))
                offset, index = end, index+1
            before = (framer.samples, framer.frames, list(framer.tail))
            self.assertEqual(framer.finish(), len(expected))
            self.assertEqual(framer.finish(), len(expected))
            self.assertEqual(before, (framer.samples, framer.frames, framer.tail))
            self.assertEqual(result, expected)
            self.assertLess(framer.maximum_buffer_samples, c.FRAME+max(sizes))

    def test_long_input_not_limited_to_original_twelve_second_clips(self):
        framer = c.BoundedFrames()
        for _ in range(3000):
            framer.push([0.0]*160)
        self.assertGreater(framer.finish(), 1200)
        self.assertLess(len(framer.tail), c.FRAME)
        self.assertEqual(framer.samples, 480000)
        state = (framer.samples, framer.frames, list(framer.tail))
        for invalid in ([], [float("nan")], [1.1], [0]*3200):
            with self.assertRaises(ValueError):
                framer.push(invalid)
            self.assertEqual(state, (framer.samples, framer.frames, framer.tail))

    def test_short_eof_does_not_manufacture_a_frame(self):
        framer = c.BoundedFrames()
        self.assertEqual(framer.push([0.0]*399), [])
        self.assertEqual(framer.finish(), 0)
        self.assertEqual(framer.samples, 399)
        self.assertEqual(len(framer.tail), 399)

    def test_finite_readout_forgets_old_peak_and_never_uses_future(self):
        pool = c.RollingPool(2)
        self.assertEqual(pool.push([100, -100]), [100, -100, 100, -100])
        for _ in range(199):
            before = pool.push([0, 0])
        self.assertEqual(before, [0.5, -0.5, 100, 0])
        self.assertEqual(pool.push([0, 0]), [0, 0, 0, 0])
        self.assertEqual(len(pool.rows), 200)

    def test_zero_proposals_is_valid_and_no_eof_event_path_exists(self):
        detector = c.ProposalDecoder()
        self.assertTrue(all(detector.consume([1.0, 0.0, 0.0]) is None for _ in range(3000)))
        self.assertEqual(detector.frame, 3000)
        self.assertFalse(hasattr(detector, "finish"))

    def test_proposal_confirmation_cooldown_and_real_sample_clock(self):
        decoder = c.ProposalDecoder()
        events = [value for _ in range(160)
                  if (value := decoder.consume([0.05, 0.90, 0.05])) is not None]
        self.assertEqual([e["frame_index"] for e in events], [52, 154])
        self.assertEqual(events[0]["available_after_sample"], 8560)
        self.assertTrue(all(e["research_proposal_only"] and not e["acoustic_truth"] for e in events))
        self.assertEqual(c.ProposalDecoder().frame, 0)
        with self.assertRaises(ValueError):
            decoder.consume([0.1, 0.9, 0.9])
        self.assertEqual(decoder.frame, 160)

    def test_summary_rejects_qualification_threshold_tuning_or_extra_training(self):
        self.assertTrue(c.validate_summary(fixture_summary()))
        for key, value in (("formal_training_admitted", True), ("human_gold", True),
                           ("threshold_selection", True), ("model_selection_made", True),
                           ("model_promotion_allowed", True), ("word_end_latency", 300),
                           ("continuous_negative_far_per_hour", 0)):
            summary = fixture_summary()
            summary[key] = value
            with self.assertRaises(ValueError):
                c.validate_summary(summary)
        for key, value in (("optimizer_updates", 49), ("train_nearphone_rows", 1),
                           ("train_development_rows", 1), ("weights_sha256", "b"*64)):
            summary = fixture_summary()
            summary["models"][0][key] = value
            with self.assertRaises(ValueError):
                c.validate_summary(summary)


if __name__ == "__main__":
    unittest.main()
