#!/usr/bin/env python3
"""Synthetic stdlib scorer-to-miner contracts; assertions survive -O/-OO."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
MINER = ROOT / "eval" / "mine_hard_negatives.py"
SCORER = ROOT / "eval" / "score_events.py"
PYTHON = [sys.executable, "-S", "-B"] + (["-" + "O" * sys.flags.optimize] if sys.flags.optimize else [])


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_wav(path, positive=True):
    pcm = b"\0\0" * 16000
    marker = int(1234).to_bytes(2, "little", signed=True) * 16000 if positive else pcm
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(pcm + marker + pcm * 4)


class HardNegativeMiningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.source = self.root / "source.wav"
        write_wav(self.source)
        self.refs = self.root / "references.jsonl"
        self.fps = self.root / "false-positives.jsonl"
        self.reference = dict(recording="r", path="source.wav", duration_s=6.0,
                              expected=[dict(keyword_id=1, start_s=1.0, end_s=2.0)])
        self.fp = dict(recording="r", path="source.wav", duration_s=6.0,
                       keyword_id=1, time_s=2.1)
        self.counter = 0

    def invoke(self, refs=None, fps=None, *, extra=(), with_references=True,
               manifest_override=None, output_override=None):
        self.counter += 1
        write_jsonl(self.refs, [self.reference] if refs is None else refs)
        write_jsonl(self.fps, [self.fp] if fps is None else fps)
        output = output_override or self.root / f"out-{self.counter}"
        manifest = manifest_override or self.root / f"manifest-{self.counter}.tsv"
        inputs = [self.source, self.refs, self.fps]
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs]
        argv = PYTHON + [str(MINER), "--false-positives", str(self.fps),
                        "--audio-root", str(self.root), "--output-dir", str(output),
                        "--manifest", str(manifest)]
        if with_references:
            argv += ["--references", str(self.refs)]
        result = subprocess.run(argv + list(extra), capture_output=True, text=True, check=False)
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs])
        return result, output, manifest

    def rejected(self, *args, error=None, **kwargs):
        result, output, manifest = self.invoke(*args, **kwargs)
        self.assertEqual(result.returncode, 2, result.stderr)
        if error:
            self.assertIn(error, result.stderr)
        self.assertFalse(output.exists(), result.stderr)
        self.assertFalse(manifest.exists(), result.stderr)

    def assert_clips(self, result, output, manifest, names, frame_counts):
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = manifest.read_text().splitlines()
        self.assertEqual(len(lines), len(names))
        self.assertEqual(sorted(p.name for p in output.glob("*.wav")), sorted(names))
        for line, name, count in zip(lines, names, frame_counts):
            path, label = line.split("\t")
            self.assertEqual(label, "")
            self.assertEqual(pathlib.Path(path), output / name)
            with wave.open(path, "rb") as reader:
                self.assertEqual(reader.getnframes(), count)
                self.assertEqual(reader.readframes(count), b"\0\0" * count)

    def test_actual_scorer_duplicate_is_not_blank_labeled(self):
        write_jsonl(self.refs, [self.reference])
        detections = self.root / "detections.jsonl"
        write_jsonl(detections, [dict(recording="r", keyword_id=1, time_s=t, confidence=.9)
                                 for t in (2.0, 2.1)])
        before = self.refs.read_bytes(), detections.read_bytes()
        summary = self.root / "summary.json"
        scored = subprocess.run(PYTHON + [str(SCORER), "--references", str(self.refs),
                                "--detections", str(detections), "--summary", str(summary),
                                "--false-positives", str(self.fps)], capture_output=True, text=True)
        self.assertEqual(scored.returncode, 0, scored.stderr)
        report = json.loads(summary.read_text())
        self.assertEqual([report[k] for k in ("expected", "matched", "false_accepts")], [1, 1, 1])
        fps = [json.loads(line) for line in self.fps.read_text().splitlines()]
        self.assertEqual(fps[0]["time_s"], 2.1)
        result, output, manifest = self.invoke(fps=fps)
        self.assert_clips(result, output, manifest, [], [])
        self.assertIn("skipped 1", result.stdout)
        self.assertEqual(before, (self.refs.read_bytes(), detections.read_bytes()))

    def test_wrong_keyword_still_overlaps_other_keyword_truth(self):
        result, output, manifest = self.invoke(fps=[dict(self.fp, keyword_id=2, time_s=1.5)])
        self.assert_clips(result, output, manifest, [], [])

    def test_far_positive_control_and_mixed_candidate_indices(self):
        result, output, manifest = self.invoke(fps=[self.fp, dict(self.fp, time_s=4.5)])
        self.assert_clips(result, output, manifest, ["hardneg_000001_kw1.wav"], [40000])
        self.assertIn("skipped 1", result.stdout)

    def test_explicit_negative_control(self):
        write_wav(self.source, positive=False)
        result, output, manifest = self.invoke(refs=[dict(self.reference, expected=[])])
        self.assert_clips(result, output, manifest, ["hardneg_000000_kw1.wav"], [40000])

    def test_empty_fp_input_and_all_overlap_manifest(self):
        for fps in ([], [self.fp]):
            with self.subTest(fps=fps):
                self.assert_clips(*self.invoke(fps=fps), [], [])

    def test_missing_reference_argument_and_unknown_recording(self):
        self.rejected(with_references=False, error="--references")
        self.rejected(refs=[], error="no recordings")
        self.rejected(fps=[dict(self.fp, recording="unknown")], error="unknown reference")
        self.rejected(fps=[{k: v for k, v in self.fp.items() if k != "recording"}])
        self.rejected(refs=[self.reference, self.reference], error="unique")
        self.rejected(refs=[dict(self.reference, recording=True)], error="non-empty text")

    def test_missing_partial_or_presence_only_annotations_fail_closed(self):
        self.rejected(refs=[{k: v for k, v in self.reference.items() if k != "expected"}], error="annotation list")
        for expected in (None, {}, ""):
            self.rejected(refs=[dict(self.reference, expected=expected)])
        for status in ("partial", "unknown", None, False):
            self.rejected(refs=[dict(self.reference, annotation_status=status)], error="complete")
        self.rejected(refs=[dict(self.reference, expected_keywords=[1])], error="clip-presence")

    def test_invalid_reference_intervals_fail_before_outputs(self):
        for start, end in ((-1, 2), (2, 1), (1, 1), (1, 7), (float("nan"), 2),
                           (1, float("inf")), (True, 2), (1, "2")):
            with self.subTest(start=start, end=end):
                event = dict(keyword_id=1, start_s=start, end_s=end)
                self.rejected(refs=[dict(self.reference, expected=[event])])
        for event in (None, {}, dict(start_s=1, end_s=2)):
            self.rejected(refs=[dict(self.reference, expected=[event])])

    def test_invalid_center_context_and_duration_fail_closed(self):
        for time in (-1, 7, float("nan"), float("inf"), True, "2.1"):
            self.rejected(fps=[dict(self.fp, time_s=time)])
        for flag, value in (("--pre-s", "nan"), ("--pre-s", "inf"), ("--pre-s", "-1"),
                            ("--post-s", "nan"), ("--post-s", "inf"), ("--post-s", "0")):
            self.rejected(extra=(flag, value))
        for duration in (0, -1, 7, float("nan"), float("inf"), True, "6"):
            self.rejected(refs=[dict(self.reference, duration_s=duration)])
        self.rejected(fps=[dict(self.fp, duration_s=5)], error="conflicts")
        self.rejected(refs=[dict(self.reference, duration_s=7)],
                      fps=[dict(self.fp, duration_s=7)], error="WAV duration")

    def test_path_identity_missing_conflicting_and_aliases(self):
        other = self.root / "other.wav"
        write_wav(other, positive=False)
        self.rejected(fps=[dict(self.fp, path="other.wav")], error="conflicts")
        self.rejected(refs=[{k: v for k, v in self.reference.items() if k != "path"}])
        self.rejected(refs=[dict(self.reference, audio_path="other.wav")], error="aliases disagree")
        self.rejected(fps=[dict(self.fp, audio="other.wav")], error="aliases disagree")
        for field in ("audio", "audio_path", "path"):
            ref = {k: v for k, v in self.reference.items() if k != "path"}
            ref[field] = "source.wav"
            fp = {k: v for k, v in self.fp.items() if k != "path"}
            fp["time_s"] = 4.5
            self.assert_clips(*self.invoke(refs=[ref], fps=[fp]), ["hardneg_000000_kw1.wav"], [40000])
        # Relative and absolute spellings (or symlinks) may identify the same file.
        alias = self.root / "alias.wav"
        alias.symlink_to(self.source)
        self.assert_clips(*self.invoke(fps=[dict(self.fp, path=str(alias), time_s=4.5)]),
                          ["hardneg_000000_kw1.wav"], [40000])

    def test_duplicate_source_recording_aliases_are_ambiguous(self):
        for kind in ("same-path", "symlink", "hardlink"):
            with self.subTest(kind=kind):
                alias = self.root / (kind + ".wav")
                if kind == "symlink":
                    alias.symlink_to(self.source)
                elif kind == "hardlink":
                    os.link(self.source, alias)
                else:
                    alias = self.source
                self.rejected(refs=[self.reference, dict(self.reference, recording="negative-alias",
                              path=str(alias), expected=[])], error="ambiguous reference source")

    def test_actual_rounded_clamped_interval_edges(self):
        cases = [
            # Clip [0, 1) exactly abuts positive [1, 2); [2, 3) abuts its end.
            (0.0, 0.0, 1.0, False, 16000),
            (2.0, 0.0, 1.0, False, 16000),
            (1.0, 0.0, 1.0, True, 16000),
            # Nominal endpoint crosses into positive, but rounds back to 1s.
            (0.0, 0.0, 1.0 + .4 / 16000, False, 16000),
            (0.0, 0.0, 1.0 + .6 / 16000, True, 16001),
            # Clamp both ends at the actual source edges.
            (0.0, 1.5, .5, False, 8000),
            (6.0, .5, 1.0, False, 8000),
        ]
        for center, pre, post, overlap, count in cases:
            with self.subTest(center=center, pre=pre, post=post):
                result, output, manifest = self.invoke(fps=[dict(self.fp, time_s=center)],
                                                       extra=("--pre-s", str(pre), "--post-s", str(post)))
                self.assert_clips(result, output, manifest,
                                  [] if overlap else ["hardneg_000000_kw1.wav"],
                                  [] if overlap else [count])
        self.rejected(fps=[dict(self.fp, time_s=0)], extra=("--pre-s", "0", "--post-s", "0.000001"),
                      error="empty hard-negative")

    def test_tiny_positive_reference_is_not_rounded_away(self):
        # Both annotation edges round to frame 16000; outward guarding must keep it.
        event = dict(keyword_id=2, start_s=1 + .1 / 16000, end_s=1 + .2 / 16000)
        result, output, manifest = self.invoke(refs=[dict(self.reference, expected=[event])],
            fps=[dict(self.fp, time_s=1.0)], extra=("--pre-s", "0", "--post-s", str(1 / 16000)))
        self.assert_clips(result, output, manifest, [], [])

    def test_all_rows_are_validated_before_any_output(self):
        self.rejected(fps=[dict(self.fp, time_s=4.5), dict(self.fp, recording="missing")])

    def test_wrong_wav_format_and_truncation_are_rejected(self):
        with wave.open(str(self.source), "wb") as writer:
            writer.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            writer.writeframes(b"\0\0" * 48000)
        self.rejected(error="16-kHz PCM16")
        write_wav(self.source, positive=False)
        self.source.write_bytes(self.source.read_bytes()[:44 + 10])
        self.rejected(refs=[dict(self.reference, expected=[])], error="truncated")

    def test_outputs_cannot_overwrite_reference_fp_or_wav_inputs(self):
        for source in (self.refs, self.fps, self.source):
            for kind in ("direct", "symlink", "hardlink"):
                with self.subTest(source=source.name, kind=kind):
                    write_jsonl(self.refs, [self.reference])
                    write_jsonl(self.fps, [self.fp])
                    alias = self.root / f"alias-{source.name}-{kind}"
                    if kind == "direct":
                        alias = source
                    elif kind == "symlink":
                        alias.symlink_to(source)
                    else:
                        os.link(source, alias)
                    result, output, _ = self.invoke(manifest_override=alias)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("output aliases", result.stderr)
                    self.assertFalse(output.exists())
        for kind in ("direct", "symlink", "hardlink"):
            with self.subTest(clip_target=kind):
                output = self.root / f"clip-alias-{kind}"
                output.mkdir()
                target = output / "hardneg_000000_kw1.wav"
                if kind == "direct":
                    # Place the original reference itself at the generated target.
                    self.refs = target
                elif kind == "symlink":
                    target.symlink_to(self.refs)
                else:
                    os.link(self.refs, target)
                result, _, manifest = self.invoke(fps=[dict(self.fp, time_s=4.5)], output_override=output)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("output aliases", result.stderr)
                self.assertFalse(manifest.exists())

    def test_output_symlink_loop_fails_without_traceback_or_writes(self):
        loop = self.root / "loop"
        loop.symlink_to(loop.name)
        result, output, _ = self.invoke(manifest_override=loop)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("cannot resolve", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())

    def test_manifest_and_clip_outputs_must_be_distinct(self):
        output = self.root / "overlapping-outputs"
        manifest = output / "hardneg_000000_kw1.wav"
        result, _, _ = self.invoke(fps=[dict(self.fp, time_s=4.5)], output_override=output,
                                  manifest_override=manifest)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("output aliases", result.stderr)
        self.assertFalse(output.exists())

    def test_afe_output_coordinate_annotations_are_preserved(self):
        ref = copy.deepcopy(self.reference)
        ref.update(timing_contract="afe-output-sample-v1", afe_latency_samples=1600)
        ref["expected"][0].update(raw_start_s=.9, raw_end_s=1.9)
        self.assert_clips(*self.invoke(refs=[ref]), [], [])


def run():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HardNegativeMiningTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful() or result.testsRun != 19 or result.skipped:
        raise RuntimeError("hard-negative mining regressions failed or were not fully executed")
    print(f"hard-negative mining: {result.testsRun} tests passed (Python optimize={sys.flags.optimize})")


if __name__ == "__main__":
    run()
