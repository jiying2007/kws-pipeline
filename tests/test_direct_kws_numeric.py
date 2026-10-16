#!/usr/bin/env python3
"""Bounded software fixture only: random tensors, no audio/corpus/optimizer run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

for name, value in {
    "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE", "MKL_NUM_THREADS": "1",
    "MKL_CBWR": "COMPATIBLE", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "ATEN_CPU_CAPABILITY": "default",
}.items():
    os.environ[name] = value
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.model import TinyStreamingRNN  # Owns existing one-thread CPU policy.
import torch
from torch.nn import functional as F
from research.direct_kws import (
    CLASS_NAMES, DirectKWSSpec, build_classifier, window_cross_entropy,
    DirectEventHead, class_target, stream_events,
)
from research.candidate_encoders.torch_models import build_encoder
from eval.score_events import validate_recordings, validate_detections, score

torch.use_deterministic_algorithms(True)
SEED = 20261016
SPEC = DirectKWSSpec(8, (3, 1, 2), (1, 3, 2))
CHUNKS = (2, 0, 1, 4, 0, 6)
ATOL, RTOL = 2e-6, 2e-5


def model():
    torch.manual_seed(SEED)
    return build_classifier(SPEC)


def inputs(batch=2):
    return torch.randn(batch, 13, 32, generator=torch.Generator().manual_seed(SEED + 1))


def state(m, batch=2):
    return m.init_state(batch, device="cpu", dtype=torch.float32)


def chunks(x):
    offset = 0
    for size in CHUNKS:
        yield x[:, offset:offset + size]
        offset += size


def head(**kwargs):
    return DirectEventHead((.7, .7), min_consecutive_frames=kwargs.get("minimum", 2),
                           cooldown_frames=kwargs.get("cooldown", 5))


class ModelTests(unittest.TestCase):
    def test_lazy_import_and_spec_identity(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import research.direct_kws; assert 'torch' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(DirectKWSSpec.from_dict(SPEC.to_dict()), SPEC)
        self.assertEqual(SPEC.context_frames, 5)
        for change in ({"classes": ["<blk>", "ni3", "hao3", "xiao3", "wo1"]},
                       {"objective": "ctc"}, {"feature_dim": 31}, {"extra": True}):
            with self.assertRaises(ValueError):
                DirectKWSSpec.from_dict({**SPEC.to_dict(), **change})
        for args in ((True, (3,), (1,)), (8, [], []), (8, (3,), (0,))):
            with self.assertRaises(ValueError):
                DirectKWSSpec(*args)

    def test_exact_three_class_head_and_reused_backbone(self):
        m = model()
        self.assertEqual(CLASS_NAMES, ("background", "你好小窝", "小窝小窝"))
        self.assertEqual(m(inputs()).shape, (13, 2, 3))
        self.assertEqual(m.classifier.weight.shape, (3, 8))
        self.assertFalse(any("out_proj" in k for k in m.state_dict()))
        torch.manual_seed(SEED)
        old = build_encoder(SPEC.backbone_spec())
        for key, value in old.state_dict().items():
            if not key.startswith("out_proj."):
                self.assertTrue(torch.equal(value, m.state_dict()[key]), key)
        x = inputs()
        hidden = torch.relu(m.in_proj(x)).transpose(1, 2)
        for block, history in zip(old.blocks, state(m)):
            hidden, _ = block.forward_chunk(hidden, history)
        expected = F.linear(hidden.transpose(1, 2), m.classifier.weight,
                            m.classifier.bias).transpose(0, 1)
        torch.testing.assert_close(m(x), expected, atol=ATOL, rtol=RTOL)

    def test_whole_step_chunk_and_nonzero_state_parity(self):
        m, x = model(), inputs()
        for nonzero in (False, True):
            start = state(m)
            if nonzero:
                start = tuple(s + .17 for s in start)
            whole, expected_state = m.forward_chunk(x, start)
            actual_state, parts = start, []
            for chunk in chunks(x):
                logits, actual_state = m.forward_chunk(chunk, actual_state)
                parts.append(logits)
            torch.testing.assert_close(torch.cat(parts), whole, atol=ATOL, rtol=RTOL)
            for a, b in zip(actual_state, expected_state):
                torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)
            actual_state, parts = start, []
            for frame in x.unbind(1):
                logits, actual_state = m.step(frame, actual_state)
                parts.append(logits)
            torch.testing.assert_close(torch.stack(parts), whole, atol=ATOL, rtol=RTOL)

    def test_causality_reset_row_isolation_and_storage(self):
        m, x = model(), inputs()
        changed = x.clone()
        changed[:, 7:] += 10
        torch.testing.assert_close(m(x)[:7], m(changed)[:7], atol=ATOL, rtol=RTOL)
        changed = x.clone()
        changed[0] += 10
        torch.testing.assert_close(m(x)[:, 1], m(changed)[:, 1], atol=ATOL, rtol=RTOL)
        initial = state(m)
        saved = tuple(s.clone() for s in initial)
        _, retained = m.forward_chunk(x, initial)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(initial, saved)))
        for s in retained:
            self.assertEqual(s.untyped_storage().nbytes(), s.numel() * s.element_size())
        self.assertEqual(retained[1].numel(), 0)
        fresh = m.reset_state(2, device="cpu", dtype=torch.float32)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(fresh, initial)))
        self.assertTrue(all(a is not b for a, b in zip(fresh, initial)))
        empty, same = m.forward_chunk(x[:, :0], retained)
        self.assertEqual(empty.shape, (0, 2, 3))
        self.assertIs(same, retained)

    def test_ce_is_direct_and_masks_padded_tails(self):
        m, x = model(), inputs()
        logits = m(x)
        targets, lengths = torch.tensor([0, 2]), torch.tensor([7, 13])
        loss = window_cross_entropy(logits, targets, lengths)
        expected = F.cross_entropy(torch.stack([logits[6, 0], logits[12, 1]]), targets)
        torch.testing.assert_close(loss, expected)
        changed = x.clone()
        changed[0, 7:] += 100
        torch.testing.assert_close(window_cross_entropy(m(changed), targets, lengths), loss)
        # Background is an ordinary CE class; there is no empty CTC target.
        raw = torch.tensor([[[3., 1., 0.]]], requires_grad=True)
        window_cross_entropy(raw, torch.tensor([0])).backward()
        self.assertLess(raw.grad[0, 0, 0], 0)
        self.assertGreater(raw.grad[0, 0, 1], 0)

    def test_finite_backward_and_chunk_gradient_parity(self):
        whole, chunked = model(), model()
        x1, x2 = inputs().requires_grad_(), inputs().requires_grad_()
        labels = torch.tensor([1, 2])
        window_cross_entropy(whole(x1), labels).backward()
        current, parts = state(chunked), []
        for chunk in chunks(x2):
            out, current = chunked.forward_chunk(chunk, current)
            parts.append(out)
        window_cross_entropy(torch.cat(parts), labels).backward()
        torch.testing.assert_close(x1.grad, x2.grad, atol=2e-5, rtol=1e-4)
        self.assertTrue(torch.isfinite(x1.grad).all())
        for a, b in zip(whole.parameters(), chunked.parameters()):
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.isfinite(a.grad).all())
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=1e-4)
        self.assertGreater(float(whole.classifier.weight.grad.abs().sum()), 0)

    def test_reject_invalid_inputs_and_losses(self):
        m, x = model(), inputs()
        for bad in (x.double(), x[:, :, :31], x[0], x * float("nan")):
            with self.assertRaises(ValueError):
                m.forward_chunk(bad, state(m))
        for bad in ([], (), tuple(s.double() for s in state(m)), state(m, 1)):
            with self.assertRaises(ValueError):
                m.forward_chunk(x, bad)
        with self.assertRaises(ValueError):
            m(x, lengths=torch.tensor([13, 13]))
        logits = m(x)
        for targets in (torch.tensor([0, 3]), torch.tensor([0., 1.]), torch.tensor([0])):
            with self.assertRaises(ValueError):
                window_cross_entropy(logits, targets)
        for lengths in (torch.tensor([0, 13]), torch.tensor([14, 13]), torch.tensor([7., 13.])):
            with self.assertRaises(ValueError):
                window_cross_entropy(logits, torch.tensor([0, 1]), lengths)
        with self.assertRaises(ValueError):
            window_cross_entropy(logits[:, :, :2], torch.tensor([0, 1]))

    def test_direct_state_dict_roundtrip(self):
        original, restored = model(), model()
        buffer = io.BytesIO()
        torch.save({"spec": SPEC.to_dict(), "state_dict": original.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        self.assertEqual(DirectKWSSpec.from_dict(saved["spec"]), SPEC)
        restored.load_state_dict(saved["state_dict"], strict=True)
        torch.testing.assert_close(restored(inputs()), original(inputs()))


class EventTests(unittest.TestCase):
    def test_explicit_class_mapping_and_ambiguity(self):
        for expected, target in (([], 0), ([1], 1), ([2], 2)):
            self.assertEqual(class_target(expected, annotation_status="complete"), target)
        for expected in (None, [0], [True], [3], [1, 2], [1, 1]):
            with self.assertRaises(ValueError):
                class_target(expected, annotation_status="complete")
        for status in (None, "partial", "unknown"):
            with self.assertRaises(ValueError):
                class_target([], annotation_status=status)

    def test_background_threshold_and_consecutive_rule(self):
        h = head()
        for p in ([.9, .05, .05], [.1, .45, .45], [.2, .6, .2], [.1, .8, .1]):
            self.assertEqual(h.accept(p), [])
        event = h.accept([.1, .8, .1])[0]
        self.assertEqual(event["keyword_id"], 1)
        self.assertAlmostEqual(event["time_s"], (400 + 4 * 320) / 16000)
        self.assertEqual(event["confidence"], .8)

    def test_latch_rearm_and_per_word_cooldown(self):
        h = head(minimum=1, cooldown=5)
        self.assertEqual(len(h.accept([.1, .8, .1])), 1)
        for _ in range(20):
            self.assertEqual(h.accept([.1, .8, .1]), [])
        h.accept([.8, .1, .1])
        self.assertEqual(len(h.accept([.1, .8, .1])), 1)
        self.assertEqual(h.accept([.1, .1, .8])[0]["keyword_id"], 2)
        self.assertEqual(h.accept([.1, .8, .1]), [])
        self.assertEqual(h.accept([.1, .8, .1]), [])
        self.assertEqual(h.accept([.1, .8, .1]), [])
        self.assertEqual(h.accept([.1, .8, .1])[0]["keyword_id"], 1)
        h.reset()
        self.assertEqual(h.frame, -1)
        self.assertEqual(h.last_hit, {})
        self.assertEqual(len(h.accept([.1, .8, .1])), 1)

    def test_invalid_probability_never_changes_state(self):
        h = head()
        for p in ([1, 1, 1], [.1, float("nan"), .9], [0, -1, 2], [0, 1], [True, 0, 0]):
            before = copy.deepcopy(h.__dict__)
            with self.assertRaises(ValueError):
                h.accept(p)
            self.assertEqual(before, h.__dict__)
        for kwargs in ({"thresholds": (float("nan"), .5)}, {"min_consecutive_frames": True},
                       {"cooldown_frames": -1}):
            settings = dict(thresholds=(.7, .7), min_consecutive_frames=2, cooldown_frames=5)
            with self.assertRaises(ValueError):
                DirectEventHead(**{**settings, **kwargs})

    def test_actual_model_streaming_adapter_and_no_eof_flush(self):
        m = model().eval()
        with torch.no_grad():
            m.classifier.weight.zero_()
            m.classifier.bias.copy_(torch.tensor([0., 3., 0.]))
        x = inputs(1)
        full = list(stream_events(m, [x[0]], head(), recording="fixture"))
        uneven = list(stream_events(m, [c[0] for c in chunks(x)], head(), recording="fixture"))
        self.assertEqual(full, uneven)
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0]["keyword_id"], 1)
        self.assertEqual(full[0]["recording"], "fixture")
        self.assertAlmostEqual(full[0]["time_s"], .045)
        h = head()
        self.assertEqual(list(stream_events(m, [x[0, :1]], h, recording="short")), [])
        self.assertEqual(h.frame, 0)  # No invented EOF frame to complete the streak.
        with self.assertRaises(ValueError):
            list(stream_events(m, [x[0]], h, recording="reuse"))
        with self.assertRaises(ValueError):
            list(stream_events(m.train(), [x[0]], head(), recording="fixture"))

    def test_stream_adapter_really_preserves_nonconstant_model_state(self):
        model = build_classifier(DirectKWSSpec(1, (2,), (1,))).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.in_proj.weight[0, 0] = 1
            model.blocks[0].depthwise.weight[0, 0, 0] = 1
            model.blocks[0].pointwise.weight[0, 0, 0] = 1
            model.classifier.bias.copy_(torch.tensor([.8, 0., -10.]))
            model.classifier.weight[1, 0] = 1
        values = torch.zeros(4, 32)
        values[0, 0] = 1
        def head():
            return DirectEventHead([.5,.5], min_consecutive_frames=2, cooldown_frames=0)
        full = list(stream_events(model, [values], head(), recording='nonconstant'))
        split = list(stream_events(model, [values[:1], values[1:1], values[1:]], head(), recording='nonconstant'))
        self.assertEqual(full, split)
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0]['time_s'], .045)
        # A reset at this boundary must destroy the event, validating sensitivity.
        broken = head()
        for chunk in (values[:1], values[1:]):
            for row in model(chunk.unsqueeze(0))[:, 0].softmax(-1).tolist():
                self.assertEqual(broken.accept(row), [])

    def test_existing_event_scorer_adapter(self):
        # Invented probabilities/reference windows verify plumbing, not quality.
        h = head(minimum=1)
        rows = [{"recording": "positive", **h.accept([.1, .8, .1])[0]}]
        h.reset()
        rows += [{"recording": "negative", **h.accept([.1, .1, .8])[0]}]
        recordings = validate_recordings([
            {"recording": "positive", "duration_s": 1.,
             "expected": [{"keyword_id": 1, "start_s": 0., "end_s": .025}]},
            {"recording": "negative", "duration_s": 1., "expected": []},
        ])
        summary, false_accepts, misses = score(
            recordings, validate_detections(rows, recordings), 0., 0.)
        self.assertEqual(summary["matched"], 1)
        self.assertEqual(summary["false_accepts"], 1)
        self.assertEqual(summary["negative_recording_false_accepts"], 1)
        self.assertEqual(summary["per_keyword"]["1"]["false_rejects"], 0)
        self.assertEqual(summary["per_keyword"]["2"]["false_accepts"], 1)
        self.assertEqual(len(false_accepts), 1)
        self.assertEqual(misses, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    source_paths = (
        "research/direct_kws/__init__.py", "research/direct_kws/model.py", "research/direct_kws/events.py",
        "research/candidate_encoders/contract.py", "research/candidate_encoders/torch_models.py",
        "training/model.py", "eval/score_events.py", "eval/context_policy.py", "tests/test_direct_kws_numeric.py",
    )
    receipt = {
        "schema": "direct-kws-software-fixture-v1", "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "elapsed_seconds": time.monotonic() - start,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        "torch_version": str(torch.__version__), "python_version": sys.version.split()[0],
        "seed": SEED, "feature_seed": SEED + 1, "spec": SPEC.to_dict(),
        "device": "cpu", "dtype": "float32", "batch": 2, "frames": 13, "chunks": list(CHUNKS),
        "state_boundary_fixture": {"hidden_dim": 1, "frames": 4, "handcrafted_parameters": True},
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "scope": {"software_fixture_only": True, "corpus_audio_used": False, "tts_asr_run": False,
                  "optimizer_updates": 0, "pretrained_weights_used": False,
                  "real_acoustic_effect_measured": False, "shipping_approved": False},
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
