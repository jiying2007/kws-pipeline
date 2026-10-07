"""Synthetic/adapter tests only: never opens either saved-logit source."""
import contextlib
from fractions import Fraction
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("saved_runner", HERE / "runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_disarmed_does_not_read_sources(self):
        with patch.object(runner, "read_bounded", side_effect=AssertionError("read")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([]), 77)

    def test_float32_recovery_and_six_way_normalization(self):
        row = [3.14159274, -0.123456789, 0, 1, -2, 4]
        recovered = [struct.unpack("!f", struct.pack("!f", x))[0] for x in row]
        ex = [math.exp(x - 4) for x in recovered]
        expected = [x / math.fsum(ex) for x in ex]
        self.assertEqual(runner.softmax6(row), expected)
        self.assertEqual(runner.softmax6([0] * 6), [1 / 6] * 6)
        for invalid in ([0] * 5, [0] * 7, [False] + [0] * 5,
                        [float("nan")] + [0] * 5, [float("inf")] + [0] * 5,
                        ["bad"] + [0] * 5, [1e300] + [0] * 5):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises((ValueError, OverflowError)):
                    runner.softmax6(invalid)

    def counts(self):
        return dict.fromkeys(("exponential_zeros", "division_only_zeros",
                              "rows_converted", "rows_with_zero_weight"), 0)

    def test_real_exponential_zero_allowed_and_small_positive_unchanged(self):
        counts = self.counts()
        weights = runner.softmax6([0, -1000, -600, 0, 0, 0], counts)
        self.assertEqual(len(weights), 6)
        self.assertEqual(weights[1], 0)
        self.assertGreater(weights[2], 0)
        terms = [math.exp(x) for x in (0, -1000, -600, 0, 0, 0)]
        self.assertEqual(weights, [x / math.fsum(terms) for x in terms])
        self.assertEqual(counts, dict(exponential_zeros=1, division_only_zeros=0,
                                     rows_converted=1, rows_with_zero_weight=1))

    def test_real_division_zero_counted_separately(self):
        self.assertGreater(math.exp(-745), 0)
        counts = self.counts()
        weights = runner.softmax6([0, 0, 0, 0, 0, -745], counts)
        self.assertEqual(weights, [0.2] * 5 + [0.0])
        self.assertEqual(counts, dict(exponential_zeros=0, division_only_zeros=1,
                                     rows_converted=1, rows_with_zero_weight=1))

    def test_spurious_zero_and_exact_guard_boundary(self):
        with patch.object(runner.math, "exp", side_effect=lambda x: 0.0 if x < 0 else 1.0):
            self.assertEqual(runner.softmax6([-512, 0, 0, 0, 0, 0])[0], 0)
            for above in (-511.9999694824219, -1):
                with self.assertRaisesRegex(ValueError, "outside underflow guard"):
                    runner.softmax6([above, 0, 0, 0, 0, 0])

    def test_failure_context_is_indexed_without_input_vector(self):
        with self.assertRaisesRegex(ValueError, "observation=MOCK callback_index=3 row_index=1"):
            list(runner.softmax_callback([[0] * 6, [False] + [0] * 5], "MOCK", 3))

    def test_unchanged_module_eof_tie_and_bound_api_on_synthetic_weights(self):
        path = HERE.parent / "k1-prefix-marginal-v1" / "k1_prefix_marginal.py"
        source = path.read_bytes()
        self.assertEqual(runner.sha(source), "e9edc93ffde5569cf7d931342b959738e3a40085957d3d6e1b275664d6729963")
        module = runner.load_module(path, source)
        verifier = module.K1PrefixMarginal()
        verifier.update_chunk([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1]])
        final = verifier.finalize()
        self.assertEqual(final.decision_status, "CERTIFIED_REJECT")
        self.assertEqual(final.decision, "REJECT_K1")
        self.assertEqual(final.snapshot.mass_bounds[0], (Fraction(1, 2), Fraction(1, 2)))
        self.assertTrue(final.emitted)
        self.assertFalse(verifier.finalize().emitted)

    def test_exclusive_ledger_and_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = runner.Outputs(root)
            try:
                output.event("attempt_started")
                with self.assertRaises(FileExistsError):
                    runner.Outputs(root)
                output.event("attempt_failed", reason="mock failure")
                output.result({"mock": True})
                with self.assertRaises(FileExistsError):
                    output.result({"mock": False})
                self.assertIn("mock failure", (root / "attempt.jsonl").read_text())
            finally:
                runner.os.close(output.fd)

    def test_source_shape_and_hash_checks_use_only_mock_bytes(self):
        manifest = {"common_raw_header": {}, "observations": [{"id": "MOCK", "eof_samples": 12,
                    "rows": 1, "callbacks": 1, "rows_per_callback": [1], "phases": ["finish"]}]}
        entries = [{"kind": "run_start", "schema": "mock"},
                   {"kind": "clip_start", "recording": "MOCK", "frames": 12},
                   {"kind": "callback", "recording": "MOCK", "call_index": 0,
                    "logits": [[0] * 6], "centers": [0], "selected_rows": 1,
                    "phase": "finish", "available_samples": 12},
                   {"kind": "clip_end", "recording": "MOCK", "complete": True,
                    "frames": 12, "model_rows": 1, "callbacks": 1},
                   {"kind": "run_end", "complete": True, "clips": 1}]
        def pack():
            raw = "\n".join(map(json.dumps, entries)).encode()
            source = {"bytes": len(raw), "sha256": runner.sha(raw), "schema": "mock", "observations": ["MOCK"]}
            return raw, source
        raw, source = pack()
        self.assertEqual(runner.extract_source(raw, source, manifest), {"MOCK": [[[0] * 6]]})
        with self.assertRaises(ValueError):
            runner.extract_source(raw + b" ", source, manifest)
        entries[2]["logits"][0].pop()
        raw, source = pack()
        with self.assertRaises(ValueError):
            runner.extract_source(raw, source, manifest)

    def test_fixed_gate_unresolved_never_counts_as_negative(self):
        manifest = json.loads((HERE / "manifest.json").read_text())
        records = []
        statuses = ["CERTIFIED_ACCEPT", "CERTIFIED_REJECT", "NUMERICALLY_UNRESOLVED",
                    "CERTIFIED_REJECT", "CERTIFIED_REJECT", "CERTIFIED_ACCEPT"]
        class MockVerifier:
            def __init__(self, token_order):
                self.frames = 0
                self.status = statuses[len(records)]
                records.append(self)
            def update_chunk(self, rows):
                self.frames += len(list(rows))
            def finalize(self):
                bounds = ((Fraction(0), Fraction(1)),) * 3
                return types.SimpleNamespace(emitted=True, decision_status=self.status,
                    decision="ACCEPT_K1" if self.status == "CERTIFIED_ACCEPT" else "REJECT_K1",
                    masses=(0., 0., 1.), log_masses=(-math.inf, -math.inf, 0.),
                    snapshot=types.SimpleNamespace(frames=self.frames, mass_bounds=bounds))
        observations = {x["id"]: [[[0] * 6 for _ in range(size)] for size in x["rows_per_callback"]]
                        for x in manifest["observations"]}
        with tempfile.TemporaryDirectory() as directory:
            output = runner.Outputs(Path(directory))
            try:
                result = runner.evaluate(types.SimpleNamespace(K1PrefixMarginal=MockVerifier), manifest, observations, output)
            finally:
                runner.os.close(output.fd)
        self.assertEqual(len(records), 6)
        self.assertEqual(sum(x.frames for x in records), 537)
        self.assertEqual([x["id"] for x in result["results"]], manifest["evaluation_order"])
        self.assertFalse(result["gate_pass"])
        self.assertFalse(result["primary_gate_pass"])
        self.assertFalse(result["results"][2]["gate_pass"])
        self.assertTrue(result["results"][-1]["gate_pass"])

    def test_release_refuses_not_released_and_wrong_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.json"
            raw = b'{"schema":"k1-first-prefix-saved-release-v2","state":"NOT_RELEASED","candidate":"k1-first-prefix-ctc-eof-v1"}'
            path.write_bytes(raw)
            args = types.SimpleNamespace(release=path, release_sha256=runner.sha(raw))
            with self.assertRaises(ValueError):
                runner.check_release(args)

    def test_reviewed_source_guards_with_mock_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in runner.CODE_FILES:
                (root / name).write_text("{}")
            module = root / "mock_module.py"
            module.write_text("MOCK = True\n")
            release = {"schema": "k1-first-prefix-saved-release-v2", "state": "RELEASED",
                       "candidate": "k1-first-prefix-ctc-eof-v1", "cumulative_attempt": 2,
                       "previous_attempt_sha256": runner.PREVIOUS_ATTEMPT_SHA256,
                       "review_ref": "mock-review", "authorization_ref": "mock-authorization",
                       "sha256": {name: runner.sha((root / name).read_bytes()) for name in runner.CODE_FILES}}
            release["sha256"]["k1_prefix_marginal.py"] = runner.sha(module.read_bytes())
            path = root / "mock-release.json"
            raw = json.dumps(release).encode()
            path.write_bytes(raw)
            args = types.SimpleNamespace(release=path, release_sha256=runner.sha(raw), module=module)
            with patch.object(runner, "HERE", root):
                self.assertEqual(runner.check_release(args)[0], {})
                for target in [root / name for name in runner.CODE_FILES] + [module]:
                    original = target.read_bytes()
                    target.write_bytes(original + b" ")
                    with self.assertRaises(ValueError):
                        runner.check_release(args)
                    target.write_bytes(original)
            args.release_sha256 = "0" * 64
            with self.assertRaises(ValueError):
                runner.check_release(args)


if __name__ == "__main__":
    unittest.main()
