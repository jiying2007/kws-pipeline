#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from statistical_bounds import poisson_rate_upper, qualification_bounds, wilson_upper  # noqa: E402


class FarOutputPaths(unittest.TestCase):
    """JSON-only CLI checks, including under -O/-OO; no removable assertions."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        self.sources = [self.root / f"shard-{seed}.json" for seed in (1, 2)]
        for seed, path in enumerate(self.sources, 1):
            path.write_text(json.dumps({
                "schema_version": 1,
                "runner_sha256": "a" * 64,
                "model_sha256": "b" * 64,
                "keyword_pack_sha256": "c" * 64,
                "seed": seed,
                "false_accepts": seed,
                "audio_hours": 12.0,
            }) + "\n", encoding="utf-8")

    @staticmethod
    def snapshot(path: pathlib.Path):
        if not path.exists() and not path.is_symlink():
            return None
        stat = path.lstat()
        try:
            content = path.read_bytes()
        except OSError as exc:
            content = ("unreadable", exc.errno)
        return (content, stat.st_dev, stat.st_ino, stat.st_mode,
                str(path.readlink()) if path.is_symlink() else None)

    def run_cli(self, output, *, sources=None, code=2, error="aliases an input summary"):
        sources = self.sources if sources is None else sources
        before = {path: self.snapshot(path) for path in [*self.sources, *sources, output]}
        entries = set(self.root.rglob("*"))
        command = [sys.executable, "-S", "-B"]
        if sys.flags.optimize:
            command.append("-" + "O" * sys.flags.optimize)
        command.append(str(ROOT / "eval" / "aggregate_far.py"))
        for source in sources:
            command.extend(["--summary", str(source)])
        command.extend(["--output", str(output), "--max-far-per-hour", "1",
                        "--max-upper-bound-per-hour", "1"])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, code, completed.stdout + completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        for path in [*self.sources, *sources]:
            self.assertEqual(self.snapshot(path), before[path], str(path))
        if code == 2:
            self.assertIn(error, completed.stderr)
            self.assertIn("no aggregate written", completed.stderr)
            self.assertIn("stale for this invocation", completed.stderr)
            self.assertEqual(completed.stdout, "")
            self.assertEqual(self.snapshot(output), before[output])
            self.assertEqual(set(self.root.rglob("*")), entries)
            return None
        self.assertEqual(completed.stderr, "")
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(completed.stdout), result)
        self.assertEqual(result["inputs"], [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sources
        ])
        self.assertEqual(result["audio_hours"], 24.0)
        self.assertEqual(result["false_accepts"], 3)
        self.assertEqual(result["far_per_hour"], 3.0 / 24.0)
        self.assertEqual(result["far_upper_bound_per_hour"], poisson_rate_upper(3, 24.0, 0.95))
        self.assertIs(result["qualified"], True)
        self.assertFalse(list(output.parent.glob(f".{output.name}.*")))
        return result

    def test_direct_first_input(self):
        self.run_cli(self.sources[0])

    def test_direct_later_input(self):
        self.run_cli(self.sources[1])

    def test_normalized_output(self):
        nested = self.root / "nested"
        nested.mkdir()
        self.run_cli(nested / ".." / self.sources[0].name)

    def test_relative_input_absolute_output(self):
        # subprocess inherits the repository cwd; the temporary path can be
        # represented relative to it without changing the test runner's cwd.
        import os
        sources = [pathlib.Path(os.path.relpath(path)) for path in self.sources]
        self.run_cli(self.sources[0], sources=sources)

    def test_input_symlink_to_output(self):
        alias = self.root / "input-link.json"
        alias.symlink_to(self.sources[0].name)
        self.run_cli(self.sources[0], sources=[alias, self.sources[1]])

    def test_output_symlink_to_input_is_conservatively_rejected(self):
        alias = self.root / "output-link.json"
        alias.symlink_to(self.sources[0].name)
        self.run_cli(alias)

    def test_output_parent_symlink(self):
        alias = self.root / "directory-link"
        alias.symlink_to(self.root, target_is_directory=True)
        self.run_cli(alias / self.sources[0].name)

    def test_input_parent_symlink(self):
        alias = self.root / "directory-link"
        alias.symlink_to(self.root, target_is_directory=True)
        self.run_cli(self.sources[0], sources=[alias / self.sources[0].name, self.sources[1]])

    def test_hardlink_to_input_is_conservatively_rejected(self):
        alias = self.root / "output-hardlink.json"
        alias.hardlink_to(self.sources[0])
        self.run_cli(alias)

    def test_output_self_symlink_loop_is_rejected_cleanly(self):
        output = self.root / "aggregate.json"
        output.symlink_to(output.name)
        self.run_cli(output, error="")

    def test_output_two_symlink_loop_is_rejected_cleanly(self):
        output = self.root / "aggregate.json"
        peer = self.root / "loop-peer.json"
        output.symlink_to(peer.name)
        peer.symlink_to(output.name)
        before = self.snapshot(peer)
        self.run_cli(output, error="")
        self.assertEqual(self.snapshot(peer), before)

    def test_input_symlink_loop_is_rejected_cleanly(self):
        alias = self.root / "input-loop.json"
        alias.symlink_to(alias.name)
        output = self.root / "aggregate.json"
        output.write_text('{"old": true}\n', encoding="utf-8")
        self.run_cli(output, sources=[alias, self.sources[1]], error="")

    def test_distinct_new_output(self):
        self.run_cli(self.root / "new" / "aggregate.json", code=0)

    def test_distinct_dangling_output_symlink_is_replaced(self):
        output = self.root / "aggregate.json"
        target = self.root / "absent.json"
        output.symlink_to(target.name)
        self.run_cli(output, code=0)
        self.assertFalse(output.is_symlink())
        self.assertFalse(target.exists())

    def test_distinct_existing_output_is_replaced(self):
        output = self.root / "aggregate.json"
        output.write_text('{"old": true}\n', encoding="utf-8")
        self.run_cli(output, code=0)

    def test_distinct_symlink_is_replaced_not_followed(self):
        target = self.root / "unrelated.json"
        target.write_text('{"keep": true}\n', encoding="utf-8")
        before = self.snapshot(target)
        output = self.root / "aggregate.json"
        output.symlink_to(target.name)
        self.run_cli(output, code=0)
        self.assertFalse(output.is_symlink())
        self.assertEqual(self.snapshot(target), before)

    def test_distinct_hardlink_is_replaced_without_changing_peer(self):
        target = self.root / "unrelated.json"
        target.write_text('{"keep": true}\n', encoding="utf-8")
        output = self.root / "aggregate.json"
        output.hardlink_to(target)
        before = self.snapshot(target)
        self.run_cli(output, code=0)
        self.assertFalse(output.samefile(target))
        self.assertEqual(self.snapshot(target), before)

    def test_invalid_input_preserves_existing_distinct_output(self):
        output = self.root / "aggregate.json"
        output.write_text('{"old": true}\n', encoding="utf-8")
        self.sources[1].write_text("not JSON\n", encoding="utf-8")
        self.run_cli(output, error="Expecting value")


class FarSeedIdentity(unittest.TestCase):
    """Seed aliases must not multiply independent exposure, even under -O/-OO."""

    def check_seeds(self, seeds, *, code=0, error="", version=2,
                    manifest=False, existing=None, negative_zero=False):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            sources = [root / f"shard-{index}.json" for index in range(2)]
            for index, (path, seed) in enumerate(zip(sources, seeds)):
                row = {
                    "schema_version": version,
                    "runner_sha256": "a" * 64,
                    "model_sha256": "b" * 64,
                    "keyword_pack_sha256": "c" * 64,
                    "seed": seed,
                    "false_accepts": index + 1,
                    "audio_hours": 12.0,
                }
                if version == 2:
                    row.update(full_negative_manifest_coverage=True,
                               negative_manifest_sha256="d" * 64 if manifest else None,
                               hard_negative_injections=1 if manifest else 0,
                               hard_negative_rate_per_minute=1.0 if manifest else 0.0,
                               hard_negative_audio_seconds=1.0 if manifest else 0.0)
                text = json.dumps(row) + "\n"
                if negative_zero and index == 1:
                    self.assertEqual(seed, 0)
                    text = text.replace('"seed": 0', '"seed": -0')
                    self.assertIn('"seed": -0', text)
                path.write_text(text, encoding="utf-8")
            before = [path.read_bytes() for path in sources]
            output = root / "aggregate.json"
            if existing is not None:
                output.write_bytes(existing)
            entries = set(root.iterdir())
            command = [sys.executable, "-S", "-B"]
            if sys.flags.optimize:
                command.append("-" + "O" * sys.flags.optimize)
            command.append(str(ROOT / "eval" / "aggregate_far.py"))
            for source in sources:
                command.extend(["--summary", str(source)])
            command.extend(["--output", str(output), "--max-far-per-hour", "1",
                            "--max-upper-bound-per-hour", "1"])
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, code, completed.stdout + completed.stderr)
            self.assertEqual([path.read_bytes() for path in sources], before)
            self.assertNotIn("Traceback", completed.stderr)
            if code == 2:
                self.assertEqual(completed.stdout, "")
                self.assertIn(error, completed.stderr)
                self.assertIn("no aggregate written", completed.stderr)
                self.assertIn("stale for this invocation", completed.stderr)
                self.assertEqual(set(root.iterdir()), entries)
                self.assertEqual(output.read_bytes() if output.exists() else None, existing)
                return
            self.assertEqual(completed.stderr, "")
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(completed.stdout), result)
            self.assertEqual(result["seeds"], sorted(seeds))
            self.assertEqual(result["shards"], 2)
            self.assertEqual(result["audio_hours"], 24.0)
            self.assertEqual(result["false_accepts"], 3)
            self.assertEqual(result["far_per_hour"], 3.0 / 24.0)
            self.assertEqual(result["far_upper_bound_per_hour"], poisson_rate_upper(3, 24.0, 0.95))
            self.assertEqual(result["inputs"], [
                {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
                for path, content in zip(sources, before)
            ])
            self.assertIs(result["qualified"], True)

    def test_opposite_signs_rejected(self):
        for seeds in ((1, -1), (-1, 1)):
            for version in (1, 2):
                for existing in (None, b'{"previous": true}\n'):
                    with self.subTest(seeds=seeds, version=version, existing=existing):
                        self.check_seeds(seeds, version=version, existing=existing, code=2,
                                         error="seed absolute values must be unique")

    def test_same_signed_seed_still_rejected(self):
        for seeds in ((1, 1), (-1, -1)):
            with self.subTest(seeds=seeds):
                self.check_seeds(seeds, code=2, error="seeds must be unique")

    def test_zero_and_negative_zero_rejected(self):
        self.check_seeds((0, 0), negative_zero=True, code=2, error="seeds must be unique")

    def test_large_integer_sign_alias_rejected(self):
        seed = (1 << 4096) + 1
        self.check_seeds((seed, -seed), code=2, error="seed absolute values must be unique")

    def test_negative_seed_and_distinct_magnitude_preserved(self):
        large = (1 << 4096) + 1
        for seeds in ((-1, 2), (-2, -1), (0, -1), (-large, large + 1)):
            for version in (1, 2):
                with self.subTest(seeds=seeds, version=version):
                    self.check_seeds(seeds, version=version)

    def test_manifest_does_not_make_background_alias_independent(self):
        self.check_seeds((17, -17), manifest=True, code=2,
                         error="seed absolute values must be unique")

    def test_distinct_positive_seeds_preserved(self):
        self.check_seeds((1103, 2207), manifest=True)


def test_aggregate_far_seed_identity() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FarSeedIdentity)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun != 7 or not result.wasSuccessful() or result.skipped:
        raise AssertionError("FAR seed-identity regression suite did not pass all 7 cases")


def test_aggregate_far_output_paths() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FarOutputPaths)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun != 18 or not result.wasSuccessful() or result.skipped:
        raise AssertionError("FAR output-path regression suite did not pass all 18 cases")


def test_aggregate_far_cli() -> None:
    """Exercise aggregation with JSON fixtures only; no audio/model execution."""
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        calls = 0
        sentinel = root / "unrelated.json"
        sentinel.write_text('{"keep": true}\n', encoding="utf-8")

        def shard(seed: int) -> dict:
            return {
                "schema_version": 2,
                "evidence_class": "synthetic-streaming-far",
                "full_negative_manifest_coverage": True,
                "runner_sha256": "a" * 64,
                "model_sha256": "b" * 64,
                "keyword_pack_sha256": "c" * 64,
                "negative_manifest_sha256": None,
                "seed": seed,
                "false_accepts": 0,
                "audio_hours": 12.0,
                "hard_negative_rate_per_minute": 0.0,
                "hard_negative_injections": 0,
                "hard_negative_audio_seconds": 0.0,
            }

        def run(
            rows: list,
            code: int = 2,
            *,
            options: tuple[str, ...] = (),
            error: str = "",
            existing: str | None = None,
        ) -> dict | None:
            nonlocal calls
            calls += 1
            case = root / str(calls)
            case.mkdir()
            output = case / "aggregate.json"
            if existing is not None:
                output.write_text(existing, encoding="utf-8")
            command = [sys.executable, str(ROOT / "eval" / "aggregate_far.py")]
            inputs = []
            for index, row in enumerate(rows):
                source = case / f"shard-{index}.json"
                # Python's permissive JSON writer intentionally supplies NaN/Inf
                # adversarial inputs; the CLI must reject them cleanly.
                source.write_text(json.dumps(row) + "\n", encoding="utf-8")
                command.extend(["--summary", str(source)])
                inputs.append({
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                })
            command.extend([
                "--output", str(output), "--max-far-per-hour", "1",
                "--max-upper-bound-per-hour", "1", *options,
            ])
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            assert completed.returncode == code, (calls, completed.stdout, completed.stderr)
            assert "Traceback" not in completed.stderr, completed.stderr
            assert sentinel.read_text(encoding="utf-8") == '{"keep": true}\n'
            assert not list(case.glob(".aggregate.json.*"))
            if code == 2:
                assert error in completed.stderr, (calls, completed.stderr)
                assert "no aggregate written" in completed.stderr
                assert "stale for this invocation" in completed.stderr
                assert completed.stdout == ""
                if existing is None:
                    assert not output.exists(), (calls, output.read_text(encoding="utf-8"))
                else:
                    assert output.read_text(encoding="utf-8") == existing
                return None
            assert completed.stderr == "", completed.stderr
            result = json.loads(output.read_text(encoding="utf-8"))
            assert json.loads(completed.stdout) == result
            assert result["inputs"] == inputs
            assert result["qualified"] is (code == 0)
            return result

        zero = run([shard(-1), shard(2)], 0)
        assert zero["false_accepts"] == 0
        assert zero["seeds"] == [-1, 2]
        assert zero["audio_hours"] == 24.0
        assert math.isclose(zero["far_upper_bound_per_hour"], -math.log(0.05) / 24.0, rel_tol=1e-12)
        assert zero["hard_negative_injections"] == 0
        assert zero["hard_negative_audio_seconds"] == 0.0
        assert zero["negative_manifest_sha256"] is None

        run([{**shard(1), "audio_hours": 12}, shard(2)], 0)
        first, second = shard(1), shard(2)
        for row, count, injections, seconds in ((first, 2, 3, 2.5), (second, 3, 4, 4.0)):
            row.update(false_accepts=count, hard_negative_injections=injections,
                       hard_negative_audio_seconds=seconds, negative_manifest_sha256="d" * 64,
                       hard_negative_rate_per_minute=60.0)
        result = run([first, second], 0, options=(
            "--min-hard-negative-injections", "7", "--min-hard-negative-audio-seconds", "6.5",
        ))
        assert result["false_accepts"] == 5
        assert result["far_per_hour"] == 5.0 / 24.0
        assert result["far_upper_bound_per_hour"] == poisson_rate_upper(5, 24.0, 0.95)
        assert result["hard_negative_injections"] == 7
        assert result["hard_negative_audio_seconds"] == 6.5
        run([first, second], 1, options=("--max-far-per-hour", "0",))
        run([first, second], 1, options=("--max-upper-bound-per-hour", "0",))
        run([first, second], 1, options=("--min-hard-negative-injections", "8",))
        run([first, second], 1, options=("--min-hard-negative-audio-seconds", "7",))

        # Schema 1's existing optional-field defaults remain supported.
        legacy = [shard(1), shard(2)]
        for row in legacy:
            row["schema_version"] = 1
            for key in ("full_negative_manifest_coverage", "hard_negative_injections",
                        "hard_negative_audio_seconds", "hard_negative_rate_per_minute",
                        "negative_manifest_sha256"):
                del row[key]
        run(legacy, 0)
        run([legacy[0], shard(2)], 0)
        run([shard(1)], error="at least two independent shards")
        run([shard(1), shard(1)], error="seeds must be unique")
        run([[], shard(2)], error="expected JSON object")

        for key in ("false_accepts", "hard_negative_injections"):
            for left, right in (
                (8, -8), (0.9, 0.9), (False, False), (0.0, 0.0), ("0", "0"), (None, 0),
            ):
                run([{**shard(1), key: left}, {**shard(2), key: right}], error=key)
        for key in ("audio_hours", "hard_negative_audio_seconds"):
            # The positive total must not hide the invalid second shard.
            run([{**shard(1), key: 25}, {**shard(2), key: -1}], error=key)
            for value in (False, "12", None, math.nan, math.inf, -math.inf, 10 ** 400):
                run([{**shard(1), key: value}, shard(2)], error=key)
            run([{**shard(1), key: 1e308}, {**shard(2), key: 1e308}], error="aggregate " + key)
        run([{**shard(1), "audio_hours": 0}, shard(2)], error="audio_hours")
        for value in (False, "0", None, -1, 61, math.nan, math.inf):
            run([{**shard(1), "hard_negative_rate_per_minute": value}, shard(2)],
                error="hard_negative_rate_per_minute")
        for key in ("false_accepts", "audio_hours", "hard_negative_injections",
                    "hard_negative_audio_seconds", "hard_negative_rate_per_minute"):
            missing = shard(1)
            del missing[key]
            run([missing, shard(2)], error=key)
        for value in (True, "2", 2.0, None, 0, 3):
            run([{**shard(1), "schema_version": value}, shard(2)], error="schema")
        for value in (True, "1", 1.0, 1.9, None):
            run([{**shard(1), "seed": value}, shard(2)], error="seed")
        for value in (False, "false", "true", 1, 0, [], {}, None):
            run([{**shard(1), "full_negative_manifest_coverage": value}, shard(2)],
                error="full hard-negative manifest")
        for key in (
            "runner_sha256", "model_sha256", "keyword_pack_sha256", "negative_manifest_sha256",
        ):
            for value in (False, "", "a" * 63, "z" * 64, "A" * 64, 123):
                run([{**shard(1), key: value}, shard(2)], error=key)
            if key != "negative_manifest_sha256":
                run([{**shard(1), key: None}, shard(2)], error=key)
                missing = shard(1)
                del missing[key]
                run([missing, shard(2)], error=key)
            run([{**shard(1), key: "e" * 64}, shard(2)],
                error="different runner/model/pack/hard-negative inputs")
        run([{**shard(1), "hard_negative_rate_per_minute": 1.0}, shard(2)],
            error="different runner/model/pack/hard-negative inputs")
        for option in ("--max-far-per-hour", "--max-upper-bound-per-hour",
                       "--min-hard-negative-audio-seconds"):
            for value in ("nan", "inf", "-1"):
                run([shard(1), shard(2)], options=(option, value))
        for value in ("nan", "inf", "0.5", "1", "-1"):
            run([shard(1), shard(2)], options=("--confidence", value), error="confidence")
        run([shard(1), shard(2)], options=("--min-hard-negative-injections", "-1"))

        # Invalid reruns must signal stale output without deleting or altering
        # either a previous result or arbitrary existing content at --output.
        for existing in (json.dumps(zero) + "\n", '{"unrelated": true}\n'):
            run([{**shard(1), "false_accepts": 8}, {**shard(2), "false_accepts": -8}],
                error="false_accepts", existing=existing)
        print(f"aggregate_far JSON CLI regressions: {calls} cases passed")


def main() -> int:
    # Zero observed false accepts is not zero true FAR. At 95% confidence the
    # one-sided Poisson upper count is -ln(0.05) ~= 2.9957 events.
    zero_far = poisson_rate_upper(0, 24.0, 0.95)
    assert math.isclose(zero_far, -math.log(0.05) / 24.0, rel_tol=1e-12)
    assert zero_far > 0.12

    frr = wilson_upper(1, 10, 0.95)
    assert 0.2 < frr < 0.5
    assert wilson_upper(0, 1000, 0.95) > 0.0

    bounds = qualification_bounds(
        false_rejects=1,
        expected_wakes=10,
        false_accepts=1,
        audio_hours=24.0,
        confidence=0.95,
    )
    assert bounds["confidence_level"] == 0.95
    assert bounds["frr_upper_bound"] > 0.1
    assert bounds["far_upper_bound_per_hour"] > 1.0 / 24.0

    try:
        poisson_rate_upper(0, 0.0, 0.95)
    except ValueError:
        pass
    else:
        raise AssertionError("zero exposure must fail")

    test_aggregate_far_output_paths()
    test_aggregate_far_seed_identity()
    test_aggregate_far_cli()
    print("test_statistical_bounds: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
