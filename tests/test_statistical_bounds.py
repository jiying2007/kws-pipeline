#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from statistical_bounds import poisson_rate_upper, qualification_bounds, wilson_upper  # noqa: E402


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

    test_aggregate_far_cli()
    print("test_statistical_bounds: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
