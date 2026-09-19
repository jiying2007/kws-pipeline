from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_bench_signal.py"

# Real `kws_bench` output from a hosted Release build, pinned so the fixture is
# production-shaped rather than invented.
BASELINE = """\
case=baseline frontend=logmel keywords=1 trie_nodes=5 geometry=32x48x420 gate=disabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033764 rtf=0.003376 us_per_audio_s=3376.4
case=baseline-gated frontend=logmel keywords=1 trie_nodes=5 geometry=32x48x420 gate=enabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033765 rtf=0.003377 us_per_audio_s=3376.5
case=product frontend=logmel keywords=4 trie_nodes=8 geometry=32x48x420 gate=disabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033800 rtf=0.003380 us_per_audio_s=3380.0
case=pcen-product frontend=pcen-lite keywords=4 trie_nodes=8 geometry=32x48x420 gate=disabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033859 rtf=0.003386 us_per_audio_s=3385.9
case=worst-case frontend=pcen-lite keywords=16 trie_nodes=23 geometry=32x48x420 gate=disabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033982 rtf=0.003398 us_per_audio_s=3398.2
case=worst-case-gated frontend=pcen-lite keywords=16 trie_nodes=23 geometry=32x48x420 gate=enabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033911 rtf=0.003391 us_per_audio_s=3391.1
"""


def run(root: pathlib.Path, text: str, *extra: str) -> int:
    payload = root / "bench.txt"
    payload.write_text(text, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(CHECKER), "--input", str(payload), *extra],
        check=False,
        capture_output=True,
        text=True,
    ).returncode


def rewrite_last_field(text: str, replacement: str) -> str:
    """Return the fixture with its final line's rtf replaced."""
    lines = text.strip().splitlines()
    parts = lines[-1].split()
    parts = [replacement if item.startswith("rtf=") else item for item in parts]
    lines[-1] = " ".join(parts)
    return "\n".join(lines) + "\n"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)

        # The measured production output is the passing case.
        assert run(root, BASELINE) == 0, "the shipped bench output must satisfy the contract"

        # A legal model update moves every case together and must still pass,
        # because structural fields are checked by agreement rather than value.
        moved = BASELINE.replace("geometry=32x48x420", "geometry=40x64x512").replace(
            "model_bytes=25944", "model_bytes=31000"
        )
        assert run(root, moved) == 0, "a uniform model update must not fail the contract"

        # Timing must be measured, not zeroed: a bench that stops measuring
        # would otherwise stay green.
        assert run(root, rewrite_last_field(BASELINE, "rtf=0.000000")) == 1

        # An order-of-magnitude regression must not hide behind runner noise.
        assert run(root, rewrite_last_field(BASELINE, "rtf=1.000000")) == 1

        # A non-numeric measurement is a broken bench, not a slow one.
        assert run(root, rewrite_last_field(BASELINE, "rtf=abc")) == 1

        # A missing field means the contract can no longer be evaluated.
        stripped = "\n".join(
            " ".join(item for item in line.split() if not item.startswith("cpu_s="))
            for line in BASELINE.strip().splitlines()
        ) + "\n"
        assert run(root, stripped) == 1

        # One case drifting structurally must fail even though its timing is
        # fine -- this is what catches a silently changed model or geometry.
        drifted = BASELINE.replace(
            "case=worst-case frontend=pcen-lite keywords=16",
            "case=worst-case frontend=pcen-lite keywords=16",
        )
        lines = drifted.strip().splitlines()
        lines[-1] = lines[-1].replace("geometry=32x48x420", "geometry=32x48x999")
        assert run(root, "\n".join(lines) + "\n") == 1

        # A single case leaves nothing to compare against.
        assert run(root, BASELINE.strip().splitlines()[0] + "\n") == 1

        # Duplicate case names would let one case masquerade as two.
        assert run(root, BASELINE + BASELINE.strip().splitlines()[0] + "\n") == 1

        # Trailing junk is not silently ignored.
        assert run(root, BASELINE + "garbage\n") == 1

        # An empty capture is not a pass: a vacuous result must be an error.
        assert run(root, "") == 2, "empty bench output must not pass"

        # Neither is an unusable ceiling.
        assert run(root, BASELINE, "--max-rtf", "0") == 2

    print("test_bench_signal: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
