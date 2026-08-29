from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        raw = root / "power.csv"
        raw.write_text("t,power_mw\n0,123\n", encoding="utf-8")
        output = root / "evidence.json"
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "tools" / "collect_target_evidence.py"),
                "--output", str(output),
                "--target", "fixture",
                "--board-revision", "A",
                "--soc", "fixture-soc",
                "--toolchain", "fixture-gcc",
                "--compiler-flags=-O3",
                "--audio-frontend", "fixture-afe",
                "--soak-hours", "1",
                "--cpu-percent", "5",
                "--rss-kib", "512",
                "--stack-high-water-bytes", "4096",
                "--max-temp-c", "55",
                "--average-power-mw", "123",
                "--power-raw", str(raw),
                "--instrument-id", "meter-1",
                "--calibration-id", "cal-1",
            ]
        )
        value = json.loads(output.read_text(encoding="utf-8"))
        assert value["schema_version"] == 2
        assert value["target"] == "fixture"
        assert len(value["collector_sha256"]) == 64
        assert len(value["power_raw_sha256"]) == 64
        assert value["raw_evidence"][0]["sha256"] == value["power_raw_sha256"]
        assert value["instrument_id"] == "meter-1"
    print("test_target_evidence: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
