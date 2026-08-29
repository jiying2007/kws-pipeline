from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CPU_PERCENT_SEMANTICS = "process_cpu_time / elapsed / online_cpu_capacity * 100"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        power = root / "power.csv"
        power.write_text("t,power_mw\n0,123\n", encoding="utf-8")
        soak = root / "runtime-soak.json"
        soak.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "command": ["fixture"],
                    "pid": 123,
                    "cpu_capacity_count": 2,
                    "cpu_percent_semantics": CPU_PERCENT_SEMANTICS,
                    "requested_hours": 1.0,
                    "elapsed_seconds": 3636.0,
                    "elapsed_hours": 1.01,
                    "completed_requested_duration": True,
                    "termination_returncode": -15,
                    "sample_seconds": 60.0,
                    "initial_cpu_seconds": 10.0,
                    "samples": [
                        {
                            "elapsed_s": 0.0,
                            "rss_kib": 500.0,
                            "cpu_seconds": 10.0,
                            "temp_c": 50.0,
                        },
                        {
                            "elapsed_s": 3636.0,
                            "rss_kib": 512.0,
                            "cpu_seconds": 373.6,
                            "temp_c": 55.0,
                        },
                    ],
                    "max_rss_kib": 512.0,
                    "average_cpu_percent": 5.0,
                    "max_temp_c": 55.0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
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
                "--runtime-soak", str(soak),
                "--stack-high-water-bytes", "4096",
                "--average-power-mw", "123",
                "--power-raw", str(power),
                "--instrument-id", "meter-1",
                "--calibration-id", "cal-1",
            ]
        )
        value = json.loads(output.read_text(encoding="utf-8"))
        assert value["schema_version"] == 2
        assert value["target"] == "fixture"
        assert value["soak_hours"] == 1.01
        assert abs(value["cpu_percent"] - 5.0) < 1e-9
        assert value["rss_kib"] == 512.0
        assert value["max_temp_c"] == 55.0
        assert len(value["collector_sha256"]) == 64
        assert len(value["runtime_soak_sha256"]) == 64
        assert value["runtime_soak_raw"] == soak.read_text(encoding="utf-8")
        assert len(value["power_raw_sha256"]) == 64
        names = {item["name"] for item in value["raw_evidence"]}
        assert names == {"runtime-soak.json", "power.csv"}
        assert value["instrument_id"] == "meter-1"
    print("test_target_evidence: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
