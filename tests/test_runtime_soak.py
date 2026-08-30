from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        output = pathlib.Path(td) / "soak.json"
        child = (
            "import time\n"
            "x = 0\n"
            "end = time.time() + 10\n"
            "while time.time() < end:\n"
            "    x += 1\n"
        )
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "tools" / "collect_runtime_soak.py"),
                "--hours",
                "0.00005",
                "--sample-seconds",
                "0.02",
                "--output",
                str(output),
                "--command",
                sys.executable,
                "-c",
                child,
            ]
        )
        value = json.loads(output.read_text(encoding="utf-8"))
        assert value["schema_version"] == 2
        assert value["completed_requested_duration"] is True
        assert value["elapsed_seconds"] > 0
        assert abs(value["elapsed_hours"] - value["elapsed_seconds"] / 3600.0) < 1e-9
        assert value["elapsed_hours"] >= value["requested_hours"]
        assert value["initial_cpu_seconds"] is not None
        assert value["cpu_capacity_count"] >= 1
        assert value["cpu_percent_semantics"] == (
            "process_cpu_time / elapsed / online_cpu_capacity * 100"
        )
        assert value["max_rss_kib"] is not None and value["max_rss_kib"] > 0
        assert value["average_cpu_percent"] is not None
        assert 0 <= value["average_cpu_percent"] <= 100
        assert isinstance(value["samples"], list) and value["samples"]
        cpu_values = [row["cpu_seconds"] for row in value["samples"] if row["cpu_seconds"] is not None]
        assert cpu_values
        expected_cpu = min(
            100.0,
            max(0.0, (cpu_values[-1] - value["initial_cpu_seconds"]) / value["elapsed_seconds"])
            / value["cpu_capacity_count"]
            * 100.0,
        )
        assert abs(value["average_cpu_percent"] - expected_cpu) < 1e-9
    print("test_runtime_soak: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
