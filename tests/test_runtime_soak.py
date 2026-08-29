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
        assert value["elapsed_hours"] >= value["requested_hours"]
        assert value["cpu_capacity_count"] >= 1
        assert value["cpu_percent_semantics"] == (
            "process_cpu_time / elapsed / online_cpu_capacity * 100"
        )
        assert value["max_rss_kib"] is not None and value["max_rss_kib"] > 0
        assert value["average_cpu_percent"] is not None
        assert 0 <= value["average_cpu_percent"] <= 100
        assert isinstance(value["samples"], list) and value["samples"]
    print("test_runtime_soak: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
