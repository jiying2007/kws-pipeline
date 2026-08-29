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
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "tools" / "collect_runtime_soak.py"),
                "--hours",
                "0.00003",
                "--sample-seconds",
                "0.02",
                "--output",
                str(output),
                "--command",
                sys.executable,
                "-c",
                "import time; x=0; end=time.time()+10;\nwhile time.time()<end: x+=1",
            ]
        )
        value = json.loads(output.read_text(encoding="utf-8"))
        assert value["schema_version"] == 2
        assert value["completed_requested_duration"] is True
        assert value["elapsed_hours"] >= value["requested_hours"]
        assert value["max_rss_kib"] is not None and value["max_rss_kib"] > 0
        assert value["average_cpu_percent"] is not None
        assert value["average_cpu_percent"] >= 0
        assert isinstance(value["samples"], list) and value["samples"]
    print("test_runtime_soak: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
