#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from refinement_progress import PhaseProgress  # noqa: E402


def main() -> int:
    values = iter([10.0, 11.0, 14.5, 15.0, 18.0, 19.0])
    with tempfile.TemporaryDirectory(prefix="refinement-progress-") as tmp:
        path = pathlib.Path(tmp) / "progress.json"
        progress = PhaseProgress(path, clock=lambda: next(values))
        progress.begin("adversarial-mining")
        active = json.loads(path.read_text(encoding="utf-8"))
        assert active["current_phase"] == "adversarial-mining"
        assert active["current_phase_started_elapsed_seconds"] == 1.0

        progress.finish("adversarial-mining")
        done = json.loads(path.read_text(encoding="utf-8"))
        assert done["current_phase"] is None
        assert done["phase_seconds"]["adversarial-mining"] == 3.5

        progress.begin("train-export")
        progress.finish("train-export")
        final = json.loads(path.read_text(encoding="utf-8"))
        assert final["phase_seconds"] == {
            "adversarial-mining": 3.5,
            "train-export": 3.0,
        }
        assert not path.with_name("progress.json.tmp").exists()

    print("refinement phase progress: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
