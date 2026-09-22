from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Callable


POLICY = "refinement-phase-progress-v1"


class PhaseProgress:
    def __init__(
        self,
        path: pathlib.Path,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = path
        self.clock = clock
        self.started = float(clock())
        self.current_phase: str | None = None
        self.current_started: float | None = None
        self.phase_seconds: dict[str, float] = {}
        self._write(now=self.started)

    def begin(self, phase: str) -> None:
        if not phase:
            raise ValueError("refinement phase name must be non-empty")
        if self.current_phase is not None:
            raise ValueError(
                f"refinement phase {self.current_phase} is still active"
            )
        if phase in self.phase_seconds:
            raise ValueError(f"refinement phase {phase} already completed")
        self.current_phase = phase
        self.current_started = float(self.clock())
        self._write(now=self.current_started)

    def finish(self, phase: str) -> None:
        if self.current_phase != phase or self.current_started is None:
            raise ValueError(
                f"cannot finish refinement phase {phase}; "
                f"active={self.current_phase}"
            )
        now = float(self.clock())
        self.phase_seconds[phase] = round(
            max(0.0, now - self.current_started),
            6,
        )
        self.current_phase = None
        self.current_started = None
        self._write(now=now)

    def snapshot(self, *, now: float | None = None) -> dict:
        value = float(self.clock()) if now is None else float(now)
        current_started_elapsed = None
        if self.current_started is not None:
            current_started_elapsed = round(
                max(0.0, self.current_started - self.started),
                6,
            )
        return {
            "schema_version": 1,
            "policy": POLICY,
            "elapsed_seconds": round(max(0.0, value - self.started), 6),
            "current_phase": self.current_phase,
            "current_phase_started_elapsed_seconds": current_started_elapsed,
            "phase_seconds": dict(self.phase_seconds),
        }

    def _write(self, *, now: float | None = None) -> None:
        payload = self.snapshot(now=now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
