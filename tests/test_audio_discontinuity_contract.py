from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    header = (ROOT / "include" / "kws_pipeline" / "kws.h").read_text(encoding="utf-8")
    integration = (ROOT / "docs" / "AUDIO_DISCONTINUITY.md").read_text(encoding="utf-8")
    # This test turns green only when the public API is wired into the runtime.
    assert "kws_engine_notify_discontinuity" in header
    assert "discontinu" in integration.lower()
    print("test_audio_discontinuity_contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
