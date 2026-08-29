#!/usr/bin/env python3
from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from acoustic_scene import render_scene  # noqa: E402


def clean_tone() -> list[int]:
    return [
        int(round(9000.0 * math.sin(2.0 * math.pi * 880.0 * index / 16000.0)))
        for index in range(16000)
    ]


def main() -> int:
    clean = clean_tone()
    afe = {"backend": "proxy"}
    near_scene = {
        "distance_m": 0.5,
        "azimuth_deg": 0.0,
        "rt60_s": 0.18,
        "snr_db": 25.0,
        "noise_profile": "fan",
        "playback_sir_db": None,
        "mic_spacing_m": 0.06,
    }
    far_scene = {
        "distance_m": 4.5,
        "azimuth_deg": 90.0,
        "rt60_s": 0.72,
        "snr_db": 2.0,
        "noise_profile": "motor",
        "playback_sir_db": 4.0,
        "mic_spacing_m": 0.06,
    }
    near_a, near_meta_a = render_scene(clean, near_scene, seed=1001, afe=afe)
    near_b, near_meta_b = render_scene(clean, near_scene, seed=1001, afe=afe)
    far, far_meta = render_scene(clean, far_scene, seed=2002, afe=afe)
    assert near_a == near_b
    assert near_meta_a == near_meta_b
    assert near_meta_a["distance_band"] == "near"
    assert far_meta["distance_band"] == "far"
    assert far_meta["reflection_count"] > near_meta_a["reflection_count"]
    assert abs(float(far_meta["itd_samples"])) > 1.0
    assert near_a != far
    assert len(near_a) >= len(clean)
    assert len(far) >= len(clean)
    assert max(abs(value) for value in far) <= 32767
    assert far_meta["afe_backend"] == "proxy"
    print("test_domain_scene: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
