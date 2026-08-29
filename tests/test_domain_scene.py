#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import struct
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from acoustic_scene import render_scene  # noqa: E402


def clean_tone() -> list[int]:
    return [
        int(round(9000.0 * math.sin(2.0 * math.pi * 880.0 * index / 16000.0)))
        for index in range(16000)
    ]


def write_pcm16(path: pathlib.Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"".join(struct.pack("<h", value) for value in samples))


def write_afe_adapter(path: pathlib.Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import argparse, hashlib, json, struct, wave
p=argparse.ArgumentParser(); p.add_argument('--left'); p.add_argument('--right'); p.add_argument('--output'); p.add_argument('--result'); a=p.parse_args()
def read(path):
    with wave.open(path,'rb') as r: return list(struct.unpack('<'+'h'*r.getnframes(), r.readframes(r.getnframes())))
l=read(a.left); r=read(a.right); n=max(len(l),len(r)); l += [0]*(n-len(l)); r += [0]*(n-len(r)); out=[int((x+y)/2) for x,y in zip(l,r)]
with wave.open(a.output,'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b''.join(struct.pack('<h',x) for x in out))
with open(a.result,'w',encoding='utf-8') as f: json.dump({'latency_samples':320,'source_sha':'fixture','toolchain':'python-fixture'},f)
""",
        encoding="utf-8",
    )


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
    assert near_meta_a["rir_backend"] == "simulated-sparse-v1"
    assert far_meta["distance_band"] == "far"
    assert far_meta["reflection_count"] > near_meta_a["reflection_count"]
    assert abs(float(far_meta["itd_samples"])) > 1.0
    assert near_a != far
    assert len(near_a) >= len(clean)
    assert len(far) >= len(clean)
    assert max(abs(value) for value in far) <= 32767
    assert far_meta["afe_backend"] == "proxy"

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        mic1 = root / "mic1-rir.wav"
        mic2 = root / "mic2-rir.wav"
        rir1 = [0] * 96
        rir2 = [0] * 96
        rir1[24] = 24000
        rir1[41] = 6000
        rir2[28] = 22000
        rir2[47] = 5500
        write_pcm16(mic1, rir1)
        write_pcm16(mic2, rir2)
        measured_scene = {
            "distance_m": 4.0,
            "azimuth_deg": 45.0,
            "rt60_s": 0.45,
            "snr_db": 80.0,
            "noise_profile": "white",
            "playback_sir_db": None,
            "mic_spacing_m": 0.06,
            "room_id": "fixture-room",
            "rir_id": "fixture-position",
            "measured_rir": {
                "mic1": str(mic1),
                "mic2": str(mic2),
                "position_id": "p01",
                "device_pose": "front",
                "manifest_sha256": "a" * 64,
                "entry_sha256": "b" * 64,
            },
        }
        measured, measured_meta = render_scene(
            clean[:4096], measured_scene, seed=3003, afe=afe
        )
        assert measured
        assert measured_meta["rir_backend"] == "measured-dual-mic-v1"
        assert measured_meta["direct_delay_samples"] == 24
        assert measured_meta["itd_samples"] == 4.0
        assert measured_meta["mic1_rir_sha256"] == hashlib.sha256(mic1.read_bytes()).hexdigest()
        assert measured_meta["mic2_rir_sha256"] == hashlib.sha256(mic2.read_bytes()).hexdigest()
        assert measured_meta["room_id"] == "fixture-room"

        adapter = root / "afe_adapter.py"
        write_afe_adapter(adapter)
        command_afe = {
            "backend": "command",
            "command": [
                sys.executable,
                str(adapter),
                "--left",
                "{left}",
                "--right",
                "{right}",
                "--output",
                "{output}",
                "--result",
                "{result}",
            ],
        }
        cmd_a, cmd_meta_a = render_scene(
            clean[:4096], near_scene, seed=4004, afe=command_afe
        )
        cmd_b, cmd_meta_b = render_scene(
            clean[:4096], near_scene, seed=4004, afe=command_afe
        )
        assert cmd_a == cmd_b
        assert cmd_meta_a["afe_identity"] == cmd_meta_b["afe_identity"]
        assert cmd_meta_a["afe_latency_samples"] == 320
        provenance = cmd_meta_a["afe_provenance"]
        assert provenance["command_template_sha256"] == cmd_meta_b["afe_provenance"]["command_template_sha256"]
        assert provenance["executable_sha256"] == hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest()
        assert provenance["left_input_sha256"] == cmd_meta_b["afe_provenance"]["left_input_sha256"]
        assert provenance["output_sha256"] == cmd_meta_b["afe_provenance"]["output_sha256"]

    print("test_domain_scene: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
