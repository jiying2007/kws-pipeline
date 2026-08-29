from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import struct
import subprocess
import tempfile
import wave

from frontend_spec import SAMPLE_RATE_HZ
from synthetic_audio import clamp16, noise_profile, write_wav

SPEED_OF_SOUND_MPS = 343.0
SUPPORTED_DISTANCE_BANDS = {"near", "mid", "far"}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def distance_band(distance_m: float) -> str:
    if distance_m <= 1.0:
        return "near"
    if distance_m <= 3.0:
        return "mid"
    return "far"


def _delay(samples: list[float], delay_samples: float) -> list[float]:
    if delay_samples < 0.0:
        raise ValueError("delay_samples must be >= 0")
    integer = int(math.floor(delay_samples))
    fraction = delay_samples - integer
    out = [0.0] * (len(samples) + integer + 2)
    for index, value in enumerate(samples):
        target = index + integer
        out[target] += value * (1.0 - fraction)
        out[target + 1] += value * fraction
    return out


def _mix_at(dst: list[float], src: list[float], offset: int, gain: float) -> None:
    if offset < 0:
        raise ValueError("mix offset must be >= 0")
    required = offset + len(src)
    if len(dst) < required:
        dst.extend([0.0] * (required - len(dst)))
    for index, value in enumerate(src):
        dst[offset + index] += value * gain


def _simulated_rir(
    samples: list[int],
    *,
    distance_m: float,
    azimuth_deg: float,
    rt60_s: float,
    mic_spacing_m: float,
    rng: random.Random,
) -> tuple[list[float], list[float], dict]:
    if distance_m <= 0.05 or distance_m > 20.0:
        raise ValueError("distance_m must be in (0.05,20]")
    if rt60_s < 0.05 or rt60_s > 2.0:
        raise ValueError("rt60_s must be in [0.05,2]")
    if mic_spacing_m <= 0.0 or mic_spacing_m > 0.5:
        raise ValueError("mic_spacing_m must be in (0,0.5]")

    base = [float(value) for value in samples]
    azimuth_rad = math.radians(azimuth_deg)
    itd_s = mic_spacing_m * math.sin(azimuth_rad) / SPEED_OF_SOUND_MPS
    direct_delay_s = distance_m / SPEED_OF_SOUND_MPS
    left_delay = (direct_delay_s - 0.5 * itd_s) * SAMPLE_RATE_HZ
    right_delay = (direct_delay_s + 0.5 * itd_s) * SAMPLE_RATE_HZ
    common_shift = -min(0.0, left_delay, right_delay)
    left_delay += common_shift
    right_delay += common_shift

    # Distance attenuation is bounded so AGC remains testable without clipping the
    # synthetic carrier into numerical silence. Reverberation still gets stronger
    # relative to the direct path as distance grows.
    direct_gain = min(1.0, 0.75 / max(distance_m, 0.25))
    left = [value * direct_gain for value in _delay(base, left_delay)]
    right = [value * direct_gain for value in _delay(base, right_delay)]

    # Sparse exponentially decaying reflection cloud. It is deterministic from the
    # scene seed and keeps the renderer dependency-free while covering DRR/RT60
    # mismatch. Measured RIRs can replace this path through render_measured_rir().
    reflection_count = 8 + int(round(rt60_s * 16.0))
    reflection_gain_total = min(1.8, 0.18 + 0.22 * distance_m + 0.55 * rt60_s)
    for reflection in range(reflection_count):
        delay_s = rng.uniform(0.008, max(0.012, min(rt60_s, 0.75)))
        decay = 10.0 ** (-3.0 * delay_s / rt60_s)
        gain = reflection_gain_total * decay / math.sqrt(reflection_count)
        gain *= rng.uniform(0.55, 1.05)
        side_jitter = rng.uniform(-0.00035, 0.00035) * SAMPLE_RATE_HZ
        source = _delay(base, direct_delay_s * SAMPLE_RATE_HZ + delay_s * SAMPLE_RATE_HZ)
        lref = _delay(source, max(0.0, 0.5 * side_jitter))
        rref = _delay(source, max(0.0, -0.5 * side_jitter))
        _mix_at(left, lref, 0, gain)
        _mix_at(right, rref, 0, gain)

    return left, right, {
        "direct_delay_samples": int(round((direct_delay_s * SAMPLE_RATE_HZ) + common_shift)),
        "itd_samples": itd_s * SAMPLE_RATE_HZ,
        "direct_gain": direct_gain,
        "reflection_count": reflection_count,
    }


def _add_noise_and_playback(
    left: list[float],
    right: list[float],
    *,
    snr_db: float,
    noise_name: str,
    playback_sir_db: float | None,
    rng: random.Random,
) -> None:
    count = max(len(left), len(right))
    if len(left) < count:
        left.extend([0.0] * (count - len(left)))
    if len(right) < count:
        right.extend([0.0] * (count - len(right)))
    signal_rms = math.sqrt(
        (sum(value * value for value in left) + sum(value * value for value in right))
        / max(1, 2 * count)
        + 1.0e-9
    )
    noise = noise_profile(noise_name, count, rng)
    noise_rms = math.sqrt(sum(value * value for value in noise) / count + 1.0e-9)
    noise_scale = signal_rms / max(1.0e-9, 10.0 ** (snr_db / 20.0) * noise_rms)
    for index, value in enumerate(noise):
        left[index] += value * noise_scale
        right[index] += value * noise_scale * rng.uniform(0.97, 1.03)

    if playback_sir_db is not None:
        # Deterministic colored playback proxy. Its delayed, spectrally dense
        # residue models post-AEC double-talk/residual energy without pretending to
        # be a full echo canceller.
        playback = noise_profile("media", count, rng)
        playback_rms = math.sqrt(sum(value * value for value in playback) / count + 1.0e-9)
        scale = signal_rms / max(1.0e-9, 10.0 ** (playback_sir_db / 20.0) * playback_rms)
        delay = int(rng.uniform(0.004, 0.035) * SAMPLE_RATE_HZ)
        for index in range(delay, count):
            residue = playback[index - delay] * scale
            left[index] += residue
            right[index] += residue * 0.92


def _normalize_pair(left: list[float], right: list[float], peak_target: float = 25000.0) -> tuple[list[int], list[int]]:
    peak = max(1.0, max(abs(value) for value in left), max(abs(value) for value in right))
    scale = min(1.0, peak_target / peak)
    return (
        [clamp16(value * scale) for value in left],
        [clamp16(value * scale) for value in right],
    )


def afe_proxy(left: list[int], right: list[int], *, azimuth_deg: float, mic_spacing_m: float) -> tuple[list[int], int]:
    # Delay-and-sum beamformer aligned to the known synthetic source direction,
    # followed by conservative peak normalization. The product path can swap this
    # for the real audio-pipeline command adapter without changing domain metadata.
    itd = mic_spacing_m * math.sin(math.radians(azimuth_deg)) / SPEED_OF_SOUND_MPS
    shift = itd * SAMPLE_RATE_HZ
    left_f = [float(value) for value in left]
    right_f = [float(value) for value in right]
    if shift >= 0.0:
        left_f = _delay(left_f, shift)
    else:
        right_f = _delay(right_f, -shift)
    count = max(len(left_f), len(right_f))
    left_f.extend([0.0] * (count - len(left_f)))
    right_f.extend([0.0] * (count - len(right_f)))
    mono = [0.5 * (a + b) for a, b in zip(left_f, right_f)]
    rms = math.sqrt(sum(value * value for value in mono) / max(1, len(mono)) + 1.0e-9)
    target_rms = 5200.0
    gain = min(6.0, max(0.5, target_rms / rms))
    return [clamp16(value * gain) for value in mono], int(math.ceil(abs(shift)))


def _read_wav(path: pathlib.Path) -> list[int]:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getframerate() != SAMPLE_RATE_HZ or reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected mono 16-kHz PCM16 WAV")
        raw = reader.readframes(reader.getnframes())
    return list(struct.unpack("<" + "h" * (len(raw) // 2), raw))


def afe_command(
    left: list[int],
    right: list[int],
    command: list[str],
    *,
    metadata: dict,
) -> tuple[list[int], int, str]:
    if not command:
        raise ValueError("AFE command must be a non-empty argv list")
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        left_path = root / "left.wav"
        right_path = root / "right.wav"
        output_path = root / "afe.wav"
        meta_path = root / "scene.json"
        write_wav(left_path, left)
        write_wav(right_path, right)
        meta_path.write_text(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        substitutions = {
            "left": str(left_path),
            "right": str(right_path),
            "output": str(output_path),
            "metadata": str(meta_path),
        }
        argv = [str(part).format(**substitutions) for part in command]
        subprocess.run(argv, check=True)
        if not output_path.is_file():
            raise ValueError("AFE command did not produce output WAV")
        mono = _read_wav(output_path)
        return mono, int(metadata.get("afe_latency_samples", 0)), hashlib.sha256("\0".join(argv).encode()).hexdigest()


def render_scene(
    clean: list[int],
    scene: dict,
    *,
    seed: int,
    afe: dict,
) -> tuple[list[int], dict]:
    rng = random.Random(seed)
    distance_m = finite(scene["distance_m"], "scene.distance_m")
    azimuth_deg = finite(scene["azimuth_deg"], "scene.azimuth_deg")
    rt60_s = finite(scene["rt60_s"], "scene.rt60_s")
    snr_db = finite(scene["snr_db"], "scene.snr_db")
    mic_spacing_m = finite(scene.get("mic_spacing_m", 0.06), "scene.mic_spacing_m")
    noise_name = str(scene.get("noise_profile", "white"))
    playback_value = scene.get("playback_sir_db")
    playback_sir_db = None if playback_value is None else finite(playback_value, "scene.playback_sir_db")

    left_f, right_f, rir_meta = _simulated_rir(
        clean,
        distance_m=distance_m,
        azimuth_deg=azimuth_deg,
        rt60_s=rt60_s,
        mic_spacing_m=mic_spacing_m,
        rng=rng,
    )
    _add_noise_and_playback(
        left_f,
        right_f,
        snr_db=snr_db,
        noise_name=noise_name,
        playback_sir_db=playback_sir_db,
        rng=rng,
    )
    left, right = _normalize_pair(left_f, right_f)
    afe_backend = str(afe.get("backend", "proxy"))
    if afe_backend == "proxy":
        mono, afe_latency = afe_proxy(
            left,
            right,
            azimuth_deg=azimuth_deg,
            mic_spacing_m=mic_spacing_m,
        )
        afe_identity = "builtin-delay-and-sum-v1"
    elif afe_backend == "command":
        command = afe.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError("command AFE backend requires afe.command argv list")
        mono, afe_latency, afe_identity = afe_command(
            left,
            right,
            [str(value) for value in command],
            metadata=scene,
        )
    else:
        raise ValueError(f"unsupported AFE backend: {afe_backend}")

    metadata = {
        "distance_m": distance_m,
        "distance_band": distance_band(distance_m),
        "azimuth_deg": azimuth_deg,
        "rt60_s": rt60_s,
        "snr_db": snr_db,
        "noise_profile": noise_name,
        "playback_sir_db": playback_sir_db,
        "mic_spacing_m": mic_spacing_m,
        "rir_id": str(scene.get("rir_id", f"sim-rt60-{rt60_s:.3f}")),
        "room_id": str(scene.get("room_id", "sim-room")),
        "afe_backend": afe_backend,
        "afe_identity": afe_identity,
        "afe_latency_samples": afe_latency,
        **rir_meta,
    }
    return mono, metadata
