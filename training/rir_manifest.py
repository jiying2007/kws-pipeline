from __future__ import annotations

import hashlib
import json
import math
import pathlib


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _text(row: dict, key: str, line_no: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"RIR manifest line {line_no}: {key} must be non-empty text")
    return value.strip()


def _finite(row: dict, key: str, line_no: int) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"RIR manifest line {line_no}: {key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"RIR manifest line {line_no}: {key} must be finite")
    return result


def _distance_band(distance_m: float) -> str:
    if distance_m <= 1.0:
        return "near"
    if distance_m <= 3.0:
        return "mid"
    return "far"


def _resolve_audio(root: pathlib.Path, value: str, label: str) -> pathlib.Path:
    path = pathlib.Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved


def load_rir_manifest(path: pathlib.Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"RIR manifest does not exist: {path}")
    root = path.parent
    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        room_id = _text(row, "room_id", line_no)
        position_id = _text(row, "position_id", line_no)
        identity = (room_id, position_id)
        if identity in seen:
            raise ValueError(f"{path}:{line_no}: duplicate room_id/position_id")
        seen.add(identity)
        distance_m = _finite(row, "distance_m", line_no)
        azimuth_deg = _finite(row, "azimuth_deg", line_no)
        rt60_s = _finite(row, "rt60_s", line_no)
        if not 0.05 < distance_m <= 20.0:
            raise ValueError(f"{path}:{line_no}: distance_m must be in (0.05,20]")
        if not -180.0 <= azimuth_deg <= 180.0:
            raise ValueError(f"{path}:{line_no}: azimuth_deg must be in [-180,180]")
        if not 0.05 <= rt60_s <= 2.0:
            raise ValueError(f"{path}:{line_no}: rt60_s must be in [0.05,2]")
        mic1 = _resolve_audio(root, _text(row, "mic1_rir", line_no), "mic1_rir")
        mic2 = _resolve_audio(root, _text(row, "mic2_rir", line_no), "mic2_rir")
        mic1_sha256 = sha256_file(mic1)
        mic2_sha256 = sha256_file(mic2)
        for key, actual in (("mic1_sha256", mic1_sha256), ("mic2_sha256", mic2_sha256)):
            declared = row.get(key)
            if declared is not None:
                if not isinstance(declared, str) or declared.lower() != actual:
                    raise ValueError(f"{path}:{line_no}: {key} does not match file bytes")
        entry_payload = {
            "room_id": room_id,
            "position_id": position_id,
            "distance_m": distance_m,
            "azimuth_deg": azimuth_deg,
            "rt60_s": rt60_s,
            "device_pose": str(row.get("device_pose", "unspecified")),
            "mic1_sha256": mic1_sha256,
            "mic2_sha256": mic2_sha256,
        }
        entry_sha256 = canonical_sha256(entry_payload)
        declared_entry = row.get("sha256")
        if declared_entry is not None and (
            not isinstance(declared_entry, str) or declared_entry.lower() != entry_sha256
        ):
            raise ValueError(f"{path}:{line_no}: sha256 does not match canonical RIR entry")
        entries.append(
            {
                **entry_payload,
                "distance_band": _distance_band(distance_m),
                "mic1": str(mic1),
                "mic2": str(mic2),
                "entry_sha256": entry_sha256,
            }
        )
    if not entries:
        raise ValueError("RIR manifest contains no entries")
    manifest_sha256 = sha256_file(path)
    return {
        "schema_version": 1,
        "path": str(path),
        "sha256": manifest_sha256,
        "entries": entries,
        "distance_histogram": {
            band: sum(1 for item in entries if item["distance_band"] == band)
            for band in ("near", "mid", "far")
        },
    }
