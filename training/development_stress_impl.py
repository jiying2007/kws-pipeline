#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import math

DEVELOPMENT_STRESS_SPLITS = ("calibration", "test")
DISTANCE_POINTS_M = {
    "0.5m": 0.5,
    "1m": 1.0,
    "2m": 2.0,
    "3m": 3.0,
    "5m": 5.0,
}
STRESS_SELECTORS = {
    "distance_azimuth": {"distance_bin", "azimuth"},
    "distance_snr": {"distance_bin", "snr"},
    "azimuth_snr": {"azimuth", "snr"},
    "distance_azimuth_snr": {"distance_bin", "azimuth", "snr"},
}
AZIMUTH_BANDS = {"front", "side", "rear"}
SNR_BANDS = {"critical", "low", "mid", "high"}


def _azimuth_band(value: float) -> str:
    if abs(value) <= 30.0:
        return "front"
    if abs(value) <= 90.0:
        return "side"
    return "rear"


def _distance_bin(value: float) -> str:
    if value <= 0.75:
        return "0.5m"
    if value <= 1.50:
        return "1m"
    if value <= 2.50:
        return "2m"
    if value <= 4.00:
        return "3m"
    return "5m"


def _snr_band(value: float) -> str:
    if value <= 6.0:
        return "critical"
    if value <= 12.0:
        return "low"
    if value <= 20.0:
        return "mid"
    return "high"


def _distance_band_for_value(domains: dict, distance_m: float) -> str:
    candidates: list[tuple[float, int, str]] = []
    order = {name: index for index, name in enumerate(("near", "mid", "far"))}
    for name, item in domains["distance_bands"].items():
        low, high = item["distance_m"]
        if low - 1.0e-9 <= distance_m <= high + 1.0e-9:
            center = (low + high) * 0.5
            candidates.append((abs(center - distance_m), order[str(name)], str(name)))
    if not candidates:
        raise ValueError(
            f"required robustness distance {distance_m:.3f}m is outside configured distance bands"
        )
    return min(candidates)[2]


def _parse_required_stress_slices(config: dict) -> list[dict]:
    gates = config.get("robustness_gates")
    if not isinstance(gates, dict):
        return []
    raw_slices = gates.get("required_stress_slices", [])
    if not isinstance(raw_slices, list):
        raise ValueError("robustness_gates.required_stress_slices must be a list")

    requirements: list[dict] = []
    seen: set[str] = set()
    for raw in raw_slices:
        key = str(raw)
        if key in seen:
            continue
        seen.add(key)
        prefix, separator, payload = key.partition(":")
        if separator != ":" or prefix not in STRESS_SELECTORS or not payload:
            raise ValueError(f"unsupported robustness stress slice: {key}")
        selectors: dict[str, str] = {}
        for part in payload.split("|"):
            name, equals, value = part.partition("=")
            if equals != "=" or not name or not value or name in selectors:
                raise ValueError(f"invalid robustness stress slice selector: {key}")
            selectors[name] = value
        if set(selectors) != STRESS_SELECTORS[prefix]:
            raise ValueError(f"invalid robustness stress slice selector set: {key}")
        if "distance_bin" in selectors and selectors["distance_bin"] not in DISTANCE_POINTS_M:
            raise ValueError(f"unsupported robustness distance bin: {selectors['distance_bin']}")
        if "azimuth" in selectors and selectors["azimuth"] not in AZIMUTH_BANDS:
            raise ValueError(f"unsupported robustness azimuth band: {selectors['azimuth']}")
        if "snr" in selectors and selectors["snr"] not in SNR_BANDS:
            raise ValueError(f"unsupported robustness SNR band: {selectors['snr']}")
        requirements.append(
            {
                "key": key,
                "selectors": selectors,
                "specificity": len(selectors),
                "order": len(requirements),
            }
        )
    return requirements


def _validate_axes(requirements: list[dict], axes: dict) -> None:
    distance_points = [float(value) for value in axes.get("distance_m", [])]
    azimuth_points = [float(value) for value in axes.get("azimuth_deg", [])]
    snr_bands = [str(value) for value in axes.get("snr_bands", [])]
    snr_points = [float(value) for value in axes.get("snr_db", [])]
    if len(snr_bands) != len(snr_points):
        raise ValueError("evaluation SNR bands/points are not aligned")

    for requirement in requirements:
        selectors = requirement["selectors"]
        if "distance_bin" in selectors:
            point = DISTANCE_POINTS_M[selectors["distance_bin"]]
            if not any(math.isclose(point, value, rel_tol=0.0, abs_tol=1.0e-9) for value in distance_points):
                raise ValueError(
                    f"required stress distance {selectors['distance_bin']} is absent from evaluation axes"
                )
        if "azimuth" in selectors:
            band = selectors["azimuth"]
            if not any(_azimuth_band(value) == band for value in azimuth_points):
                raise ValueError(
                    f"required stress azimuth band {band} is absent from evaluation axes"
                )
        if "snr" in selectors and selectors["snr"] not in snr_bands:
            raise ValueError(
                f"required stress SNR band {selectors['snr']} is absent from evaluation axes"
            )


def _axis_coordinates(axes: dict, ordinal: int) -> dict:
    distance_points = [float(value) for value in axes["distance_m"]]
    azimuth_points = [float(value) for value in axes["azimuth_deg"]]
    snr_bands = [str(value) for value in axes["snr_bands"]]
    snr_points = [float(value) for value in axes["snr_db"]]
    distance_m = distance_points[ordinal % len(distance_points)]
    azimuth_deg = azimuth_points[ordinal % len(azimuth_points)]
    azimuth_cycle = ordinal // len(azimuth_points)
    snr_index = (ordinal + azimuth_cycle) % len(snr_points)
    return {
        "distance_bin": _distance_bin(distance_m),
        "distance_m": distance_m,
        "azimuth": _azimuth_band(azimuth_deg),
        "azimuth_deg": azimuth_deg,
        "snr": snr_bands[snr_index],
        "snr_db": snr_points[snr_index],
    }


def _matches(coordinates: dict, selectors: dict[str, str]) -> bool:
    return all(str(coordinates.get(name)) == value for name, value in selectors.items())


def _project_coordinates(
    coordinates: dict,
    selectors: dict[str, str],
    axes: dict,
    variant_index: int,
) -> dict:
    result = dict(coordinates)
    distance_name = selectors.get("distance_bin")
    if distance_name is not None and result["distance_bin"] != distance_name:
        result["distance_bin"] = distance_name
        result["distance_m"] = float(DISTANCE_POINTS_M[distance_name])

    azimuth_name = selectors.get("azimuth")
    if azimuth_name is not None and result["azimuth"] != azimuth_name:
        candidates = [
            float(value)
            for value in axes["azimuth_deg"]
            if _azimuth_band(float(value)) == azimuth_name
        ]
        if not candidates:
            raise ValueError(f"stress azimuth band {azimuth_name} is absent from evaluation axes")
        result["azimuth"] = azimuth_name
        result["azimuth_deg"] = candidates[variant_index % len(candidates)]

    snr_name = selectors.get("snr")
    if snr_name is not None and result["snr"] != snr_name:
        snr_bands = [str(value) for value in axes["snr_bands"]]
        if snr_name not in snr_bands:
            raise ValueError(f"stress SNR band {snr_name} is absent from evaluation axes")
        index = snr_bands.index(snr_name)
        result["snr"] = snr_name
        result["snr_db"] = float(axes["snr_db"][index])
    return result


def _pairwise_keys(coordinates: dict) -> tuple[tuple, tuple, tuple]:
    return (
        (coordinates["distance_bin"], coordinates["azimuth_deg"]),
        (coordinates["distance_bin"], coordinates["snr"]),
        (coordinates["azimuth_deg"], coordinates["snr"]),
    )


def _pairwise_unique(coordinates: list[dict]) -> dict[str, int]:
    return {
        "distance_azimuth": len(
            {(item["distance_bin"], item["azimuth_deg"]) for item in coordinates}
        ),
        "distance_snr": len(
            {(item["distance_bin"], item["snr"]) for item in coordinates}
        ),
        "azimuth_snr": len(
            {(item["azimuth_deg"], item["snr"]) for item in coordinates}
        ),
    }


def _support_counts(requirements: list[dict], coordinates: list[dict]) -> dict[str, int]:
    return {
        item["key"]: sum(_matches(scene, item["selectors"]) for scene in coordinates)
        for item in requirements
    }


def build_development_stress_plan(
    config: dict,
    axes: dict,
    *,
    split: str,
    positive_scene_count: int,
) -> dict:
    requirements = _parse_required_stress_slices(config)
    gates = config.get("robustness_gates", {})
    target = int(gates.get("min_expected_wakes", 1)) if isinstance(gates, dict) else 1
    if target <= 0:
        raise ValueError("robustness_gates.min_expected_wakes must be > 0")
    if positive_scene_count < 0:
        raise ValueError("positive_scene_count must be >= 0")
    _validate_axes(requirements, axes)

    baseline = [_axis_coordinates(axes, ordinal) for ordinal in range(positive_scene_count)]
    baseline_support = _support_counts(requirements, baseline)
    baseline_pairwise = _pairwise_unique(baseline)
    empty = {
        "split": split,
        "positive_scene_count": positive_scene_count,
        "target_per_slice": target,
        "required_slices": [item["key"] for item in requirements],
        "baseline_support": baseline_support,
        "resulting_support": baseline_support,
        "pairwise_unique_before": baseline_pairwise,
        "pairwise_unique_after": baseline_pairwise,
        "reserved_scenes": 0,
        "slots": {},
    }
    if split not in DEVELOPMENT_STRESS_SPLITS or not requirements:
        return empty

    current = [dict(item) for item in baseline]
    pair_counts = [Counter() for _ in range(3)]
    for coordinates in current:
        for index, key in enumerate(_pairwise_keys(coordinates)):
            pair_counts[index][key] += 1
    used: set[int] = set()
    slots: dict[int, dict] = {}
    ordered = sorted(requirements, key=lambda item: (-item["specificity"], item["order"]))

    for requirement in ordered:
        selectors = requirement["selectors"]
        while sum(_matches(scene, selectors) for scene in current) < target:
            variant_index = len(slots)
            candidates: list[tuple[tuple[int, int, int], int, dict]] = []
            for ordinal, coordinates in enumerate(current):
                if ordinal in used or _matches(coordinates, selectors):
                    continue
                projected = _project_coordinates(
                    coordinates, selectors, axes, variant_index
                )
                if not _matches(projected, selectors):
                    continue
                lost_pairwise = 0
                for index, (old_key, new_key) in enumerate(
                    zip(_pairwise_keys(coordinates), _pairwise_keys(projected))
                ):
                    if old_key != new_key and pair_counts[index][old_key] == 1:
                        lost_pairwise += 1
                mutations = sum(
                    coordinates[name] != projected[name]
                    for name in ("distance_bin", "azimuth_deg", "snr")
                )
                candidates.append(
                    ((lost_pairwise, mutations, ordinal), ordinal, projected)
                )
            if not candidates:
                raise ValueError(
                    "development positive support cannot represent robustness stress contract: "
                    f"split {split} cannot reach {target} for {requirement['key']}"
                )
            _, ordinal, projected = min(candidates, key=lambda item: item[0])
            original = current[ordinal]
            for index, key in enumerate(_pairwise_keys(original)):
                pair_counts[index][key] -= 1
            current[ordinal] = projected
            for index, key in enumerate(_pairwise_keys(projected)):
                pair_counts[index][key] += 1
            used.add(ordinal)
            slots[ordinal] = {
                "source_slice": requirement["key"],
                "selectors": dict(selectors),
                "variant_index": variant_index,
            }

    resulting_support = _support_counts(requirements, current)
    for key, value in resulting_support.items():
        if value < target:
            raise ValueError(
                f"development stress scheduler failed support contract for {key}: {value} < {target}"
            )
    return {
        **empty,
        "resulting_support": resulting_support,
        "pairwise_unique_after": _pairwise_unique(current),
        "reserved_scenes": len(slots),
        "slots": slots,
    }


def apply_development_stress_scene(
    scene: dict,
    domains: dict,
    axes: dict,
    spec: dict,
) -> dict:
    selectors = spec.get("selectors")
    if not isinstance(selectors, dict):
        raise ValueError("development stress spec is missing selectors")
    variant_index = int(spec.get("variant_index", 0))
    result = dict(scene)

    distance_name = selectors.get("distance_bin")
    if distance_name is not None and _distance_bin(float(result["distance_m"])) != distance_name:
        if distance_name not in DISTANCE_POINTS_M:
            raise ValueError(f"unsupported robustness distance bin: {distance_name}")
        distance_m = float(DISTANCE_POINTS_M[distance_name])
        if not any(
            math.isclose(distance_m, float(value), rel_tol=0.0, abs_tol=1.0e-9)
            for value in axes["distance_m"]
        ):
            raise ValueError(f"stress distance {distance_name} is absent from evaluation axes")
        band = _distance_band_for_value(domains, distance_m)
        result.update(
            {
                "distance_m": distance_m,
                "distance_band": band,
                "room_id": f"sim-{band}-eval",
                "rir_id": f"sim-{band}-eval",
            }
        )

    azimuth_name = selectors.get("azimuth")
    if azimuth_name is not None and _azimuth_band(float(result["azimuth_deg"])) != azimuth_name:
        candidates = [
            float(value)
            for value in axes["azimuth_deg"]
            if _azimuth_band(float(value)) == azimuth_name
        ]
        if not candidates:
            raise ValueError(f"stress azimuth band {azimuth_name} is absent from evaluation axes")
        result["azimuth_deg"] = candidates[variant_index % len(candidates)]

    snr_name = selectors.get("snr")
    if snr_name is not None and _snr_band(float(result["snr_db"])) != snr_name:
        snr_bands = [str(value) for value in axes["snr_bands"]]
        if snr_name not in snr_bands:
            raise ValueError(f"stress SNR band {snr_name} is absent from evaluation axes")
        result["snr_db"] = float(axes["snr_db"][snr_bands.index(snr_name)])

    return result
