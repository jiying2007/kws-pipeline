#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

DIMENSION_PREFIXES = {
    "distance": "distance:",
    "azimuth": "azimuth:",
    "snr": "snr:",
    "rt60": "rt60:",
    "noise": "noise:",
    "playback": "playback:",
    "composite": "composite:",
    "distance_azimuth": "distance_azimuth:",
    "distance_snr": "distance_snr:",
    "azimuth_snr": "azimuth_snr:",
    "distance_azimuth_snr": "distance_azimuth_snr:",
}
INTERACTION_DIMENSIONS = {
    "distance_azimuth",
    "distance_snr",
    "azimuth_snr",
    "distance_azimuth_snr",
}
DISTANCE_BIN_TO_BAND = {
    "0.5m": "near",
    "1m": "near",
    "2m": "mid",
    "3m": "far",
    "5m": "far",
}
HARDNESS_EPSILON = 1.0e-12


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def metric_hardness(item: dict, label: str) -> float:
    frr = finite(item.get("frr", 0.0), f"{label}.frr")
    far = finite(item.get("far_per_hour", 0.0), f"{label}.far")
    latency = finite(item.get("p95_post_end_latency_ms", 0.0), f"{label}.latency")
    # Acoustic curriculum is driven by recognition errors, not harmless latency
    # jitter. Otherwise a completely zero-error calibration pass still has a
    # non-zero maximum, and relative normalization amplifies the slowest normal
    # slice to full curriculum strength. Keep latency only as a secondary signal
    # among slices that already contain FR/FAR evidence; candidate selection and
    # the strict latency gate remain responsible for latency itself.
    error_hardness = frr * 20.0 + min(far, 20.0) * 0.05
    if error_hardness <= HARDNESS_EPSILON:
        return 0.0
    return error_hardness + min(latency, 1500.0) / 3000.0


def previous_dimension(previous: dict | None, name: str) -> dict[str, float]:
    if not isinstance(previous, dict):
        return {}
    dimensions = previous.get("dimension_weights", previous)
    if not isinstance(dimensions, dict):
        return {}
    value = dimensions.get(name, {})
    if not isinstance(value, dict):
        return {}
    return {str(key): finite(weight, f"previous.{name}.{key}") for key, weight in value.items()}


def previous_keyword_dimension(
    previous: dict | None, keyword_id: str, name: str
) -> dict[str, float]:
    if not isinstance(previous, dict):
        return {}
    keyword_dimensions = previous.get("keyword_dimension_weights", {})
    if not isinstance(keyword_dimensions, dict):
        return {}
    keyword = keyword_dimensions.get(str(keyword_id), {})
    if not isinstance(keyword, dict):
        return {}
    value = keyword.get(name, {})
    if not isinstance(value, dict):
        return {}
    return {
        str(key): finite(weight, f"previous.keyword.{keyword_id}.{name}.{key}")
        for key, weight in value.items()
    }


def _weights_from_hardness(
    hardness: dict[str, float],
    *,
    base: dict[str, float],
    strength: float,
    max_weight: float,
    playback_floor: bool,
) -> dict[str, float]:
    maximum = max(hardness.values()) if hardness else 0.0
    weights: dict[str, float] = {}
    for value_name, score in hardness.items():
        relative = score / maximum if maximum > HARDNESS_EPSILON else 0.0
        target = 1.0 + strength * relative
        old = base.get(value_name, 1.0)
        weights[value_name] = min(
            max_weight,
            max(1.0, 0.55 * old + 0.45 * target),
        )
    if playback_floor and {"playback", "no-playback"}.issubset(weights):
        weights["playback"] = max(weights["playback"], weights["no-playback"])
    return weights


def _interaction_fields(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in value.split("|"):
        if "=" not in field:
            continue
        key, item = field.split("=", 1)
        if key and item:
            result[key] = item
    return result


def _project_interactions(
    dimension_hardness: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Project joint-slice hardness into marginals consumed by scene sampling.

    Exact joint slices remain intact for adaptive positive-stress replay. The
    projection uses max, not sum, so one hard triple can raise the relevant
    distance/azimuth/SNR marginals without multiplying the same evidence several
    times or recreating the rejected hard far/rear floors.
    """
    projected = {
        dimension: dict(values)
        for dimension, values in dimension_hardness.items()
    }
    for dimension in INTERACTION_DIMENSIONS:
        for value, score in dimension_hardness.get(dimension, {}).items():
            fields = _interaction_fields(value)
            distance_bin = fields.get("distance_bin")
            if distance_bin is not None:
                band = DISTANCE_BIN_TO_BAND.get(distance_bin)
                if band is not None:
                    target = projected.setdefault("distance", {})
                    target[band] = max(target.get(band, 0.0), score)
            azimuth = fields.get("azimuth")
            if azimuth is not None:
                target = projected.setdefault("azimuth", {})
                target[azimuth] = max(target.get(azimuth, 0.0), score)
            snr = fields.get("snr")
            if snr is not None:
                target = projected.setdefault("snr", {})
                target[snr] = max(target.get(snr, 0.0), score)
    return projected


def _dimension_views(
    domains: dict,
    *,
    base_lookup,
    strength: float,
    max_weight: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], list[tuple[float, str]]]:
    raw_hardness: dict[str, dict[str, float]] = {}
    ranked_domains: list[tuple[float, str]] = []
    for dimension, prefix in DIMENSION_PREFIXES.items():
        hardness: dict[str, float] = {}
        for key, item in domains.items():
            if not key.startswith(prefix) or not isinstance(item, dict):
                continue
            value_name = key[len(prefix) :]
            score = metric_hardness(item, key)
            hardness[value_name] = score
            ranked_domains.append((score, key))
        if hardness:
            raw_hardness[dimension] = hardness

    dimension_hardness = _project_interactions(raw_hardness)
    dimension_weights: dict[str, dict[str, float]] = {}
    for dimension, hardness in dimension_hardness.items():
        dimension_weights[dimension] = _weights_from_hardness(
            hardness,
            base=base_lookup(dimension),
            strength=strength,
            max_weight=max_weight,
            playback_floor=dimension == "playback",
        )
    ranked_domains.sort(key=lambda item: (-item[0], item[1]))
    return dimension_hardness, dimension_weights, ranked_domains


def update_curriculum(
    domain_metrics: dict,
    *,
    previous: dict | None = None,
    strength: float = 2.0,
    max_weight: float = 6.0,
) -> dict:
    if strength < 0.0 or max_weight < 1.0:
        raise ValueError("curriculum strength/max_weight is invalid")
    domains = domain_metrics.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("domain metrics is missing domains")

    dimension_hardness, dimension_weights, ranked_domains = _dimension_views(
        domains,
        base_lookup=lambda dimension: previous_dimension(previous, dimension),
        strength=strength,
        max_weight=max_weight,
    )

    keyword_dimension_hardness: dict[str, dict] = {}
    keyword_dimension_weights: dict[str, dict] = {}
    keyword_worst_domains: dict[str, list[dict]] = {}
    keyword_domains = domain_metrics.get("keyword_domains", {})
    if keyword_domains is not None and not isinstance(keyword_domains, dict):
        raise ValueError("keyword_domains must be an object")
    for keyword_id, keyword_metrics in sorted(keyword_domains.items()):
        if not isinstance(keyword_metrics, dict):
            raise ValueError("keyword domain metric must be an object")
        keyword_domain_map = keyword_metrics.get("domains", {})
        if not isinstance(keyword_domain_map, dict):
            raise ValueError("keyword domain metric is missing domains")
        hardness, weights, ranked = _dimension_views(
            keyword_domain_map,
            base_lookup=lambda dimension, kid=str(keyword_id): previous_keyword_dimension(
                previous, kid, dimension
            ),
            strength=strength,
            max_weight=max_weight,
        )
        keyword_dimension_hardness[str(keyword_id)] = hardness
        keyword_dimension_weights[str(keyword_id)] = weights
        interaction_ranked = [
            (score, key)
            for score, key in ranked
            if score > HARDNESS_EPSILON
            and key.split(":", 1)[0] in INTERACTION_DIMENSIONS
        ]
        keyword_worst_domains[str(keyword_id)] = [
            {"domain": key, "hardness": score}
            for score, key in interaction_ranked[:12]
        ]

    return {
        "schema_version": 2,
        "hardness_policy": "recognition-error-gated-latency-v1",
        "dimension_weights": dimension_weights,
        "dimension_hardness": dimension_hardness,
        "worst_domains": [
            {"domain": key, "hardness": score}
            for score, key in ranked_domains
            if score > HARDNESS_EPSILON
        ][:12],
        "keyword_dimension_weights": keyword_dimension_weights,
        "keyword_dimension_hardness": keyword_dimension_hardness,
        "keyword_worst_domains": keyword_worst_domains,
        "interaction_projection": "max-to-distance-azimuth-snr-marginals-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--previous", type=pathlib.Path)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=6.0)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    previous = None
    if args.previous:
        value = json.loads(args.previous.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("previous curriculum must be a JSON object")
        previous = value
    result = update_curriculum(
        metrics,
        previous=previous,
        strength=finite(args.strength, "strength"),
        max_weight=finite(args.max_weight, "max_weight"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
