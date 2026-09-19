from __future__ import annotations

import math

SIGNAL_MODES = {"legacy-counts-v1", "normalized-rates-v1"}


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def validate_controller_config(raw: dict) -> str:
    if not isinstance(raw, dict):
        raise ValueError("loss_controller must be an object")
    for prefix in ("positive_example_weight", "ordered_token_loss_weight"):
        initial = finite(raw.get(f"{prefix}_initial"), f"{prefix}_initial")
        low = finite(raw.get(f"{prefix}_min"), f"{prefix}_min")
        high = finite(raw.get(f"{prefix}_max"), f"{prefix}_max")
        step = finite(raw.get(f"{prefix}_step"), f"{prefix}_step")
        if not 0.0 < low <= initial <= high or step <= 0.0:
            raise ValueError(f"{prefix} controller bounds are invalid")
    mode = str(raw.get("signal_mode", "legacy-counts-v1"))
    if mode not in SIGNAL_MODES:
        raise ValueError(f"unsupported loss controller signal_mode: {mode}")
    if mode == "normalized-rates-v1":
        initial = finite(raw.get("wake_example_weight_initial"), "wake_example_weight_initial")
        low = finite(raw.get("wake_example_weight_min"), "wake_example_weight_min")
        high = finite(raw.get("wake_example_weight_max"), "wake_example_weight_max")
        step = finite(raw.get("wake_example_weight_step"), "wake_example_weight_step")
        if not 0.0 < low <= initial <= high or step <= 0.0:
            raise ValueError("wake_example_weight controller bounds are invalid")
        frr_scale = finite(raw.get("normalization_frr"), "normalization_frr")
        far_scale = finite(
            raw.get("normalization_far_per_hour"),
            "normalization_far_per_hour",
        )
        deadband = finite(raw.get("pressure_deadband", 0.0), "pressure_deadband")
        if frr_scale <= 0.0 or far_scale <= 0.0:
            raise ValueError("normalized controller scales must be > 0")
        if not 0.0 <= deadband < 1.0:
            raise ValueError("normalized controller pressure_deadband must be in [0,1)")
    return mode


def initial_controller(policy: dict) -> dict:
    raw = policy["loss_controller"]
    mode = validate_controller_config(raw)
    return {
        "positive_example_weight": float(raw["positive_example_weight_initial"]),
        "wake_example_weight": float(raw.get("wake_example_weight_initial", 1.0)),
        "ordered_token_loss_weight": float(raw["ordered_token_loss_weight_initial"]),
        "failure_replay_repeat": 0,
        "controller_signal_mode": mode,
        "controller_severity": 0.0,
    }


def next_controller(
    policy: dict,
    current: dict,
    false_rejects: int,
    false_accepts: int,
    *,
    frr: float | None = None,
    far_per_hour: float | None = None,
    latch_after_failure: bool = False,
) -> dict:
    raw = policy["loss_controller"]
    mode = validate_controller_config(raw)
    positive = float(current["positive_example_weight"])
    wake = float(current.get("wake_example_weight", raw.get("wake_example_weight_initial", 1.0)))
    ordered = float(current["ordered_token_loss_weight"])
    p_step = float(raw["positive_example_weight_step"])
    w_step = float(raw.get("wake_example_weight_step", 0.0))
    o_step = float(raw["ordered_token_loss_weight_step"])

    severity = 0.0
    frr_pressure: float | None = None
    far_pressure: float | None = None
    decision = "balanced"
    if mode == "normalized-rates-v1":
        if frr is None or far_per_hour is None:
            raise ValueError("normalized controller requires FRR and FAR/hour signals")
        frr_value = finite(frr, "controller.frr")
        far_value = finite(far_per_hour, "controller.far_per_hour")
        if not 0.0 <= frr_value <= 1.0 or far_value < 0.0:
            raise ValueError("controller FRR/FAR signals are outside valid ranges")
        frr_scale = float(raw["normalization_frr"])
        far_scale = float(raw["normalization_far_per_hour"])
        deadband = float(raw.get("pressure_deadband", 0.0))
        frr_pressure = frr_value / frr_scale
        far_pressure = far_value / far_scale
        severity = max(frr_pressure, far_pressure)
        if frr_pressure > far_pressure * (1.0 + deadband):
            wake += w_step
            ordered -= 0.5 * o_step
            decision = "recall"
        elif far_pressure > frr_pressure * (1.0 + deadband):
            wake -= w_step
            ordered += o_step
            decision = "precision"
    else:
        if false_rejects > false_accepts:
            positive += p_step
            ordered -= 0.5 * o_step
            decision = "recall"
        elif false_accepts > false_rejects:
            positive -= p_step
            ordered += o_step
            decision = "precision"
        elif false_accepts > 0:
            ordered += 0.5 * o_step
        severity = float(false_rejects + false_accepts)

    positive = clamp(
        positive,
        float(raw["positive_example_weight_min"]),
        float(raw["positive_example_weight_max"]),
    )
    if mode == "normalized-rates-v1":
        wake = clamp(
            wake,
            float(raw["wake_example_weight_min"]),
            float(raw["wake_example_weight_max"]),
        )
    ordered = clamp(
        ordered,
        float(raw["ordered_token_loss_weight_min"]),
        float(raw["ordered_token_loss_weight_max"]),
    )

    if mode == "normalized-rates-v1":
        if severity <= 1.0:
            repeat = 0
        elif severity <= 2.0:
            repeat = 1
        elif severity <= 4.0:
            repeat = 2
        else:
            repeat = int(policy["failure_replay_repeat_max"])
    else:
        failures = int(false_rejects) + int(false_accepts)
        previous_repeat = int(current.get("failure_replay_repeat", 0))
        if failures <= 0:
            repeat = previous_repeat if latch_after_failure else 0
        elif failures <= 4:
            repeat = 1
        elif failures <= 16:
            repeat = 2
        else:
            repeat = int(policy["failure_replay_repeat_max"])
    repeat = min(repeat, int(policy["failure_replay_repeat_max"]))

    return {
        "positive_example_weight": positive,
        "wake_example_weight": wake,
        "ordered_token_loss_weight": ordered,
        "failure_replay_repeat": repeat,
        "controller_signal_mode": mode,
        "controller_severity": severity,
        "controller_decision": decision,
        "frr_pressure": frr_pressure,
        "far_pressure": far_pressure,
    }
