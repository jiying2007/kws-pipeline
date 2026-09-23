from __future__ import annotations

import math

from objective_contract import (
    PATH_PURITY_LOSS_WEIGHT_DEFAULT,
    PATH_PURITY_MARGIN_DEFAULT,
    PATH_PURITY_MARGIN_MAX,
)


def path_purity_settings(train: dict) -> tuple[float, float, bool]:
    if not isinstance(train, dict):
        raise ValueError("train config must be an object")
    configured = (
        "path_purity_loss_weight" in train or "path_purity_margin" in train
    )
    weight = float(
        train.get("path_purity_loss_weight", PATH_PURITY_LOSS_WEIGHT_DEFAULT)
    )
    margin = float(train.get("path_purity_margin", PATH_PURITY_MARGIN_DEFAULT))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("train.path_purity_loss_weight must be finite and >= 0")
    if not math.isfinite(margin) or not 0.0 <= margin <= PATH_PURITY_MARGIN_MAX:
        raise ValueError(
            f"train.path_purity_margin must be finite and in [0,{PATH_PURITY_MARGIN_MAX}]"
        )
    return weight, margin, configured


def ordered_token_exact_wake_only_setting(train: dict) -> tuple[bool, bool]:
    if not isinstance(train, dict):
        raise ValueError("train config must be an object")
    configured = "ordered_token_exact_wake_only" in train
    raw = train.get("ordered_token_exact_wake_only", False)
    if not isinstance(raw, bool):
        raise ValueError("train.ordered_token_exact_wake_only must be boolean")
    return raw, configured


def optional_objective_cli_args(train: dict) -> list[str]:
    weight, margin, path_purity_configured = path_purity_settings(train)
    exact_wake_only, ordered_scope_configured = ordered_token_exact_wake_only_setting(
        train
    )
    result: list[str] = []
    if path_purity_configured:
        result.extend(
            [
                "--path-purity-loss-weight",
                str(weight),
                "--path-purity-margin",
                str(margin),
            ]
        )
    if ordered_scope_configured and exact_wake_only:
        result.append("--ordered-token-exact-wake-only")
    return result
