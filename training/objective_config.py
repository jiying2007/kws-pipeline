from __future__ import annotations

import math

from objective_contract import (
    ORDERED_TOKEN_SCOPE_DEFAULT,
    ORDERED_TOKEN_SCOPES,
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


def ordered_token_scope_setting(train: dict) -> tuple[str, bool]:
    if not isinstance(train, dict):
        raise ValueError("train config must be an object")
    configured = "ordered_token_scope" in train
    scope = str(train.get("ordered_token_scope", ORDERED_TOKEN_SCOPE_DEFAULT))
    if scope not in ORDERED_TOKEN_SCOPES:
        raise ValueError(
            "train.ordered_token_scope must be one of "
            + ", ".join(sorted(ORDERED_TOKEN_SCOPES))
        )
    return scope, configured


def optional_objective_cli_args(train: dict) -> list[str]:
    args: list[str] = []

    weight, margin, path_purity_configured = path_purity_settings(train)
    if path_purity_configured:
        args.extend(
            [
                "--path-purity-loss-weight",
                str(weight),
                "--path-purity-margin",
                str(margin),
            ]
        )

    ordered_scope, ordered_scope_configured = ordered_token_scope_setting(train)
    if ordered_scope_configured:
        args.extend(["--ordered-token-scope", ordered_scope])
    return args
