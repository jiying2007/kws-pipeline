from __future__ import annotations

import math

from path_purity import PATH_PURITY_MARGIN_DEFAULT, PATH_PURITY_MARGIN_MAX


PATH_PURITY_LOSS_WEIGHT_DEFAULT = 0.0


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


def optional_objective_cli_args(train: dict) -> list[str]:
    weight, margin, configured = path_purity_settings(train)
    if not configured:
        return []
    return [
        "--path-purity-loss-weight",
        str(weight),
        "--path-purity-margin",
        str(margin),
    ]
