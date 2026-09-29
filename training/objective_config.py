from __future__ import annotations

import math

from objective_contract import (
    CTC_OBJECTIVE_POLICIES,
    CTC_OBJECTIVE_POLICY_DEFAULT,
    ORDERED_TOKEN_SCOPE_DEFAULT,
    ORDERED_TOKEN_SCOPES,
    PATH_PURITY_LOSS_WEIGHT_DEFAULT,
    PATH_PURITY_MARGIN_DEFAULT,
    PATH_PURITY_MARGIN_MAX,
    SEQUENCE_MARGIN_NEGATIVE_POLICIES,
    SEQUENCE_MARGIN_NEGATIVE_POLICY_DEFAULT,
    SEQUENCE_MARGIN_POSITIVE_POLICIES,
    SEQUENCE_MARGIN_POSITIVE_POLICY_DEFAULT,
)


def ctc_objective_policy_setting(train: dict) -> tuple[str, bool]:
    if not isinstance(train, dict):
        raise ValueError("train config must be an object")
    configured = "ctc_objective_policy" in train
    policy = str(train.get("ctc_objective_policy", CTC_OBJECTIVE_POLICY_DEFAULT))
    if policy not in CTC_OBJECTIVE_POLICIES:
        raise ValueError(
            "train.ctc_objective_policy must be one of "
            + ", ".join(sorted(CTC_OBJECTIVE_POLICIES))
        )
    return policy, configured


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


def sequence_margin_negative_policy_setting(train: dict) -> tuple[str, bool]:
    if not isinstance(train, dict):
        raise ValueError("train config must be an object")
    configured = "sequence_margin_negative_policy" in train
    policy = str(
        train.get(
            "sequence_margin_negative_policy",
            SEQUENCE_MARGIN_NEGATIVE_POLICY_DEFAULT,
        )
    )
    if policy not in SEQUENCE_MARGIN_NEGATIVE_POLICIES:
        raise ValueError(
            "train.sequence_margin_negative_policy must be one of "
            + ", ".join(sorted(SEQUENCE_MARGIN_NEGATIVE_POLICIES))
        )
    return policy, configured


def sequence_margin_positive_policy_setting(train: dict) -> tuple[str, bool]:
    if not isinstance(train, dict):
        raise ValueError("train config must be an object")
    configured = "sequence_margin_positive_policy" in train
    policy = str(
        train.get(
            "sequence_margin_positive_policy",
            SEQUENCE_MARGIN_POSITIVE_POLICY_DEFAULT,
        )
    )
    if policy not in SEQUENCE_MARGIN_POSITIVE_POLICIES:
        raise ValueError(
            "train.sequence_margin_positive_policy must be one of "
            + ", ".join(sorted(SEQUENCE_MARGIN_POSITIVE_POLICIES))
        )
    return policy, configured


AUXILIARY_LOSS_WEIGHT_NAMES = (
    "ordered_token_loss_weight",
    "keyword_sequence_margin_loss_weight",
    "prefix_completion_loss_weight",
    "recurrent_release_loss_weight",
    "suffix_root_suppression_loss_weight",
)

LEGACY_AUXILIARY_LOSS_WEIGHT_NAMES_V1 = (
    "ordered_token_loss_weight",
    "keyword_sequence_margin_loss_weight",
    "prefix_completion_loss_weight",
    "recurrent_release_loss_weight",
)


def ctc_vad_align_setting(train: dict) -> tuple[bool, bool]:
    if not isinstance(train, dict):
        raise ValueError("train config must be an object")
    configured = "ctc_vad_align" in train
    value = train.get("ctc_vad_align", False)
    if not isinstance(value, bool):
        raise ValueError("train.ctc_vad_align must be boolean")
    return value, configured


def auxiliary_loss_weights(train: dict) -> dict[str, float]:
    """Return only explicitly supplied controls, retaining an explicit zero."""
    if not isinstance(train, dict):
        raise ValueError("train config must be an object")
    result: dict[str, float] = {}
    for name in AUXILIARY_LOSS_WEIGHT_NAMES:
        if name not in train:
            continue
        raw = train[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"train.{name} must be numeric, not boolean/text")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"train.{name} must be finite and >= 0")
        result[name] = value
    return result


def verify_auxiliary_loss_readback(train: dict, recorded: dict) -> dict[str, float]:
    """Verify resolved trainer controls against the independent effective config."""
    if not isinstance(recorded, dict):
        raise ValueError("auxiliary loss readback must be an object")
    names = set(recorded)
    full = set(AUXILIARY_LOSS_WEIGHT_NAMES)
    legacy = set(LEGACY_AUXILIARY_LOSS_WEIGHT_NAMES_V1)
    if names == legacy:
        normalized_recorded = dict(recorded)
        normalized_recorded["suffix_root_suppression_loss_weight"] = 0.0
    elif names == full:
        normalized_recorded = dict(recorded)
    else:
        raise ValueError("auxiliary loss readback has unsupported fields")
    actual = auxiliary_loss_weights(normalized_recorded)
    for name, expected in auxiliary_loss_weights(train).items():
        if actual[name] != expected:
            raise ValueError(f"auxiliary loss readback mismatch: {name}")
    return actual


def optional_objective_cli_args(train: dict) -> list[str]:
    args: list[str] = []

    ctc_policy, ctc_policy_configured = ctc_objective_policy_setting(train)
    if ctc_policy_configured:
        args.extend(["--ctc-objective-policy", ctc_policy])
    for name, value in auxiliary_loss_weights(train).items():
        args.extend(["--" + name.replace("_", "-"), str(value)])

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

    negative_policy, negative_policy_configured = sequence_margin_negative_policy_setting(train)
    if negative_policy_configured:
        args.extend(["--sequence-margin-negative-policy", negative_policy])

    positive_policy, positive_policy_configured = sequence_margin_positive_policy_setting(train)
    if positive_policy_configured:
        args.extend(["--sequence-margin-positive-policy", positive_policy])

    ctc_vad_align, ctc_vad_configured = ctc_vad_align_setting(train)
    if ctc_vad_configured and ctc_vad_align:
        args.append("--ctc-vad-align")
    return args
