#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from hard_negative_replay import (  # noqa: E402
    normalize_hard_negative_replay,
    normalize_positive_stress_replay,
)


def main() -> int:
    token_map = {"a": 1, "b": 2, "c": 3}
    hard = normalize_hard_negative_replay(
        [
            {
                "tokens": ["a", "c"],
                "examples": 1,
                "focus_keyword_id": 0,
            }
        ],
        active_tokens=["a", "b", "c"],
        forbidden=[["a", "b"]],
        token_map=token_map,
        keyword_ids={0},
    )
    assert hard[0]["focus_keyword_id"] == 0

    keywords = [
        {
            "id": 0,
            "text": "wake-zero",
            "tokens": ["a", "b"],
            "token_ids": [1, 2],
        }
    ]
    positive = normalize_positive_stress_replay(
        [
            {
                "keyword_id": 0,
                "examples": 1,
                "focus": "adaptive",
                "fallback": {},
            }
        ],
        keywords=keywords,
    )
    assert positive[0]["keyword_id"] == 0

    try:
        normalize_positive_stress_replay(
            [{"examples": 1, "focus": "adaptive", "fallback": {}}],
            keywords=keywords,
        )
    except ValueError as exc:
        assert "keyword_id is required" in str(exc)
    else:
        raise AssertionError("missing positive-stress keyword_id defaulted to zero")

    try:
        normalize_hard_negative_replay(
            [
                {
                    "tokens": ["a", "c"],
                    "examples": 1,
                    "focus_keyword_id": 0x1_0000_0000,
                }
            ],
            active_tokens=["a", "b", "c"],
            forbidden=[["a", "b"]],
            token_map=token_map,
            keyword_ids=None,
        )
    except ValueError as exc:
        assert "must fit uint32" in str(exc)
    else:
        raise AssertionError("out-of-range replay keyword id was accepted")

    print("replay keyword id contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
