#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from synthetic_audio import keyword_render_context, token_carriers  # noqa: E402
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

    from development_failure_replay import select_failure_specs
    from adversarial_refinement import _repeat_focus_rows

    specs = [
        {
            "focus_keyword_ids": [0],
            "source_keyword_id": 0,
        },
        {
            "focus_keyword_ids": [0],
            "source_keyword_id": 0,
        },
    ]
    selected_specs = select_failure_specs(
        specs,
        max_unique=10,
        max_per_keyword=1,
    )
    assert len(selected_specs) == 1

    expanded: list[tuple[int, ...]] = []
    _repeat_focus_rows(
        expanded,
        [0],
        2,
        label="keyword-zero",
    )
    assert expanded == [(0,), (0,)]

    wide_keywords = [
        {
            "id": index,
            "text": f"wake-{index}",
            "tokens": [f"t{index}"],
            "token_ids": [index + 1],
        }
        for index in range(25)
    ]
    active, command_carriers = keyword_render_context(
        wide_keywords,
        32,
        {"backend": "command"},
    )
    assert len(active) == 25
    assert command_carriers == {}
    try:
        token_carriers(wide_keywords, 32)
    except ValueError as exc:
        assert "tone backend supports at most 24" in str(exc)
    else:
        raise AssertionError("tone carrier limit unexpectedly disappeared")

    print("replay keyword id contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
