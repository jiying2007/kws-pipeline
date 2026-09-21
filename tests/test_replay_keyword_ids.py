#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from synthetic_audio import (  # noqa: E402
    command_tts_surface_forms,
    command_tts_text,
    command_tts_timeout_seconds,
    keyword_render_context,
    token_carriers,
)
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

    surface_keywords = [
        {
            "id": 1,
            "text": "你好小窝",
            "tokens": ["ni3", "hao3", "xiao3", "wo1"],
            "token_ids": [1, 2, 3, 4],
        },
        {
            "id": 2,
            "text": "小窝小窝",
            "tokens": ["xiao3", "wo1", "xiao3", "wo1"],
            "token_ids": [3, 4, 3, 4],
        },
    ]
    forms = command_tts_surface_forms(surface_keywords, {"backend": "command"})
    assert forms == {
        "ni3": "你",
        "hao3": "好",
        "xiao3": "小",
        "wo1": "窝",
    }
    assert command_tts_text(
        ["hao3", "ni3", "xiao3", "wo1"],
        forms,
    ) == "好你小窝"
    try:
        command_tts_surface_forms(
            [
                {"id": 1, "text": "甲", "tokens": ["a"], "token_ids": [1]},
                {"id": 2, "text": "乙", "tokens": ["a"], "token_ids": [1]},
            ],
            {"backend": "command"},
        )
    except ValueError as exc:
        assert "ambiguous surface forms" in str(exc)
    else:
        raise AssertionError("ambiguous command-TTS surface form was accepted")
    explicit = command_tts_surface_forms(
        [
            {"id": 1, "text": "甲", "tokens": ["a"], "token_ids": [1]},
            {"id": 2, "text": "乙", "tokens": ["a"], "token_ids": [1]},
        ],
        {"backend": "command", "token_surface_forms": {"a": "啊"}},
    )
    assert explicit == {"a": "啊"}

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

    three_keywords = [
        {"id": 0, "text": "zero", "tokens": ["a"], "token_ids": [1]},
        {"id": 1, "text": "one", "tokens": ["b"], "token_ids": [2]},
        {"id": 2, "text": "two", "tokens": ["c"], "token_ids": [3]},
    ]
    explicit = [
        {
            "keyword_id": 0,
            "examples": 4,
            "focus": "adaptive",
            "fallback": {"distance_bin": "5m"},
        },
        {
            "keyword_id": 1,
            "examples": 4,
            "focus": "adaptive",
            "fallback": {"distance_bin": "5m"},
        },
    ]
    no_fill = normalize_positive_stress_replay(
        explicit,
        keywords=three_keywords,
    )
    assert [row["keyword_id"] for row in no_fill] == [0, 1]
    filled = normalize_positive_stress_replay(
        explicit,
        keywords=three_keywords,
        auto_fill_missing=True,
        auto_examples=6,
        auto_fallback={
            "distance_bin": "5m",
            "azimuth": "rear",
            "snr": "critical",
        },
    )
    assert [row["keyword_id"] for row in filled] == [0, 1, 2]
    assert filled[0].get("auto_filled") is None
    assert filled[1].get("auto_filled") is None
    assert filled[2]["auto_filled"] is True
    assert filled[2]["examples"] == 6
    assert filled[2]["focus"] == "adaptive"

    assert command_tts_timeout_seconds({"backend": "command"}) == 120
    assert command_tts_timeout_seconds(
        {"backend": "command", "timeout_seconds": 45}
    ) == 45
    for invalid_timeout in (0, 301, True, 1.5):
        try:
            command_tts_timeout_seconds(
                {"backend": "command", "timeout_seconds": invalid_timeout}
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid command TTS timeout accepted: {invalid_timeout!r}"
            )

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
        {
            "backend": "command",
            "token_surface_forms": {
                f"t{index}": f"wake{index}" for index in range(25)
            },
        },
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
