from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_decoder_boundary_references import (  # noqa: E402
    FRAME_HOP_SAMPLES,
    ctc_viterbi_alignment,
    internal_split_from_alignment,
    paused_samples,
    select_boundary_sources,
    stitched_samples,
)


def row(
    *,
    voice: str,
    source: str,
    kind: str,
    target: list[int],
    keyword_id: int | None,
) -> dict:
    return {
        "kind": kind,
        "keyword_id": keyword_id,
        "target_ids": target,
        "wav_sha256": "a" * 64,
        "path": f"{source}.wav",
        "speech_like_provenance": {
            "voice_id": voice,
            "source_id": f"speech-like:calibration:slot:{source}",
        },
    }


def logits_for(labels: list[int], vocab: int) -> list[list[float]]:
    result: list[list[float]] = []
    for label in labels:
        values = [-8.0] * vocab
        values[label] = 8.0
        result.append(values)
    return result


def expect_failure(call, needle: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected failure containing {needle!r}")


def main() -> int:
    rows = [
        row(
            voice="voice-a",
            source="kw1-exact",
            kind="positive",
            target=[1, 2, 3, 4],
            keyword_id=1,
        ),
        row(
            voice="voice-a",
            source="kw1-pause",
            kind="positive",
            target=[1, 2, 3, 4],
            keyword_id=1,
        ),
        row(
            voice="voice-a",
            source="kw2-exact",
            kind="positive",
            target=[3, 4, 3, 4],
            keyword_id=2,
        ),
        row(
            voice="voice-a",
            source="neg-short-nihao",
            kind="negative",
            target=[1, 2],
            keyword_id=None,
        ),
        row(
            voice="voice-a",
            source="neg-short-xiaowo",
            kind="negative",
            target=[3, 4],
            keyword_id=None,
        ),
    ]
    selected = select_boundary_sources(rows)
    assert [(item["keyword_id"], item["target_ids"]) for item in selected] == [
        (1, (1, 2, 3, 4)),
        (2, (3, 4, 3, 4)),
    ]
    assert selected[0]["positive"]["speech_like_provenance"]["source_id"].endswith(
        ":kw1-exact"
    )
    assert selected[0]["left"]["target_ids"] == [1, 2]
    assert selected[0]["right"]["target_ids"] == [3, 4]
    assert selected[1]["left"] is selected[1]["right"]

    missing = [value for value in rows if value["target_ids"] != [1, 2]]
    expect_failure(
        lambda: select_boundary_sources(missing),
        "lacks standalone half-word negatives",
    )

    labels = [0, 1, 0, 2, 0, 3, 0, 4, 0]
    logits = logits_for(labels, vocab=5)
    states = ctc_viterbi_alignment(logits, (1, 2, 3, 4))
    assert len(states) == len(logits)
    for state in (1, 3, 5, 7):
        assert state in states

    alignment = internal_split_from_alignment(
        logits,
        (1, 2, 3, 4),
        event_start_sample=0,
        event_end_sample=3200,
        lead_samples=0,
    )
    assert alignment["split_samples"] == 1280
    assert alignment["split_samples"] % FRAME_HOP_SAMPLES == 0

    raw = list(range(4000))
    paused = paused_samples(
        raw,
        alignment["split_samples"],
        gap_samples=6400,
        lead_samples=320,
        tail_samples=320,
    )
    assert len(paused) == 320 + len(raw) + 6400 + 320
    assert paused[320 : 320 + 1280] == raw[:1280]
    assert paused[320 + 1280 : 320 + 1280 + 6400] == [0] * 6400
    assert paused[320 + 1280 + 6400 : 320 + 6400 + len(raw)] == raw[1280:]

    left = [1] * 1000
    right = [2] * 1200
    stitched, silence = stitched_samples(
        left,
        right,
        gap_samples=6400,
        lead_samples=320,
        tail_samples=320,
    )
    assert silence == 6680
    right_start = 320 + len(left) + silence
    assert right_start % FRAME_HOP_SAMPLES == 0
    assert stitched[right_start : right_start + len(right)] == right

    expect_failure(
        lambda: internal_split_from_alignment(
            logits,
            (1, 2, 3),
            event_start_sample=0,
            event_end_sample=3200,
            lead_samples=0,
        ),
        "even-length wake target",
    )

    print("test_decoder_boundary_references: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
