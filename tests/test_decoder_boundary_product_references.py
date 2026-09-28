from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_decoder_boundary_product_references import (  # noqa: E402
    NEGATIVE_ROLE,
    POSITIVE_ROLE,
    positive_event,
    select_product_boundary_sources,
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
        "event_start_frame": 100,
        "event_end_frame": 1000,
        "speech_like_provenance": {
            "voice_id": voice,
            "source_id": f"speech-like:calibration:slot:{source}",
        },
    }


def main() -> int:
    assert POSITIVE_ROLE == "natural-full-phrase-pause-v1"
    assert NEGATIVE_ROLE == "cross-utterance-long-gap-v1"
    rows = [
        row(
            voice="voice-a",
            source="kw1-pause",
            kind="positive",
            target=[1, 2, 3, 4],
            keyword_id=1,
        ),
        row(
            voice="voice-a",
            source="kw1-exact",
            kind="positive",
            target=[1, 2, 3, 4],
            keyword_id=1,
        ),
        row(
            voice="voice-a",
            source="kw2-pause",
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
    selected = select_product_boundary_sources(rows)
    assert [(item["keyword_id"], item["target_ids"]) for item in selected] == [
        (1, (1, 2, 3, 4)),
        (2, (3, 4, 3, 4)),
    ]
    assert selected[0]["positive"]["speech_like_provenance"]["source_id"].endswith(
        ":kw1-pause"
    )
    assert not selected[0]["positive"]["speech_like_provenance"]["source_id"].endswith(
        ":kw1-exact"
    )
    assert selected[1]["left"] is selected[1]["right"]

    event = positive_event(rows[0], keyword_id=1, sample_count=1600)
    assert event["start_s"] == 100 / 16000
    assert event["end_s"] == 1000 / 16000
    assert event["match_not_before_s"] == event["end_s"]

    no_pause = [item for item in rows if "-pause" not in item["speech_like_provenance"]["source_id"]]
    try:
        select_product_boundary_sources(no_pause)
    except ValueError as exc:
        assert "no natural full-keyword pause" in str(exc)
    else:
        raise AssertionError("exact-only positives were accepted as natural pause authority")

    print("test_decoder_boundary_product_references: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
