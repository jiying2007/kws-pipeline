# Melo6 saved human adjudication

This small, offline supplement joins the unchanged machine comparison with
explicit, hash-bound human transcript revisions and native/derived audio
bindings. It uses Python 3.10+ and the standard library only. It performs no
playback, audio transformation, inference, download, or training. It does not
import or modify the existing saved-result checker or model runners.

The current review has five full text labels matching the planned strings and
one partial label. M6 is `[首字听不清]天天气很好`, with full text null, first
character UNKNOWN and prompt match UNKNOWN. Its earlier `明天天气很好` label
is superseded history only. The highest explicit revision wins, including
when a partial correction replaces a full transcript. Saved ASR agreement
cannot fill an unknown character or revive the older label.

## Reproduce

Let `CHECKER` be this directory in the source supplement and `REVIEW` be
`research/2026-10-06-melo6-human-adjudication` in the data supplement. Set
`COMPARISON` to the data checkout's unchanged
`research/2026-10-06-melo6-blind-asr-results/comparison-result.json`.

```sh
python -B "$CHECKER/adjudicate_saved.py" \
  --comparison "$COMPARISON" \
  --labels "$REVIEW/human-labels.json" \
  --bindings "$REVIEW/audio-bindings.json" \
  --vocabulary "$REVIEW/vocabulary-binding.json" \
  --out "$NEW_OUTPUT"
python -B "$CHECKER/test_adjudicate_saved.py" \
  --comparison "$COMPARISON" --review-root "$REVIEW" -v
```

`NEW_OUTPUT` must not exist. The pure function `adjudicate_saved` takes four
byte strings and returns a new result. All four inputs have pinned SHA256
identities; byte changes, alias swaps, wrong audio hashes, and rollback to
the older full M6 snapshot fail closed. A later human review must be preserved
in a new reviewed snapshot, with corresponding explicit bindings. Nine tests
cover exact reproduction, revision precedence and rollback, ASR/human text
coexistence, binding rejection, OOV/partial handling, unchanged inputs,
exclusive output creation, and the remaining research gaps.

## Interpretation

The original machine gate remains FAIL at 2/6 dual exact-intent matches.
M1/M2/M4 now have human text matching intent alongside saved recognizer
disagreement. This adds human transcript evidence without changing the
preregistered machine result. The reviewer had seen ASR hypotheses, so this
is not a fully blinded review. Human listening concerned native 44.1-kHz
mono float32 WAVs; the ASRs decoded bound 16-kHz PCM16 derivatives. The
latter were not independently heard. Byte linkage does not prove acoustic
equivalence or independently establish recognizer errors on those derivatives.
No confidence, tone analysis, or acoustic tail completeness is inferred.

M1–M5 are textually representable under the bound research alphabet
`<blank>, 你, 好, 小, 窝, 屋`. That is a clip-level text property only.
M6 lacks a full transcript and its observed suffix also contains OOV
characters. OOV or partial speech is never encoded as a blank CTC target.
Every CTC target stays null and every training admission stays false.
M1/M2 provide word-level K1/K2 positive coverage. This increment leaves split
allocation UNASSIGNED and does not establish frozen split-integrity qualification
or a qualified positive development set. One identity group alone cannot fill
all isolated train/development/held-source partitions. Identity-held-out
evaluation, cleared voice rights, and KWS improvement remain unestablished.

Original comparison, triage, source freezes, and audio files remain unchanged.
Their earlier UNKNOWN human fields describe that historical snapshot; this
separate supplement records the subsequent limited review.
