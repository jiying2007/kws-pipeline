# Dataset-driven iteration lane

This is the fourth lane, alongside Research, Confirm and Qualification.

## Why it exists

The governed qualification gate asks for this:

```json
"domain_gates": {
  "max_frr": 0.0, "max_far_per_hour": 0.0,
  "max_p95_latency_ms": 800.0, "max_far_frr": 0.0
}
```

`max_far_per_hour: 0.0` is not a measurable target. Observing zero false accepts
over `T` hours does not establish a rate of zero; it establishes an upper bound
of roughly `3/T` per hour at 95% confidence. Measured on this repository's own
fixtures, **12 seconds of negative exposure produces an upper bound of about
900 false accepts per hour** -- reported by the governed gate as a clean `0.0`.

Asking for zero false rejects *and* zero false accepts at the same time also
removes the solution space: lowering the threshold satisfies recall and breaks
the false-accept rate, raising it does the reverse, and the only remaining
point -- reject everything -- still fails recall.

So this lane does not judge. It **measures**, in a form that can be compared.

## The two controlled variables

Every run declares exactly two things and which of them is under test:

| Variable | Identity | Meaning |
| --- | --- | --- |
| `dataset` | `dataset_id` | Content hash over audio bytes + expected events + domain bins |
| `model` | `model_id` | Hash over the model **and** the keyword pack (the pack carries the threshold, so it is part of the operating point) |

A run also records `code_sha`, `seed` and `protocol`.

## Rules

1. **One variable at a time.** If both `dataset_id` and `model_id` move, the
   delta has no interpretation and the comparison is refused.
2. **The declared variable must be the one that moved.** Declaring `dataset`
   and shipping a new model is refused.
3. **Same protocol or no comparison.** Numbers from different protocols are not
   the same measurement.
4. **Code changes confound.** If `code_sha` differs, re-baseline first.
5. **False accepts are compared on the upper bound, not the point estimate.**
   Zero over 30 seconds and zero over 30 hours are wildly different evidence.

## Interpreting the exposure number

| Negative exposure | 95% upper bound if zero false accepts |
| --- | --- |
| 12 s (0.0033 h) | ~900 / h |
| 10 min (0.167 h) | ~18 / h |
| 1 h | ~3 / h |
| 10 h | ~0.3 / h |

Treat the bound, not the point estimate, as the measurement. A dataset too small
to support the product's false-accept budget cannot demonstrate that budget --
that is a property of the dataset, not of the model.

## Running it

Dispatch `.github/workflows/dataset-driven-iteration.yml` with a dataset
(`references.jsonl`), a model, a keyword pack, and the variable under test.
Optionally point `baseline_card` at a previously recorded scorecard.

Locally the same thing is two commands:

```bash
python3 tools/run_dataset_iteration.py \
  --runner build/kws_wav --model model.kwm --keywords keywords.kwk \
  --references datasets/v3/references.jsonl --audio-root datasets/v3 \
  --work-dir build/di --output card.json --variable dataset

python3 tools/compare_dataset_iterations.py \
  --baseline evidence/dataset-iterations/<dataset>/<old>.json \
  --candidate card.json --far-budget 1.0
```

Exit codes for the comparison: `0` no regression, `1` regression,
`2` refused or unusable input.

## What this lane does not do

It does not qualify, freeze, or promote anything, and it consumes no protected
evidence. A verdict of `improved` means the measured operating point moved in
the right direction on this dataset under this protocol -- nothing more. Product
claims still need real-human and physical-target evidence.

## Recording

Scorecards are committed to `evidence/dataset-iterations/<dataset-id>/<run-id>.json`
so results outlive artifact retention and remain reviewable.
