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

Zero observed false rejects *and* zero observed false accepts can coexist on a
finite, separable dataset. Such a result does not establish zero population
risk or generalization to other speakers and conditions. Threshold changes often
trade recall against false accepts, but that tradeoff does not mathematically
rule out a zero-error sample. Report positive support, negative exposure and
confidence bounds; rejecting everything still fails recall when positives exist.

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

## FAR populations and scorecard compatibility

New scorecards use schema 3, default protocol `dataset-iteration-v3`, and the
fixed FAR identity policy `negative-only-recordings-iteration-approx95-v1`.
The policy is part of `run_id` even with a custom `--protocol` label.

`far_rate` uses only complete recordings with `expected: []`: its
`observed_false_accepts`, `point_estimate_per_hour` and upper bound all use that
population and `exposure.exposure_hours`. Missing negative metrics or zero
negative exposure are errors, not zero FAR. The upper-bound calculation is
unchanged: `2.9957 / T` for zero events, otherwise
`(N + 1.959964 * sqrt(N)) / T`, labelled as a normal approximation. It does not
switch to the scorer's separate exact-Poisson upper bound. Schema-2/3 comparison
checks that the point and bound exactly match this count/exposure calculation;
do not round stored values. A non-finite computed bound is rejected, including
when a finite positive exposure is too small to represent its bound.

`metrics.false_accepts` and `metrics.far_per_hour` remain the scorer's global
all-recording diagnostics, including unmatched detections in positive
recordings. The comparison's `deltas.far_per_hour` remains that global
diagnostic; its FAR budget and improvement decision use the negative-only
upper bound. This lane still does not qualify or promote a model.

## Effective-domain dataset identity

Schema 3 fixes `identity.dataset_identity_policy` to
`effective-domain-keys-v1`, including with a custom `--protocol`. Each dataset
hash entry contains `sorted(domain_keys(row))` from `eval/domain_metrics.py`,
the same helper used to group scoring slices. The dataset hash is over a tagged
envelope `{"dataset_identity_policy": "effective-domain-keys-v1", "entries": [...]}`.
The policy also participates in `run_id` through the identity block. Even
all-only datasets therefore receive new IDs; historical IDs are not rewritten.

Only a dictionary in nested `domain` contributes slices. Missing, non-dictionary
or empty `domain` means `all` only. Top-level distance fields, `scene`,
`domain_id`, and unknown nested fields do not contribute or fill sparse axes.
Named `distance_band` and numeric `distance_m` contribute independently.
All effective distance, azimuth band/point, SNR, RT60, noise, playback,
pairwise/triple and composite slices are bound without a second normalization.

This is effective-slice identity, not raw-metadata identity: within-bin changes,
equivalent azimuth wraps and non-null playback magnitude changes intentionally
retain an ID when their effective keys are identical. Numeric validation,
categorical `str()` labels (including null/container behavior), and playback
presence semantics remain those of the scoring helper. Raw input provenance
continues to be retained in `artifacts.references_sha256`. Audio bytes,
expectations, recording sorting and canonical audio-path aliases retain their
existing contracts. This change does not claim that every other reference
metadata, declared exposure or timing field is bound by `dataset_id`.

The current comparator supports schema-1/schema-1 historical mixed-population
FAR cards and schema-2/schema-2 fixed-FAR cards under their original contracts,
including matching custom protocols. Both are classified as legacy
top-level-distance dataset identity, not silently promoted. Schema-3/schema-3
requires the exact effective-domain policy and the same fixed negative-only FAR
policy as schema 2; all schema-2 FAR tuple and bound checks also apply to schema 3.

Mixed legacy/new domain policies, unknown/missing schema-3 policies, unexpected
domain-policy fields on schemas 1/2, and unknown or boolean schemas are refused
before interpreting moved dataset/model IDs, even with matching custom protocol
labels. Schema-1/schema-2 pairs still fail FAR-policy compatibility. Re-run both
sides from the original inputs under the same current code and policies; do not
relabel or re-hash historical scorecards. Keep archived cards unchanged. An old
comparator does not implement these checks: use the current comparator.

### Software regression scope

`tests/test_dataset_iteration.py` retains required-metric, alias and FAR checks,
and adds literal-metadata effective-domain, compatibility and framing checks.
Strict unittest assertions and comparator children run in normal, `-O` and
`-OO` modes. The main-assembly fixture always uses a normal-interpreter child
with all three backend commands mocked; it is not optimized-main or acoustic
evidence. The existing RUN CLI `-OO` startup/docstring limitation is not changed.
These focused checks do not replace the repository's full required checks or
hosted/product qualification.

## Running it

Dispatch `.github/workflows/dataset-driven-iteration.yml` from the exact
current protected `main` with a dataset (`references.jsonl`) and the variable
under test.

Leave `model_path` and `keywords_path` empty to use the model tuple pinned by
`configs/shipping.xiaowo.json` and mirrored under `models/registry/`. This is
the default product-facing path. To compare another deployable model, supply
**both** paths explicitly; supplying only one is refused.

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

Because `main` is protected and has no bypass actors, the workflow never pushes
evidence directly to `main`. It creates a deterministic
`dataset-iteration/<run-id>` branch and opens a pull request. This keeps the
measurement lane compatible with the repository's required-PR/status-check
ruleset.
