# Dataset-driven iteration lane

> **Current scope: technical research only.** The frozen synthetic model and keyword source is `configs/research/two-keyword-synthetic-comparator.json`. Synthetic results do not establish natural-speech FAR/FRR or physical SSC305 measurements. Historical restrictions in archived plans do not govern this active research lane.

This lane measures data and model iterations under an explicit comparison protocol.

## Why it exists

Observing zero false accepts over `T` hours does not establish a population rate
of zero; it establishes an upper bound of roughly `3/T` per hour at 95%
confidence. For example, **12 seconds of negative exposure produces an upper
bound of about 900 false accepts per hour**, even when the point estimate is zero.

Zero observed false rejects *and* zero observed false accepts can coexist on a
finite, separable dataset. Such a result does not establish zero population
risk or generalization to other speakers and conditions. Threshold changes often
trade recall against false accepts, but that tradeoff does not mathematically
rule out a zero-error sample. Report positive support, negative exposure and
confidence bounds; rejecting everything still fails recall when positives exist.

This lane records measurements and a versioned research comparison. Its verdict
does not promote a model or require stopping further research.

New runs produce schema-5 exact-Poisson scorecards. Their comparator now emits
comparison-report schema 3 with policy
`research-operating-point-wilson-bonferroni-nominal95-v1` and four possible
outcomes. Scorecards 1–4 retain their historical comparison-report schema 1,
three outcomes, and default FRR tolerance of zero; saved historical reports are
not rewritten. The [multilingual model tournament](KWS_MULTILINGUAL_MODEL_TOURNAMENT.md)
accepts a homogeneous v4 or v5 ablation cohort and retains its separate observed
Pareto ranking policy. For systems with different source code or native
frontends, use the explicit system comparison below. These are research results,
not independent-human or SSC305 qualification.

## The two controlled variables

An ablation declares these identities and which one is under test:

| Variable | Identity | Meaning |
| --- | --- | --- |
| `dataset` | `dataset_id` | Content hash over audio bytes + expected events + domain bins + verified AFE sidecar presence/content |
| `model` | `model_id` | Hash over the model **and** the keyword pack (the pack carries the threshold, so it is part of the operating point) |

A run also records `code_sha`, `seed` and `protocol`.

## Default ablation rules

1. **One variable at a time.** If both `dataset_id` and `model_id` move, the
   delta has no interpretation and the comparison is refused.
2. **The declared variable must be the one that moved.** Declaring `dataset`
   and evaluating a new model is refused.
3. **Same protocol or no comparison.** Numbers from different protocols are not
   the same measurement.
4. **Code changes confound.** If `code_sha` differs, re-baseline first.
5. **Both systems must demonstrate the same FAR budget before an operating-point
   conclusion.** The exact upper bounds may differ while both meet that budget.
   Zero over 30 seconds and zero over 30 hours provide different evidence.

## Comparing complete systems

`--mode system` allows different model tuples, source revisions, runners and
native frontends. It is a system-quality comparison, without attributing a delta
to one particular code or architecture change. The default `ablation` mode keeps
the single-variable and identical-code checks; neither mode is inferred from
which identities happen to differ.

Produce both cards with `--mode system --variable system`, a native-frontend
description and the same `--system-context` JSON, for example:

```json
{
  "sample_rate_hz": 16000,
  "feed_chunk_samples": 160,
  "reset": "per-recording",
  "appended_context_samples": 0,
  "eof": "no-extra-audio",
  "event_time": "input-wav-sample-seconds",
  "keyword_map": {"1": "你好小窝", "2": "小窝小窝"}
}
```

This is the runner's declared **input/event contract**, not its internal feature
geometry. Each adapter must implement that contract; native acoustic hops,
lookahead, models, decoding and calibrated thresholds may differ. This tool does
not reconfigure or attest the inside of an arbitrary runner. It currently uses
the existing `run_corpus.py` direct-runner interface; a new multi-file model
runtime still needs its adapter and complete model-package binding.
The hosted iteration workflow uses the built-in `kws_wav` default 160-sample
block and rejects any other declared block size. Other adapters must document
and implement their own actual feed contract.

```sh
python tools/run_dataset_iteration.py \
  --runner "$RUNNER" --model "$MODEL" --keywords "$KEYWORDS" \
  --references "$REFERENCES" --audio-root "$AUDIO_ROOT" \
  --work-dir "$WORK" --output "$CARD" \
  --mode system --variable system --system-context "$CONTEXT" \
  --native-frontend "native-80-bin-v1"

python tools/compare_dataset_iterations.py \
  --baseline "$BASE_CARD" --candidate "$CANDIDATE_CARD" \
  --mode system --far-budget 0.13 --frr-margin 0.02 --latency-budget 500
```

System comparison requires the same original/effective reference bytes, measured
WAV/PCM inputs, verified AFE metadata and scoring protocol. It reopens the saved
input/provenance/detection files and recomputes event metrics. Matching a manually
chosen `dataset_id`, protocol name or metrics table is insufficient. Preserve the
private files referenced by `system_evaluation`; missing or changed evidence is
an error. These pre/post checks are not an atomic filesystem snapshot or an
attestation of native frontend execution. Scorecards from ablation mode cannot
be passed as system results merely by adding the comparison CLI flag.

### Saving and restoring the measured inputs

System runs automatically copy the required replay inputs into
`<scorecard-directory>/evidence/<run_id>/`. The bundle contains original/effective
references, saved events, corpus provenance, WAV and AFE files. It does not copy
model weights or runner binaries: those are needed for new inference, while this
comparison rescoring uses the saved events. Keep the actual model artifacts in
the experiment's separate checkpoint record.

`system_evaluation.origin_root` is the common ancestor of the bound files. The
bundle mirrors paths relative to that root, including absolute audio/AFE paths
from the references. Inputs outside the checkout can make the root broader,
even `/`; only the listed files are copied, never the entire root. Copies are
hashed before/after and fully rescored before the completed bundle is exposed.
An existing bundle is reverified and is not overwritten with different bytes.

Move the card and its sibling `evidence/` directory together. The comparison CLI
automatically uses that sibling bundle. To recover a differently placed bundle,
pass `--baseline-evidence-root /restore/baseline` and/or
`--candidate-evidence-root /restore/candidate`, each naming the directory that
mirrors the recorded `origin_root`. Scorecard, reference and provenance bytes
remain unchanged. Missing or altered restored inputs fail verification. Producer
and comparator reject output aliases to their inputs, including symlinks and
hardlinks, and publish the final JSON by atomic replacement after validation.
These checks assume no concurrent writer replaces paths between checks.

The latency budget still applies to the existing reference-end sample-coordinate
metric, not application callback wall time. The example budgets are explicit
research choices, not evidence that a candidate meets them. Compare separately
calibrated systems at a common FAR budget; equal confidence scores across models
are not required or assumed.

### Four outcomes and the meaning of uncertainty

For schema-5 cards, choose an absolute FRR material-difference margin before
examining candidate outcomes. The initial research default is **0.02**, meaning
two percentage points, and is not claimed to be optimal. `--frr-margin` and its
existing alias `--frr-tolerance` record that value. The tool cannot authenticate
when an explicitly supplied margin was chosen.

The comparator verifies integer expected/matched/missed counts, their sum and
the exact `false_rejects / expected` FRR. The same evaluation dataset must have
the same positive denominators. It constructs a two-sided **97.5% Wilson score
interval** for each arm using the existing one-sided helper at 0.9875. If the
baseline interval is `[L_b, U_b]` and candidate interval is `[L_c, U_c]`, the
difference interval for candidate FRR minus baseline FRR is
`[L_c - U_b, U_c - L_b]`. This is a Bonferroni construction with **approximate,
nominal 95% coverage**, because the marginal Wilson coverage is approximate.
It does not require independence between the two systems. See the NIST
[Wilson interval formulas](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
and [Bonferroni inequality](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm).

| FRR evidence result | Interval rule with margin `m` |
| --- | --- |
| `improved` | Entire difference interval is below `-m` |
| `regressed` | Entire difference interval is above `+m` |
| `neutral` | Entire difference interval lies within `[-m, +m]` |
| `inconclusive` | Any remaining overlap, or no positive support on either side |

`frr_direction` separately reports the observed point-estimate direction:
`lower`, `higher`, `unchanged`, or `not_estimated`. `frr_assessment` retains the
counts, marginal intervals, difference interval and FRR evidence result even
when the final operating-point `verdict` is inconclusive.

For example, a change from 20/100 misses to 10/100 misses has a lower observed
FRR but remains inconclusive under this interval rule. At 200/1000 versus
100/1000, the difference interval is approximately `[-0.1491, -0.0499]`, so its
FRR evidence result is improved at margin 0.02. These are count examples,
not claims about independent human recordings.

The final verdict also requires both exact, one-sided 95% FAR upper bounds to
meet the same declared budget and the candidate's latency point to meet its
ceiling. Otherwise it is `inconclusive`. FAR budget status is reported separately
for each arm as `met`, `not_demonstrated`, or `observed_rate_above_budget`.
The last status describes an observed rate, not a proved population excess or
candidate regression. With zero false accepts, the default 1 FA/h budget needs
about 2.996 hours of qualifying negative exposure; shorter zero-error runs do
not demonstrate that budget. The latency report has no confidence interval, so
crossing its point-value ceiling is not called a statistical latency regression.

These are conditional event-level research intervals. Independent Bernoulli
events with a common miss probability within each arm are an assumption, not
something an aggregate scorecard establishes. Repeated speakers, related source
clips and synthetic voices can
violate that assumption; the comparator does not count them as independent
human evidence or apply a source-cluster correction. It also does not use paired
event discordances or correct for repeated examination and candidate selection.
Independent-human and final-target claims need their corresponding data and
analysis. All new reports explicitly keep `independent_human_evidence=false`.

The existing per-keyword table is retained with counts, intervals, observed
direction and false accepts. Its positive counts must sum to the aggregate;
system mode requires that table and rescores it from the saved events. Missing
per-word evidence in a metadata-only ablation card is reported as unavailable.
Per-word intervals are diagnostic,
not a simultaneous guarantee across all words, and a small slice's fluctuation
does not automatically veto the aggregate result. Report accent and scene
slices alongside them. Complete accented, linked or lightly pronounced wake
words remain positives; ASR disagreement alone must not turn them into negatives.
Near-sounding and partial-word challenges remain separate from natural continuous
background FAR; synthetic or repeated challenge duration cannot establish that
natural-audio population. Historical results retain their original interpretation.

## Interpreting the exposure number

| Negative exposure | 95% upper bound if zero false accepts |
| --- | --- |
| 12 s (0.0033 h) | ~900 / h |
| 10 min (0.167 h) | ~18 / h |
| 1 h | ~3 / h |
| 10 h | ~0.3 / h |

Treat the bound, not the point estimate, as the measurement. A dataset too small
to support the experiment's false-accept budget cannot demonstrate that budget --
that is a property of the dataset, not of the model.

## FAR populations and scorecard compatibility

New scorecards use schema 5, default protocol `dataset-iteration-v5`, and the
fixed FAR identity policy `negative-only-recordings-exact-poisson-one-sided95-v1`.
The policy is part of `run_id` even with a custom `--protocol` label.

`far_rate` uses only complete recordings with `expected: []`: its
`observed_false_accepts`, `point_estimate_per_hour` and upper bound all use that
population and `exposure.exposure_hours`. Missing negative metrics or zero
negative exposure are errors, not zero FAR. Producer and comparator both use
`tools/statistical_bounds.py:poisson_rate_upper` with confidence 0.95. For zero
events the bound is `-ln(0.05)/T`; otherwise the helper solves the one-sided
Poisson CDF equation. The count is an integer in [0,10000], bounding work on
untrusted scorecards. Do not round stored values; the comparator checks the exact
count/exposure calculation and rejects non-finite bounds.

Historical schemas 2--4 retain `2.9957/T` for zero and
`(N + 1.959964*sqrt(N))/T` otherwise, under their original approximate policy.
That approximation incorrectly decreases from zero to one event. Old card bytes
are not rewritten and their replay reports include `historical_statistics: true`;
use schema 5 for new iteration decisions. A v4/v5 pair is rejected even with
identical custom protocol labels. Regenerate both cards from original inputs to
move a comparison to the exact policy.

`metrics.false_accepts` and `metrics.far_per_hour` remain the scorer's global
all-recording diagnostics, including unmatched detections in positive
recordings. The comparison's `deltas.far_per_hour` remains that global
diagnostic; its FAR budget and improvement decision use the negative-only
upper bound. This lane still does not qualify or promote a model.

## Effective-domain dataset identity

Schema 3 introduced `identity.dataset_identity_policy`
`effective-domain-keys-v1`. Schema 4 retains the same effective-domain semantics
under its new AFE-aware policy described below. Each dataset
hash entry contains `sorted(domain_keys(row))` from `eval/domain_metrics.py`,
the same helper used to group scoring slices. The dataset hash is over a tagged
envelope `{"dataset_identity_policy": "effective-domain-keys-verified-afe-v2", "entries": [...]}`
for schema 4 (schema 3 used `effective-domain-keys-v1`).
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

## Verified AFE sidecar dataset identity

Schemas 4 and 5 fix `identity.dataset_identity_policy` to
`effective-domain-keys-verified-afe-v2`, even with a custom `--protocol`. Each
entry also contains `afe_metadata: {"present": false}` when both metadata keys
are absent, or `afe_metadata: {"present": true, "sha256": "..."}` for a present
sidecar. The SHA is computed from the actual file and verified against the
reference's `metadata_sha256`, not merely copied from that declaration.
`metadata_path` resolves against `--audio-root` unless absolute, using the
corpus runner's existing semantics. Missing files, malformed paths, incomplete
pairs and mismatched digests fail before evaluation starts; none means absence.

Different verified bytes, or presence versus absence, change dataset and run
IDs. Relocating identical bytes preserves both, given the same other identity
fields. Thus a sidecar-only change is a dataset experiment; a sidecar plus model
change is rejected as two variables moving. No-metadata datasets also migrate
to new IDs because both the policy tag and explicit absence marker are new.
Historical golden IDs/cards remain archived, not rewritten or relabelled.

The producer binds inputs before evaluation, retains corpus per-file pre/post
checks, and rechecks both effective and original reference bindings after the
Python evaluation stages before emitting the scorecard. A persistent sidecar
or relevant binding change fails closed and preserves an existing scorecard.
This is bounded persistent-change detection, **not** immutable staging, an
atomic snapshot, an automatic lock, or proof of exact bytes consumed. Changes
restored between checks and broader reference/WAV/model races remain outside
this guarantee. Corpus timing validation and posterior-cache sidecar rejection
are unchanged; the identity helper does not replace runtime timeline validation.

The comparator retains schema-1/schema-1 historical mixed-population FAR,
schema-2/schema-2 fixed-FAR, and schema-3/schema-3 effective-domain contracts,
including matching custom protocols. Schemas 1/2 retain legacy top-level-distance
identity; schema 3 requires exactly `effective-domain-keys-v1`. Schemas 4 and 5 require
the same AFE-aware identity policy. Schemas 2/3/4 share all fixed negative-only FAR tuple
and bound checks. Schema-1/schema-2 pairs still fail FAR-policy compatibility.

Mixed dataset policies, unknown/missing policies, unexpected policy fields on
schemas 1/2, and unknown or boolean schemas fail before interpreting moved IDs,
even with matching custom protocol labels. Old cards never imply verified AFE
absence. Re-run both sides from original inputs under the same current code and
policies; do not relabel or re-hash archived cards. An old comparator does not
implement these checks: use the current comparator.

### Software regression scope

`tests/test_dataset_iteration.py` retains required-metric, alias and FAR checks,
and adds literal-metadata effective-domain, verified-sidecar, historical
compatibility, persistent-mutation and four-way framing checks.
Strict unittest assertions and comparator children run in normal, `-O` and
`-OO` modes. The main-assembly fixture always uses a normal-interpreter child
with all three backend commands mocked. A second normal-child fixture invokes
the actual corpus, scorer and domain Python mains, mocking only the external
runner on literal tiny PCM/TSV inputs. The v5 suite additionally covers an
independent high-precision Poisson oracle, the zero-to-one FA counterexample,
independent Decimal inversion of the binomial score equation, all four outcomes,
count/denominator checks, separate budget status, and per-word diagnostics,
mode separation, WAV-duration rechecking, unchanged-card bundle restoration,
damaged evidence and output aliases, including under optimized interpreters.
No real acoustic model or C timeline-parser execution is established by these
fixtures. These checks supplement the repository's required CI and later real
speech/device measurements.

## Running it

Dispatch `.github/workflows/dataset-driven-iteration.yml` from the exact
current protected `main` with a dataset (`references.jsonl`) and the variable
under test.

Leave `model_path` and `keywords_path` empty to use the model tuple pinned by
`configs/research/two-keyword-synthetic-comparator.json` and mirrored under `models/registry/`. This is
the default frozen research baseline. To compare another deployable model, supply
**both** paths explicitly; supplying only one is refused.

Optionally point `baseline_card` at a previously recorded scorecard. For a
system comparison, also supply `baseline_artifact_run_id` so the complete saved
input bundle is restored into a directory separate from the new candidate.
If both inputs are supplied, the downloaded card must match `baseline_card`
byte for byte. Supplying only the explicit artifact run ID selects the card
from that completed main-branch iteration.

Set `variable=system`, `system_context_path`, and `native_frontend` for a system
comparison; dataset/model choices retain the causal-ablation mode. Set
`resume_artifact_run_id` to finish publication from a completed iteration's
saved outputs without building the runner or repeating measurements.

The full evidence is uploaded before publication and after computation failures
when partial output exists. The artifact retention window is 90 days. Preserve
valuable replay bundles in a durable authorized location before that window
expires; a public scorecard alone is insufficient for future system rescoring.

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

Exit codes for the comparison: `0` improved, neutral or inconclusive;
`1` regressed; `2` refused or unusable input. An inconclusive result can proceed
to the next research measurement; it does not claim that the budgets passed.

## What this lane does not do

It does not qualify, freeze, or promote anything, and it consumes no protected
evidence. A new-schema verdict of `improved` means the conditional FRR difference
interval clears the predeclared margin while the common FAR condition and
candidate latency point budget are met. It is not proof of generalization from
synthetic or repeated-source observations. Natural-speech and physical-target
claims still need their corresponding measurements.

## Recording

Scorecards are committed to `evidence/dataset-iterations/<dataset-id>/<run-id>.json`
so results outlive artifact retention and remain reviewable.
For system runs, preserve the replay bundle as well; a scorecard hash cannot
replace its audio, events or references. The bundle can contain private
evaluation audio and must remain in the dataset's authorized storage scope.

Because `main` is protected and has no bypass actors, the workflow never pushes
evidence directly to `main`. It creates a deterministic
`dataset-iteration/<run-id>` branch and opens a pull request. This keeps the
measurement lane compatible with the repository's required-PR/status-check
ruleset.
