# kws-data bulk KWS development effect boundary (cross-repository)

This is the **single consumption boundary** for the already-merged
`jiying2007/kws-data` #47/#48/#49 native Qwen3-TTS and 3-ASR bulk source
pipeline. We intentionally do **not** duplicate its TTS runner, model-cache,
source-rights importer, native recognizer dispatcher or retry semantics in
`kws-pipeline`. The earlier stacked alternative PRs #692/#694 are closed,
unmerged to avoid divergent generators and evidence formats. #690 remains
merged as an independent read-only audit for its own legacy private campaign.

## Status and invariant

Entry point: `tools/kws_data_bulk_effect_boundary.py`. It runs no model,
does not access GitHub or upload private media, and **cannot** admit training.
It rechecks the source-derived model plan and saved machine-screen consistency
against actual 16-kHz PCM16 WAV file bytes plus generation receipts, then
aggregates counts by both wake words, exact fixed requested negative text
classes, and source family.

Follow the independent [bounded feedback priority plan](KWS_DATA_FEEDBACK_PRIORITIES.md) after producing the private effect inventory; it never authorizes another model call or training row.

The output is a **development-only** effect inventory. It cannot re-run native
ASR, authenticate external source rights, prove a synthetic utterance was
spoken exactly, infer true FAR/FRR, or substitute for real-human qualification.
Even 3/3 matching native transcripts are **machine candidates**, not gold
labels. The source-owner `kws-data` checks full native ASR sidecars, adapter
locks, full frame coverage and consumed-attempt semantics; this reader
deliberately does not claim to repeat those checks.

## Source producer and the minimum private input set

The source producer is the pinned `kws-data` checkout containing:
- `research/auto-tts-voice-clone-20261102/bulk_pool.py`
- `bulk_runner.py` with verified output `plan-lock.json` and
  `cells/<cell_id>/started.json + completed.json + audio-16k.wav`
- `bulk_asr.py` with `receipt_contract.py`, original private native
  ASR receipts and the `--report` output `kws-bulk-clone-machine-screen-v1`

Before transferring any derived summary, pin the **exact kws-data Git commit**
and exact `plan`, `screen` plus private audio-root content digests outside
this repository. A mutable branch, caller-edited JSON or an ASR string is not
source authority. Do not copy reference recordings, sensitive transcripts or
rights/identity documents into public GitHub.

First obtain a machine-screen report from the existing `kws-data` code,
after genuinely authorized and pinned local TTS / native-ASR executions
(if they have happened):

```bash
# This invocation is screen readback only; no new TTS or ASR calls.
python3 research/auto-tts-voice-clone-20261102/bulk_asr.py \
  --plan /private/bulk-plan.json \
  --generated /private/bulk-generated \
  --asr-output /private/bulk-asr \
  --adapter-lock /private/native-asr-lock.json \
  --report /private/bulk-machine-screen.json
```

Then, from a pinned `kws-pipeline` checkout:

```bash
python3 tools/kws_data_bulk_effect_boundary.py \
  --plan /private/bulk-plan.json \
  --screen /private/bulk-machine-screen.json \
  --generated-root /private/bulk-generated \
  --output /private/reports/kws-bulk-effect.json
```

`--output` must target a pre-existing private directory and a **new** file;
no existing output or original input is overwritten. Its JSON is created with
mode 0600. Both pinned input documents are read only. The tool independently
re-verifies each saved generated WAV + decoded PCM, checks source-family
train/development grouping and rejected status combinations, and produces
per-phrase challenge counts. A negative recognized as a full wake is never
silently recoded into a correct hard negative.

## Output interpretation

- `total_generated_audio_verified`: bytes/hash/shape checked here, **not**
  intelligibility or label correctness.
- `screen_by_keyword`: machine screen support for kw1/kw2. A positive machine
  candidate is still not a reviewed training label or real-user recall.
- `screen_by_challenge`: intended near-homophone, incomplete, repeated and
  ordinary negatives, stratified by screen outcome. The challenge key is
  derived only from the exact immutable 16-prompt `bulk_pool` v1 schedule.
- `unique_source_families_declared`: source metadata count, **not**
  independently authenticated real human speakers.
- `gaps`: unresolved source, disagreement, false-wake confusion and missing
  real-human/continuous/SSC305 evidence. Every report explicitly has
  `source_rights_authenticated=false`,
  `asr_source_authenticated=false`, `training_admitted=false`,
  `qualification_allowed=false`. No product-release approval is part of this research evidence.

The fixed 16-prompt challenge taxonomy is a derived **analytics mapping**,
not a change to `kws-data` model input or its historical byte identities.
If the upstream bulk plan changes, update the mapping via a separately
reviewed version rather than inferring arbitrary new negative labels.

## What actually unblocks new CTC training

This report is deliberately not a `training_admission` receipt. Real
training must have independently controlled and verified source/rights plus
actual-label evidence, a complete frozen three-ASR native policy/root where
appropriate, family-isolated training splits and matched ancestry accepted by
`tools/asr_source_resolver.py` /
`training/training_admission.py`. The existing verified research training
roots remain empty; do not insert `kws-data` machine-screen strings as
authenticated training roots without completing those checks.

After a truly admitted train-only base exists, compare a fixed model and
decoder under one data variable on independent development recordings, then
measure independent real Mandarin/final-AFE performance and physical
SSC305 resources as separate research evaluations. These measurements are
not prerequisites for continuing controlled source-only experiments; none
can be replaced with generated clips or fake CI inference.
