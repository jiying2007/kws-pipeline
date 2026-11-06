# Context v3 full2601 observations

`training/research_context_v3_inputs.py` is the no-model consumer of the new24
context-v3 weak packages. It requires the independently verified 72-cell v3
plan, its exact new24 D2 and context-v2 ancestry, the source-family split and
the previously fixed full2601 vocabulary. It does not generate audio or labels,
extend that vocabulary, execute training, or alter earlier packages.

## Replay before projection

`load_data_module` accepts the v3 data adapter and hashes all 14 producer
dependencies. Every weak package must have an explicitly pinned manifest.
`aggregate` invokes the full native/ASR replay before reading or filtering rows,
checks the producer's exact source files, and joins the original plan, native
bytes, owner stage and recipe, selective-ASR evidence, decision and audio.

The aggregator uses global source IDs and original native PCM hashes across
packages. A batch-local `clip-000` identifier cannot establish a new source.
Source-family, speaker and PCM aliases cannot cross train/development roles,
including for zero-weight rows. `prior_sources` carries those existing aliases
into the same check. Duplicate packages, reused source IDs, duplicated native
PCM, mismatched plan pins and altered evidence are rejected.

Coverage reports retain all observed rows, the fixed planned count of 72, and
the exact unobserved cell IDs. An observed subset never becomes a claim that
all requests returned. Failed-native and uncertain rows stay in the audit
population before eligibility is decided.

## Complete observed text as the target

`verified_projection` authenticates both the exact vocabulary file and original
FSMN token bytes. The resulting map must equal the plan's fixed map, with 2601
outputs and blank ID zero. `project_row` computes targets from the complete
observed normalized transcript; it never reads the requested prompt to build a
target and never truncates text to the wake phrase or a legacy filtered export.

Positive-intent rows require an observed intended full keyword. Ordinary-speech
rows require both complete keywords to be absent, both keyword masks confirmed
and a nonempty full transcript. Their CTC target is every character of that
transcript. The `[0, 0]` keyword labels describe observed absence; they do not
mean an empty CTC target. This projection does not execute keyword loss.

Uncertain and failed-native labels have zero objective weights and unknown
keyword targets. A confirmed transcript that contains an out-of-vocabulary
character or cannot fit the unpadded C1 CTC geometry is also quarantined as a
whole: no CTC target, zero CTC and keyword weights, and no invented negatives.
The complete text, ASR observations and original weak decision remain intact
under `observed_label`. Geometry includes one additional frame for each pair of
adjacent equal target IDs, so repeated characters do not evade the CTC check.

The report gives eligible counts by category, projection rejection reasons and
explicit ordinary-speech target semantics. Each projected row carries its
parent manifest/report/input pins, source role, fixed vocabulary pins, request
hashes, historical-reuse flag and observed context metadata. Character offsets
remain transcript positions; no acoustic event times are inferred.

## Use and limits

Call `verified_projection(packages, loader, source_files, ...)` with exact
package declarations, expected canonical plan hash, vocabulary file hash and
original token path. The result contains every observed row, with separate
`ctc_mask`, `ctc_weight`, `target_ids` and `quarantined` fields. A later training
preparation must preserve those fields and the train/development separation.

The existing D2 consumption probe remains its fixed 25-step mechanical test.
It is not a context-v3 trainer, and this module does not add an extended-step
switch to it. Actual context training requires a separately declared preparation
and bounded consumer over these authenticated full-text rows.

The standard-library tests cover ordinary full-text targets, unknown/native
failure/OOV quarantine, contradictory keyword labels, fixed vocabulary pins,
CTC geometry and old schema rejection. With `KWS_CONTEXT_DATA_ROOT` set to the
matching data checkout, the suite also creates and replays a real-format
fictional native/selective-ASR package, then checks full2601 projection, partial
coverage, source separation, duplicate evidence and tampering. No test imports
a model or performs model inference.
