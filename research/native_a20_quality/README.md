# Native A20 offline data admission and saved-prediction quality tools

Research-only, standard-library Python. This directory is independent of
`research/native_a20`, default CMake, the product ABI/registry and shipping gates.
It does not run inference, train, tune thresholds, sweep parameters, download data,
play audio, or require a board/SDK. It accepts already recorded decisions only.
Fictional examples and unit tests are software-contract tests, never quality evidence.

## Quick start

From the repository root:

```sh
python3 -B -m unittest discover -s research/native_a20_quality/tests -v
python3 -B research/native_a20_quality/quality.py admit \
  --manifest research/native_a20_quality/inventory/existing-50.manifest.json
python3 -B research/native_a20_quality/quality.py score \
  --manifest research/native_a20_quality/examples/toy-manifest.json \
  --predictions research/native_a20_quality/examples/toy-predictions.json \
  --output /tmp/native-a20-toy-report.json
```

Output paths must be new and must not be input paths. `admit` validates metadata;
`score` validates metadata and saved predictions before emitting any report. Exit 2
means invalid/unqualified input, not a numerical pass. Neither command changes
admission records or opens an inference path. The dedicated path-filtered workflow
runs only these metadata tests/commands. No new dependency is installed.

## Evidence contract

`schemas/manifest.schema.json` and `schemas/predictions.schema.json` are closed,
versioned JSON Schemas. The CLI enforces their used subset with stdlib code plus
cross-record semantic checks. Unknown fields, duplicate JSON keys, non-finite
numbers, booleans in numeric fields, malformed hashes and incomplete records fail
closed. There is no legacy-format guessing.

Every input has original WAV and decoded PCM hashes, frame count, sample rate and
exact duration; logical source reference and hash; source tier; data-use license
review and explicitly approved roles; speaker/generator revision/voice/seed/prompt
family; session, room, device and AFE; label strength and binding; prior training,
selection or observation exposure; train/dev/heldout/inventory role; leakage and
derivation families; and noise/far-field/hard-negative strata. Unknown fields remain
explicitly unknown. `declared_keywords` represents source intent/weak information,
not annotated positive events. A recorded model license does not establish audio
output rights, training rights or product qualification. Admission does not grant
permissions, authenticate legal claims or substitute for source review.

Independent scoring requires a reviewed role permission, known audio identity,
reviewed exposure, a pre-result split freeze, no prior exposure, and source-bound
verified positive/negative annotations. Opening heldout also requires an explicit
protocol authorization. Source-family, speaker, session, generator voice/prompt,
leakage-group, derivation-family and exact PCM/WAV identity conflicts across
train/dev/heldout are rejected. Exposed inventory siblings cannot be hidden while
a related asset is called heldout. The grouping checks are deliberately conservative:
use documented source-family/recording identities, never unique aliases to defeat them.

Byte-identical scored assets are rejected even within one split. More than one
negative-bearing dev/heldout asset sharing a derivation family or original recording
session is rejected, including crops/augmentations with different PCM bytes. This
version conservatively requires a single complete recording record with disjoint
verified-negative intervals instead of summing overlapping or separately processed
crops. Replayed/looped or noncontinuous audio cannot contribute negative hours.
Source/exposure claims still need independent audit; software cannot discover a
false declaration or detect unrecorded prior model observation.

The exact manifest file SHA-256, protocol SHA-256, expected model payload SHA-256,
source-manifest SHA-256 and decoder-configuration SHA-256 bind saved predictions.
The last three must match protocol pins. For a new candidate or baseline, freeze
an explicitly reviewed protocol rather than change identities after seeing results.
The frozen input order must match both manifest and prediction order. State resets
once per complete asset, and persists within that asset; resetting every chunk of
a continuous negative recording is unsupported. The saved run must explicitly declare
this state policy. Real discontinuities need separately bounded recording assets and
a reviewed continuity protocol, not silent state bridging. Every manifest asset must
have one complete prediction record, including an empty
decision list when nothing fired; omitted/partial output never implies a miss or
zero false alarms. The run must process the full declared duration.

## Metrics and exact denominators

Reports separate fresh eligible development and qualified research-heldout partitions.
Each gets per-keyword and dataset/source-tier/speaker/generator-voice/noise/far-field/
hard-negative strata. These are pointwise views, not independent experiments or a
simultaneous confidence guarantee. Synthetic/open-source measurements only describe
their admitted data domain. They do not imply real microphone/AFE/device performance.

- FRR = missed eligible, verified positive **events** / all such events; recall is
  one minus FRR. No clip count or weak/unknown intent is used as this denominator
- Annotated matching windows must be disjoint, including across keywords. The first
  correct-keyword decision in an event window is one hit; additional same-keyword
  decisions are duplicates. Wrong-keyword decisions in positive windows are separate
  counts. They cannot rescue another event or become continuous-negative FA evidence
- Negative hours are the sum of disjoint, bounded intervals of original continuous
  audio explicitly verified to contain **none of the entire scoped keyword set**.
  False-alarm events are saved decisions within those intervals, reported per keyword
  and pooled. Other unannotated time/decisions are unscored, never presumed negative.
  Clip FPR is not calculated or substituted for FA/h
- All interior time windows are `[start, end)`. A finish callback exactly at recording
  EOF is attributed to the interval ending at EOF, avoiding dropped final detections.
  A verified true word tail exactly at EOF is likewise valid; an EOF decision then
  has zero latency from that tail.
  Times outside `[0, duration]` are rejected, never clipped. Runs requiring a different
  delayed/asynchronous timeline need a separately reviewed mapping/protocol
- The empirical zero count produces observed FA/h = 0, but **does not establish zero
  population FAR**. With the stated stationary homogeneous Poisson/independent-count
  assumption, an exact one-sided upper mean solves `P(Poisson(mu) <= k) = 1-c`.
  Divide by verified negative hours; for zero events it is `-ln(1-c) / hours`
- A two-sided Wilson FRR interval is supplied only when the protocol explicitly
  asserts independent, identically distributed Bernoulli event trials. Speaker,
  generator and repeated-session correlations may invalidate that model. Poisson
  assumptions may likewise fail for clustered alarms. If not justified, set the
  assumption false: descriptive rates remain, confidence bounds are null. No valid
  positive denominator or negative exposure means null, not zero, for that metric

The repository's pinned `tools/statistical_bounds.py` was reviewed. Poisson inversion
uses the same exact one-sided definition; this standalone implementation uses a
constant-memory reverse log-domain recurrence. The FRR output is a **two-sided**
Wilson interval, whereas the product helper's `wilson_upper` is one-sided. Its upper
endpoint equals that helper evaluated at `(1+c)/2`, subject to floating-point precision.
There is no claim that product qualification thresholds or deployment identities apply.

Latency is `first correct callback-available audio timestamp - verified true word-tail
annotation`, on the same audio timeline. A decoder's token-end/keyword-end estimate is
not a callback availability time. Predictions must explicitly declare this semantics.
There is no clip-end fallback. Missing tails yield `NOT_QUALIFIED` when none are valid,
plus counts of detected events missing tail truth. Signed early values are retained,
not clamped. Nearest-rank p50/p95/p99/max are descriptive and restricted to detected
annotated events, with explicit sample counts; missed events are still in FRR. These
are offline audio-timeline metrics, not capture-to-wake wall latency or board deadlines.

Weak, unknown, inventory/train and exposed records are kept in an exploratory section
with decision counts and provenance. They contribute no independent FRR, FA/h or
latency denominator. Source-unknown labels and labels not publicly bound to rows are
different conditions. Human-confirmed does not mean independent, heldout or product-gold.

## Resource evidence and next experiments

`recorded_resources` only echoes named recorded CPU, wall/service time, I/O, fault,
model/arena/cache/stack and RSS fields with units, environment, sample count, correlation
note and an evidence reference/hash. The tool checks types/units but does not remeasure
or authenticate the underlying evidence. Missing values are not estimated. Cache
included in arena must not be added a second time. Static layout, startup and steady
state are separate phases. No host-to-A32 extrapolation is made.

All quality/resource thresholds are **unset**. CPU/I/O are the primary optimization
priorities; a modest RAM increase can be considered only against an explicitly
measured benefit and reviewed budget. No 5%/10% target has been adopted here. A review
must establish a development baseline, goals, operating threshold, coverage/sample
plan and stopping rule before an authorized one-time heldout evaluation. A future
model run/download/generation/training/optimization needs its own approved scope.
Vendor SDK absence blocks vendor-specific builds; board/AFE absence blocks the
corresponding physical qualification, not this metadata tooling.

## Existing data inventory

`inventory/existing-50.manifest.json` is an inventory-only projection from the bounded
archive, not newly admitted data. `existing-50.summary.json` retains original versus
public source hashes and verified local public-stage identities. The full public-file
manifest is pinned to SHA-256
`d0c265888b8e4799101a226eb568f4592f5e63d824493fae4214b0d577b5ed01`.
Remote archive publication/completeness is not asserted.

- D20: 20 WAVs, 31.12 s, all weak (11 historical weak positives and 9 weak foils).
  The source audio, trainer protocol/manifest, checkpoints and saved training evidence
  were restored; no retraining was run, and public excerpts are not a revalidated
  standalone retraining package. Restoration does not recover the permanently missing
  old fresh native traces or grant ordinary training/product admission
- Serena18: 18 WAVs, 27.28 s; 4 historical resolved weak labels, 14 source-unknown.
  All are exposed diagnostics, excluded from training, threshold/endpoint/model
  selection and fresh heldout use
- fixed12: 12 WAVs, 15.4 s; exposed regression. Existing review context reports an
  aggregate of 2 human-confirmed, 5 ASR-weak and 5 unknown. The inspected public byte
  allowlist/provenance binds none of those labels per row. Therefore all 12 row labels
  remain `not_publicly_bound`/unknown here. The aggregate is explicitly not independently
  derivable from that public byte provenance and cannot be used as scoring truth.
  Generator revision/reference assertions separately bind the hashed public license/
  provenance scope document; a source/model license is not an output-rights guarantee

All 50 WAV and decoded PCM hashes and frame counts were checked against already
materialized archive bytes, totaling 73.8 s. None contributes an independent FRR or
continuous-negative FA/h denominator, and none has true word-tail truth. Replacement8
is invented numeric DSP validation and is not part of these 50 speech WAVs.

To reproduce from an already materialized bounded archive (no download is performed):

```sh
python3 -B research/native_a20_quality/inventory_existing.py \
  --materialized-root MATERIALIZED_ARCHIVE_ROOT \
  --archive-root BOUNDED_ARCHIVE_PUBLIC_ROOT \
  --public-file-manifest PUBLIC_FILE_MANIFEST_JSON \
  --source-manifest IMMUTABLE_NATIVE_SOURCE_MANIFEST_JSON \
  --output-dir NEW_OUTPUT_DIRECTORY
```

The source manifest must be the immutable baseline at
[PR463 head 8045532](https://github.com/jiying2007/kws-pipeline/blob/804553286fd9caa294769cc4d44cc0c43dc66e88/research/native_a20/SOURCE_MANIFEST.json),
SHA-256 `4a90303837f75183d55adf21bb3fd6bf2e7cf0c79a62e10d582e9f809612227e`.
It is not the publication allowlist hash or an ARM adaptation's new manifest. The
inventory's decoder pin is null and heldout opening is false, so it cannot silently
serve as an execution protocol. Reconstruction checks the top-level manifest, source
manifest, archive index, asset-index shards, relevant source documents and every audio
identity. Outputs contain only public logical paths/hashes and bounded technical
metadata, not machine paths, private storage IDs, conversations or listening quotations.
