# Independent ASR observations of the frozen 124 acoustic variants

`tools/research_acoustic_asr.py` adds actual recognition evidence to the previously
rendered acoustic package. It reads the original 124 WAVs and writes a separate
validation view. It never edits their original rows, inherited labels, audio,
training weights, source families or splits.

## Fixed scope and interpretation

| Property | Bound |
| --- | --- |
| Original variants | 124: 28 onset offsets, 28 digital levels, 28 measured RIR convolutions, 40 recorded-noise mixtures |
| Original split | 100 training, 24 development; source and speaker families unchanged |
| ASR engines | Existing Qwen06, SenseVoice and Whisper native workers, exact existing model/runtime identities |
| Chunks | 16: fifteen of 8 and one of 4, in original row order |
| New recognition calls | At most 372, one call per original variant and engine |
| Worker/queue time | 570 seconds per worker, 10 minutes per job, 160 runner-minutes aggregate |
| Other computation | Zero TTS calls and zero optimizer steps |
| Concurrency and retry | One canonical shared heavy-model lock; zero automatic retries |
| Read-only labels | All 40 noise-mixture full-text CTC masks remain disabled, even when ASR agrees |

The adapter authenticates the original package manifest and exact file roster,
original augmentation source, archived parent row/audio/proof, retained parent
metadata, source/speaker/family/split joins and RIR/noise assets. The unique
declaration registry is keyed by the original package manifest SHA. A new output
directory cannot allocate another recognition budget for the same package.

Each original mono PCM16 16-kHz WAV is retained byte for byte. Its samples are
decoded as signed integer divided by 32768, exactly represented as little-endian
float32. There is no resampling, gain normalization, trimming or RMS rejection.
This preserves intentional digital attenuation. Digital gain remains digital
gain; it does not establish a human soft-spoken or physical-distance condition.

The unchanged native ASR controller receives its normal audio-only manifest and
opaque clip IDs. Full parent text is kept in the owner plan for comparison after
recognition; it is never passed to a recognizer. Every variant has the explicit
source-quality-uncertain flag, which requires the third engine under the existing
selective policy even when both primary engines agree. Native decoder termination
and quality flags are retained without filtering them away.

## Separate full-transcript validation view

For each variant, `validation/rows.jsonl` retains raw and normalized full text
from all three engines, native evidence hashes, PCM identities, parent/source
lineage and original inherited labels. The comparison uses the unchanged native
orthographic normalization, including punctuation and CJK spacing rules.

Conclusions distinguish unanimous complete parent-text matches, unanimous
different text, complete-transcript disagreement, quality/termination uncertainty,
and all-empty native transcripts. A keyword match alone is insufficient for a
full-transcript match. Disagreement and uncertainty produce a quarantine
suggestion in this new view, with no automatic change to the original dataset.
Machine agreement remains weak evidence and never becomes human transcription
or training/production approval.

## Preparation and prerequisites

Freeze the complete source tree before preparing the actual stage. Preparation
authenticates and copies input bytes, creates audio-only manifests and exact
native requests, and declares the 16-job queue without a model call:

```sh
python tools/research_acoustic_asr.py prepare \
  --variants /absolute/variants-001 \
  --variants-manifest-sha256 ORIGINAL_MANIFEST_SHA256 \
  --runtime /absolute/restored-asr-runtime \
  --output /absolute/variants-asr-124-001 \
  --context-campaign /absolute/context-v3/campaign/campaign.json \
  --context-validator /absolute/frozen-context-source/tools/research_context_v3_campaign.py \
  --context-handoff /absolute/context-v3/continuation-001/handoff.json \
  --handoff-validator /absolute/frozen-handoff-source/tools/research_context_v3_metadata_handoff.py \
  --ingestion-selection /absolute/ingestion/selection.json \
  --ingestion-validator /absolute/frozen-ingestion-source/training/research_context_v3_consume.py
```

The v2 campaign schema explicitly pins the original campaign, its metadata
handoff and both frozen validators. It keeps generation five read-only and binds
the original four and continuation five chunk identities. The ASR runtime must
be the same canonical runtime as the context campaign. The exact frozen
mixed-reader ingestion validator is pinned at preparation. Recognition cannot
start while either preceding phase remains active or lacks its required durable
closures. Once those phases terminate, bind their evidence:

```sh
python tools/research_acoustic_asr.py bind-prerequisites \
  --stage /absolute/variants-asr-124-001/stage.json \
  --ingestion-stage /absolute/ingestion/stage/stage.json \
  --ingestion-validator /absolute/frozen-ingestion-source/training/research_context_v3_consume.py
```

This invokes the exact frozen ingestion consumer in its core runtime. That
status-only subprocess verifies its source, runtime, recipe and queue, then
authenticates the pinned handoff's original four and continuation five chunk
acknowledgements plus the final handoff acknowledgement. The complete returned
closure must equal the one retained in the ingestion preparation. Its manifest,
request and closure bytes are bound to the original training recipe; the
request must name these exact campaign, handoff, validators and pre-ASR
selection. Original packages keep the r1 reader and continuation packages keep
the r2 metadata reader. Both cumulative and continuation budgets must be closed
without unresolved/failed context jobs or overrun.

The v2 prerequisite receipt also authenticates the actual fixed ingestion
terminal result and terminal acknowledgement. A completed training job
additionally requires its durable readback or explicit readback-failure closure.
A failed, timed-out or unknown ingestion job is admitted only as a durably
stopped prerequisite; it remains failed, timed-out or unknown. The acoustic
observer never repeats that training job or presents it as successful training.

## Guarded execution and persistence

```sh
python tools/research_acoustic_asr.py run --stage /absolute/variants-asr-124-001/stage.json
python tools/research_acoustic_asr.py status --stage /absolute/variants-asr-124-001/stage.json
```

One live `Gate` survives each prelaunch acknowledgement and queue invocation.
The queue recipe is the exact unchanged native selective-ASR request; a separate
owner recipe binds that request to the stage, chunk, selected samples and budget.
The worker verifies its inherited job/recipe/output environment and exact live
intent scope before calling the native controller. A consumed, incomplete native
attempt cannot be restarted. A saved pending intent belongs to its original live
owner and cannot grant a new process another launch.

Every chunk requires an externally acknowledged `before-asr` capsule, then an
authenticated `asr-complete` or `asr-terminal` capsule. The completed native output
is saved before deriving the validation view. Its `chunk-complete` acknowledgement
is required before the next chunk can launch. Dead claimed jobs are reconciled to
authentic unknown terminal receipts and saved before the owner halts. Status
authenticates acknowledgement scope and result/view pins before reporting a
chunk as durably closed.

Capsules contain the fixed source/plan/queue metadata, current chunk audio and
native evidence, current parent/source metadata, and exact preceding closure
descriptors, acknowledgements, provider receipts and phase markers. They do not
accumulate earlier chunks' audio or embed earlier capsule ZIPs. The final
aggregate is separately saved after all 16 chunk closures. An actual external
save receipt is required; a local ZIP alone does not satisfy the barrier.

The prerequisite chain retains the exact old/new context source files, both
data-reader rosters, mixed-ingestion source files, preparation metadata, handoff
history and all chunk/final acknowledgement metadata. Their separately saved
capsules remain identifiable without copying earlier native audio into each
new chunk. The frozen v1 owner is preserved separately; no v1 execution has
been relabelled as v2, and this source revision creates no recognition claim.

## Verification

The focused suite runs in normal, `-O` and `-OO` stdlib-only modes. It covers exact
PCM decoding and low-level audio preservation, full-text/keyword counterexamples,
noise-mask retention, native input boundaries and forced third-engine routing,
unique declaration, prerequisite closure failures, exact native queue recipes,
live worker/environment/lease verification in fresh toy queue children, missing
acknowledgements, old intents, unknown terminal closure, view failure, busy-slot
handling and authenticated status. All subprocess/native fixtures are explicitly
no-model fixtures. These tests do not claim acoustic quality or model accuracy.
