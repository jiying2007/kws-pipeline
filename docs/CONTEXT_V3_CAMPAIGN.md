# Context v3 generation, ASR and full2601 campaign

`tools/research_context_v3_campaign.py` closes all 72 declared requests in nine
ordered chunks of eight. Each chunk reaches native generation, independent
selective ASR, machine-weak evidence replay, full2601 projection and externally
acknowledged durable closure before the next generation job can start.

## Fixed execution scope

The create-only declaration binds the exact generation stage and queue, data
adapter and producer sources, pipeline/controller/Gate/ASR/projection sources,
selective policy, original vocabulary and token bytes, ASR setup receipt and
driver interpreter/package identity. The stage retains the reference assets,
original source-family split and complete plan ancestry. Its campaign-driver
registration and lock also exclude the standalone generation entry point.

The allocation is exactly nine generation jobs and at most nine ASR jobs:
72 TTS claims, at most 216 ASR claims, 90 generation plus 90 ASR runner minutes,
and one shared heavy-model lock. Every queue job has a ten-minute cap and each
model worker has 570 seconds. Persisted actual queue durations count toward the
180-minute allocation. Consumed claims and saved results are reported separately;
unknown duration is never reported as a measured zero-duration success.

The driver checks its 10,800-second monotonic allocation before each chunk and
reserves a full 600 seconds after each intent acknowledgement before launching
a new model job. Durable closure of already-consumed evidence may finish after
that launch deadline. Failed, timed-out, unknown and ambiguously interrupted
inference is never retried automatically.

## Durable phase sequence

Each phase uses the existing checkpoint `Gate` and requires the actual provider
save response before it can advance:

1. `before-generation`: the exact live intent, recipe, sources and queue.
2. `generation-complete`: authentic queue terminal and all current native data.
3. `before-asr`: native export, audio-only derivation, source map, request and queue.
4. `asr-complete`: native recognizer results before pure weak/projection work.
5. `chunk-complete`: weak evidence, full-text projection, coverage and receipt.

Capsules contain static dependencies and the current chunk. Previous capsules
remain separate immutable objects, linked by their descriptors, acknowledgements,
provider responses and phase markers. Earlier audio, future job outputs and
nested checkpoint ZIPs are excluded from each current chunk's package.

One live Gate instance owns pending intents for the invocation. A restored
intent, including an acknowledged intent without a visible claim, cannot
authorize a new invocation. Completed phases are reusable only after their
original queue receipts authenticate. A local ZIP alone is not a persistent
provider acknowledgement.

Dead generation or ASR claims are reconciled into unknown terminal receipts,
saved with exact receipt pins, then halted. Failed and timed-out jobs likewise
receive terminal capsules before stopping. A completed generation with zero
eligible native returns closes without ASR or invented weak rows. Failed-native
audio may be transcribed for audit while retaining zero training weight.

## Source-preserving full text

The existing preparation preserves original 24-kHz FLOAT WAVs and derives
audio-only 16-kHz inputs by its declared polyphase resampling rule. The source
map is prepared before ASR. Recognizer workers receive audio and IDs without
requested text, reference text or another recognizer's transcript.

Source-map, weak-view and full2601 work run in fresh no-model subprocesses. This
prevents cached modules from the generator's copied producer from masquerading
as the pinned weak adapter. Every current and preceding weak package is replayed
before projection, checking global source/PCM deduplication and the unchanged
train/development source-family split. Partial coverage remains explicit against
all 72 planned cells.

Ordinary speech uses its complete nonempty observed transcript as the CTC target.
Both keyword zeros require confirmed observed absence. Uncertain, failed-native,
OOV and infeasible rows retain their text/provenance with zero weights. The
campaign does not train a model or turn machine agreement into human truth.

## Commands

`declare` requires `--stage`, `--data-root`, `--asr-runtime`, `--vocabulary`,
`--fsmn-tokens`, `--output` and optionally the existing `--policy` path. Use the
pinned driver interpreter with NumPy, SciPy and soundfile; prepare generation
from that same frozen pipeline tree. Then call `run --campaign /private/campaign.json`.
`status --campaign /private/campaign.json` reports durable completion and actual
queue/call accounting. The `worker` actions are pure replay subprocesses.

Tests use real queue claims, dummy processes, actual capsule/ack formats and
explicitly fictional native/ASR evidence. They cover unacknowledged and saved
but unlaunched intent restart rejection, post-ack deadline exhaustion, dead
generation/ASR claims, live intent propagation, ASR closure before weak work,
zero exports and complete ordinary targets in fresh subprocesses. They perform
no model inference or downloads.
