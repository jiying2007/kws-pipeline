# Selective ASR v2 for the active D1 experiment

The initial eight generated clips and separate four paired-mode diagnostics
retain their completed three-recognizer evidence. New D1 batches use the fixed
policy in `configs/research/selective-asr-v2.json`, declared before expansion:

- Qwen 0.6B and SenseVoice recognize every clip.
- Whisper recognizes every primary disagreement, absent/incomplete primary,
  primary quality flag, or source-quality uncertainty.
- Otherwise Whisper audits a deterministic 20% of agreeing primary results.
  Compute SHA256 over UTF-8
  `kws-d1-whisper-agreement-audit-v2|20261105|PLAN_SHA256|GLOBAL_SOURCE_ID`;
  select when `int(hash, 16) * 5 < 2**256`. Batch-local clip numbers never enter
  this sample. Twenty percent is the sampling probability, not an exact quota.

Normalization matches `kws_data_contract.reference_text.normalize`: NFKC,
casefold, punctuation as word separators except apostrophes, whitespace collapse,
and removal of spaces only between CJK characters. Whole observed text remains
available. Distinct phrases, English word boundaries and contractions are not
collapsed into keyword spellings.

## Source and native boundaries

`active-asr-inputs-v1` still has exactly six top-level fields and audio identities
only. The unchanged `active_asr_worker.py` receives only this manifest, the model,
the selected opaque clip IDs and runtime/output options. Qwen has empty context
and automatic language, SenseVoice has no text/hotword prompt, and Whisper has
the existing explicit Chinese language with no prompt or prefix.

The data producer separately creates `active-asr-source-map-v2` by joining the
original archive member, frozen plan and generation receipts. It binds each
global source ID to original WAV, original PCM and derived PCM hashes. Native
generation failures that returned audio retain that raw waveform and explicit
quality reasons. They remain uncertain, regardless of any ASR agreement.
No output from a failure without returned audio is invented.

The routing file binds both primary result files, source map, policy, plan and
native worker identities. Its decision is recomputed from the native evidence
on every audit. The worker never receives the routing file or peer transcripts.
This is an exact command/input and callsite boundary, not an OS file-access trace.

## Bounded execution and restart behavior

`tools/active_asr_selective.py declare` creates an immutable request containing
all paths, hashes, model/runtime identities, source snapshots, CPU threads and
wall-clock budget. `run --request ...` requires the inherited heavy-model lease
from `tools/research_queue.py`. Each job allows at most 64 clips, four CPU threads,
600 seconds and three calls per clip; the actual expected matrix is smaller when
Whisper is unrequested. Schedule jobs sequentially with an explicit total queue
budget. The controller installs the native offline restriction before model
imports; children inherit the same filter, lease and process group.

One original attempt is permitted per output directory. Completed sealed output
is fully re-read without new model processes or file changes. An unsealed or
interrupted attempt is consumed and fails closed; it cannot reset its deadline
or automatically retry a possibly consumed model call.
If all native calls and the controller attempt finished but final metadata
sealing was interrupted, the `audit` subcommand can write only the missing seal.
Any existing inventory must remain byte-identical; no decoder is invoked.

The final v2 audit has explicit `requested_matrix` and `not_requested_matrix`
inside `state`. Missing requested results, surplus calls, duplicate identities,
changed native projections, incorrect worker receipts or changed routing prevent
completion. A persisted result with an incomplete decoder is an executed call,
but remains unusable as a weak full-text label. A crash with no requested result
is an incomplete evidence matrix even if Whisper produced another observation.
Selected models must also retain their actual native load receipts. The audit
checks complete Qwen/SenseVoice tensor descriptors against the recorded checkpoint
application and independent architecture schema, and checks Whisper's exact
checkpoint hash, implementation revision and dimensions. Unrequested Whisper
does not require or acquire a model-load receipt.

For an unrequested Whisper observation, consumers use the explicit
`status: not_requested`, `usable: false`, `text: null` state and retain the routing
hash. It is neither a failed observation nor a third vote. Two matching usable
whole-text observations can be machine weak evidence under the data contract;
they never become human gold or an independent acoustic completeness claim.

## Verification scope

Offline regression tests exercise the real controller and audit with fictional
native receipts, including identity-based sampling, incomplete primaries,
quality uncertainty, native projection tampering, extra calls, shared leases and
byte/inode/mtime-preserving completed resume. These tests make zero model calls.
The existing supervised-ASR and native-network tests cover the process and
kernel restriction mechanisms used here. Actual new D1 calls require a separate
immutable execution receipt; test success is not recognition accuracy evidence.
