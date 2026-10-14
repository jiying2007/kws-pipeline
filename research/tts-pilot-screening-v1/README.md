# New TTS pilot blind ASR screening

This module does not execute the consumed r5 campaign. It copies only the locked
model/runtime adapters and pure Docker isolation helpers. There is no old release,
claim, workflow or history admission. Root owns the new one-time workflow trigger.

## Interface / execution

1. Collect TTS outputs under `INPUT/tqp-qwen06/artifact` and
   `INPUT/tqp-qwen17/artifact`, retaining receipts and original FLOAT WAV files.
2. In an environment with NumPy and SciPy (no Torch required), run:

   `python research/tts-pilot-screening-v1/derive.py --plan research/tts-quality-pilot-v1/plan.json --artifact-root INPUT --output DERIVED`

3. In a fresh free public Ubuntu Actions job with existing Docker/cgroup v2, run:

   `python research/tts-pilot-screening-v1/host.py --manifest DERIVED/manifest.json --audio-root DERIVED --output "$RUNNER_TEMP/pilot-asr-$GITHUB_RUN_ID"`

The output directory must not exist. Never rerun this operation to recover
failures. No workflow_dispatch, automatic retry or publication is implemented.
Integrate the command into the *new* pilot workflow after artifact acquisition;
do not point any old r5 workflow/control at these inputs. Offline tests:

`python -m unittest discover -s research/tts-pilot-screening-v1 -p 'test_*.py'`

## Scientific interpretation

- Denominators: design 48; currently executable 32 Qwen; successful generated /
  decodable count separately. Each engine comparison retains its 16 design slots.
  Cosy blocked slots are an authorization/reference gap, not acoustic errors.
- Quality is a technical audio gate (finite/nonclipping/nonsilent/termination),
  **not** verified pronunciation or naturalness. Rates separately report technical
  audio quality, primary transcript agreement, target consistency and overall
  machine-screen qualification. Missing/failed slots stay in every fixed rate.
- Qwen3-ASR-0.6B and SenseVoiceSmall independently receive only opaque hash WAVs.
  Models see neither targets nor peer text. Only successful primary transcript
  disagreement selects Whisper-small. Errors do not authorize a retry or third
  recognizer replacement. Equal primary outputs are two-model consensus only.
- Third outputs never erase original disagreement. A 2/3 majority remains review
  tier. No row is human gold; no row is automatically training-admitted. High
  confidence candidate requires both primary exact target matches, decoder
  completeness and technical audio pass. This is a heuristic, not calibrated
  confidence. Human linguistic/naturalness review remains separate.
- Target normalization removes only Unicode punctuation and whitespace after
  NFKC. It does not convert homophones, map 小我 to 小窝, or pass hotwords/prompts.

## Budget / operational limits

- 48 maximum design clips, <=16/engine; at most 96 primary + 48 conditional third
  calls = 144 ASR slots. Currently <=64 primary + 32 third = 96 calls for Qwen.
  Duplicate identical WAV bytes may share one decode, but their design slots
  remain separate. No synthetic backfill. First actual call is the stage resource
  probe and is included; failure stops that family before the next batch.
- Sequential containers; original immutable model/runtime locks; 12 GiB RAM,
  no swap, 4 CPUs, 256 PIDs, read-only code/runtime, no network during inference,
  no capabilities, non-root, 300 sec/cell and 1200 sec load/setup guards. Each
  <=16 subset is capped at 4800 sec; output bound 64 MiB per container. Require
  >=14 GiB currently free RAM and 25 GiB disk. Public runner feasibility is not
  established by local unit tests; resource failure must stop, not upgrade to a
  paid runner. Reserve at most 360 job minutes, not an inference-time prediction.
- Acquisition uses retained pinned setup and download limits, and immutable
  official Python image. No model/container execution has been done locally.
- Suggested rollout: the two new TTS resource probes first; inspect their actual
  cost and quality before authorizing remaining TTS slots and ASR stage. Do not
  auto-start ASR merely because this code is merged.

## Known verification boundary

The local environment has neither Torch nor Docker. Tests cover accounting,
blind projection, fixed denominator, conditional third policy and command
construction/source staging. They do not establish hosted runtime success. The
SenseVoice adapter uses the original Docker process policy. It does not install
r5's additional process-level fork-denial seccomp filter, so the optional ffmpeg
availability probe may spawn a bounded process. Missing ffmpeg follows the
original FileNotFoundError branch. This is intentionally not a claim of identical
r5 process isolation; Docker no-network, capability, memory and PID limits remain.
Native sources/weights remain unchanged. Raw evidence is retained for review.

## Review fixes and public export

The fixed campaign design is `expected-plan.json`, SHA-bound by `campaign.py`.
All manifests must retain exactly 48 identified slots (16 each) and their exact
engine/target association. Derived TTS rows must match every frozen plan field,
model/revision, single attempt, native geometry and native float-value hash.
Deleting failed slots or substituting another plan is rejected.

Whisper emits atomic load/cell progress for the existing host watchdog. After a
hard process kill, primary results are reconstructed by the retained read-only
`terminal_outcomes` receipt validator; Whisper results are reconstructed from
per-clip completion/start/raw hashes bound to the input job and config. Aggregate
outcomes files are not authoritative. Finished clips remain successful, started
unfinished clips consume their attempt, and untouched clips remain not_run.

Public artifact export is a separate offline operation:

`python research/tts-pilot-screening-v1/export_public.py --host-root HOST_OUTPUT --derived-root DERIVED --output PUBLIC_EXPORT`

Upload only `PUBLIC_EXPORT`, never HOST_OUTPUT or a runtime/code directory. The
exporter writes explicit schemas for results, recomputed report and generation
receipt projections, plus hash-verified Qwen preset-derived PCM WAVs. No raw
setup logs, arbitrary JSON, checkpoints, source tree, cache, original FLOAT WAVs
or unreviewed native decoder payloads are exported. This operation only prepares
files; publication must remain within the parent's authorized workflow.

A fresh output directory is only a local collision guard. **Changing directories
is not global authorization for another campaign/run.** Root must bind the new
workflow to its durable first-trigger campaign guard before enabling execution;
this module does not authorize reruns or replenish consumed attempts.

CI entrypoint: `python3 -I -S -B research/tts-pilot-screening-v1/test_suite.py`
(and `-O` / `-OO`). It enforces a nonzero test inventory. No-site runs execute
22 stdlib contracts and explicitly skip the separate SciPy-dependent resampling
suite; a dependency-enabled run executes all 26 tests. These are test fixtures,
not new inference or audio-generation calls against any model.
For the required numerical CI job, run `test_suite.py --require-numeric` in the
pinned NumPy/SciPy environment. Missing dependencies fail immediately, and any
skipped test makes this full-suite invocation fail. Never use the no-site skip
as evidence that numerical derivation was validated.

## Optional explicit third confirmation

See `TRAINING_CONFIRMATION.md` for the inactive
`third_policy=training_candidate_confirmation` extension. The compatibility
default remains `disagreement_only`; expanded selection requires a new explicit
activation/admission field. It can produce actual three-ASR machine training
candidates within the same 96 current / 144 design slot ceiling, never human gold.
The updated suite has 40 numeric-enabled tests (36 stdlib contracts).
