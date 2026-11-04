# Prospective strict native-v2 Whisper capture

Local/private implementation only. No model was loaded, no new recognition was
run, and no input, transcript, raw archive or output was published while building
or testing this helper. Its tests are explicitly invented stdlib-only objects.
A passing test proves schema and state-machine consistency, not an authentic ASR
observation, source rights, or training admission.

## What is implemented

`whisper_capture.py` imports only the standard library. Its explicit `prepare()`
operation uses the unchanged, SHA-pinned `FixedWhisper` loader and its existing
native `observed_effective()` observer. The loader still checks the original
upstream tree, installed-package inventory, exact Whisper-small checkpoint,
CPU-only runtime and model dimensions. The upstream revision is
`86098128c0b4f24f0e2aa2994de830614b474227`; the checkpoint SHA-256 remains
`9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794`.

Preparation accepts no audio and does not call the frontend or decoder. It
constructs a real native `DecodingTask` to observe the resolved tokenizer,
initial tokens, blank/permanent suppression, sample limit, language/task and
actual model device/dtype. It produces **new** strict model/source lock bytes,
then binds that observed configuration to those locks before the first decode.
It never edits the old adapter, pins, diagnostic config, outputs or receipts.
The original loader's diagnostic schemas stay under
`original-loader-diagnostics/`, clearly separate from the new native-v2 chain.

The caller then freezes one policy containing all three actual prepared model
identities. `bind_policy()` verifies the exact policy identity, source bytes,
checkpoint and loaded configuration. No audio is accepted before this step.

For each `infer()`, the helper reads and hashes the complete bounded WAV itself,
checks mono 16 kHz PCM16 and at most 480000 frames, writes the strict start
**before** calling the retained decoder, then calls that decoder exactly once.
The retained decoder already records each actual greedy update before native
`finalize()`, which is important because finalize may append a non-native EOT.
The helper rechecks effective/source/asset identity and the persisted input,
trace and pre-finalize boundary before the shared `NativeEvidenceSink` can
write the strict raw output and successful completion. Failure poisons the live
helper; there is no retry, retrospective import, receipt promotion or repair.

The raw strict output embeds the original native projection, input capture,
all observed update steps and pre-finalize capture. It also binds the retained
native files by path, byte count and SHA-256. Full-source input processing and
natural decoder completion are separate checks. Acoustic verbatim completeness
always remains `unknown`, even after natural EOT.

## API and deployment inputs

The implementation has no implicit campaign launch or downloader. A separately
authorized bounded runner must stage these existing source closures:

- `retained_repo`: the unchanged source-screen32 Whisper directory, its sibling
  `source-screen32-v1/runtime_scope.py`, and `tools/asr_native_contracts.py`.
  Exact old file pins are in `PINS`; mismatch is fatal, not permission to update.
- `runtime`: the existing reviewed Python 3.12 runtime containing `source/`,
  `model/small.pt`, `asset-lock.json`, `installed-inventory.json` and the installed
  package tree. Missing runtime bytes must not be replaced by defaults.
- `sink_source`: `native_sink.py` from the shared Qwen native-capture directory,
  with sibling `generation_capture.py`. It must expose `NativeEvidenceSink`.
- `consensus_tools`: actual `speech_asr_consensus.py` and
  `speech_label_admission.py`. All those executed helper bytes are retained in
  the new source lock. Their independent code/provenance review is still needed.
- `executed_head_sha`: the actual executed repository commit, supplied by the
  runner and later independently checked by the resolver. Writing this value
  here is not proof that the value is authentic.
- A new private output directory, the frozen audio-only source/request identity
  map, and the three real prepared identities used to freeze the policy.

The retained kernel-scope verifier is called before model loading and before/
after each decode. It requires an unprivileged, read-only-root, network-disabled
container with its original CPU/RAM/PID limits. A runner must preserve the
original 1200-second load and 300-second per-clip deadlines and bounded output
retention. The supplied `worker.py` implements those per-operation deadlines;
callers using the lower-level API directly must supply them. Neither file
launches a container, establishes network isolation, installs dependencies,
creates a GitHub run, or sets a timeout on its parent process. Its `maximum` is an explicit single-attempt count capped
at 64; the caller must additionally enforce the campaign's total budget.

In an already authorized isolated worker, call:

```python
prepared = prepare(
    retained_repo, runtime, fresh_private_output, actual_executed_commit,
    sink_source, consensus_tools, recognizer_id="whisper-small", maximum=64,
)
# prepared.identity is the actual native-v2 policy row, also in identity.json.
# Coordinate all three zero-forward preparations, then freeze one full policy.
prepared.bind_policy(original_policy_bytes, canonical_policy_sha256)
receipt = prepared.infer(
    source_wav_path, opaque_id, independently_frozen_wav_sha256,
    source_id, unique_request_id,
)
```

The lower-level API is an in-memory prepare/freeze/infer interface. It does not
accept a saved configuration instead of a newly loaded runtime, and does not
deserialize a previous live helper. The CLI described below deliberately
reloads in phase two and compares the fresh native configuration to the already
frozen policy before recognition. It must run in a dedicated model process: imported
`common`/validator modules from other source paths are explicitly rejected.

## Exact retention layout

`prepare()` writes `identity.json`, strict `model-lock.json`, `source-lock.json`,
`native-config.json`, `native-dependencies.json`, a preparation receipt and the
exact source/dependency bodies under `bindings/`. It does not copy weights.
The model asset inventory hashes the actual loaded checkpoint. The tokenizer
is retained once in the source inventory, avoiding an ambiguous duplicate
asset/source match in the strict resolver.

After policy binding, `capture/` is the self-contained capture-member root:

- `recognizer-policy.json`, `model-lock.json`, `source-lock.json`,
  `native-config.json`, `native-dependencies.json`.
- Every source lock `files[].path`, including the dependency at
  `bindings/upstream/whisper/assets/multilingual.tiktoken`.
- Per opaque ID: `started.json`, `raw-output.json`, `completed.json` only after
  successful validation, and original native files under `native/`.

`native-dependencies.json` maps the config field `tokenizer_sha256` to that
exact relative member. When an independently authorized private retention
pipeline forms a bundle, it must use that member and the same archive identity
as `source-lock.json`. Strict resolver references have `archive`/`member` keys;
this local helper does not invent an archive name or trusted source root.
The executed source lock includes all copied helper/upstream bytes and the
runtime inventory/asset-lock bytes. Runtime package bodies are checked by the
unchanged loader against the inventory, not copied into this bounded capture.

Do not upload these private payloads to a public repository or public artifact
store. Source metadata publication, if separately authorized, is distinct from
publication of raw transcripts or complete ASR payloads.

## Verified here and remaining blockers

The stdlib tests check fresh prepare without forward, unchanged old diagnostics,
policy gating, exact common start/completion fields and file hashes, dependency
retention, private native trace binding, path safety, bad WAV/source/checkpoint
identity, config changes, duplicate attempts, and failure without completion.
All four Python files are parsed with the Python 3.8 grammar. The production
model runtime intentionally remains the retained Python 3.12 environment.

Run the local invented tests with `python -m unittest discover -s DIRECTORY`.
The same suite must pass under normal Python, `-O`, and `-OO`.

Still not established: actual prepared runtime measurements; real frozen
three-model policy; new source audio; a launched campaign and externally enforced total/output budgets; any new model output; independent source/asset/authenticity review;
source rights/lineage/allowed purpose; or registration of an independently
reviewed training trust root. No observed config, transcript or success field
is fabricated to close any of these gaps.


## Two-phase CLI worker

`worker.py` is an actual integration entry point, **not** authorization to launch
it. It has two separate phases using the same frozen source closure:

- `--phase prepare` loads and resolves the real native configuration without
  receiving any audio, source map or policy, and leaves the real identity at
  `/output/whisper-prepare/identity.json`.
- After the host has prepared all three identities and frozen one policy,
  `--phase infer` loads the model afresh, re-observes the real settings, requires
  exact identity/config equality with that policy, then decodes each blind clip
  once. It never reuses a saved config as if it were a loaded observation.

Common required arguments:

```
--phase prepare|infer
--retained-repo /code
--runtime /runtime
--output /output/whisper-prepare|whisper-infer
--executed-head ACTUAL_EXECUTED_COMMIT
--sink-source /code/research/qwen-native-capture-20261104/native_sink.py
--consensus-tools /code/tools
```

Both output subdirectories must initially be absent. The output mount itself
already exists. `--recognizer-id` defaults to `whisper-small`.

Inference additionally requires:

```
--input /input
--job-sha256 FROZEN_JOB_FILE_BYTE_SHA256
--identity-map /code/runtime-policy/identity-map.json
--policy /code/runtime-policy/recognizer-policy.json
--policy-sha256 FROZEN_CANONICAL_POLICY_SHA256
```

No extra `/policy` mount is introduced: the retained container mount contract
only admits `/code`, `/runtime`, `/input` and `/output`. The host freezes the
small policy/map in its readonly code stage between preparation and inference;
that stage must contain no TTS targets or peer output.

`/input` contains exactly `job.json` and `audio/`. The job retains
`blind-source-screen32-job-v1` and has 1–16 clips with exactly `audio_id`,
`audio_path`, `wav_sha256`; IDs are `clip-NNNNNN`, with unique content-addressed
`audio/SHA256.wav` members. The separate identity map has exactly:

```json
{"schema":"blind-asr-source-request-map-v1","clips":[
  {"audio_id":"clip-000001","source_id":"opaque-source",
   "request_id":"unique-request","wav_sha256":"FROZEN_WAV_SHA256"}
]}
```

Unknown fields, duplicate sources/requests and any target text fields fail
before model loading. The worker hashes the original map bytes into its final
private input binding; source/request IDs are then included in each strict
start and completion.

The worker writes the retained progress shape (`stage`, `index`,
`started_monotonic`) atomically to `/output/progress.json`. Only `load` with
index 0 and `cell` with indices 1–16 are emitted; completion leaves the last
valid progress untouched, matching the existing supervisor contract. It enforces 1200
seconds for load and policy checking and 300 seconds per clip, stops on the
first failure, marks the live object unusable, and preserves original partial
native evidence. A failed phase gets a separate `*-failed.json` terminal,
never a successful all-clips terminal or repaired receipt. The 27 invented
unit tests pass in normal Python and with `-O` and `-OO`, including mocked
CLI phase ordering, no-audio prepare, policy-before-decode, deadlines,
stop-on-failure, frozen job identity and blind input/map membership.


## Guarded preparation import support

`verify_prepare_imports(retained_repo, runtime)` verifies the unchanged retained
source pins and kernel scope, then the original source inventory, installed
package inventory and checkpoint/tokenizer/frontend asset bytes before importing
the official Whisper package definitions. It returns that module so an external
no-audio preparation worker can install its forward fence **before** calling the
actual retained model constructor. This helper does not call `load_model`, decode,
transcribe or a model forward. The original constructor still repeats all of its
existing checks.

When `research/native-asr-integration-20261104` exists in the supplied code root,
both `prepare_worker.py` and `zero_forward.py` are required source members and
are retained under `bindings/capture/`. Both prepare and inference include the
same closure, preventing phase-only code from escaping the strict source lock.
The import-order and closure tests use invented package objects, not ML imports.
The CLI is additionally tested with `python -I -B worker.py --help`.
