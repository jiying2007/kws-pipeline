# Bounded CosyVoice3 generation gate

This directory contains runner-only execution code and stdlib-only tests. It does
not establish that the model fits, runs, or improves pronunciation. No runtime,
weights, inference, or network acquisition was used to build the harness.

## Integration contract

An outer controller must first enforce the approved public repository, standard
free runner, exact reviewed head, one admitted run/nonce, first attempt, overall
wall time, and ownership/cleanup of the complete process group. Inner workers
deliberately inherit that process group. The controller must write
`JOB/controller-state.json` with integer `baseline_free_bytes` before downloads or
installation, and own all new runtime, source, model and output files under JOB.
The controller's full-filesystem monitoring also covers temporary expansion
outside JOB. Do not run generation commands as detached processes.

Use these commands in order, with the exact dependency lock and successful
qualification receipt from dependency-prep. Python means the stdlib interpreter
for acquisition/staging and the qualified isolated interpreter for other steps:

1. `python -I -S pilot.py acquire-source --job-root JOB`
2. `VENV/bin/python -I pilot.py qualify-source --job-root JOB --runtime-lock LOCK --runtime-receipt RECEIPT`
3. `VENV/bin/python -I pilot.py acquire-model --job-root JOB --runtime-lock LOCK --runtime-receipt RECEIPT`
4. `VENV/bin/python -I pilot.py run --job-root JOB --runtime-lock LOCK --runtime-receipt RECEIPT`
5. After success or failure: `python -I -S pilot.py verify-output --job-root JOB --artifact-dir NEW_ARTIFACT_DIR`

Every operational CLI checks GitHub Actions, the exact public repository, Linux
x64, and run attempt 1. This is defense in depth, not a substitute for external
authorization and one-shot admission. Acquisition and generation claims are
exclusive-create files, never erased or reused. Every failure stops its stage.
There is no retry, seed search, model fallback, paid resource, or local path to
the original reference. All inputs use revision-qualified public URLs.

## Source, runtime, and model gates

- 53 source/license files, 416,785 bytes: only the retained reviewed import graph
  plus the two applicable licenses. All executable source files have both
  SHA256 and official Git blob identities. The source/Matcha revisions match the
  frozen plan. No git clone, source archive, install hook, or model snapshot runs
- Public reference: the one pinned GitHub clip, exact SHA256, Git blob and length
- Runtime receipt: status, lock SHA256, exact interpreter, isolated environment,
  unchanged complete distribution set, no network attempts and fresh pip check
- Source qualification is a separate supervised offline process. The real
  frontend-bound SoundFile adapter is numerically tested on 11 deterministic
  mono/stereo PCM cases, including 16→24 kHz. ORT sessions and torch.load are
  forbidden during this phase; all retained source modules are imported without
  constructing a model
- Model acquisition starts only after qualification and both actual/effective
  remaining disk budgets reach 6,635,020,288 bytes. All 12 model files are fetched
  sequentially to bounded .part files, hash-checked, then atomically renamed.
  LFS bodies use their SHA256; sidecars use official Git blob IDs. Receipts also
  record actual SHA256 for every file
- Stock BlankEN safetensors are retained. Extra files, spk2info, alternate model
  types, custom-code/attention config keys and custom tokenizer classes fail
  closed. No `weights_only=False` or arbitrary deserialization workaround is
  present. TORCH_FORCE_WEIGHTS_ONLY_LOAD is set

## Frozen inference

The original six phrase rows, seeds 610201–610206 and alternating paired order
are frozen in pilot-config.json and protected by a compiled-in SHA256. There
are exactly twelve possible calls. The first plain/inpaint pair is included in
that maximum. Runtime IDs for `[w]`, `[ō]`, `[ū]` must each be one registered
added-special-token ID, with matching encode/decode identity and multiplicity.
All whole-text IDs are saved before synthesis.

The stock AutoModel selects CosyVoice3 with only supported kwargs: load_trt=False,
load_vllm=False, fp16=False. No device or load_jit kwargs are passed. CPU/FP32,
24 kHz, model tensors, ORT CPU providers and stock RAS sampling are asserted.
The loader adapter binds file_utils before frontend import, then verifies the
actual frontend alias. Hugging Face constructors are restricted to the exact
local BlankEN directory with local_files_only=True and trust_remote_code=False.
Python network connections are blocked; offline environment flags and disabled
ORT telemetry apply before source/model imports.

Official add_zero_shot_spk is called once in memory with the unchanged public
prompt. No speaker cache is saved to the model. Immediately before each call,
random/NumPy/Torch seeds reset to that phrase's seed, after all initialization
and prompt feature extraction. Both arms use stream=False, speed=1.0 and
text_frontend=False. No frontend text normalization, duplicate target character,
phoneme substitution, sweep, crop, padding or post-hoc normalization is used.

## Resource enforcement and artifacts

The inner supervisor enforces 900-second startup, 300-second per-clip deadlines,
75-minute phase maximum, 12 GiB process-tree RSS, at least 2 GiB host available
memory, 8 MiB worker log ceiling and 128 MiB outputs-plus-generation-receipts.
Torch has four compute threads and one interop thread; both ORT pools have one.
The outer controller must additionally enforce job wall time and stop/clean its
whole process group after each phase. Acquisition is also bounded by immutable
file size, single-transfer timeout, free disk, 14,000,000,000-byte allocation and
whole-filesystem-delta envelopes, and the untouched 1 GiB reserve.

Each complete output has native float32 NPY, 24 kHz PCM16 and deterministic
16 kHz PCM16. Empty, nonfinite, over-20-second, clipped-range or silent output
stops generation without retry; overlength is never trimmed. Raw native bytes
may remain locally for a failed amplitude gate but are excluded from the public
artifact unless the complete clip passes validation.

verify-output uses only the standard library. It stages exact approved basenames
after validating clip ID, attempt, frozen text/seed/condition, hashes, NPY format,
finite amplitude, PCM headers, sample rates and exact durations. It emits all
twelve outcome rows, including attempted_failed, invalid_excluded and
not_attempted. Failure receipts are retained. Only validated complete clip trios
are included; source, model, dependencies, reference audio, caches and arbitrary
files are never recursively copied. Public artifact upload remains the outer
workflow's explicitly authorized action; this code never uploads anything.
Any additional controller/runtime technical receipts must be separately
allowlisted and rechecked against the 128 MiB final artifact limit.

## Local checks

`python -I -S test_pilot_contract.py` and
`python -I -S test_soundfile_adapter.py` pass 44 tests total, also under `-O`.
These use stdlib logic and fake modules only. Real fixture qualification,
dependency compatibility, model startup, memory/speed and audio quality remain
unrun until the admitted runner executes. No generated data is admitted for
training. An eventual acoustic conclusion must retain the frozen paired design,
three-ASR evaluation, reference confounds and exploratory interpretation.

The SoundFile adapter preserves the upstream Apache notice; LICENSE.CosyVoice
contains the applicable full license. Source/license bodies are acquired only
on the runner. PUBLIC-FILES.json is the publication allowlist for this directory.
