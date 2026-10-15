# Bounded class-balanced memorization diagnostic (reviewed, execution enabled)

This is an exposed, machine-label research experiment, not product training,
independent evaluation, FRR/FAR measurement, human-gold evidence or shipping
qualification. No model, optimizer state, checkpoint, KWM, ONNX, training receipt
or reusable weights are written. Initial and learned tensors remain in memory;
only an initial-weight hash and scalar/token diagnostics are retained. Raw argmax token paths and input frame counts support independently recomputing collapse, blank fractions and class/macro rates. NLL and mean blank probabilities are runner observations, not independently recomputable without logits/weights.

The enabled plan has completed independent review. Real-recording optimizer calls remain zero at this source-publication checkpoint. The operator must publish the exact reviewed source, verify protected-main CI, then invoke the fixed campaign once.

## Fixed experiment

The reviewed candidate registry supplies exactly 16 recordings (5 KW1 = 你好小窝,
11 KW2 = 小窝小窝). `verify_inputs` regenerates its complete byte/hash/evidence
binding against four fixed archives before returning PCM in memory. Thirteen
historical ASR records are **ID joins only**, not retrospective proof of decoder
PCM; three newer records have same-PCM recognizer receipt chains. Neither group
is admitted through the production source resolver. This separate research
experiment does not add a production trust root or bypass production admission.
A future product training attempt must establish its own legitimate source root,
complete capture/admission and required evaluation roles. Never use the
production trainer's fixture option as a fallback for these real recordings.

Pair selection is deterministic: first common preset voice, then stable IDs,
one per class. The current pair is qwen06-05 (KW1) and qwen06-06 (KW2).
`plan.json` pins their full IDs. Preset identity is not a human-speaker identity.

- Existing TinyStreamingRNN, 32 features / 48 hidden / 5 tokens; existing log-mel
  frontend (400-sample windows, 320-sample hop), mono s16le 16 kHz scaled by 32768. No new architecture.
- Seed 20261015; Adam lr .003, betas .9/.999, eps 1e-8, no weight decay; clip 5.
- Pure CTCLoss(blank=0, reduction=none, zero_infinity=False), each NLL divided by
  its own input frame count, then fixed weights summed. No auxiliary losses,
  VAD, augmentation, warm start, adaptive search, retries or extra TTS/ASR calls.
- Pair: batch 2, exactly 300 updates; observations 0/50/100/200/300. Continue only
  if both full greedy CTC collapses equal their target at the final checkpoint.
- A and B: full batch 16, 600 updates each; observations
  0/50/100/200/300/400/500/600. Both restart from the identical original random
  initialization and optimizer settings. A each 1/16; B each KW1 1/10 and KW2
  1/22. Thus class balance is the only A/B objective change.
- At most 1500 optimizer calls. A reviewed fixed relative campaign_claim_path under one controlled --claim-root claims the campaign even if --output changes; an exclusive output directory additionally claims its artifacts;
  each call is atomically reserved before `step`. Failure consumes its reserved
  slot, records a minimal diagnostic and stops. Existing evidence is not
  overwritten, resumed or automatically retried. The eventual launch guard must
  permit only one run for this fixed plan; changing `--output` is not permission
  to rerun. No previous fit20/fit200 budget is reused.

Every checkpoint has per-record target/decode, raw and frame-normalized CTC NLL,
argmax blank fraction and mean blank probability, per-class complete rates and
macro rate. Every successful update records finite loss/gradient/updated-weight
checks. No chunked path is used: the whole-sequence frontend/model is evaluated.
These are training-set memorization observations, never generalization claims.

## Runtime and launch prerequisites

`runtime-candidate.lock.json` extracts the complete default Torch CPU dependency
closure from an existing locked research runtime: CPython 3.12.14,
Torch 2.12.1+cpu, ten wheels, 202843438 bytes. Historical screen32 ASR setup and
model-load records establish successful use of this Torch version, **not** a
successful installation or CTC backward qualification of this newly reduced
closure. Every wheel URL/size/hash is preserved. The runner rechecks installed wheel payload bytes (excluding generated RECORD and non-importable .data scripts), installed versions and distribution allowlist; numeric source modules execute already hash-checked bytes, with checked import roots. The registry input-pins and all dynamically executed source modules are checked before invocation. No new download, installation,
Torch import, numeric preflight or optimizer call was performed while authoring.

This is explicitly different from `training/requirements.lock` and its product
qualification (Python 3.12.11 / Torch 2.13). Do not claim that qualification.
No ASR/TTS model package or model weights are needed by this experiment.

Before enabling: independently review code, plan, candidate registry and source
verifier hashes; materialize all fixed source archives; use official pinned
CPython 3.12.14 on Linux x86_64; download each exact wheel once to a private
wheelhouse and verify size/SHA256 (no unbounded retries, no index resolution).
In a fresh isolated venv, install only the pinned wheels:

```
python -m pip install --no-index --no-deps --require-hashes --find-links WHEELHOUSE -r requirements.research.lock
python -m pip check
python numeric_preflight.py --repo REPO
```

The numeric preflight uses synthetic tensors, forward/backward only, zero
optimizer calls. It does not replace the actual source verification. Record
installation/verification results and actual runtime identity. Only after root's
one-run approval may the independently reviewed plan set execution_enabled true
and create its fixed research-budget directory under one private --claim-root. Never change that ledger root to retry the campaign. This local claim is not a cross-machine lifetime guarantee. The source plan hash must be externally pinned; a single-run launch guard may then invoke:

```
python run_diagnostic.py --repo REPO --artifacts ARCHIVES --wheelhouse WHEELHOUSE --output NEW_PRIVATE_OUTPUT --claim-root FIXED_PRIVATE_LEDGER_ROOT --plan-sha256 REVIEWED_ENABLED_PLAN_SHA256
```

No workflow is included. This iteration uses one controlled local research execution with a fixed ledger root after exact protected-main verification. Preserve command/source/plan identities, stdout/stderr, exit status and resource observations outside the result directory even if a prerequisite fails before the campaign claim. Do not change the ledger root or output path to retry. No secrets or paid resources are needed.

## Current execution preparation

The ten official pinned wheels were downloaded with matching byte sizes and SHA256 values and installed offline in a fresh isolated research venv. `pip check` and installed payload verification passed. Actual identity is CPython 3.12.14 / Torch 2.12.1+cpu on Linux x86_64, with CUDA unavailable. Synthetic frontend and CTC forward/backward preflight passed without any optimizer call or real input audio; independent CTC path enumeration also passed. Torch reports the absent optional NumPy bridge; this runner does not use it. These observations do not qualify the production training runtime.

## Verification performed

`python -m unittest discover -s research/kws-balanced-fit-diagnostic-v1 -p 'test_*.py'`
(and `python -O`, `python -OO`) checks collapse semantics, class weights, stable
pair selection, exclusive results/run claim and budget exhaustion. These are
stdlib tests, not numeric/model validation. `py_compile` checks syntax only.
The execution-disabled path fails before importing Torch.

This new logmel/lr/seed plan is not claimed to be a controlled comparison with any old fit or PCEN experiment.
