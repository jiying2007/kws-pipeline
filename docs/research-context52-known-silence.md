# Fixed context52 known-silence training

This is one new research learning point using every eligible training row in the
sealed context72 projection. It does not replace the original 17-row consumption
smoke, the original tiny8 experiment, or the fixed two-checkpoint development
diagnostic. Its entry point is `training/research_context52_known_silence.py`.

This document describes the fixed declaration, execution procedure and evidence
contract. It is not a record that the new training job has run or passed a quality
gate. Actual execution requires the new job's original claim, terminal result,
saved-output manifest and subsequent audit receipt.

## Declaration and interpretation

The prior settings were fixed on 2026-11-06 before the current paired development
results were consulted. The tracked plan is
`configs/research/context52-known-silence-20261106.json`, SHA-256
`87ef781b8c95bccb6158633a4c0d8d506b8e5351be38cdf38415bd18c163bda4`.
The earlier root decision, retained at
`/workspace/scratch/d18b42cd7071/runtime-c1-context-development/next-stage-fixed-contract-decision.json`,
has SHA-256
`08b37c8f71112265c6b20a4aa8827aa516c966deb3de736bfb41220ec555cdbe`.
Both are authenticated by declaration. The paired108 prerequisite must have a
completed hash-bound queue receipt and its real saved-logit audit must be verified;
its accuracy, loss and keyword counts do not select a setting or admission rule.

The stage starts from the original zero-step C1 checkpoint, SHA-256
`da7671fe71fef9ecd6822c747e3fe153b9ee563a6784c18eded90d3354486ecf`, and keeps the
complete 2601-symbol vocabulary, SHA-256
`104575dab3a9f7ed622d8fa29aa8a897e0590cca8aed9214b26ef4cd1c8b4e4b`.
It keeps seed 20261105, AdamW lr 0.001 / weight decay 0.01, clip norm
5, batch size 4, full-text CTC normalizer 1 and known-silence coefficient 1. The
original `research_c1_known_silence.train` numerical loop is called directly.
Durable accounting wrappers delegate each model and optimizer call once.

| Item | Fixed contract |
| --- | --- |
| Training | All 52 eligible original train rows, in projection order; 18 families |
| Composition | 17 K1, 17 K2, 18 ordinary speech negatives; original full text and weak weights |
| Exclusions | The two original failed-native rows retain zero weight and unknown labels |
| Development | All 18 rows in the original consumer's declared order; 6 separate families |
| Optimization | Exactly 2000 steps × 4 draws = 8000 training forwards |
| Prefix schedule | Each row's own visits rotate through 0, 200 and 500 ms |
| Draw coverage | First 44 rows: 154 visits; last 8 rows: 153 visits |
| Offset totals | 2696 / 2652 / 2652 training draws at 0 / 200 / 500 ms |
| Final readback | Train52 + dev18, once at each prefix: 156 + 54 = 210 forwards |
| Model construction | One C1 instance; at most one heavy model job concurrently |
| Total model forwards | 8000 training + 210 final observations = 8210; development contributes zero optimizer draws |
| Other calls | No before readback, ASR/TTS, threshold search or automatic retry |
| Time | Worker 540 seconds; one queue job and total queue budget of 10 minutes |

The original nonempty transcript remains the primary CTC target, including ordinary
speech negatives. Added PCM16 zeros supply only the auxiliary blank target: zero,
nine or 24 wholly added-zero feature windows. The 200/500 ms auxiliary term is mean
blank NLL from the **same** full-sequence forward. It is not multiplied by the weak
speech-label weight, and it does not label quiet original speech as silence.
Windows crossing into original speech are outside the added-zero auxiliary mask.
Unknown keyword targets retain `None`, mask `False` and weight zero. They are not
counted as negative examples. The two excluded native failures retain their
original `full_text=None`, false CTC/keyword masks and zero weights; they enter
neither the 8000 training draws nor the 210 final observations.

The complete 52-row roster, development18 roster, source/family/speaker and PCM
identities, both keyword known masks, labels, initial checkpoint, vocabulary and
source hashes are in the immutable recipe. The existing frozen consumer and
correction verifiers run in isolated standard-library processes. Existing sealed
sources and manifests are preserved; a changed numerical/frontend/reader source
still fails verification. The two narrow historical metadata differences retain
the same handling documented by `research-c1-context-development.md`.

The total budget stays at 2000 updates and 8000 draws while the roster grows.
The first 44 rows receive 154 visits and the last eight receive 153, whereas the
original eight-row correction used 1000 visits per row. Source membership and full
text change together with text lengths, weak weights and coverage. Consequently,
this is a fixed-budget learning point on a broader roster, and cannot isolate the
causal effect of data quantity as a single-variable data-size ablation.
Training and development are reported separately. The development families and
original PCM identities are disjoint from this training roster and the original
tiny8 set, but this development
set has already been observed in previous research. It is not an independent final
test. Weak machine transcripts are not human gold. Quality winner and generalization
pass remain null; no tiny8 accuracy gate is reused.

The final observations use the one fixed final checkpoint. There is no checkpoint
selection, threshold search, development-driven roster change or tuning. Complete
K1/K2 token-sequence presence in greedy CTC is a clip-level diagnostic; it is not a
timestamped wake-event decoder result, a natural-speech FAR denominator, or a claim
that any development row is an independently admitted negative example.

## Evidence and failure behavior

Each training forward has a durable `started` claim before entry and `completed`
record after a finite float32 result; its row identity, original split, family,
prefix and feature hash are bound. Each optimizer update has the same paired
accounting after exactly four completed training forwards. `consumption.jsonl`
retains all per-draw full texts, CTC masks/weights, primary and auxiliary losses,
per-step gradient norms and clipping contract. Readback also checks each individual
draw's total equals primary plus auxiliary, preventing errors that cancel inside
a batch. AdamW parameter order is independently bound to the old canonical order
and the exact zero-step checkpoint; every saved moment and step count is checked.

Full unquantized float32 features and logits are retained for **all 210 final
observations**, with complete labels, keyword known masks, PCM and checkpoint
bindings. The frozen declaration reports their exact raw data sizes in
`final_logits_payload_bytes` and `final_features_payload_bytes`. These count
`frames × 2601 × 4` logit bytes and `frames × 32 × 4` feature bytes over the complete
final roster, before serialization metadata. Use those recipe fields for storage
preparation, and preserve all original float32 values in the final artifacts.

The 8000 intermediate training logits and intermediate gradient tensors are not
retained. Training evidence consists of durable forward and optimizer ledgers,
per-draw consumption/loss and per-step gradient observations, plus the final model
state and complete AdamW state. Recomputing scalar loss decomposition and ledger
arithmetic is not a replay of every backward pass or optimizer update.

The final 2000-step checkpoint is saved and read back before the 210 observations.
A later evaluation timeout therefore leaves that checkpoint. A caught earlier
failure attempts to save a clearly marked partial checkpoint and retains all
existing claims, ledgers and partial artifacts. A hard process interruption can
leave only the durable evidence written so far. Any unmatched started claim is
unknown, never silently treated as an uncalled model or a completed update. No
failure is resumed or retried automatically; the existing queue claim prevents
reentry. A checkpoint or summary file alone does not establish successful job
completion; the original terminal queue result and audit remain authoritative.
Failure evidence supplies no quality decision.

The saved-tensor `audit` command constructs no model and performs no forward or
optimizer step. It authenticates the whole manifest, rebuilds native features from
the retained original PCM, checks every full float32 tensor, recomputes complete
K1/K2 sequence presence, blank and added-zero statistics, full-text CTC NLL and edit
counts, and checks the 8000/2000/210 call/consumption/state ledgers. It does not claim
that gradient tensors or the full training trajectory were numerically replayed.
It sets `complete_optimization_numerically_replayed` to `False`.
The audit loads the saved tensors and reconstructs features using the pinned Torch
runtime. It also acquires the canonical shared heavy-model lock, so it must be
scheduled separately from another heavy job even though its new model
construction, forward and optimizer counts are all zero.

## Commands and runtime

Use the fixed CPython 3.12.16 runtime. The historical metadata verifier also needs
its shared-library directory in `LD_LIBRARY_PATH`; the queue must inherit it.
The paths below are examples for one new stage identity. Choose unused preflight,
queue and receipt paths; create the parent directories before the create-only
commands. Run from the reviewed frozen source checkout. Declaration and
queue-request creation use metadata and the original isolated verifiers; they
create no model, acquire no heavy-model slot and initialize no queue.

```bash
export LD_LIBRARY_PATH=/workspace/scratch/84a81b0d7941/runtime-recovery/python/actions-3.12.16/lib
export KWS_STAGE_PYTHON=/workspace/scratch/84a81b0d7941/runtime-recovery/venvs/training/bin/python
cd /workspace/scratch/d18b42cd7071/context52-known-silence
"$KWS_STAGE_PYTHON" -I -S -B training/research_context52_known_silence.py declare \
  --diagnostic-output /workspace/scratch/d18b42cd7071/runtime-c1-context-development/queue/jobs/c1-context-development18/outputs \
  --diagnostic-manifest-sha256 8c5b14b8186206d56d7b0224b8f78b93e4291ab199bd897c1c9a168c5b746e2f \
  --diagnostic-audit /workspace/scratch/d18b42cd7071/runtime-c1-context-development/readback.json \
  --prior-decision /workspace/scratch/d18b42cd7071/runtime-c1-context-development/next-stage-fixed-contract-decision.json \
  --output /absolute/new/preflight/recipe.json
"$KWS_STAGE_PYTHON" -I -S -B training/research_context52_known_silence.py queue-request \
  --recipe /absolute/new/preflight/recipe.json \
  --python "$KWS_STAGE_PYTHON" --output /absolute/new/preflight/queue-request.json
```

### Save the prelaunch evidence before queue creation

The owner reviews the frozen source, exact recipe, exposure plan and input lineage,
then uses the existing `research_d2_checkpoints.create_capsule` mechanism to save
the prelaunch source/input/declaration/review evidence. Verify the capsule against
current sources and retain the actual storage provider response and acknowledgement.
The complete older dependency graph remains explicit; a local ZIP alone is not a
provider save. This persistence barrier is part of the execution procedure, not a
new CLI subcommand or an automatic storage check inside `run`.

Only after that saved prelaunch evidence is confirmed, initialize the single-job
queue and run its one claim:

```bash
"$KWS_STAGE_PYTHON" -I -S -B tools/research_queue.py init \
  --root /absolute/new/runtime/queue \
  --request /absolute/new/preflight/queue-request.json \
  --plan configs/research/active-plan-20261105.json \
  --heavy-lock /workspace/scratch/84a81b0d7941/runtime-recovery/roles/asr/heavy-model.lock
"$KWS_STAGE_PYTHON" -I -S -B tools/research_queue.py run \
  --root /absolute/new/runtime/queue --max-jobs 1
```

The queue records the claim, launch command, logs, elapsed time and terminal
result. Its generated worker command uses `run --recipe ... --output ...
--heavy-lock ...` without `-S`, so the pinned numerical dependencies are available.
The runner checks its inherited queue/job/recipe/output binding and canonical
shared lock before importing the model runtime. Direct unsupervised invocation is
not the supported launch path. `--max-jobs 1` bounds this invocation to one new
claim; failed or unknown claims do not create an automatic retry allowance.

The worker's 540-second deadline covers validation, training and final readback;
the queue supplies the outer 10-minute deadline. Model construction, 8000 training
forwards and 210 final forwards are bounded separately. No elapsed-time estimate
authorizes raising these limits if the job fails to complete.

### Audit the actual sealed result

After the original queue result reports `completed`, obtain its bound
`result["outputs"]["MANIFEST.json"]["sha256"]` value and independently
check the corresponding manifest bytes. Use that exact value below; the hash of
the older paired108 diagnostic is not the new training output's manifest hash.
Keep the audit receipt outside the sealed output tree:

```bash
export KWS_CTX52_MANIFEST_SHA256=REPLACE_WITH_ACTUAL_COMPLETED_OUTPUT_MANIFEST_SHA256
"$KWS_STAGE_PYTHON" -I -B training/research_context52_known_silence.py audit \
  --output /absolute/new/runtime/queue/jobs/context52-known-silence-2000/outputs \
  --manifest-sha256 "$KWS_CTX52_MANIFEST_SHA256" \
  --receipt /absolute/new/runtime/readback.json
```

The audit requires the pinned numerical runtime and shared lock. It does not
retrain or run the model. Preserve the actual command, return code and audit
receipt. Only a successful real audit can support a `verified` assertion. Terminal
storage must retain all 210 original float32 feature/logit artifacts, final and
initial states, complete ledgers, frozen source/input bindings and manifest, with
actual provider responses. Failure evidence remains separate from a completed
result and never receives an inferred success status.

Contract tests use only invented metadata/call fixtures. They do not import Torch,
construct models, create an actual queue or acquire the shared heavy-model lock.
