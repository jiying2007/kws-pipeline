# Grouped A/B diagnostic (consumed; execution disabled)

The unique 1200-update/1218-forward campaign completed: evidence PASS; scientific validation DESCRIPTIVE_ONLY. Both fixed step-600 training endpoints complete 1/12 targets (KW1 1/4; KW2 0/8), with blank prefixes passing. Exposed Vivian probe: A 0/4, B 1/4. No checkpoint selection, generalization or production claim. Budget fully consumed; no retry or restart. See [closure](closure.json) and [original evidence](result-20261016/README.md). Current plan changes only execution_enabled to false.

## Historical reviewed source and preregistration

The following text records the original enabled source before execution; its not-yet-executed and enabled statements do not grant current execution permission.

# Fixed grouped-balance diagnostic v1

This independently reviewed source package is **enabled for one new 1200-update campaign, not yet executed**. Integration changes only execution_enabled from false to true in the plan; the disabled review plan SHA-256 is 93504d23a6be34c4e0927dcab478cca0fa389bdb784b1aa543882ead31361661. Only the parent executor may launch once after verifying the exact protected-main source identity and a separately frozen single-run launcher. No model, optimizer, ASR or TTS was run to prepare this package. Synthetic tests never import Torch.

## Preregistered question and limitations

Compare uniform sample weighting A against class-balanced weighting B on a fixed 12-record training cohort: Serena 3, Uncle_Fu 5, Dylan 4, comprising 4 KW1 and 8 KW2. Vivian's 4 records (1 KW1, 3 KW2) form a preset-group-disjoint **previously exposed diagnostic probe**, not an independent test, human gold, production admission or proven speaker-independent evaluation. Historical exposure, machine labels, 13 historical ASR ID-only joins and three stronger PCM chains remain unchanged. Never use probe results to pick a model, threshold, step count or new retry.

A assigns each training record 1/12; B assigns KW1 1/8 and KW2 1/16. Each weight multiplies the complete per-record objective: (CTC NLL + lambda=1 prefix-blank NLL) / input frames. Probe descriptive loss uses the SAME macro-class weights for both arms (KW1 1/2, KW2 1/6), never backpropagated. Raw per-class and macro complete-match rates are reported for both arms.

Both arms reset the same seeded (20261015) TinyStreamingRNN state, use fresh Adam (lr .003), fixed pinned frontend/runtime, and execute exactly 600 full-batch updates. No early stop, extension, warm start, retry, resume, augmentation, VAD, threshold selection or exported reusable weights. Final states remain only in memory.

## Frozen prefix and identity

cohort-prefix-freeze.json freezes all 16 registry IDs, PCM hashes, targets, preset groups, lengths and all scalar PCM RMS values. Each prefix is the maximal initial run of complete 400-sample windows at hop 320 (25 ms / 20 ms at 16 kHz), with unwindowed normalized-PCM RMS strictly below 1e-4. This exactly matches the historical rule and frontend window geometry. A first frame at or above threshold yields prefix=0, anchor=0, and null prefix probability extrema. No trimming, per-class substitute prefix or selected range. Original four ZIP hashes, registry/verifier/input pins, model/frontend pins and runtime wheel lock are enforced before numerical execution.

## Strict event order and budget

New campaign: kws-grouped-balance-20261015-v1; old campaign budgets cannot be reused. The reviewed launcher must provide an empty canonical output and pre-create claim-root/research-budget. An exclusive, non-overwriting campaign claim is durable before model work.

A training observation at step 0, then updates 1..600 with observations after 50/100/200/300/400/500/600; same for B. Only after all 1,200 optimizer updates: one A endpoint probe then one B endpoint probe. Total: 1200 training + 16 training observations + 2 probes = 1218 model forward calls. Each call reserves and returns durably in strict order; optimizer calls similarly reserve and return, and incomplete events block subsequent ones. Evaluation uses eval/no_grad and verifies parameter/RNG preservation. There is only one model(x) call site, using the full selected batch. Alternative-target CTC computations reuse those logits.

All 1218 forwards retain every sample's raw log probabilities, actual frame length, original target CTC NLL (plus alternate KW1/KW2/empty NLL at the 18 observation/probe calls), CTC/anchor/total weighted losses, raw argmax and collapsed decode, prefix blank statistics and rates. payload-NNNN.json is authoritative; optimizer receipts hash-bind it rather than duplicating LP. Observation A-0000.json etc and probe-A-0600.json duplicate the corresponding payload for convenient reading. Failures preserve completed receipts and failure.json; process interruption still leaves pending reservations. No repair, retry or continuation is authorized.

## Bounded local evidence cost

Each training forward contains 1276 valid frames × 5 logits = 6380 numbers. Across 1200 steps that is 7,656,000 LP numbers. Using a conservative 32 bytes per JSON float gives about 245 MB for training logits; metadata and observations suggest less than 350 MB uncompressed overall. This is an estimate, not a compression guarantee. The runner enforces a 512 MiB cumulative directory-write ceiling (reserving 1 MiB for failure evidence). Each evidence JSON write has an enforced 1 MiB ceiling, well below a 16 MiB individual-file limit. Bundle delivery requires a supported large-file route or reviewed sharding; do not drop audit fields to fit an upload limit. Training does not compute alternate-target CTC again: its saved LP supports independent offline reconstruction. Only the 18 observation/probe calls compute alternative targets from existing LP. No wall-clock bound is claimed without running the experiment. Free local CPU only; new ASR/TTS=0.

## Validation

Run python numeric_preflight.py and python [-O|-OO] -m unittest -v test_contract.py. These checks only inspect static source/pins, synthetic PCM and mock durable receipt sequences. They do not establish numerical experiment success. Independent evidence validation must reconstruct CTC and aggregate metrics from saved LP without loading a model or creating extra forward calls. The enabled plan must be independently reviewed again, including its exact SHA-256 and source manifest. The author-reviewed disabled plan is preserved separately; the integrated plan has execution_enabled=true. No numerical experiment was executed during integration.
