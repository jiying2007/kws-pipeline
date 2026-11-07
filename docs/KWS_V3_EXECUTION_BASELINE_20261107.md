# KWS V3 execution baseline — 2026-11-07

## Purpose and scope

This is the execution handoff for the V3 research plan. It records the verified starting point and the next bounded actions. It does not itself authorize a model invocation, admit data, or claim acoustic/SSC305 success.

Scope: public `jiying2007/kws-data` and `jiying2007/kws-pipeline`; two Mandarin phrases only: `你好小窝` and `小窝小窝`. Research-only use. No Private repositories, no ChatGPT Work/Codex shared quota, no Scheduled/hourly polling.

## Frozen starting identities

- `kws-pipeline/main`: `f097f6667fb08fc16c4c0a4c7a3ee1518421f5f8`
- `kws-data/main`: `d4c2a39e5c358bdd0770c8d2f8695898c0f54110`
- Research runtime: CPython `3.12.16`, using the existing core/TTS/ASR role locks. Do not introduce a new Python compatibility matrix.
- Current pipeline main CI run `42329708937`: success. The frozen-model FAR nightly `42312402130`: failure; preserve as a regression observation, not as a new-model result.
- Existing open PRs #729 (pipeline) and #70 (data) are not presumed to be V3 blockers. Inspect their exact dependency before touching; do not duplicate or supersede their work without evidence.

## Evidence already completed — reuse, do not repeat

1. Balanced first-64 C1 augmentation is completed and retained. It used 96 eligible train rows (52 old + 44 new), 34 development rows, 4 uncertain zero-loss rows, 2,000 optimizer steps / 8,000 draws, and 402 final observations. The checkpoint and paired evidence are retained.
2. The fixed C1 3:1 old/new allocation experiment is completed and **not promoted**. Its change includes sampler composition/order as well as marginal quota; do not describe it as a pure 3:1 ratio ablation and do not repeat it.
3. Saved C1 posterior diagnostics show residual K1 weaknesses at both “你” and “好”; they do not establish that a decoder change alone can recover all misses. Reuse saved tensors/logits; no duplicate forwards.
4. Natural ASR evidence has 73 windows / 219 observations, 216 quality-usable observations, zero full-text quorum, zero known keyword-presence/absence labels under the frozen rule, and zero accepted natural-negative seconds. All 73 windows remain development. Reuse these observations; do not regenerate or relabel them as sealed test.
5. The first synthetic 64-cell source batch and its ASR observations are already accounted for. Do not regenerate those 64 WAVs or repeat their completed ASR calls. Remaining 192 cells are not automatically authorized: first establish expected information gain, rights, source identity, and a bounded execution declaration.
6. D2 unknown external-call history remains unknown. Never blindly replay unknown calls.

## Corrected research matrix

The first three active model candidates must reuse existing code/assets, not create parallel frameworks:

- Control: existing RNN64/CTC research baseline with exact checkpoint and vocabulary identity.
- Candidate A: existing C1 causal encoder + CTC, using the completed balanced64 checkpoint as a retained comparator; any new training must have a distinct pre-registered hypothesis and new immutable experiment identity.
- Candidate B: existing FSMN head/full-text CTC research path, with source/checkpoint/feature identities verified before execution.
- Candidate C: existing direct-KWS prototype is a conditional challenger only. It requires valid event/window supervision and its own direct-event calibration; clip-presence labels must not be fabricated into event windows.

Keep one heavy-model slot per shared host and the existing cross-repository lock. Do not start all models simultaneously on one constrained host. Start with one bounded feasibility run per eligible family, then retain at most two candidates for multi-seed confirmation. New BC-ResNet/CRNN architectures, loss sweeps, distillation and QAT are deferred until the current candidates identify a measured gap.

## M0 — baseline and execution ledger

- [x] Verify both repositories are public and accessible.
- [x] Pin the current main commits and active Python runtime.
- [x] Record completed C1, 3:1 negative decision, natural 73-window result and frozen nightly failure.
- [ ] Produce a single machine-readable V3 experiment ledger that references existing immutable artifacts; do not copy audio or duplicate registries.
- [ ] Reconcile the current D1/D2 queue and local heavy-model lock from the execution owner. GitHub source alone cannot prove a private/local process is idle.
- [ ] Confirm exact checkpoint, training-source hashes, admission receipts and available compute for each candidate.

M0 is not complete until the owner-side queue/lock and exact checkpoint assets have been read back. A documentation merge alone does not complete M0.

## M1 — labels and continuous-event evaluation

- [ ] Preserve the current full-text quorum consumer unchanged.
- [ ] Implement a separately versioned keyword-presence evidence view; retain `present / absent / unknown`, source PCM binding, ASR quality and per-recognizer provenance. No quorum relaxation based solely on “all ASRs missed the phrase”.
- [ ] Keep event timing as a separate label capability. CTC forced alignment/posterior alignment is diagnostic only unless independently validated against event annotations.
- [ ] Validate the new label protocol on a small blinded, stratified audit set covering both complete phrases, ordinary negatives, near-homophones, incomplete prefixes, clipped speech and ASR-incomplete outputs.
- [ ] Keep all 73 existing natural windows development-only. Any new audio window must be predeclared by source interval before recognition.
- [ ] Use the existing event scorer for continuous event FAR/FRR. Clip-presence mode and event-window mode must not be mixed.

M1 success means the new view has measured precision/coverage and explicit rejection reasons. It does not mean natural FAR is available until valid negative exposure exists.

## M2 — bounded parallel model experiments

For each eligible family, freeze: source and data manifest hashes, group split, vocabulary, frontend, initial checkpoint, training objective, seed, sample order, optimizer, step/draw budget, checkpoint-selection rule, C runtime and evaluation protocol.

- C1-CTC: no repeat of balanced64 or 3:1. New training requires a distinct evidence-backed hypothesis, such as a diagnosed label-quality or token-support change.
- FSMN-CTC: verify the exact source weights and feature contract; train only on admitted rows; preserve old and new cohort metrics separately.
- Direct-KWS: first prove event/window target validity and whole-vs-chunk state parity; otherwise keep it as software prototype only.

Use one seed for bounded feasibility. Only candidates with real retained checkpoints and valid evaluation proceed to three-seed confirmation. No mass architecture/threshold/weight sweep.

## M3 — candidate selection

Use a two-stage decision:

1. Validity gates: provenance, split isolation, numerical/runtime correctness, finite training, valid event semantics and resource feasibility.
2. Quality comparison: per-keyword FRR at calibration-selected matched FAR operating points, valid continuous negative exposure and one-sided confidence bounds, latency distribution, subgroup support, memory/CPU cost.

A candidate with insufficient independent exposure is `inconclusive`, not PASS. Do not use the sealed test set for iterative tuning. Preserve every negative result.

## M4 — R1 engineering and independent research

Only the selected development candidate proceeds to quantization/export and full C streaming implementation. Validate float C parity first, then INT8 separately, including chunk boundaries, discontinuity/reset, EOF, state persistence, event timestamps and worst-case processing time.

Independent human/final-AFE evaluation and physical SSC305 evaluation remain separate evidence lanes. Until actually measured, statuses remain `NOT_RUN`. The research targets (per-word FRR, continuous FAR with confidence bound, latency, distance/accent/noise slices and board resource/soak) must be frozen before collection.

## Event-driven orchestration rules

- Advance only on a verified terminal receipt and exact input hashes.
- Idempotency key: campaign + stage + input hash + recipe hash.
- A completed stage is reused. A failed infrastructure attempt is retried only under its own bounded retry rule. An unknown external call is quarantined until reconciled.
- Persist terminal evidence before enqueueing the next stage.
- One heavy-model slot and shared lock across both repositories; record runner-minutes, wall time, CPU/RAM and model-load cost separately.
- No new Scheduled/hourly polling. Use existing PR/push/workflow-dispatch and explicit stage-completion events; do not add an unbounded workflow_run chain.
- Only an actual process/job may be marked `in_progress`; plan/preview/queue declaration remains `planned` or `queued`.

## Immediate next actions

1. Read the execution-owner's D1/D2 campaign manifests, terminal receipts, PIDs, locks and budget. This is required before any model launch.
2. Read back the completed balanced64 checkpoint and 3:1 result manifests; verify their exact hashes and decision records.
3. Inspect existing FSMN and direct-KWS model/data contracts and map each to the common event scorer. Do not train until actual-label admission and a compatible evaluation cohort are proven.
4. Implement M1's new versioned keyword-presence view and blinded audit tests as a narrow PR; do not change the frozen full-text consumer or existing 73-window roles.
5. After M1 and M0 owner-side gates pass, launch one bounded feasibility experiment on the best-supported eligible candidate; then schedule the next family through the shared heavy-model slot.

## Status vocabulary

- `planned`: no execution declaration yet.
- `queued`: immutable declaration exists; no worker has started.
- `in_progress`: worker/process identity and start receipt verified.
- `completed`: terminal receipt, outputs and hashes independently read back.
- `failed`: terminal failure retained with classification.
- `unknown`: invocation may have occurred; do not retry until reconciled.
- `NOT_RUN`: external/physical evidence not collected.

No row may be promoted based on a PR merge, CI pass, generated clip count, ASR call count, or successful preparation alone.
