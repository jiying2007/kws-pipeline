# LR-stability independent evidence adapter

This is a small stdlib-only adapter. It pins and reuses the existing grouped
validator at SHA256 cf974939751107b8e6e6c15d34bff1e5830e173a1a45cb3c34d1642681d36bcc.
It does not import a runner, Torch, model, optimizer, audio, ASR or TTS. The shared
module provides safe I/O, strict JSON/type validation, log-domain CTC, blank-loss
checks, and unchanged numerical tolerances. The new code supplies only the new
fixed source contract, single-arm ledger, history/LR gates and outer bindings.

Contract: one fresh A arm, seed/model/Adam/batch/weights/prefix objective unchanged;
600 optimizer updates, 608 forwards, observations at 0/50/100/200/300/400/500/600.
No probe, retries, resume, early stop, extension or exported weights. LR is .003
for updates 1..500 and .0003 from update 501; the optimizer is not recreated.
The transition receipt requires all 500 steps and observation500 to be complete.

All saved LP payloads are checked using independent float64 CTC/blank loss.
Every first500 training payload and every observation through500 compares exactly
with the accepted historical A payload after removing only the campaign-specific
schema name. Every successful step through500 compares exactly after excluding
its newly added LR/replay flags and the payload digest (the new schema changes
that digest). The new digest is separately checked against actual payload bytes.
Initialization is exactly the historical hash/seed. A mismatch must stop before
the next optimizer reservation/event and have a consistent failure context.

The history is read-only grouped-balance-result, checked against its accepted
manifest 4260c936d8249c59f2d5b270d40274b5f2b29bb45d930dd31c338b240d666b25.
The pinned public archive manifest binds that same manifest. Root separately
verifies the historical publication at PR589/main
f86ffb57234c1083c03d25c4ae89bfdbac2f733c. Equality establishes saved, visible
numerical-trajectory reproduction; it cannot independently prove unsaved Adam
state or parameter tensors at every step. Source/receipts are not hardware
attestation.

The pinned historical A600 LP observation is independently checked, and the
summary includes its endpoint plus fixed new600-minus-historicalA600 per-class
and macro differences. Only predeclared step600 is the endpoint. Complete evidence is DESCRIPTIVE_ONLY,
including negative accuracy results; partial/timeout failures are NOT_EVALUATED.
The 12 exposed machine-labelled training items support no generalization, recall,
FA/h, human-gold or production-readiness claim. This schedule was proposed after
seeing historical results and is labelled as such; it is not a held-out test.

The root-owned 8-key freeze has schema lr-stability-verifier-trusted-freeze-v1,
plan_sha256, launcher_sha256, runner_sha256, reviewed_source_commit,
reviewed_source_pins, tolerances_sha256, verifier_sha256. Use the unchanged shared
tolerances digest b701856f6bae2c93aa377cece1549a4afcab03921940dbc49ece6b3ddc01cf9b.
The adapter itself pins its shared-code dependency. Never regenerate a production
freeze from a result, or loosen a tolerance after seeing it.

Fixed siblings: lr-stability-integrated, lr-stability-result,
lr-stability-campaign-ledger, lr-stability-execution-evidence,
launch_reviewed_lr_stability.py. The shared grouped verifier directory and
accepted historical result/manifest remain alongside them. Outer limits are
3600 seconds CPU/wall, 8 GiB address space, 16 MiB per file and 512 MiB evidence.

Example (absolute paths recommended):

    python -I -B lr-stability-result-verification/verify_result.py \
      --repo lr-stability-integrated --result lr-stability-result \
      --campaign-claim lr-stability-campaign-ledger/research-budget/kws-lr-stability-20261016-v1.claim.json \
      --process-result lr-stability-execution-evidence/process-result.json \
      --trusted-freeze lr-stability-result-verification/trusted-freeze.json \
      --report-dir lr-stability-result-acceptance

A preflight failure with no durable intent/claim is conservatively rejected,
not accepted as proof of zero or consumed budget. Preserve its failure/outer logs
for root review; this adapter does not add a special preflight acceptance branch.

Reports are exclusive-created. Exit0 means valid completed evidence; exit2 means
partial or rejected evidence. All tests are synthetic or read-only hash/source
checks, and never spend model/optimizer budget:

    python [-O|-OO] -B -m unittest discover -s lr-stability-result-verification -v
