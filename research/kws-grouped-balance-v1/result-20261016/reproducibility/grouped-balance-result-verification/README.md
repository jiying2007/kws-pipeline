# Independent grouped-balance result verification

This stdlib-only verifier reads pinned source and evidence bytes. It never imports
or executes Torch, the runner, model, optimizer, audio processing, ASR, or TTS.
All tests are synthetic. Existing horizon evidence and verification are untouched.

## Frozen diagnostic contract

- A and B: 600 fresh Adam updates each; identical initial hash, seed, model,
  optimizer settings, train batch and exact unweighted initial observations.
- Train: 12 exposed machine-labelled items, 4 KW1 + 8 KW2, with conservative
  Serena / Uncle_Fu / Dylan preset groups. Probe: 4 exposed non-test Vivian
  items, 1 KW1 + 3 KW2. PCM identities and conservative groups cannot overlap.
- A weights: each 1/12. B weights: KW1 1/8, KW2 1/16. CTC/input_frames and
  lambda=1 prefix blank NLL/input_frames receive the same sample weight.
- Training observations: 0, 50, 100, 200, 300, 400, 500, 600 for each arm.
  After both arms finish, exactly one A600 probe and one B600 probe; probe
  description uses the same macro weights for both arms (1/2, 1/6).
- Exactly 1200 successful updates and 1218 returned forwards are necessary for
  completion: 1200 training + 16 train observations + 2 probe forwards.
- No retries, resume, early stopping, extension, new ASR/TTS, or weight export.
  Probe never changes training, selects a checkpoint, or selects a model.

Completed evidence reports DESCRIPTIVE_ONLY. It reports fixed-step per-class and
macro rates, train/probe fit and prefix results, and paired B-minus-A differences.
Neither a positive synthetic endpoint nor real 4/4 probe result is a claim of
held-out evaluation, generalization, recall, FA/h, human gold, or product readiness.
A negative endpoint can still be fully valid completed evidence.

## Independent numerical checks

All 1218 forward payloads retain raw per-sample log probabilities. The verifier
independently computes float64 log-domain CTC for KW1, KW2, and empty sequences,
including repeated-label skip restrictions and direct empty-target blank sums.
Training stores its target-class NLL; observations store all three NLLs. All
stored NLLs, prefix blank loss, loss weights, normalized loss sums, first-index
argmax, collapse, first nonblank frame, prefix counts, blank probabilities, and
per-class/macro rates are checked. Zero-length prefixes require null min/max.
Step receipts bind original payload bytes by SHA256 rather than duplicating LP.

Frozen tolerances, unchanged from horizon:
- CTC NLL: 0.001 absolute + 0.00001 relative
- Derived arithmetic: 0.000001 absolute + 0.000002 relative
- Per-row logsumexp normalization: 0.000002 absolute

These are fixed before execution and may not be relaxed after observing results.

## Sequence and failure semantics

The result must be a prefix of the exact fixed forward event schedule, including
contiguous reservation, return and payload ledgers. A missing return forbids an
optimizer reservation. A missing payload, step or observation forbids the next
forward. At most one forward or optimizer reservation can be unfinished. No
probe event is accepted before both training arms finish. Failures and timeouts
report actual reserved/returned/succeeded counts and NOT_EVALUATED, never success.
If the next reservation fails before creating its file, failure position may name
only the immediately next scheduled event, and only after the prior event has a
complete return, payload and step/observation receipt. This covers A600 to B0 and
B600 to probe A600 without permitting skipped checkpoints or unfinished events.

Closed JSON schemas, duplicate keys, nonfinite numbers, booleans masquerading as
numbers, symlinks, hardlinks, special files, changed reads and report overwrites
are rejected. Single reads are limited to 16 MiB; total evidence to 512 MiB.
The reviewed runner imposes the tighter 1 MiB JSON limit and preserves room for
its failure receipt. These are evidence limits, not additional numerical budget.

## Root-owned prelaunch freeze

Root supplies an external 8-key trusted freeze; fixtures do not generate a
production freeze:

- schema = grouped-balance-verifier-trusted-freeze-v1
- plan_sha256 (final enabled plan)
- launcher_sha256 (final enabled launcher)
- runner_sha256
- reviewed_source_commit (verified public 40-character commit)
- reviewed_source_pins (complete reviewed relative-path SHA256 map)
- tolerances_sha256
- verifier_sha256

Retain its pre-execution hash outside evidence. It does not authenticate itself.
Source closure and runtime lock are independently hash-checked, while remote
publication identity and CI remain root responsibilities.

Fixed sibling paths: grouped-balance-integrated, grouped-balance-result,
grouped-balance-campaign-ledger, grouped-balance-execution-evidence, and
launch_reviewed_grouped_balance.py. Outer receipts bind the fixed command,
3600-second CPU/wall limits, 8 GiB address space, 16 MiB file limit, 512 MiB
runner evidence limit, public source commit, launcher, logs and process outcome.

Run with absolute paths:

    python -I -B grouped-balance-result-verification/verify_result.py \
      --repo grouped-balance-integrated --result grouped-balance-result \
      --campaign-claim grouped-balance-campaign-ledger/research-budget/kws-grouped-balance-20261015-v1.claim.json \
      --process-result grouped-balance-execution-evidence/process-result.json \
      --trusted-freeze grouped-balance-result-verification/trusted-freeze.json \
      --report-dir grouped-balance-result-acceptance

Exit 0 means completed valid evidence, regardless of observed accuracy. Exit 2
means partial or rejected evidence. Reports use exclusive creation.

Tests: python [-O|-OO] -B -m unittest discover -s grouped-balance-result-verification -v

Receipts and LP checks establish consistency with the reviewed source, not
independent hardware or weight-state attestation. Preset group separation is a
conservative proxy, not proof of distinct speakers. No production authority is
created by passing this validator.
