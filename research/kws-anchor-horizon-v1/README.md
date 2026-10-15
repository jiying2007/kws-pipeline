# Fixed 600-step horizon diagnostic (consumed; execution disabled)

The unique campaign completed with evidence PASS and scientific PASS: all 600 updates and 608 forwards completed; the first 300 updates exactly reproduced historical T; both full targets and blank prefixes passed at the sole primary endpoint R600. R400/R500 are descriptive only. See [closure](closure.json) and [original observations](observations/2026-10-15-completed-scientific-pass/README.md). The budget is fully consumed; no retry or restart is authorized. The current plan changes only execution_enabled back to false. Scientific PASS is limited to these two exposed machine-labeled training samples, without true recall, false-activation-rate, generalization, independent-evaluation or shipping claims.

## Historical reviewed source and preregistration

The following record describes enabled source 7ab25606c66b829668bb0b796b3debc0d93abd1a before its unique run. Its historical not-yet-executed and execution-enabled statements do not describe current execution permission.

# Fixed 600-step horizon diagnostic (reviewed source enabled; not yet executed)

Independent campaign `kws-anchor-horizon-20261015-v1`. This is a new bounded diagnostic chosen within the user's optimization request, not a user-authored explicit 600-step instruction and not reopening either closed campaign. Only training horizon changes. Original two exposed machine-labelled PCM samples, targets, seed, initialization, Adam, full 71/83 frames, original CTC plus lambda=1 prefix blank anchor graph, prefix lengths 23/10, raw greedy decoder remain fixed. No augmentation, VAD, cropped audio, ASR/TTS, exports, production changes or independent-evaluation claim.

One R arm: exactly 600 updates and 608 reserved model forwards. Checkpoints 0/50/100/200/300/400/500/600. Early success does not stop execution; R600 alone is primary: both full raw greedy targets must match and every prefix argmax must be blank. Intermediate checkpoints are descriptive, never a best-checkpoint selection.

Historical baseline is copied exclusively from the existing published raw archive, recording its hash and each projected member hash. Old global step301..600 means historical T local update1..300; it is NOT a previous T600 result. R updates1..300 compare exact typed per-step loss, gradient norm, CTC/anchor components and all sample components before optimizer reservation; five checkpoint complete sample objects including full log probabilities compare exactly after observation. Mismatch stops before any next optimizer call. Baseline history never counts as new evidence.

Exclusive immutable claim, forward reservations, optimizer reservations before calls, successful finite-update receipts and failure context are retained. Claimed/reserved execution is consumed on failure; no retries, resume, alternate arm or additional updates. Initialization is fresh and deterministic. The independently reviewed source is now enabled solely by changing execution_enabled from false to true; the disabled review plan SHA256 is 5a16fdd9707ca9e9b9e6f01889a133ea21ad14f102852cd9e45e0bba7cde051b. No numerical execution has occurred during integration. Exact protected-main source verification and a separately frozen single-run launcher are required before execution. `numeric_preflight.py` is static only despite its inherited name.

Validation uses only Python stdlib and never imports Torch. No runtime/model calls have been performed by this implementation task. Source manifest excludes itself.

Forward receipts verify the runner's declared ordered invocation ledger, not an independent proof against arbitrary runtime behavior outside this pinned source.

Failure at 600 does not establish convergence or architectural impossibility. Success does not rule out delayed alignment or reliance on silence; full-target versus missing-initial-token posterior margins can be descriptive post-hoc analyses of saved probabilities, never a changed decoder or primary endpoint.
