# Explicit offline research mode

This separately loadable Linux ELF library activates the archived candidate for
explicit research callers. It does not change production build/defaults or promote
the candidate. The earlier strict development screen remains failed.

From repository root:

    python3 -B research/bounded-local-retention-2026-10-20/activation/run_checks.py --cc gcc
    python3 -B research/bounded-local-retention-2026-10-20/activation/build_research.py --mode bounded-retention-research-v1 --repo-root . --output NEW_BUILD_DIRECTORY

Use `--cc clang` for the other compiler. `patch`, `nm`, and `readelf` are required.
These commands compile and test invented inputs only. No audio, models, real
posterior inputs or scientific replays are read or run. The output path must not
exist. Source reconstruction is bound to the parent BUILD-IDENTITY.json and retains
the exact previously reviewed candidate bytes and floating-point compiler flags.

The facade exports only10kr_* symbols. kr_create requires the exact mode string
`bounded-retention-research-v1` and fixes threshold0.7. It cannot accept arbitrary
thresholds. Legacy constructors, reset, finish and diagnostic capture are hidden.
No new arithmetic policy is introduced. This isolates the fixed0.7 policy from
the base decoder's inherited arbitrary-threshold/default0 constructors.

API, with original types/signatures from copied kws3.h/kws3_retention.h:
- kr_create(mode, recording_id, &decoder), kr_destroy(decoder).
- kr_step_f32, kr_summary, kr_retention, kr_threshold.
- kr_storage_bytes, kr_event_bytes, kr_summary_bytes, kr_retention_bytes.

Always check every status, retain all events, and require summary.valid and
numerical_support_ok at terminal. kr_retention can return a last committed snapshot
after sticky-invalid failure; a successful getter is not a successful experiment.
No reset/finish/padding or EOF extension is exposed. Retention capture is disabled;
only aggregate committed snapshots are provided, not per-episode ledgers.

Measured native decoder allocation is533032bytes, but the ELF library also has
6029360bytes of TLS per participating thread, including the inactive diagnostic
buffer. The builder reports the actual ELF TLS size. Neither count is total RSS,
and the small state delta is not total deployment overhead. Instrumentation has
not been stripped because that would change the verified source identity.

Tests cover31 configuration checks,130 invented blank-row public API steps plus
NaN rollback/sticky failure. Blank rows do not exercise retention; the separate
parent run_synthetic.py retains33 policy cases/142assertions. Tests are not acoustic
validation, FAR, recall, paired CPU ratios or target-device benchmarks.

Five rows mean150ms nominal audio progression. The product extra-wait cap500ms is
NOT enforced or validated by this offline library. Do not use it as evidence of
wall-clock latency compliance. Scientific acquisition requires its own fixed
inputs/labels/plan, resource limits and independent review; this entry does not
reset or authorize consumed one-time experiments.
