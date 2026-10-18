# Minimal retention and integration plan

Retain the four source cores, their synthetic contract tests, the three-tier research policy, and the anonymous evidence JSON. Retain `README.md`, this plan, validation metadata and the explicit SHA manifest with them. Do not copy execution directories, acquisition tools, model files, tensor outputs, individual annotations, or run logs.

Suggested repository root: `research/causal-kws-increment-2026-10-18/`. The core modules share one directory so their original imports remain valid. The local test command is standard-library-only. The constructor's architecture dependency is already maintained separately; integrate against its then-current public interface instead of duplicating an entire model/training runner here.

`SOURCE-RETENTION-CANDIDATE.json` is an offline proposed metadata addition. Its paths describe this candidate under the suggested repository root, not an already-pushed tree. It records per-file SHA/size and actual local contract-test scope. The eventual publisher must record the resulting commit and CI run only after those really exist. This batch is independent of any currently open integration branch.

Reuse scope:

1. The live head hook, merger and timing loss are exact copies. Event state is an exact class-source span. Compare `evidence/core-provenance.json` before making integration edits. Any adaptation requires new local identities and appropriate rerun tests.
2. The tier record is a small declaration, not an automatic labeling pipeline. A future trainer must separately bind admitted rows, source hashes, voice-lineage split and a predeclared weak-weight ablation. Publication grants no additional training permission.
3. Operating-point evidence is already-exposed development research. Keep B 0.70 and B 0.10 tradeoffs, synthetic-tail dependence, post-hoc OR status and unequal step-240 exposure visible.
4. The S aggregate is retained byte-for-byte from the reviewed anonymous export. Its review/receipt hashes are recorded separately. No old-S numerical reference parity is claimed.
5. Official source and license declarations have separate scopes. A missing legalcode snapshot digest is explicit; do not replace it with a fabricated or unrelated hash. No audio, transcript or model artifact is included for redistribution.

A future release still needs separately authorized data access, frozen runtime and model identities, real audio model execution, appropriate positive/negative evaluation, and target-device testing. This lightweight public asset set is intentionally insufficient for offline end-to-end training reproduction.
