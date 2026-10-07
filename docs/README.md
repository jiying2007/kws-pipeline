# Documentation index

## Start here

- [Build and use the engine](../README.md#build-and-install)
- [Current product status](KWS_LANDING_STATUS.md) and [research roadmap](KWS_RESEARCH_PRODUCT_ROADMAP.md)
- [Reusable research modules and outcomes](../research/README.md)
- [Public research evidence in kws-data](https://github.com/jiying2007/kws-data/blob/main/docs/RESEARCH_INDEX.md)
- [Engineering tool index](../tools/README.md)
- [Complete offline keyword-set identity](KEYWORD_SET_IDENTITY.md): parsed vocabulary/policy identity; no deployment or training approval
- [Historical branch outcomes](../research/consolidation/history/BRANCH_OUTCOMES.md) and [reading older notes](../research/consolidation/history/READING_NOTES.md)

## Product contracts

Current machine-readable product authority is `configs/shipping.xiaowo.json`. The immutable promoted model is `model-749187ec1d66`, qualified for exactly `你好小窝` and `小窝小窝`. The frozen commercial candidate is `deployment-c20f3eb88e43`. The product remains `shipping_approved=false` until real-human final-AFE acoustic evidence and physical target-board evidence both pass and the explicit terminal shipping promotion succeeds.

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — runtime/offline architecture and hard bounds
- [`RUNTIME_CONFIG.md`](RUNTIME_CONFIG.md) — parameter contract, L0-L3 layers, ranges, tuning and change checklist
- [`CUSTOMIZATION.md`](CUSTOMIZATION.md) — framework L0/L1/L2 capability versus the current qualified two-word SKU
- [`EVALUATION.md`](EVALUATION.md) — continuous FAR/FRR and domain scoring
- [`INTEGRATION.md`](INTEGRATION.md) — `audio-pipeline`, final AFE evidence and application integration
- [`REAL_HUMAN_QUALIFICATION.md`](REAL_HUMAN_QUALIFICATION.md) — restricted real-Mandarin corpus, frozen final-AFE, no-retry exposure and Phase-A runbook
- [`TARGET_AND_SHIPPING_PROMOTION.md`](TARGET_AND_SHIPPING_PROMOTION.md) — Phase-B per-DUT/multi-DUT target qualification and Phase-C terminal shipping approval
- [`PERFORMANCE.md`](PERFORMANCE.md) — hosted and target-board performance contracts
- [`RELEASE_QUALIFICATION.md`](RELEASE_QUALIFICATION.md) — artifact-bound shipping qualification
- [`SYNTHETIC_TRAINING.md`](SYNTHETIC_TRAINING.md) — deterministic synthetic/domain loop
- [`CORPUS_IDENTITY.md`](CORPUS_IDENTITY.md) — byte-complete training/evaluation corpus identity
- [`TARGET_EVIDENCE.md`](TARGET_EVIDENCE.md) — machine-collected physical target evidence contract
- [`AUDIO_DISCONTINUITY.md`](AUDIO_DISCONTINUITY.md) — XRUN/route/clock reset semantics
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — traceable/rebuildable/bit-reproducible claims
- [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md) — layered regression and product evidence
- [`KWS_LANDING_EXECUTION.md`](KWS_LANDING_EXECUTION.md) — staged training diagnosis, algorithm comparison and product-evidence closure
- [`PRODUCT_MODEL_TRAINING.md`](PRODUCT_MODEL_TRAINING.md) — canonical governed model-training/preflight handoff, provenance and promotion boundaries
- [`DATASET_ITERATION.md`](DATASET_ITERATION.md) — dataset-driven iteration for the registered deployable model without changing release authority
- [`research/`](research/) — retained historical experiment/negative-result evidence; archived files are not current executable runbooks
- [`REPOSITORY_GOVERNANCE.md`](REPOSITORY_GOVERNANCE.md) — current repository/platform governance state
- [`GOVERNANCE_TARGET.md`](GOVERNANCE_TARGET.md) — enforced terminal governance target
- [`TERMINAL_HARDENING.md`](TERMINAL_HARDENING.md) — final software hardening contract

Commercial-candidate contracts:

- [`configs/parameter-contract.json`](../configs/parameter-contract.json) — single source of truth for every tunable parameter, its layer, unit and validation range; consumed by the C build and by the python tools
- [`configs/shipping.xiaowo.json`](../configs/shipping.xiaowo.json) — exact two-word product/model authority and remaining shipping evidence boundary
- [`configs/nightly.xiaowo-frozen-model.json`](../configs/nightly.xiaowo-frozen-model.json) — independent frozen-model synthetic regression namespace; no formal qualification seed fields
- [`commercial/afe-evidence.schema.json`](../commercial/afe-evidence.schema.json) — fail-closed final command-AFE evidence schema
- [`commercial/real-human-qualification.policy.json`](../commercial/real-human-qualification.policy.json) — frozen Phase-A sample, confidence, FAR/FRR/latency and critical-slice gates
- [`commercial/real-human-corpus.schema.json`](../commercial/real-human-corpus.schema.json) — private sealed held-out human corpus contract
- [`commercial/final-afe-adapter.schema.json`](../commercial/final-afe-adapter.schema.json) — private final audio-pipeline command adapter contract
- [`commercial/real-human-qualification-summary.schema.json`](../commercial/real-human-qualification-summary.schema.json) — aggregate Phase-A result contract; `shipping_approved` is always false
- [`commercial/target-qualification.policy.json`](../commercial/target-qualification.policy.json) — physical per-DUT and multi-DUT cohort target policy
- [`commercial/target-bundle.schema.json`](../commercial/target-bundle.schema.json) — controlled target evidence bundle profile contract
- [`governance/require_current_main.sh`](../governance/require_current_main.sh) — shared manual-promotion gate requiring exact current protected main and live terminal ruleset
- [`.github/workflows/far-nightly.yml`](../.github/workflows/far-nightly.yml) — immutable released-model nightly FAR regression with no training/qualification
- [`.github/workflows/deployment-release.yml`](../.github/workflows/deployment-release.yml) — immutable content-addressed SDK/model/evidence tuple; publication requires exact protected `main`
- [`.github/workflows/real-human-qualification.yml`](../.github/workflows/real-human-qualification.yml) — manual self-hosted restricted-data Phase-A workflow; raw/post-AFE WAVs are never uploaded
- [`.github/workflows/target-dut-qualification.yml`](../.github/workflows/target-dut-qualification.yml) — manual self-hosted single-DUT physical evidence gate; raw lab evidence stays controlled
- [`.github/workflows/target-cohort-promotion.yml`](../.github/workflows/target-cohort-promotion.yml) — immutable multi-DUT Phase-B cohort promotion
- [`.github/workflows/shipping-approval.yml`](../.github/workflows/shipping-approval.yml) — only terminal promotion path allowed to emit `shipping_approved=true`

## Repository layout

- [`src/`](../src/), [`include/`](../include/), [`cmake/`](../cmake/) and [`CMakeLists.txt`](../CMakeLists.txt): device library and SDK build
- [`configs/`](../configs/), [`keywords/`](../keywords/), [`models/`](../models/): current machine-readable product inputs and model registry
- [`training/`](../training/), [`eval/`](../eval/) and [`tools/`](../tools/README.md): offline preparation, evaluation and engineering utilities
- [`commercial/`](../commercial/), [`governance/`](../governance/): product-evidence and promotion contracts
- [`tests/`](../tests/), [`fuzz/`](../fuzz/), [`bench/`](../bench/): software verification
- [`research/`](../research/README.md): isolated reusable research code plus frozen source/evidence archives
- [`docs/research/`](research/): earlier narrative records; the current conclusions are in the research map above

The [workflow directory](../.github/workflows/) serves different purposes: ordinary
CI and model-free contract tests; retained-evidence diagnostics with explicit
admission; frozen-model regression; and separately governed training, release or
physical qualification. A dispatch entry or historical approval is not permission
to run it. Workflow names and test coverage are retained during navigation cleanup.
The newly retained historical acquisition definitions live under
[`research/consolidation/`](../research/consolidation/SOURCE_INTEGRATION.md), outside
the active workflow directory.
