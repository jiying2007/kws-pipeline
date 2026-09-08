# Documentation index

Current machine-readable product authority is `configs/shipping.xiaowo.json`. The immutable promoted model is `model-749187ec1d66`, qualified for exactly `你好小窝` and `小窝小窝`. The frozen commercial candidate is `deployment-c20f3eb88e43`. The product remains `shipping_approved=false` until real-human final-AFE acoustic evidence and physical target-board evidence both pass and the explicit terminal shipping promotion succeeds.

- `ARCHITECTURE.md` — runtime/offline architecture and hard bounds
- `CUSTOMIZATION.md` — framework L0/L1/L2 capability versus the current qualified two-word SKU
- `EVALUATION.md` — continuous FAR/FRR and domain scoring
- `INTEGRATION.md` — `audio-pipeline`, final AFE evidence and application integration
- `REAL_HUMAN_QUALIFICATION.md` — restricted real-Mandarin corpus, frozen final-AFE, no-retry exposure and Phase-A runbook
- `TARGET_AND_SHIPPING_PROMOTION.md` — Phase-B per-DUT/multi-DUT target qualification and Phase-C terminal shipping approval
- `PERFORMANCE.md` — hosted and target-board performance contracts
- `RELEASE_QUALIFICATION.md` — artifact-bound shipping qualification
- `SYNTHETIC_TRAINING.md` — deterministic synthetic/domain loop
- `CORPUS_IDENTITY.md` — byte-complete training/evaluation corpus identity
- `TARGET_EVIDENCE.md` — machine-collected physical target evidence contract
- `AUDIO_DISCONTINUITY.md` — XRUN/route/clock reset semantics
- `REPRODUCIBILITY.md` — traceable/rebuildable/bit-reproducible claims
- `TESTING_STRATEGY.md` — layered regression and product evidence
- `REPOSITORY_GOVERNANCE.md` — current repository/platform governance state
- `GOVERNANCE_TARGET.md` — enforced terminal governance target
- `TERMINAL_HARDENING.md` — final software hardening contract

Commercial-candidate contracts:

- `configs/shipping.xiaowo.json` — exact two-word product/model authority and remaining shipping evidence boundary
- `configs/nightly.xiaowo-frozen-model.json` — independent frozen-model synthetic regression namespace; no formal qualification seed fields
- `commercial/afe-evidence.schema.json` — fail-closed final command-AFE evidence schema
- `commercial/real-human-qualification.policy.json` — frozen Phase-A sample, confidence, FAR/FRR/latency and critical-slice gates
- `commercial/real-human-corpus.schema.json` — private sealed held-out human corpus contract
- `commercial/final-afe-adapter.schema.json` — private final audio-pipeline command adapter contract
- `commercial/real-human-qualification-summary.schema.json` — aggregate Phase-A result contract; `shipping_approved` is always false
- `commercial/target-qualification.policy.json` — physical per-DUT and multi-DUT cohort target policy
- `commercial/target-bundle.schema.json` — controlled target evidence bundle profile contract
- `governance/require_current_main.sh` — shared manual-promotion gate requiring exact current protected main and live terminal ruleset
- `.github/workflows/far-nightly.yml` — immutable released-model nightly FAR regression with no training/qualification
- `.github/workflows/deployment-release.yml` — immutable content-addressed SDK/model/evidence tuple; publication requires exact protected `main`
- `.github/workflows/real-human-qualification.yml` — manual self-hosted restricted-data Phase-A workflow; raw/post-AFE WAVs are never uploaded
- `.github/workflows/target-dut-qualification.yml` — manual self-hosted single-DUT physical evidence gate; raw lab evidence stays controlled
- `.github/workflows/target-cohort-promotion.yml` — immutable multi-DUT Phase-B cohort promotion
- `.github/workflows/shipping-approval.yml` — only terminal promotion path allowed to emit `shipping_approved=true`
