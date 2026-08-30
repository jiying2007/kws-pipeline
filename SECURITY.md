# Security policy

## Supported code

Security fixes target the current `main` branch. The project is pre-1.0 and does not maintain compatibility/security backports for older unreleased ABI/evidence variants.

## Reporting a vulnerability

Do not publish exploit details, malformed artifact samples that trigger memory corruption, or sensitive product recordings in a public issue. Use GitHub private vulnerability reporting when available; otherwise coordinate a private disclosure path with the repository maintainer before sharing sensitive material.

Public issues are appropriate for non-sensitive hardening requests, documentation gaps and already-fixed findings.

## Security-relevant boundaries

### `.kwm` and `.kwk`

Model and keyword-pack blobs are binary inputs and must pass strict ABI/version/size/layout/fingerprint/finite-value checks before use. Do not bypass `kws_model_open()` or `kws_keyword_pack_open()` for externally supplied artifacts.

Product update systems should authenticate/sign release bundles and verify approved release hashes before exposing artifacts to the runtime. Defensive parsing is not a replacement for an authenticated OTA/update chain.

### PCM input and discontinuities

`kws_engine_accept_pcm16()` consumes caller-owned PCM memory and performs no I/O. The caller must provide readable memory for the declared sample count and obey `KWS_MAX_PCM_BLOCK_SAMPLES`.

When the capture timeline loses continuity because of XRUN, route change, clock reset or suspend/resume, the caller must use `kws_engine_notify_discontinuity()`. Continuing to feed post-gap PCM into pre-gap acoustic state can create invalid semantic continuity even when memory safety is unaffected.

### Offline/hosted tools

`kws_wav`, `kws_board_bench`, training, corpus/evaluation and evidence tooling perform filesystem I/O and may allocate memory. They are engineering tools, not privileged daemons. Run them with least privilege and do not pass untrusted paths through privileged automation.

Training images used for shipping candidates should be immutable OCI-digest identities. `training/Dockerfile` does not fetch/upgrade Python packages from the network; the prebuilt training base owns the complete dependency tree. Real `torch_ctc` workflow integration uses digest-pinned `KWS_TRAINING_IMAGE`.

### Corpus identity

A path or manifest hash alone is not sufficient identity for acoustic evidence. v0.3 records both file SHA256 and decoded mono-16-kHz PCM16 SHA256 for training and evaluation WAVs. This detects renamed/rewrapped duplicates and detects a WAV replaced underneath an unchanged manifest/reference file.

Evaluation also verifies declared recording duration against real WAV frame count. This prevents a short negative file from being represented as a much larger FAR exposure.

Do not weaken these checks for convenience in final qualification.

### Dataset separation

Human qualification data must be audited for decoded-PCM and identity leakage. Speaker/session/source overlap across train/calibration/qualification invalidates the independence assumption. A final held-out recording must not be mined into hard-negative/false-reject replay and then reused as unbiased release evidence.

### Product-board evidence

Shipping resource evidence is accepted only through target evidence schema v2 with `evidence_class=product-board`. The tuple binds exact `sku`, `source_sha`, distinct builder/DUT identities, collector/station identity, the exact repository collector, board runner, model, keyword pack and board audio.

The retained raw measurement set is frozen through canonical `evidence-raw.jsonl`. Its rows are exact `{name, sha256, bytes}` identities and must equal runtime-soak + every additional raw artifact + the raw power file. Missing, extra, duplicate or substituted raw files are rejected.

The collector also requires `--attestation-verification`: a schema-v1 result from the controlled product trust layer that reports `verified=true` and binds the raw-manifest subject, collector, board runner, model and keyword pack. The target-evidence collector verifies that result; it does not create its own trust assertion.

Builder and DUT identity must be distinct. External instrument measurements must retain original raw files plus instrument/calibration identity. Runtime-soak CPU/RSS/thermal summaries are recomputed from embedded retained samples, so modifying summary fields alone does not create valid evidence.

A self-consistent manually typed JSON is not proof that a physical measurement occurred. Hosted fixtures only test the contract and must never be represented as product-board evidence for a shipping claim.

### Qualification evidence

`qualification_manifest.py` schema v2 independently re-hashes/reopens the release-candidate tuple, including:

- model/pack/token/config/checkpoint artifacts;
- model provenance schema v3 and actual training-corpus WAV/decoded-PCM identity;
- clean dataset audit covering the selected manifests;
- evaluation runner/references/detections plus actual held-out WAV identity/duration;
- target benchmark runner/board audio/summary;
- target evidence schema v2;
- exact evidence collector;
- canonical `--evidence-raw` manifest;
- exact `--attestation-verification` result;
- every selected `--raw-evidence` file;
- exact `--sku` and `--source-sha` identity.

`qualification_gate.py` validates cross-links, requires the matching SKU policy with `shipping_approved=true`, applies acoustic/statistical/resource thresholds and binds its schema-v3 result to the exact manifest/policy.

These SHA256 relationships provide **integrity and internal consistency**. External authenticity still depends on the approved attestation issuer/policy, controlled DUT/qualification infrastructure, authenticated release signing and OTA/update trust roots. A party able to replace every artifact and the trust policy can construct a new internally consistent bundle; cryptographic identity does not replace organizational trust management.

## Supply-chain controls

Formal source/SDK releases use deterministic packaging, SHA256SUMS, SPDX SBOM and GitHub attestations. CI pins GitHub Actions by commit SHA. The SDK is built twice independently and installed trees must be byte-identical before release publishing.

Repository rulesets/branch protection are a GitHub platform control separate from source code. The strict terminal-governance target is documented in `docs/GOVERNANCE_TARGET.md`; current platform configuration must be audited independently rather than inferred from repository files.

## Sensitive data

Wake-word qualification corpora can contain voices, household speech and device playback recordings. Do not commit private production recordings, personally identifying metadata, credentials or proprietary datasets to this public repository. Store private audio/raw measurement bytes in controlled immutable storage and retain only required cryptographic identities/approved references in public source control.
