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

Training images used for shipping candidates should be immutable OCI-digest identities. `training/Dockerfile` does not fetch/upgrade Python packages from the network; the prebuilt training base owns the complete dependency tree.

### Corpus identity

A path or manifest hash alone is not sufficient identity for acoustic evidence. v0.3 records both file SHA256 and decoded mono-16-kHz PCM16 SHA256 for training and evaluation WAVs. This detects renamed/rewrapped duplicates and detects a WAV replaced underneath an unchanged manifest/reference file.

Evaluation also verifies declared recording duration against the real WAV frame count. This prevents a short negative file from being represented as a much larger FAR exposure.

Do not weaken these checks for convenience in final qualification.

### Dataset separation

Human qualification data must be audited for decoded-PCM and identity leakage. Speaker/session/source overlap across train/calibration/qualification invalidates the independence assumption. A final held-out recording must not be mined into hard-negative/false-reject replay and then reused as unbiased release evidence.

### Target evidence

Target resource/thermal/power evidence must be generated/retained with `collect_target_evidence.py` or an equally controlled product collector. v0.3 qualification binds the exact collector and every declared raw evidence file.

External instrument measurements must retain raw files plus instrument/calibration identity. A self-consistent manually typed JSON is not proof that a physical measurement occurred.

### Qualification evidence

`qualification_manifest.py` schema v2 independently re-hashes/reopens the release-candidate evidence tuple, including:

- actual model/pack/token/config/checkpoint artifacts;
- model provenance schema v3;
- actual training-corpus WAV/decoded-PCM identity;
- clean dataset audit;
- evaluation runner/references/detections plus actual held-out WAV identity/duration;
- target benchmark runner/board audio/summary;
- target evidence schema v2, exact collector and raw evidence files.

`qualification_gate.py` validates cross-links, applies the explicit SKU policy and binds its schema-v3 result to the exact manifest/policy.

These SHA256 relationships provide **integrity and internal consistency**, not an external trust root. Product distribution must additionally use authenticated signing/update mechanisms. A malicious party able to replace every evidence artifact plus the approved policy can create a new internally consistent unsigned bundle; authenticity must come from the release/signing system and controlled evidence acquisition.

## Supply-chain controls

Formal source/SDK releases use deterministic packaging, SHA256SUMS, SPDX SBOM and GitHub attestations. CI pins GitHub Actions by commit SHA. The SDK is built twice independently in CI and the installed trees must be byte-identical before release publishing.

Repository rulesets/branch protection are a GitHub platform control separate from source code. The strict terminal-governance target is documented in `docs/GOVERNANCE_TARGET.md`; current platform configuration must be audited independently rather than inferred from repository files.

## Sensitive data

Wake-word qualification corpora can contain voices, household speech and device playback recordings. Do not commit private production recordings, personally identifying metadata, credentials or proprietary datasets to this public repository. Store private audio/raw measurement bytes in controlled immutable storage and retain only the required cryptographic identities/approved references in public source control.
