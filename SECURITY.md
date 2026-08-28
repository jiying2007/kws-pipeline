# Security policy

## Supported code

Security fixes target the current `main` branch. The project is pre-1.0 and does not maintain compatibility/security backports for older unreleased ABI variants.

## Reporting a vulnerability

Do not publish exploit details, malformed artifact samples that trigger memory corruption, or sensitive product recordings in a public issue. Use GitHub private vulnerability reporting when available; otherwise contact the repository maintainer through GitHub and coordinate a private disclosure path before sharing sensitive material.

Public issues are appropriate for non-sensitive hardening requests, documentation gaps and already-fixed findings.

## Security-relevant boundaries

### `.kwm` and `.kwk`

Model and keyword-pack blobs are binary inputs and must pass strict ABI/version/size/layout/fingerprint/finite-value checks before use. Do not bypass `kws_model_open()` or `kws_keyword_pack_open()` for externally supplied artifacts.

Product update systems should authenticate/sign release bundles and verify the approved release hashes before exposing artifacts to the runtime. The runtime parser is defensive, but it is not a replacement for an authenticated OTA/update chain.

### PCM input

`kws_engine_accept_pcm16()` accepts caller-owned PCM memory only and performs no I/O. The caller must provide readable memory for the declared sample count and obey `KWS_MAX_PCM_BLOCK_SAMPLES`.

### Offline/hosted tools

`kws_wav`, `kws_board_bench`, training and evaluation tooling perform filesystem I/O and may allocate memory. They are engineering tools, not privileged daemons. Run them with least privilege and do not pass untrusted paths through privileged automation.

### Qualification evidence

`qualification_manifest.py` is a provenance/integrity verifier for a release-candidate evidence tuple. It independently re-hashes the actual model, pack, token/config files, evaluation runner, references, detections, target benchmark runner and board audio; it also revalidates ABI/config/metric/board-statistic consistency. `qualification_gate.py` validates artifact cross-links, applies the explicit SKU policy, and SHA256-binds its result to the exact manifest and policy.

These SHA256 relationships detect substitution/inconsistency inside the retained qualification bundle, but they are **not a cryptographic signature or trust root**. Product distribution must additionally use the platform's authenticated signing/update mechanism. A malicious party able to replace every evidence file plus the approved policy can still construct a self-consistent unsigned bundle; authenticity must therefore come from the product release/signing system.

## Sensitive data

Wake-word qualification corpora can contain voices, household speech and device playback recordings. Do not commit private production recordings, personally identifying metadata, credentials or proprietary datasets to the public repository. Keep only schemas, synthetic fixtures and reproducible hashes in source control.
