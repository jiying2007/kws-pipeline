# Security policy

## Supported code

Security fixes target the current `main` branch. The project is pre-1.0 and does not maintain compatibility/security backports for older unreleased ABI variants.

## Reporting a vulnerability

Do not publish exploit details, malformed artifact samples that trigger memory corruption, or sensitive product recordings in a public issue.

Use GitHub private vulnerability reporting when it is available for this repository. If it is not available, contact the repository maintainer through GitHub first and coordinate a private disclosure path before sharing sensitive details.

Public issues are appropriate for non-sensitive hardening requests, documentation gaps and already-fixed findings.

## Security-relevant boundaries

The real-time library intentionally has a small attack surface, but callers still own the surrounding trust boundary.

### `.kwm` and `.kwk`

Model and keyword-pack blobs are parsed as binary inputs and must pass strict ABI/version/size/layout/fingerprint/finite-value checks before use. Do not bypass `kws_model_open()` or `kws_keyword_pack_open()` for externally supplied artifacts.

Even though loaders reject malformed formats, the library is not a sandbox for hostile model-generation pipelines. Product update systems should additionally authenticate/sign release bundles and verify the expected SHA256 from the product release manifest before exposing artifacts to the runtime.

### PCM input

`kws_engine_accept_pcm16()` accepts caller-owned PCM memory only and never performs I/O. The caller must provide valid readable memory for the declared sample count and obey `KWS_MAX_PCM_BLOCK_SAMPLES`.

### Offline/hosted tools

`kws_wav`, `kws_board_bench`, training and evaluation tools perform filesystem I/O and may allocate memory. They are engineering tools, not privileged daemons. Run them with least privilege and do not feed untrusted paths through privileged automation.

### Release evidence

`release_manifest.py` uses SHA256 and vocabulary fingerprints to detect accidental/malicious artifact substitution inside a qualification bundle. The manifest is provenance evidence, not a cryptographic signature. Product distribution should use the platform's authenticated update/signing mechanism in addition to these hashes.

## Sensitive data

Wake-word qualification corpora can contain voices, household speech and device playback recordings. Do not commit private production recordings, personally identifying metadata, credentials or proprietary datasets to the public repository. Keep only schemas, synthetic fixtures and reproducible hashes in source control.
