# Third phased attempt: acquisition design blocked

This is source-only preparation for the separately authorized third campaign
attempt. No workflow, execution release, or new source freeze is created here.
Neither previous activation may be reused. The already prepared fail-fast setup
remains dormant. No hosted run or network asset acquisition was performed by
this preparation.

## Preserved experiment

- Same 16 retained clips, Whisper small, fixed decoder, no training or model change.
- One free GitHub-hosted CPU job, at most 45 workflow minutes; existing stricter
  2400-second job and 1200-second setup budgets remain unchanged.
- Source and runtime locks are unchanged. In particular torch remains
  `2.12.1+cpu`, 192268841 bytes, SHA256
  `ae4bb28409f5370852bd71af221066236c38d647f780d9b0a7240c330a9c12df`,
  at the original exact URL:
  `https://download-r2.pytorch.org/whl/cpu/torch-2.12.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl`.
- Runtime acquisition, offline installation, `pip check`, and installed inventory
  must succeed before requesting the 483617219-byte model checkpoint.
- Any failure stops; no retry, mirror, version fallback, User-Agent spoofing,
  credential lookup, or access-denial bypass.
- Run 38105471896 remains a setup failure of unknown cause. Run 38107706348
  remains the separately retained wheel-19 torch HTTP 403 failure. Both have
  zero audio attempts. Neither history is rewritten or relabeled.

## Official pip candidate and specific missing guarantees

`locked-download-requirements.txt` is an inert requirements candidate generated
from every unchanged runtime-lock row. It uses exact direct URLs and SHA256
hashes, with no package index or floating version resolution. It is not an
approved command or transport implementation.

Official pip's documented CLI provides `--require-hashes`, `--no-deps`,
`--no-index`, `--only-binary=:all:`, `--no-cache-dir`, and version-check disabling.
Connection retries and incomplete-download restarts have separate controls:
`--retries 0` and, in versions supporting it, `--resume-retries 0`. The embedded
bootstrap pip version must be established before relying on these options;
unsupported options must fail closed, never be silently dropped.

The documented CLI does not expose either a redirect destination allowlist or
per-response/shared response-body byte quotas. This is the blocking limitation:

1. Checking only the initial URL does not stop pip following a redirect to an
   unapproved destination. `--trusted-host` is a TLS trust exception, not an
   outbound allowlist, and is forbidden here.
2. Validating final file size or hash happens after bytes have been acquired.
   File-size resource limits and periodic directory polling do not bound pip's
   network reads, temporary copies, buffered data, or cumulative body bytes.
3. `--timeout` is a socket timeout, not an absolute job deadline. Existing outer
   watchdogs must still cover the entire phase.
4. `--retries 0` alone is insufficient for pip versions that separately resume
   incomplete downloads by default.

The current downloader's accounting is explicitly selected response body bytes,
not all wire traffic; headers, unreturned error/redirect bodies, TLS/protocol
bytes, and internal buffering are excluded. This preparation does not broaden
that claim or claim that polling disk usage measures network transfer.

No private pip API patch is used. A wrapper around pip's vendored requests or
urllib3 would need a separately version-bound, adversarially tested interception
of every request, redirect, retry/resume path, and body-consumption path, including
error/redirect bodies. Mocking a high-level download call cannot establish that
coverage. pip explicitly does not provide a supported in-process library API.
A proxy that terminates TLS, a new trust exception, or changed network/security
settings is not an acceptable silent substitute.

## Admission remains blocked

Before a third activation can be prepared, a genuine pip transport must preserve
at least the existing approved HTTPS destinations and body-byte accounting
boundary, fixed original identities, 1536 MiB shared body budget, exact wheel
sizes and hashes, zero retries, and existing time budgets. Tests must exercise
unapproved redirects, oversize bodies, aggregate overflow, hash/size mismatch,
403, incomplete downloads, and phase failure preventing model acquisition.
Neither changing only a User-Agent nor running stock pip and checking afterwards
meets that bar. No claim is made that official pip will resolve the previous 403.

The safe completed part is dependency-first ordering, whose 47 offline contract
tests pass. Four additional candidate checks are embedded in the same
`test_contract.py` entrypoint, so all 51 run in existing CI in normal, `-O`,
and `-OO` modes without adding a workflow. Third admission/freeze/workflow preparation and activation wait on a
reviewed acquisition design; this record is not an execution approval.

## Official documentation reviewed 2026-10-11

- https://pip.pypa.io/en/stable/cli/pip_download/
- https://pip.pypa.io/en/stable/cli/pip/
- https://pip.pypa.io/en/stable/user_guide/#using-pip-from-your-program

These references describe current public CLI behavior. They are not evidence
that the fixed container's bootstrap pip has the same version or capabilities.
