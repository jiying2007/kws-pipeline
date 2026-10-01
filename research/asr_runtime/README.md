# Hosted ASR dependency / CPU qualification

This research lane qualifies a locked offline ASR Python environment on a standard
public GitHub-hosted Ubuntu 24.04 runner. It does not change embedded KWS code.
It retrieves/loads no target ASR/TTS/KWS weights, processes no user audio, and runs
no speech-model forward pass. Package imports may initialize bundled tokenizers
or statistical data; receipts explicitly distinguish those side effects.

## Current status and exact input identities

Static review has passed and source admission is enabled. Real execution remains
dormant until exact owner PR-body and label authorization. The actual runtime is
unverified: no real Docker, source build, install or ASR dependency import has run.
Ordinary PRs run only dependency-free fixtures. The owner-armed label job is the
**only** real-execution trigger.

The complete v2 PyPI metadata inventory is newly resolved, not represented as a
byte-identical recovery of an older inventory:

- `inventory.json` SHA256: `626b8c827ee63f95f07e828ae8351ef582197ab253cc68d80b29288679c796e1`
- 140 runtime packages: 135 upstream wheels and five source distributions
- 143 unique runtime/build inputs: 138 wheels and five source archives
- Exact unique input bodies: **3,060,297,109 bytes**
- Python 3.12.14 / Linux x86_64 / glibc 2.36 markers, Requires-Python and wheel tags
- SenseVoice numerical recipe pins NumPy 1.26.4; Librosa 0.11.0 and SciPy 1.17.1
  satisfy the re-resolved graph. FunASR[knf] is explicit. Do not mix an unverified
  torchaudio ABI: torchaudio is imported only if the exact lock contains it

The Docker Official Image metadata is independently hash-checkable:

- `docker.io/library/python@sha256:c90be507635af19768837aa7eeb2f4ce89a74d62962a335497b9df8edfb7f19d`
- Raw platform manifest: 2,325 bytes; raw config: 6,562 bytes
- Config/image ID: `sha256:bb31e660647633ecd0a394013306057121b4fd091e2bf214010558249c5e1eb9`
- Seven compressed layers plus config: **380,580,320 bytes**
- Exact bulk package + image descriptor budget: **3,440,877,429 bytes**
- Metadata/HTTP/checkout overhead is separate. Actual expanded image size is
  unmeasured, with a 2 GiB admission cap checked after the approved pull

Only the ten small build/source inputs have been downloaded for static review:
21,050,826 bytes. Their verified scientific audit is separate from the unchanged
metadata inventory, whose body flags describe its earlier metadata-only stage.
See [SOURCE_REVIEW.md](SOURCE_REVIEW.md). No layer bodies were fetched in preparation.

## One owner-approved label event

This is a bounded operational single-run authorization, not a claim that a
privileged owner can never deliberately authorize another run.

1. Publish a disabled draft PR containing only `research/asr_runtime/**` and the
   dedicated workflow. Obtain its actual PR number/head and finish static review
2. After explicit execution approval, the publishing owner enables the reviewed
   admission/build-plan flags and updates the PR source. The final head and
   source-manifest hash must be known before arming. A synchronize event alone
   cannot run the real job
3. Put exactly one `KWS_ASR_RUNTIME_APPROVAL=` JSON line in the PR body. Required
   fields are armed=true, exact PR number, base/head SHA, source-manifest SHA,
   the committed activation nonce, exact dependency/image/total bytes and scope
   `single_no_model_dependency_cpu_qualification`. It is parsed as data, never
   interpolated into shell code
4. The owner adds the committed `asr-runtime-qualify-v3` label. The job requires
   that precise labeled event, owner actor/author, public same repository, fixed
   branch `codex/github-asr-runtime-once-20261001`, run_attempt=1, exact source/
   changed paths/identities and all reviewed locks. Any mismatch fails before
   image/package downloads. Checkout and the exact public base-tree fetch are
   small source reads. No action writes, persistent credentials or secrets enter
   an execution container
5. The owner monitors the admitted run, then changes armed to false and removes
   the label. Ordinary pushes, PR opening/reopening, unrelated labels and manual
   reruns do not re-execute. Re-adding the label is a new consequential action
   requiring another explicit approval; no automatic retry is configured

The body values can be constructed from `admit_run.source_identity(repo_root)`,
the final GitHub head/base/PR identities and `locks/admission.json`. No dispatch
API, guessed future PR number, unsigned shell fragment or extra activation path
is needed.

## Execution sequence

1. Validate every exact official PyPI URL/size/hash, model numerical recipe,
   container target, raw OCI identity and independently reviewed build plan
2. Read the existing Docker daemon only. Retain an allowlisted early receipt of
   client/server versions, client hash, Linux architecture, cgroup support,
   storage driver and security options. Missing access fails; no sudo, Docker
   installation, daemon setting changes, new credentials or paid runner
3. Explicitly pull only the reviewed digest, verify official repository digest
   aliases, exact config ID/platform and actual expanded size. `create` always
   uses `--pull=never`. A tiny stdlib container verifies real UID, capabilities,
   NNP/seccomp, cgroup2 limits, input/root write failures and no external route
4. Fetch one copy of each exact dependency artifact, enforcing body length,
   deadline and SHA without redirects. Fully stream archive payloads on the host
   without extraction or execution. Check cumulative expansion/member caps,
   path/link safety, top-level wheel identity, RECORD and PEP658 METADATA hashes
5. In two fresh network-none containers, install only the five source-reviewed
   build tools and construct the five sdists from fixed, unchanged source trees.
   Verify declared/native/static/license payloads and byte-identical wheels.
   crcmod's upstream pure-Python fallback is explicitly rejected
6. Install the complete runtime closure once, including torch's declared
   CUDA-capable PyPI dependencies, using no-index, require-hashes, only-binary and
   normal dependency resolution. `--no-deps` is used only for individual wheel
   construction, never for runtime installation
7. Run pip check, exact actual version/import checks and tiny CPU FP32/BF16
   matrix, layer-normalization and convolution operations. Receipts bind input
   contracts, scripts, real package versions, CPU probe, resource peaks, exits,
   OOM state and cleanup. This cannot establish full-model BF16, ASR quality or
   latency, nor KWS device performance. Model execution is a separate stage

## Isolation, limits and failure behavior

Each container uses network none, read-only root/inputs, nonroot numeric user,
ALL capabilities dropped, no-new-privileges, Docker's default seccomp, private
namespaces, init, 10 GiB memory and equal memory-swap (zero swap), 2 CPU quota,
256 PIDs and a 512 MiB noexec/nosuid/nodev `/tmp` tmpfs. A CPU quota is not a total
CPU-time limit. Effective settings and immutable ownership/image/command/mount
identity are read back; container-local loopback is not called a socket ban.

A dedicated disk-backed `/work` avoids placing the full installation in RAM.
Its 16 GiB logical byte cap, 512 MiB output cap and 32 GiB host free-disk floor
are **sampled**, not filesystem quotas. Work scanning uses fd-relative nofollow
access, counts normal venv links without following them and tolerates disappearing
entries. Frozen outputs/inputs reject links and special files. Actual cumulative
archive expansion plus 2 GiB install headroom must fit scratch before install.
No home, credentials, user audio, models, host devices or Docker socket is mounted.

The controller has a 50-minute whole-qualification deadline inside a 60-minute
Actions job. Each container also has a bounded wall limit; SIGTERM is handled so
known containers reach cleanup. CLI process groups have an absolute deadline and
bounded pipe drain. The previously verified immutable container ID is separately
stopped/killed/inspected even when later diagnostics fail, with early and final
receipts; forced removal closes a remaining running-state uncertainty. Unknown
create results or never-verified ownership are left untouched and fail without
blind retry. Exit 137 is not presumed OOM. Failure cannot become a green skip.

Before/after input manifests and parent Python executable/system-package metadata
are compared. This does not claim the whole host is unchanged: expected Docker
image/container state changes are outside that narrow identity observation.

## Logs, retention and fees

Logs retain bounded JSON-escaped prefixes/tails, byte counts and stream hashes;
truncation is explicit. Individual receipts are at most 64 KiB, and emitted receipt
JSON has a 512 KiB aggregate limit. The job uploads no artifacts, wheel/source
bodies, venv, model/audio data or cache. No recurring schedule or paid runner is
configured. Public standard-runner availability does not imply all storage or
all runner tiers are free; any future artifact upload needs its own reviewed
size/retention/cost boundary.

## Dependency-free local validation

```sh
python3 -I -S research/asr_runtime/tests/test_qualify.py
python3 -I -S tools/test_inventory.py --root research/asr_runtime/tests --workflow .github/workflows/research-asr-runtime.yml
python3 -I -S research/asr_runtime/qualify.py --candidate
```

Fixtures use synthetic archives/commands and read frozen metadata; they do not
run Docker, build/install dependencies or import ASR packages. The unrelated C
build/full repository CI suite has not been run for this candidate.

Official references:

- [Docker Hub platform manifest](https://hub.docker.com/layers/library/python/3.12.14-bookworm/images/sha256-c90be507635af19768837aa7eeb2f4ce89a74d62962a335497b9df8edfb7f19d)
- [Pinned Dockerfile](https://github.com/docker-library/python/blob/f2c5d1b8a6adecb5b00b3c9331d4f863beade6b3/3.12/bookworm/Dockerfile)
- [Pinned SenseVoice numerical recipe](https://github.com/QwenAudio/SenseVoice/blob/586bbdb2b052f4fe19322581825228b6810a9fb3/requirements.txt)
- [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)
