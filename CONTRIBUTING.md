# Contributing

`kws-pipeline` is an always-on embedded audio component. Changes are reviewed against product invariants, evidence integrity and real-time constraints, not only functional correctness.

Repository merge/release policy is defined in [`docs/REPOSITORY_GOVERNANCE.md`](docs/REPOSITORY_GOVERNANCE.md). The stricter platform-governance target is in [`docs/GOVERNANCE_TARGET.md`](docs/GOVERNANCE_TARGET.md). Product acoustic and physical-board evidence remains a separate qualification boundary.

## Required local checks

At minimum:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 tests/test_frontend_parity.py ./build/kws_feature_dump
python3 tests/test_dataset_audit.py
python3 tests/test_corpus_identity.py
python3 tests/test_keyword_compile.py
python3 tests/test_eval.py
python3 tests/test_run_corpus.py
python3 tests/test_board_bench.py --runner ./build/kws_board_bench
python3 tests/test_release_qualification.py
python3 tests/test_target_evidence.py
python3 tests/test_audio_discontinuity_contract.py
python3 tests/test_training_supply_chain.py
python3 tests/test_reproducible_sdk.py
python3 tests/test_terminal_docs.py
python3 tests/test_test_inventory.py
python3 -m py_compile tools/*.py training/*.py eval/*.py tests/*.py
```

CI additionally runs Clang static analysis, a C line-coverage gate, ASan/UBSan, `.kwm/.kwk` libFuzzer smoke, Cortex-A32 ARMv7 hard-float cross-build, two independent SDK builds with byte-for-byte install-tree comparison, the full synthetic/domain/FAR matrix and clean SDK consumer checks.

Do not add a new `tests/test_*.py` without wiring it into an official workflow. `tools/test_inventory.py` enforces this.

## Real-time data-plane invariants

Changes under `src/` must preserve these defaults unless an explicit architecture change is approved:

- no heap allocation in `kws_engine_accept_pcm16()` or its callees;
- no mutexes, condition variables, hidden worker threads, filesystem access or text conversion in the real-time library;
- fixed 16-kHz / 400-sample frame / 320-sample hop KWSP-v2 geometry;
- one call accepts at most `KWS_MAX_PCM_BLOCK_SAMPLES` and never partially consumes an invalid block;
- caller-owned engine arena with alignment reported by `kws_engine_required_alignment()`;
- caller-owned model blob remains read-only and valid for the engine lifetime;
- no global mutable runtime state;
- deterministic hard bounds for features, hidden state, vocabulary, keywords and Trie nodes;
- capture timeline gaps are represented explicitly through `kws_engine_notify_discontinuity()` rather than silently bridging acoustic state across missing samples.

The decoder must retain the structural CTC repeated-label rule. Adjacent identical token transitions require a blank-separated prefix state; nonblank and blank-separated Viterbi states must not be collapsed.

If a proposal needs to violate a real-time invariant, document the CPU/RAM/latency reason and update architecture, tests, performance expectations and qualification requirements in the same change.

## Frontend contract

`training/frontend_spec.py` is the dependency-free feature specification. Changes to window/FFT/mel/energy/normalization/PCEN behavior must update the reference contract and keep C/reference/Torch parity green.

Any performance approximation or future activation quantization requires measured target motivation plus parity/acoustic requalification; do not optimize only from theoretical MAC counts.

## Binary ABI and public API contracts

`.kwm` and `.kwk` are product artifacts. Format changes must not be smuggled into an existing ABI version.

- reject malformed/non-canonical inputs rather than guessing;
- keep reserved fields zero and validate them on load;
- keep vocabulary identity bound by the 64-bit fingerprint;
- bump the relevant ABI for incompatible binary-layout/semantic changes;
- update exporter/compiler/parser/tests/docs together;
- generated C keyword tables must satisfy the same vocabulary identity checks as `.kwk`;
- parser fuzz targets must stay sanitizer-clean.

A public SDK/API behavior change must also bump the software/package version when appropriate even if KWSP/KWKP layouts remain unchanged. v0.3 is an example: the deployable ABIs remain v2/v3 while the public discontinuity API and evidence schemas changed.

The project uses a hard-cut model: unsupported legacy formats/evidence schemas are rejected instead of accumulating compatibility branches.

## Training, corpus and evaluation invariants

- `train_ctc.py` requires the actual token vocabulary; do not reintroduce size-only vocabulary configuration;
- TSV and JSONL training manifests must resolve to real mono 16-kHz PCM16 WAVs;
- checkpoints must bind the canonical real training corpus by file SHA256 + decoded PCM SHA256 + frames and reject incompatible warm starts/exports;
- model provenance schema v3 must carry that real corpus identity;
- run `training/audit_dataset.py` before release training/qualification;
- final human qualification must require speaker/session/source identity metadata;
- CTC targets must be valid and alignable;
- `--head-only` requires an explicit compatible warm start;
- final qualification recordings must not be recycled into replay/model/threshold tuning;
- evaluation provenance schema v2 must bind every real held-out WAV;
- `references.duration_s` must equal the real WAV duration;
- release FAR/hour and FRR come from continuous real audio evidence, not isolated clip accuracy;
- hosted/synthetic FAR results remain regression signals only.

## Training supply-chain invariant

Shipping training should use an immutable OCI base referenced with `@sha256:`. `training/Dockerfile` must not fetch/upgrade dependencies from the network. The final training image digest must be recorded in shipping checkpoints with `--require-container-digest`.

Do not replace this with a floating image tag or an unrecorded local Python environment for a release candidate.

## Target evidence invariant

Final CPU/RSS/stack/thermal/power/soak evidence must be connected to a physical target and retained raw measurements. Use the repository collector or an equally controlled product collector; qualification binds the exact collector and raw evidence hashes.

A manually typed JSON that contains plausible values is not sufficient final evidence.

## Release qualification

A software change is not a shipping acoustic qualification. v0.3 release qualification requires:

- exact source/model/pack/config/checkpoint identity;
- model provenance schema v3 and real training-corpus identity;
- clean dataset audit;
- exact evaluation runner/references/original held-out WAVs/detections/provenance/metrics;
- real target-board benchmark summary;
- target evidence schema v2 + exact collector + raw evidence files;
- approved SKU policy;
- qualification manifest schema v2 and gate result schema v3.

See `docs/RELEASE_QUALIFICATION.md`.

## Release engineering

Formal release workflow must not publish unless the full hosted, coverage, sanitizer, fuzz, Cortex-A32 and reproducible-SDK gates pass. Release artifacts must retain SHA256SUMS, SPDX SBOM and GitHub attestations.

A released `vX.Y.Z` tag is immutable; never move an existing release tag to a different commit.

## Scope and style

Keep the device data plane small and C11-focused. Offline Python/hosted tools may use richer facilities, but those dependencies must not leak into the always-on runtime.

Prefer explicit validation, machine-verifiable evidence and testable contracts over conventions. Avoid transitional compatibility code in unreleased interfaces; update all callers/tests/docs in the same pull request.
