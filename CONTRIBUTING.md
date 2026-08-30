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
python3 tests/test_false_reject_mining.py
python3 tests/test_keyword_compile.py
python3 tests/test_eval.py
python3 tests/test_statistical_bounds.py
python3 tests/test_run_corpus.py
python3 tests/test_board_bench.py --runner ./build/kws_board_bench
python3 tests/test_release_qualification.py
python3 tests/test_runtime_soak.py
python3 tests/test_target_evidence.py
python3 tests/test_audio_discontinuity_contract.py
python3 tests/test_training_supply_chain.py
python3 tests/test_reproducible_sdk.py
python3 tests/test_terminal_docs.py
python3 tests/test_test_inventory.py
python3 -m py_compile tools/*.py training/*.py eval/*.py tests/*.py
```

CI additionally runs the complete synthetic/domain/FAR matrix, Clang static analysis, a C line-coverage gate, ASan/UBSan, `.kwm/.kwk` libFuzzer smoke, Cortex-A32 ARMv7 hard-float cross-build, two independent SDK builds with byte-for-byte install-tree comparison and clean SDK consumer checks.

Do not add a new `tests/test_*.py` without wiring it into an official workflow. `tools/test_inventory.py` enforces this.

## Real-time data-plane invariants

Changes under `src/` must preserve these defaults unless an explicit architecture change is approved:

- no heap allocation in `kws_engine_accept_pcm16()` or its callees;
- no mutexes, condition variables, hidden worker threads, filesystem access or text conversion in the real-time library;
- fixed 16-kHz / 400-sample frame / 320-sample hop KWSP-v2 geometry;
- one call accepts at most `KWS_MAX_PCM_BLOCK_SAMPLES` and never partially consumes an invalid block;
- caller-owned engine arena and read-only model blob;
- no global mutable runtime state;
- deterministic hard bounds for features, hidden state, vocabulary, keywords and Trie nodes;
- capture timeline gaps are represented through `kws_engine_notify_discontinuity()` rather than silently bridging acoustic state across missing samples.

The decoder must retain the structural CTC repeated-label rule. Adjacent identical token transitions require a blank-separated prefix state; nonblank and blank-separated Viterbi states must not be collapsed.

## Frontend, ABI and public API contracts

`training/frontend_spec.py` is the dependency-free feature specification. Changes to window/FFT/mel/energy/normalization/PCEN behavior must update the reference contract and keep C/reference/Torch parity green.

`.kwm` and `.kwk` are product artifacts. Reject malformed/non-canonical inputs rather than guessing; keep reserved fields zero; keep vocabulary identity bound; bump the relevant ABI for incompatible binary-layout/semantic changes; update exporter/compiler/parser/tests/docs together; keep parser fuzz targets sanitizer-clean.

A public SDK/API behavior change must also bump the software/package version when appropriate even if KWSP/KWKP layouts remain unchanged. v0.3 is an example: deployable ABIs remain v2/v3 while the discontinuity API and evidence schemas changed.

The project uses a hard-cut model: unsupported legacy formats/evidence schemas are rejected instead of accumulating compatibility branches.

## Training, corpus and evaluation invariants

- `train_ctc.py` requires the actual token vocabulary; do not reintroduce size-only vocabulary configuration;
- TSV and JSONL training manifests must resolve to real mono 16-kHz PCM16 WAVs;
- checkpoints must bind canonical real training-corpus file/decoded-PCM/frame identity;
- model provenance schema v3 must carry that corpus identity;
- run `training/audit_dataset.py` before release training/qualification;
- final human qualification must require speaker/session/source metadata;
- final qualification recordings must not be recycled into replay/model/threshold tuning;
- evaluation provenance schema v2 must bind every held-out WAV;
- `references.duration_s` must equal real WAV duration;
- release FAR/hour and FRR come from continuous real audio evidence, not isolated clip accuracy;
- hosted/synthetic FAR results remain regression signals only.

## Training supply-chain invariant

Shipping training should use an immutable OCI base referenced with `@sha256:`. `training/Dockerfile` must not fetch/upgrade dependencies from the network. The final training image digest must be recorded in shipping checkpoints when `--require-container-digest` is used.

The repository variable for real `torch_ctc` integration is `KWS_TRAINING_IMAGE` and must contain a digest-pinned image reference.

## Product-board evidence invariant

Shipping CPU/RSS/stack/thermal/power/soak evidence must use the exact repository `tools/collect_target_evidence.py` contract and `evidence_class=product-board`.

The evidence tuple must bind:

- exact SKU and source SHA;
- distinct builder and DUT identities plus collector/station identity;
- exact collector, board runner, model, keyword pack and board audio;
- runtime-soak bytes and independently recomputable CPU/RSS/thermal metrics;
- canonical `evidence-raw.jsonl` with exact `{name, sha256, bytes}` set equality;
- external attestation-verification schema v1 from the controlled product trust layer;
- raw power evidence plus instrument/calibration identity;
- any stack/thermal/audio counters required by the SKU as retained raw artifacts.

A manually typed JSON with plausible values is not final evidence. Do not weaken `--evidence-raw`, `--attestation-verification`, `--builder-id`, `--dut-id`, `--collector-id`, `--board-runner`, `--model`, `--keyword-pack`, `--board-audio`, `--sku` or `--source-sha` requirements for convenience.

## Release qualification

A software change is not a shipping acoustic qualification. v0.3 release qualification requires:

- exact source/model/pack/config/checkpoint identity;
- model provenance schema v3 and training-corpus byte identity;
- clean dataset audit covering exact training/final references manifests;
- exact evaluation runner/references/original held-out WAVs/detections/provenance/metrics;
- exact target board benchmark runner/audio/summary;
- product-board evidence schema v2;
- exact collector, canonical raw evidence manifest, attestation verification and raw evidence files;
- matching `shipping_approved=true` SKU policy;
- qualification manifest schema v2 and gate result schema v3.

See `docs/RELEASE_QUALIFICATION.md`.

## Release engineering

Formal release workflow must not publish unless the full hosted, coverage, sanitizer, fuzz, Cortex-A32 and reproducible-SDK gates pass. Release artifacts retain SHA256SUMS, SPDX SBOM and GitHub attestations.

A released `vX.Y.Z` tag is immutable; never move an existing release tag to a different commit.

## Scope and style

Keep the device data plane small and C11-focused. Offline Python/hosted tools may use richer facilities, but those dependencies must not leak into the always-on runtime.

Prefer explicit validation, machine-verifiable evidence and testable contracts over conventions. Avoid transitional compatibility code in unreleased interfaces; update all callers/tests/docs in the same pull request.
