# Contributing

`kws-pipeline` is an always-on embedded audio component. Changes are reviewed against real-time invariants, source rights, reproducible evidence and scientific validity, not only functional correctness. **This repository is KWS technical research only.** Do not treat an old product policy, GitHub Release or Shipping field as current data/model admission or deployment authorization.

Protected-main review and CI rules are described in [`docs/REPOSITORY_GOVERNANCE.md`](docs/REPOSITORY_GOVERNANCE.md); the [current research execution boundary](docs/RESEARCH_ONLY_EXECUTION.md) governs actual experiments. Independent human speech, final-AFE and physical SSC305 measurements remain separate scientific evidence tasks, never silently inferred from synthetic CI.

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
python3 tests/test_decoder_policy_replay.py --path-runner ./build/kws_decoder_path_replay
python3 tools/prepare_eval_context_fixtures.py
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
python3 tools/test_inventory.py --official-workflows
python3 -m py_compile tools/*.py training/*.py eval/*.py tests/*.py
```

CI additionally runs the complete synthetic/domain/FAR matrix, Clang static analysis, a C line-coverage gate, ASan/UBSan, `.kwm/.kwk` libFuzzer smoke, Cortex-A32 ARMv7 hard-float cross-build, two independent SDK builds with byte-for-byte install-tree comparison and clean SDK consumer checks. Historical release-named fixture tests in the command list are regression checks only, not a live qualification or publication workflow.

Do not add a new `tests/test_*.py` without wiring it into an official workflow. `tools/test_inventory.py` checks executable declarations and reports their scope; this is not proof of execution. See [actual test execution evidence](docs/TEST_EXECUTION_EVIDENCE.md) for the exact guarded suites, modes and fail-closed outcome checks.

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

`.kwm` and `.kwk` are binary acoustic-model and keyword-pack research assets. Reject malformed/non-canonical inputs rather than guessing; keep reserved fields zero; keep vocabulary identity bound; bump the relevant ABI for incompatible binary-layout/semantic changes; update exporter/compiler/parser/tests/docs together; keep parser fuzz targets sanitizer-clean.

A public C library/API behavior change must also bump the software/package version when appropriate even if KWSP/KWKP layouts remain unchanged. The historical v0.3 ABI/schema transition is an example of keeping source and binary compatibility claims explicit, not a product release requirement.

The project uses a hard-cut model: unsupported legacy formats/evidence schemas are rejected instead of accumulating compatibility branches.

## Training, corpus and evaluation invariants

- `train_ctc.py` requires the actual token vocabulary; do not reintroduce size-only vocabulary configuration;
- TSV and JSONL training manifests must resolve to real mono 16-kHz PCM16 WAVs;
- checkpoints must bind canonical real training-corpus file/decoded-PCM/frame identity;
- model provenance schema v3 must carry that corpus identity;
- run `training/audit_dataset.py` before research training or claiming independent evaluation;
- real-human research measurements require declared source rights, speaker/session/source metadata and a disjoint holdout;
- final held-out recordings must not be recycled into replay/model/threshold tuning and later presented as independent;
- evaluation provenance schema v2 must bind every held-out WAV;
- `references.duration_s` must equal real WAV duration;
- independently measured real-audio FAR/hour and FRR require actual continuous exposure/event counts, not isolated clip accuracy;
- hosted/synthetic FAR results remain regression signals only.

## Reproducible research training

For `torch_ctc` research integration, use an immutable OCI image reference with `@sha256:` where a container is required. `training/Dockerfile` must not silently upgrade/download unpinned dependencies. Record the actual research image digest and runtime environment in the experiment receipt. `KWS_TRAINING_IMAGE` identifies the digest-pinned image; it does **not** confer publication or Shipping authority.

## Independent physical-board research evidence

Use `tools/collect_target_evidence.py` and `tools/collect_runtime_soak.py` for independently reproducible SSC305 CPU, RSS, stack, latency, thermal/power, audio-counter and soak observations when actual hardware and measurement sources exist.

Bind the precise source SHA, board/DUT, collector and measurement station, tool versions, model/keyword hashes, original board-audio inputs, raw observations and canonical per-file SHA256/byte counts. Report unavailable quantities explicitly as NOT_RUN or null; a plausible typed JSON, hosted cross-build or self-declared device summary is not a physical measurement.

The old `evidence_class=product-board`, SKU fields, raw attestation flags and Phase A/B/C policy belong to retained historical format replay. They are **not** active requirements for importing lawful research samples, training a new candidate or comparing scientific model outcomes. Do not silently weaken underlying raw-metric or source-identity tests when reusing a historical scorer.

## Archived qualification-format regression (not an active gate)

`tests/test_release_qualification.py` and the historical `qualification_gate.py` / target scoring tests remain as offline negative/positive replay coverage for old immutable schemas. Historical `shipping_approved=true` policy examples describe prior product-era receipts only; the current repository has no Shipping approval authority or Release-based model admission.

For current experiments, evaluate source/corpus eligibility, disjoint speaker-family splits, actual final-AFE audio, independent real-speech FAR/FRR and SSC305 resource observations under the [research execution guide](docs/RESEARCH_ONLY_EXECUTION.md). Never relabel a fixture PASS or frozen source hash as proof that an unmeasured real-world goal passed. See [archived historical qualification contracts](docs/RELEASE_QUALIFICATION.md) only when independently replaying that history.

## Research CI, reproducibility and provenance

Changes to runtime/library/source contracts require the applicable hosted compiler, coverage, sanitizer, fuzz, ARM and reproducible-build checks to pass before protected-main merge. Current research CI **does not publish commercial model/SDK GitHub Releases**, issue Shipping authorization or approve a product. Retain SHA256 identity and source/provenance evidence for any separately authorized research assets. Historical tags and immutable prior Releases must not be moved or rewritten.

## Scope and style

Keep the device data plane small and C11-focused. Offline Python/hosted tools may use richer facilities, but those dependencies must not leak into the always-on runtime.

Prefer explicit validation, machine-verifiable evidence and testable contracts over conventions. Avoid transitional compatibility code in unreleased interfaces; update all callers/tests/docs in the same pull request.
