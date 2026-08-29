# Contributing

`kws-pipeline` is an always-on embedded audio component. Changes are reviewed against product invariants, not only functional correctness.

Repository merge/release policy is defined in [`docs/REPOSITORY_GOVERNANCE.md`](docs/REPOSITORY_GOVERNANCE.md). Product acoustic and physical-board evidence remains a separate qualification boundary.

## Required local checks

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 tests/test_frontend_parity.py ./build/kws_feature_dump
python3 tests/test_dataset_audit.py
python3 tests/test_keyword_compile.py
python3 tests/test_eval.py
python3 tests/test_run_corpus.py
python3 tests/test_board_bench.py --runner ./build/kws_board_bench
python3 tests/test_release_qualification.py
python3 -m py_compile tools/*.py training/*.py eval/*.py tests/*.py
```

Run ASan/UBSan for C changes. Parser changes must also pass the Clang libFuzzer targets configured with `-DKWS_BUILD_FUZZ=ON`. CI cross-builds the core and target qualification tools for Cortex-A32 ARMv7 hard-float.

## Real-time data-plane invariants

Changes under `src/` must preserve these defaults unless an explicit architecture change is approved:

- no heap allocation in `kws_engine_accept_pcm16()` or its callees;
- no mutexes, condition variables, hidden worker threads, filesystem access or text conversion in the real-time library;
- fixed 16-kHz / 400-sample frame / 320-sample hop ABI-v2 geometry;
- one call accepts at most `KWS_MAX_PCM_BLOCK_SAMPLES` and never partially consumes an invalid block;
- caller-owned engine arena with alignment reported by `kws_engine_required_alignment()`;
- caller-owned model blob remains read-only and valid for the engine lifetime;
- no global mutable runtime state;
- deterministic hard bounds for features, hidden state, vocabulary, keywords and trie nodes.

The keyword decoder must retain the structural CTC repeated-label rule. Adjacent identical token transitions require a blank-separated prefix state; nonblank and blank-separated Viterbi scores must not be collapsed into a single ambiguous state.

If a proposal needs to violate one of these properties, document the CPU/RAM/latency reason and update architecture, tests and qualification requirements in the same change.

## Frontend contract

`training/frontend_spec.py` is the dependency-free feature specification. Changes to the C frontend, window, FFT geometry, mel bins, energy scale or normalization must update the reference spec and keep `tests/test_frontend_parity.py` green. The PyTorch frontend must reuse the shared constants rather than fork its own geometry.

## Binary ABI and artifact contracts

`.kwm` and `.kwk` are product artifacts. Format changes must not be smuggled into an existing ABI version.

- reject malformed/non-canonical inputs rather than guessing;
- keep reserved fields zero and validate them on load;
- keep vocabulary identity bound by the 64-bit fingerprint;
- bump the relevant ABI version for incompatible layout/semantic changes;
- update exporter/compiler/parser/tests/docs together;
- keep generated C keyword tables subject to the same vocabulary identity checks as `.kwk`;
- keep `.kwm/.kwk` parser fuzz targets buildable and sanitizer-clean.

The project currently uses a hard-cut model: unsupported legacy formats are rejected instead of accumulating compatibility branches.

## Training and evaluation invariants

- `train_ctc.py` requires the actual token vocabulary; do not reintroduce size-only vocabulary configuration;
- checkpoints must retain exact vocabulary fingerprint/token hash and reject incompatible warm starts/exports;
- training/evaluation speaker, session and decoded audio must remain disjoint;
- run `training/audit_dataset.py` before release training/qualification; the audit compares decoded PCM payload hashes so renamed or rewrapped copies still count as leakage;
- CTC manifests must not contain invalid token IDs or unalignable targets;
- `--head-only` requires an explicit compatible warm start;
- final held-out qualification recordings must not be recycled into hard-negative training;
- release FAR/hour and FRR come from continuous audio, not isolated clip accuracy;
- hosted benchmark values are regression signals, never target-board claims.

## Release qualification

A software change is not a shipping acoustic qualification. For product release evidence use `docs/RELEASE_QUALIFICATION.md` and retain:

- exact source/artifact/corpus hashes;
- evaluation provenance and metrics;
- real target-board benchmark summary;
- soak/resource/thermal/power evidence;
- approved SKU policy;
- deterministic qualification manifest and gate result.

## Scope and style

Keep the real-time library small and C11-focused. Offline Python and hosted tools may use richer facilities, but they must not leak dependencies into the device data plane.

Prefer explicit validation and testable contracts over implicit conventions. Avoid transitional compatibility code in unreleased interfaces; update all callers and documentation in the same pull request.
