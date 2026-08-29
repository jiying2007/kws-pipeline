# kws-pipeline

[English](README.md) | [简体中文](README.zh-CN.md)

`kws-pipeline` is a low-compute, always-on keyword spotting engine for embedded Linux/RTOS-class products. It targets Cortex-A32/A7-class CPU budgets, supports configurable Mandarin wake phrases such as **“你好小窝”**, **“小窝”** and **“小窝小窝”**, and is designed to consume mono PCM16 16-kHz audio after a lightweight BF/AEC/RES/NS/AGC chain such as [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline).

The runtime is open-token KWS rather than one classifier per wake phrase:

```text
PCM16 16 kHz
 -> 25 ms / 20 ms-hop frontend (log-mel or PCEN-lite)
 -> tiny int8-weight streaming RNN
 -> pinyin-token logits
 -> shared-prefix keyword trie
 -> CTC repetition + prefix arbitration
 -> speech / threshold / refractory gates
 -> wake event
```

A normal phrase change is an L0 keyword-pack update, not a model retrain. If field data misses FAR/FRR targets, the repository also provides calibration, hard-negative/false-reject replay, shallow output-head tuning, domain-aware synthetic iteration and artifact-bound release qualification.

## v0.3 software hardening

`v0.3.x` keeps the deployable **KWSP model ABI v2** and **KWKP keyword-pack ABI v3**, but hard-cuts the software evidence contracts around production qualification:

- training checkpoints and model provenance bind every training WAV by file SHA256, decoded-PCM SHA256, frame count and stable corpus identity;
- evaluation provenance binds every held-out qualification WAV by the same byte/PCM identity and rejects a declared `duration_s` that differs from the real WAV duration;
- release qualification requires a clean dataset-audit artifact and hard speaker/session/source separation for human qualification data;
- qualification manifest schema v2 re-reads the original training/evaluation audio instead of trusting self-reported hashes;
- target evidence schema v2 binds the evidence collector and retained raw measurement files; target CPU/thermal/resource JSON is no longer sufficient as an unbound manual declaration;
- `kws_engine_notify_discontinuity()` explicitly resets partial frontend/PCEN/RNN/decoder state on XRUN, route, clock or suspend/resume discontinuities without losing configured keywords or monotonic telemetry;
- CI adds C coverage, Clang static analysis, test-inventory enforcement and byte-for-byte independent SDK-build comparison;
- the optional production `torch_ctc` integration workflow runs in a digest-pinned training image and validates the real train/export/runtime path.

These changes make software qualification evidence byte-bound and auditable. They do **not** substitute for real Mandarin speakers, shipping microphone/enclosure/audio-pipeline recordings or physical Cortex-A32 measurements; those remain Issue #2 gates.

## Product properties

- C11 + libm only in the real-time library; PyTorch and `pypinyin` stay offline.
- No heap, hidden thread, lock, filesystem or text/pinyin conversion in the real-time path.
- Caller-owned aligned engine arena; model tensors are zero-copy views into a read-only `.kwm` blob.
- **`KWSP` model ABI v2**: fixed 16-kHz / 400-sample / 320-sample geometry, vocabulary fingerprint and model-bound frontend identity.
- **`KWKP` keyword-pack ABI v3**: per-keyword threshold, trailing-blank requirement, priority and `immediate` / `longest` / `grace` prefix policy.
- Model, keyword pack, training checkpoint and generated C keyword table are bound to the same 64-bit token-vocabulary identity; same-sized but differently mapped vocabularies are rejected.
- `logmel` and `pcen-lite` are both implemented in the C runtime, dependency-free reference frontend and training path. A release cannot silently run a model with a different frontend than the one recorded in its model lineage.
- Adjacent repeated acoustic tokens follow the structural CTC rule: a repeated token can advance only from a blank-separated prefix state.
- Shared-prefix phrases can be resolved deliberately instead of depending on file order. Longer candidates, priority and grace/trailing-blank policy are bounded runtime metadata.
- L0 keyword-only update, L1 threshold/replay calibration, L2 `--head-only` shallow customization.
- Default 32-feature / 48-hidden / ~420-token geometry is about **1.2 MMAC/s** and about **26 KB** of model weights+biases. These are design estimates, not Cortex-A32 board measurements.

## Build and install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/prefix
```

Installed consumers can use CMake package metadata or `pkg-config`:

```cmake
find_package(KwsPipeline CONFIG REQUIRED)
target_link_libraries(app PRIVATE KwsPipeline::core)
```

```bash
pkg-config --cflags --libs kws-pipeline
```

## Compile custom wake phrases

Production should pin explicit pinyin tokens. The keyword TSV accepts up to eight columns:

```text
id  text  threshold  explicit-pinyin  min_trailing_blanks  priority  prefix_policy  grace_frames
```

Example with a prefix conflict:

```text
1	你好小窝	0.55	ni3 hao3 xiao3 wo1
2	小窝	0.55	xiao3 wo1	1	10	grace	3
3	小窝小窝	0.55	xiao3 wo1 xiao3 wo1	1	20	longest
```

- `immediate`: emit as soon as the terminal meets its threshold.
- `longest`: hold a terminal until its trailing-blank condition is satisfied so a longer shared-prefix path can supersede it.
- `grace`: keep a bounded pending terminal for `grace_frames`, while still honoring `min_trailing_blanks`.
- When simultaneous immediate candidates exist, priority, then path depth, then confidence provides deterministic arbitration.

Compile one vocabulary-bound `.kwk` and optional firmware-linked C table:

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-pack build/xiaowo.kwk \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

The fourth column may be omitted during exploration and generated with `pypinyin`; shipping manifests should keep it explicit.

## Frontend and model identity

`training/frontend_spec.py` is the dependency-free feature contract. It implements `logmel` (`frontend_kind=0`) and `pcen-lite` (`frontend_kind=1`). Both use 16-kHz / FFT-512 / mel 80–7600 Hz / 400-sample frame / 320-sample hop geometry. PCEN-lite adds bounded streaming smoothing and gain normalization before common feature-vector normalization.

The model header records the frontend kind. Checkpoints and export provenance record frontend identity and frontend-spec version, and release qualification cross-checks the runtime frontend against model lineage. CI compares the real C frontend against the reference implementation for both modes.

## Base training and shallow customization

For human release data, audit decoded PCM and identity leakage before training. Shipping qualification should require speaker/session/source metadata:

```bash
python3 training/audit_dataset.py \
  --split train=data/train.jsonl \
  --split calibration=data/calibration.jsonl \
  --split qualification=data/qualification.jsonl \
  --require-metadata speaker_id \
  --require-metadata session_id \
  --require-metadata source_id \
  --report build/dataset-audit.json \
  --fail-within-split
```

`train_ctc.py` accepts TSV (`WAV<TAB>token_ids`) and schema-rich JSONL manifests. Train and export with the exact token vocabulary and chosen frontend:

```bash
python3 training/train_ctc.py \
  --manifest data/train.jsonl \
  --tokens keywords/tokens.zh.txt \
  --frontend logmel \
  --output build/base.pt

python3 training/export_model.py \
  --checkpoint build/base.pt \
  --tokens keywords/tokens.zh.txt \
  --output build/base.kwm
```

The checkpoint records a canonical training-corpus identity. The exporter writes model provenance schema v3 and carries that corpus identity into the released model lineage. Warm starts require compatible vocabulary, geometry and frontend identity.

For L2 shallow customization use `--warm-start ... --head-only`. Final qualification recordings must never be recycled into replay/tuning and then reused as unbiased evidence.

## Training environment

Shipping training should run in a prebuilt immutable OCI image. `training/Dockerfile` requires a base reference containing `@sha256:<digest>` and does not resolve/install packages from the network. Build the repository wrapper with `training/build_container.py`; pass the final image digest as `KWS_TRAINING_IMAGE_DIGEST` and use `train_ctc.py --require-container-digest` so the checkpoint records the exact training environment identity.

The weekly/manual `.github/workflows/training-integration.yml` can execute the production `torch_ctc` loop when repository variable `KWS_TRAINING_BASE_IMAGE` is configured with a digest-pinned image.

## Domain-aware synthetic loop

The repository includes a deterministic software-validation loop for near/mid/far acoustic domains and frontend A/B:

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

The example config covers nominal **0.3–5.0 m** distances, azimuth, RT60, SNR, white/fan/motor/media noise and optional local playback/AEC residual. Calibration/test/qualification positives use deterministic `far -> mid -> near` rotation. Complete rendered utterances run through the real C runtime and produce domain FRR/FAR/latency/confusion metrics.

This is **synthetic-domain evidence only**. It does not establish real 3–5 m human-speech performance, real robot AFE behavior or target-board acoustic qualification.

## Streaming and discontinuities

Normal integration can pass 160-sample/10-ms blocks directly to `kws_engine_accept_pcm16()`. If capture loses samples or changes timeline continuity, notify the engine explicitly:

```c
kws_engine_notify_discontinuity(kws, KWS_DISCONTINUITY_XRUN);
```

Use the matching reason for route change, clock reset or suspend/resume. This prevents pre-gap acoustic state from being joined to post-gap audio. See `docs/AUDIO_DISCONTINUITY.md`.

## Continuous and long-FAR evaluation

Run held-out continuous recordings through the real runtime:

```bash
python3 eval/run_corpus.py \
  --runner build/kws_wav \
  --model build/base.kwm \
  --keywords build/xiaowo.kwk \
  --references qualification/references.jsonl \
  --audio-root qualification/audio \
  --detections qualification/detections.jsonl \
  --provenance qualification/detections.provenance.json
```

Evaluation provenance schema v2 includes a canonical identity for every audio file. `duration_s` in each reference row must equal the real WAV duration. `eval/domain_metrics.py` adds near/mid/far, angle, RT60, noise, playback and keyword-confusion views. `eval/long_far_stream.py` and `.github/workflows/far-nightly.yml` provide synthetic streaming FAR regression; hosted/generated exposure is not a shipping FAR claim.

## Artifact-bound release qualification

A green source CI is a software baseline, not a shipping acoustic claim. v0.3 qualification re-hashes and cross-checks the exact model/provenance/checkpoint/training corpus, dataset audit, keyword pack, release vocabulary/config, evaluation runner/references/**real evaluation WAVs**/detections, target board benchmark and raw target evidence.

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --model-provenance release/base.kwm.provenance.json \
  --checkpoint release/base.pt \
  --training-tokens release/training-tokens.txt \
  --training-manifest release/train.jsonl \
  --dataset-audit qualification/dataset-audit.json \
  --keywords release/xiaowo.kwk \
  --tokens release/tokens.txt \
  --config release/runtime.json \
  --eval-runner qualification/kws_wav.eval \
  --references qualification/references.jsonl \
  --eval-audio-root qualification/audio \
  --detections qualification/detections.jsonl \
  --eval-summary qualification/eval-summary.json \
  --eval-provenance qualification/detections.provenance.json \
  --board-summary qualification/board-summary.json \
  --board-runner qualification/kws_board_bench.target \
  --board-audio qualification/board-audio.wav \
  --evidence qualification/evidence.json \
  --evidence-collector qualification/collect_target_evidence.py \
  --raw-evidence qualification/power.csv \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json

python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Repeat `--training-manifest` and `--raw-evidence` as needed. The gate requires model ABI v2, keyword-pack ABI v3, frontend-spec v2, exact runtime↔model-lineage frontend identity, corpus byte identity, dataset-audit coverage, target-evidence identity and the SKU acoustic/performance/resource thresholds.

## Validation boundary

CI gates GCC/Clang, CTest, Clang static analysis, C line coverage, ASan/UBSan, libFuzzer parser smoke, Cortex-A32 ARMv7 hard-float cross-build, both frontend parity modes, decoder/prefix contracts, dataset leakage, corpus byte identity, domain-aware multi-frontend iteration, streaming long-FAR smoke, schema-v2 byte-complete release qualification, machine target-evidence contracts, independent SDK reproducibility and clean SDK consumption.

Those results prove software contracts and deterministic/synthetic regressions. Shipping qualification still requires a real Mandarin model, independent human/device held-out recordings and physical Cortex-A32 evidence for FAR/hour, FRR, latency, CPU, memory, thermal/power and soak behavior. Repository issue #2 remains open for that evidence.

See `docs/README.md`, `docs/ARCHITECTURE.md`, `docs/CORPUS_IDENTITY.md`, `docs/AUDIO_DISCONTINUITY.md`, `docs/TARGET_EVIDENCE.md`, `docs/REPRODUCIBILITY.md`, `docs/TESTING_STRATEGY.md` and `docs/RELEASE_QUALIFICATION.md`.

## License

Apache-2.0. See `LICENSE`.
