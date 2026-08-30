# kws-pipeline

[English](README.md) | [简体中文](README.zh-CN.md)

`kws-pipeline` is a low-compute, always-on keyword spotting engine for embedded Linux/RTOS-class products. It targets Cortex-A32/A7-class CPU budgets, supports configurable Mandarin wake phrases such as **“你好小窝”**, **“小窝”** and **“小窝小窝”**, and consumes mono PCM16 16-kHz audio after a lightweight BF/AEC/RES/NS/AGC chain such as [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline).

```text
PCM16 16 kHz
 -> 25 ms / 20 ms-hop log-mel or PCEN-lite
 -> tiny int8-weight streaming RNN
 -> pinyin-token logits
 -> shared-prefix keyword Trie
 -> CTC repetition + prefix arbitration
 -> speech / threshold / refractory gates
 -> wake event
```

A normal phrase change is an L0 keyword-pack update, not a model retrain. If field data misses FAR/FRR targets, the repository provides threshold calibration, hard-negative/false-reject replay, shallow output-head tuning, domain-aware iteration and artifact-bound release qualification.

## v0.3 software baseline

`v0.3.x` keeps the deployable **KWSP model ABI v2** and **KWKP keyword-pack ABI v3**, while hard-cutting the qualification/evidence contracts:

- training checkpoints and model provenance bind every selected training WAV by file SHA256, decoded-PCM SHA256 and frame count;
- evaluation provenance binds every held-out WAV and rejects a declared `duration_s` that differs from the real WAV duration;
- final human qualification requires a clean dataset audit with speaker/session/source isolation;
- qualification manifest schema v2 reopens original training/evaluation audio instead of trusting self-reported hashes;
- product-board evidence binds SKU, source SHA, builder/DUT/collector identity, the exact collector, runtime-soak bytes, raw evidence manifest, externally verified attestation result, board runner, model, keyword pack and board audio;
- runtime-soak CPU/RSS/thermal summaries are independently recomputed from retained samples;
- `kws_engine_notify_discontinuity()` clears partial acoustic state on XRUN, route, clock or suspend/resume discontinuities;
- CI gates GCC/Clang, static analysis, C coverage, ASan/UBSan, libFuzzer, Cortex-A32 cross-build, deterministic SDK reproducibility and test inventory;
- the optional real `torch_ctc` integration workflow runs inside a digest-pinned training image.

These changes close the **software and evidence-engineering path**. They do not substitute for real Mandarin speakers, the final microphones/enclosure/AFE, genuine 0.3–5 m acoustic qualification or physical Cortex-A32 measurements; those remain Issue #2 gates.

## Product properties

- C11 + libm only in the real-time library; PyTorch and `pypinyin` remain offline.
- No heap, hidden thread, lock, filesystem or text/pinyin conversion in the real-time path.
- Caller-owned aligned engine arena; model tensors are zero-copy views into a read-only `.kwm` blob.
- **KWSP ABI v2**: fixed 16-kHz / 400-sample / 320-sample geometry, vocabulary fingerprint and frontend identity.
- **KWKP ABI v3**: per-keyword threshold, trailing-blank requirement, priority and `immediate` / `longest` / `grace` prefix policy.
- Adjacent repeated acoustic tokens obey structural CTC blank-separation semantics.
- Shared-prefix phrases are resolved deterministically rather than by TSV order.
- L0 keyword-only update, L1 threshold/replay calibration, L2 `--head-only` customization.
- Default 32-feature / 48-hidden / ~420-token geometry is roughly **1.2 MMAC/s** and **26 KB** weights+biases; these are design estimates, not target-board measurements.

## Build and install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/prefix
```

Installed consumers can use either CMake package metadata or `pkg-config`:

```cmake
find_package(KwsPipeline CONFIG REQUIRED)
target_link_libraries(app PRIVATE KwsPipeline::core)
```

## Compile custom wake phrases

Production should pin explicit pinyin tokens:

```text
id  text  threshold  explicit-pinyin  min_trailing_blanks  priority  prefix_policy  grace_frames
1   你好小窝  0.55       ni3 hao3 xiao3 wo1
2   小窝      0.55       xiao3 wo1             1                    10        grace          3
3   小窝小窝  0.55       xiao3 wo1 xiao3 wo1   1                    20        longest
```

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-pack build/xiaowo.kwk \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

## Training and dataset isolation

`train_ctc.py` accepts TSV (`WAV<TAB>token_ids`) and schema-rich JSONL. Human release data should use JSONL identity metadata. Audit the exact final references manifest that will later be qualified:

```bash
python3 training/audit_dataset.py \
  --split train=data/train.jsonl \
  --split calibration=data/calibration.jsonl \
  --split qualification=qualification/references.jsonl \
  --audio-root qualification=qualification/audio \
  --require-metadata speaker_id \
  --require-metadata session_id \
  --require-metadata source_id \
  --report qualification/dataset-audit.json \
  --fail-within-split
```

Train and export:

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

The checkpoint records canonical training-corpus identity; model provenance schema v3 carries it into the release lineage. Final qualification recordings must never be recycled into tuning/replay and then reused as unbiased evidence.

## Immutable training environment

Shipping training should use an immutable OCI image referenced as `name@sha256:<digest>`. `training/Dockerfile` performs no network dependency installation. `training/build_container.py` validates the immutable base and records a build receipt; shipping training can require `KWS_TRAINING_IMAGE_DIGEST`.

The real `torch_ctc` integration workflow uses repository variable **`KWS_TRAINING_IMAGE`** (or the manual `training_image` input), and accepts only a digest-pinned image reference.

## Domain-aware self-validation

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

The example matrix spans nominal 0.3–5.0 m distance, azimuth, RT60, SNR, white/fan/motor/media noise and optional playback/AEC residual. Complete rendered utterances run through the real C runtime.

This remains **synthetic-domain evidence**. It does not establish real 3–5 m human-speech performance, real robot AFE behavior or target-board qualification.

## Streaming and discontinuities

Normal integration may pass 160-sample/10-ms blocks to `kws_engine_accept_pcm16()`. If capture loses timeline continuity, notify the engine before accepting new audio:

```c
kws_engine_notify_discontinuity(kws, KWS_DISCONTINUITY_XRUN);
```

Use the matching reason for route change, clock reset or suspend/resume. See `docs/AUDIO_DISCONTINUITY.md`.

## Continuous evaluation

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

Evaluation provenance schema v2 binds every actual WAV, and every reference `duration_s` must equal the real WAV duration. Hosted/synthetic long-FAR is a regression signal, not a shipping FAR claim.

## Product-board evidence contract

First supervise the actual process under test:

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

Freeze the exact raw files in `qualification/evidence-raw.jsonl` as `{name, sha256, bytes}` rows, and obtain an approved external `qualification/attestation-verification.json` that verifies the raw-manifest/collector/board-runner/model/keyword-pack tuple.

Then assemble target evidence with the complete v0.3 contract:

```bash
python3 tools/collect_target_evidence.py \
  --output qualification/evidence.json \
  --target product-sku-a \
  --board-revision A \
  --soc cortex-a32 \
  --toolchain arm-linux-gnueabihf-gcc-... \
  --compiler-flags '-O3 -mcpu=cortex-a32 ...' \
  --audio-frontend audio-pipeline-vX \
  --audio-frontend-sha256 <sha256> \
  --runtime-soak qualification/runtime-soak.json \
  --stack-high-water-bytes <measured> \
  --average-power-mw <measured> \
  --raw-evidence qualification/stack-watermark.txt \
  --power-raw qualification/power.csv \
  --evidence-raw qualification/evidence-raw.jsonl \
  --attestation-verification qualification/attestation-verification.json \
  --board-runner qualification/kws_board_bench.target \
  --model release/base.kwm \
  --keyword-pack release/xiaowo.kwk \
  --board-audio qualification/board-audio.wav \
  --sku product-sku-a \
  --source-sha "$(git rev-parse HEAD)" \
  --builder-id qualification-builder-01 \
  --dut-id product-dut-01 \
  --collector-id qualification-station-01 \
  --instrument-id <meter-id> \
  --calibration-id <calibration-id>
```

`builder-id` and `dut-id` must be distinct. The collector independently derives CPU/RSS/thermal/soak metrics from the retained runtime-soak samples and binds the exact raw/artifact identities. See `docs/TARGET_EVIDENCE.md`.

## Artifact-bound release qualification

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --model-provenance release/base.kwm.provenance.json \
  --checkpoint release/base.pt \
  --training-tokens release/training-tokens.txt \
  --training-manifest data/train.jsonl \
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
  --evidence-collector tools/collect_target_evidence.py \
  --evidence-raw qualification/evidence-raw.jsonl \
  --attestation-verification qualification/attestation-verification.json \
  --raw-evidence qualification/runtime-soak.json \
  --raw-evidence qualification/stack-watermark.txt \
  --raw-evidence qualification/power.csv \
  --source-sha "$(git rev-parse HEAD)" \
  --sku product-sku-a \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json

python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Repeat `--training-manifest` / `--raw-evidence` as needed. The gate requires model ABI v2, keyword-pack ABI v3, frontend-spec v2, exact corpus byte identity, audit coverage, product-board evidence identity, shipping-approved SKU policy and the configured acoustic/performance/resource thresholds.

## Validation boundary

CI proves software contracts and deterministic/synthetic regressions: GCC/Clang, CTest, static analysis, coverage, sanitizers, fuzzing, Cortex-A32 cross-build, frontend/decoder contracts, corpus identity, qualification integrity, runtime-soak/target-evidence validation, reproducible SDK and clean SDK consumption.

A green repository and a signed `v0.3.0` release still do **not** prove a shipping Mandarin SKU until independent real human/device acoustic evidence and physical target-board measurements exist. Issue #2 is intentionally the only product-evidence gate left open.

See `docs/README.md`, `docs/RELEASE_QUALIFICATION.md`, `docs/TARGET_EVIDENCE.md`, `docs/CORPUS_IDENTITY.md`, `docs/AUDIO_DISCONTINUITY.md` and `docs/TESTING_STRATEGY.md`.

## License

Apache-2.0. See `LICENSE`.
