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

## v0.3 software hardening

`v0.3.x` keeps the deployable **KWSP model ABI v2** and **KWKP keyword-pack ABI v3**, but hard-cuts the qualification evidence contracts:

- training checkpoints/model provenance bind every real training WAV by file SHA256, decoded-PCM SHA256 and frame count;
- evaluation provenance binds every held-out WAV and rejects declared `duration_s` values that differ from the real WAV duration;
- final human qualification requires a clean dataset audit with speaker/session/source isolation;
- qualification manifest schema v2 reopens original training/evaluation audio rather than trusting self-reported hashes;
- target resource evidence is assembled from a retained runtime-soak trace, power/raw measurement files and the exact evidence collector;
- `kws_engine_notify_discontinuity()` explicitly clears partial acoustic state on XRUN, route, clock or suspend/resume discontinuities;
- CI adds Clang static analysis, C coverage, test-inventory enforcement and independent SDK byte-reproducibility checks;
- the optional production `torch_ctc` integration workflow executes inside a digest-pinned training image.

These changes make the **software evidence path** byte-bound and auditable. They do not substitute for real Mandarin speakers, the shipping microphones/enclosure/audio-pipeline or physical Cortex-A32 measurements; those remain Issue #2 gates.

## Product properties

- C11 + libm only in the real-time library; PyTorch and `pypinyin` stay offline.
- No heap, hidden thread, lock, filesystem or text/pinyin conversion in the real-time path.
- Caller-owned aligned engine arena; model tensors are zero-copy views into a read-only `.kwm` blob.
- **KWSP ABI v2**: fixed 16-kHz / 400-sample / 320-sample geometry, vocabulary fingerprint and frontend identity.
- **KWKP ABI v3**: per-keyword threshold, trailing-blank requirement, priority and `immediate` / `longest` / `grace` prefix policy.
- Adjacent repeated acoustic tokens obey the structural CTC blank-separation rule.
- Shared-prefix phrases are resolved deterministically instead of depending on TSV order.
- L0 keyword-only update, L1 threshold/replay calibration, L2 `--head-only` customization.
- Default 32-feature / 48-hidden / ~420-token geometry is about **1.2 MMAC/s** and about **26 KB** of model weights+biases; these are design estimates, not Cortex-A32 measurements.

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

## Compile custom wake phrases

Production should pin explicit pinyin tokens:

```text
id  text  threshold  explicit-pinyin  min_trailing_blanks  priority  prefix_policy  grace_frames
```

Example:

```text
1	你好小窝	0.55	ni3 hao3 xiao3 wo1
2	小窝	0.55	xiao3 wo1	1	10	grace	3
3	小窝小窝	0.55	xiao3 wo1 xiao3 wo1	1	20	longest
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

`train_ctc.py` accepts TSV (`WAV<TAB>token_ids`) and schema-rich JSONL. Human release data should use JSONL identity metadata.

Before a shipping candidate, audit the **same final qualification references file** that will be passed to `qualification_manifest.py`:

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

This direct `references.jsonl` coverage is intentional: v0.3 qualification verifies that the dataset-audit artifact covers the exact training manifests and the exact final held-out reference manifest.

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

The checkpoint records a canonical real training-corpus identity. Model provenance schema v3 carries that identity into the released model lineage. Final qualification recordings must never be recycled into tuning/replay and then reused as unbiased evidence.

## Immutable training environment

Shipping training should use a prebuilt immutable OCI image. `training/Dockerfile` requires a base reference containing `@sha256:<digest>` and performs no network dependency installation. Build the wrapper with `training/build_container.py`, pass the final digest as `KWS_TRAINING_IMAGE_DIGEST`, and use `train_ctc.py --require-container-digest` for a shipping candidate.

`.github/workflows/training-integration.yml` can execute the real `torch_ctc` loop when repository variable `KWS_TRAINING_BASE_IMAGE` points to a digest-pinned image.

## Domain-aware development loop

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

The example config covers nominal **0.3–5.0 m** distance, azimuth, RT60, SNR, white/fan/motor/media noise and optional playback/AEC residual. Complete rendered utterances run through the real C runtime.

This is **synthetic-domain evidence only**. It does not establish real 3–5 m human-speech performance, real robot AFE behavior or target-board qualification.

## Streaming and discontinuities

Normal integration may pass 160-sample/10-ms blocks directly to `kws_engine_accept_pcm16()`. If capture loses timeline continuity, notify the engine:

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

Evaluation provenance schema v2 binds every actual WAV. Each reference `duration_s` must equal the real WAV duration. Hosted/synthetic long-FAR remains a regression signal, not a shipping FAR claim.

## Physical-target evidence

First supervise the **actual process under test**:

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

The soak trace derives child-process CPU/RSS and thermal observations. Then assemble target evidence:

```bash
python3 tools/collect_target_evidence.py \
  --output qualification/evidence.json \
  --target product-sku-a \
  --board-revision A \
  --toolchain arm-linux-gnueabihf-gcc-... \
  --compiler-flags '-O3 -mcpu=cortex-a32 ...' \
  --audio-frontend audio-pipeline-vX \
  --audio-frontend-sha256 <sha256> \
  --runtime-soak qualification/runtime-soak.json \
  --stack-high-water-bytes <measured> \
  --average-power-mw <measured> \
  --raw-evidence qualification/stack-watermark.txt \
  --power-raw qualification/power.csv \
  --instrument-id <meter-id> \
  --calibration-id <calibration-id>
```

`soak_hours`, CPU, RSS and max temperature come from the retained runtime-soak file rather than command-line declarations. Stack high-water remains a product-harness measurement and should have retained raw evidence. Power requires the raw instrument export plus instrument/calibration identity.

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
  --evidence-collector qualification/collect_target_evidence.py \
  --raw-evidence qualification/runtime-soak.json \
  --raw-evidence qualification/stack-watermark.txt \
  --raw-evidence qualification/power.csv \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json

python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Repeat `--training-manifest` / `--raw-evidence` as needed. The gate requires model ABI v2, keyword-pack ABI v3, frontend-spec v2, exact corpus byte identity, audit coverage, target raw-evidence identity and the SKU acoustic/performance/resource thresholds.

## Validation boundary

CI gates GCC/Clang, CTest, Clang static analysis, C coverage, ASan/UBSan, libFuzzer, Cortex-A32 cross-build, frontend parity, decoder/prefix contracts, dataset leakage, corpus byte identity, domain iteration, long-FAR regression, v0.3 qualification, runtime-soak/target-evidence collectors, independent SDK reproducibility and clean SDK consumption.

Those results prove software contracts and regression mechanisms. Shipping qualification still requires real Mandarin held-out recordings through the final product audio path and physical Cortex-A32 evidence. Issue #2 remains open for those measurements.

See `docs/README.md`, `docs/RELEASE_QUALIFICATION.md`, `docs/TARGET_EVIDENCE.md`, `docs/CORPUS_IDENTITY.md`, `docs/AUDIO_DISCONTINUITY.md` and `docs/TESTING_STRATEGY.md`.

## License

Apache-2.0. See `LICENSE`.
