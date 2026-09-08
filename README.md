# kws-pipeline

[English](README.md) | [简体中文](README.zh-CN.md)

`kws-pipeline` is a low-compute, always-on keyword spotting engine for embedded Linux/RTOS-class products. The **currently qualified synthetic product SKU** is deliberately narrow and ships exactly two four-character Mandarin wake phrases:

- **你好小窝** — `ni3 hao3 xiao3 wo1`
- **小窝小窝** — `xiao3 wo1 xiao3 wo1`

The standalone two-character phrase **小窝 is not a shipping wake word** for this SKU. The qualified model uses a dedicated five-token vocabulary (`<blk>`, `ni3`, `hao3`, `xiao3`, `wo1`). The framework supports larger vocabularies and configurable keyword packs, but arbitrary-Mandarin L0 phrase replacement has **not** been qualified for this released model.

The engine consumes mono PCM16 16-kHz audio after a lightweight BF/AEC/RES/NS/AGC chain such as [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline):

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

## Current engineering-qualified model

The current immutable model release is `model-749187ec1d66` and is bound to:

- model-training run `34134789576`;
- exact trained HEAD `749187ec1d6662658f06aa9c76d47fde835968db`;
- `model.kwm` SHA256 `ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`;
- exact two-keyword shipping pack above;
- untouched synthetic qualification seed `271838`, now **consumed and permanently frozen**;
- qualification `256/256`, `0 FR`, `0 FA`;
- strict robustness pass;
- continuous synthetic hard-negative FAR evidence with `0 FA` and full manifest coverage.

`configs/shipping.xiaowo.json` is the machine-readable product contract. It intentionally records `shipping_approved=false`: synthetic qualification is engineering evidence, not a substitute for final real-human/final-AFE acoustic qualification and physical target-board evidence.

## Runtime and product properties

- C11 + libm only in the real-time library; PyTorch and `pypinyin` remain offline.
- No heap, hidden thread, lock, filesystem or text/pinyin conversion in the real-time path.
- Caller-owned aligned engine arena; model tensors are zero-copy views into a read-only `.kwm` blob.
- **KWSP ABI v2**: fixed 16-kHz / 400-sample / 320-sample geometry, vocabulary fingerprint and frontend identity.
- **KWKP ABI v3**: per-keyword threshold, trailing-blank requirement, priority and `immediate` / `longest` / `grace` prefix policy.
- Adjacent repeated acoustic tokens obey structural CTC blank-separation semantics.
- Shared-prefix phrases are resolved deterministically rather than by TSV order.
- `kws_engine_notify_discontinuity()` clears partial acoustic state on XRUN, route, clock or suspend/resume discontinuities.
- Runtime accepts versioned external-AFE metadata and exposes bounded telemetry suitable for product diagnostics.

The larger ~420-token / ~26-KB model described in design notes is a **framework sizing option**, not the identity of the currently qualified five-token model.

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

## Shipping keyword pack and customization boundary

The qualified shipping TSV is exactly:

```text
id  text      threshold  explicit-pinyin
1   你好小窝  0.55       ni3 hao3 xiao3 wo1
2   小窝小窝  0.55       xiao3 wo1 xiao3 wo1
```

Compile it with:

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-pack build/xiaowo.kwk \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

L0 keyword-pack changes are valid only when every requested acoustic token already exists in the loaded model vocabulary and the resulting product tuple is revalidated. For the current dedicated five-token shipping model, only the two phrases above are release-qualified. A Mandarin phrase that needs other tokens requires an intentionally broader/new acoustic model and fresh qualification; it must not be presented as a zero-training field update.

See `docs/CUSTOMIZATION.md` for L0/L1/L2 rules.

## Domain-aware self-training and self-validation

The repository provides a deterministic offline loop:

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.torch-domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

The formal loop uses real C-runtime evaluation, four fixed training rounds, calibration/test gates, hard-negative replay, adaptive domain curriculum, untouched qualification, robustness slices and continuous-FAR evidence. Its synthetic matrix spans nominal 0.3–5.0 m distance, azimuth, RT60, SNR, white/fan/motor/media noise and playback/AEC-residual proxies.

The active formal qualification seed `271838` is already exposed by the accepted release and must never be retried. Future genuinely new model candidates reserve a new formal seed; current maintenance must not consume `271839`.

### Nightly regression is intentionally separate

`.github/workflows/far-nightly.yml` no longer retrains a model and no longer renders formal qualification. It downloads the exact immutable `model-749187ec1d66` Release, verifies release checksums/provenance/acceptance evidence, renders an independent `nightly-frozen-model-v1` synthetic negative corpus and runs four long-FAR shards against the frozen released model.

`configs/nightly.xiaowo-frozen-model.json` contains **no formal qualification seed or FAR-holdout namespace**. Nightly evidence is regression evidence only; it can never become fresh qualification evidence.

## Final AFE integration contract

Synthetic formal training currently uses the built-in proxy AFE, but the renderer already provides a fail-closed `command` AFE backend for the final product pipeline. The backend binds:

- exact invoked executable SHA256;
- command-template SHA256;
- exact ordered AFE config-file bundle SHA256;
- left/right input hashes and output hash;
- result-sidecar hash;
- reported latency in samples;
- optional final pipeline SHA/source SHA/toolchain identity.

The final real `audio-pipeline` binary/config will be inserted through this contract during the later real-human/device qualification phase. Until that exact final AFE identity exists, `shipping_approved` remains false.

## Immutable training environment

Shipping training should use an immutable OCI image referenced as `name@sha256:<digest>`. `training/Dockerfile` performs no network dependency installation. `training/build_container.py` validates the immutable base and records a build receipt. The real `torch_ctc` integration workflow uses repository variable **`KWS_TRAINING_IMAGE`** (or the manual `training_image` input) and accepts only a digest-pinned image reference.

## Continuous evaluation and statistical boundary

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

Evaluation provenance binds every actual WAV and validates duration from decoded frames. Synthetic/nightly long-FAR is a strict regression signal, not a commercial real-world FAR claim. A zero observed count is interpreted with statistical confidence bounds rather than as proof that the true rate is zero.

## Real-human corpus contract — deferred next phase

When real data is introduced, the exact final references manifest must be audited for speaker/session/source isolation:

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

Recordings used for tuning, hard-negative mining or false-reject replay may not later be reused as unbiased final qualification evidence.

## Product-board evidence contract — prepared, physical execution deferred

The repository already has a fail-closed evidence schema for the later physical-board phase. First supervise the actual product process:

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

Then freeze raw evidence and assemble target identity:

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

`builder-id` and `dut-id` must be distinct. CPU/RSS/thermal/soak summaries are recomputed from retained raw samples; power requires original instrument evidence and calibration identity. See `docs/TARGET_EVIDENCE.md`.

## Artifact-bound release qualification

Final human/device shipping qualification uses the same exact artifact contract rather than hand-written summaries:

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

The shipping gate requires model ABI v2, keyword-pack ABI v3, frontend-spec v2, exact corpus identity, dataset-audit coverage, product-board evidence identity and a policy with `shipping_approved=true`. The repository keeps that last state false until the deferred real-human/final-AFE and physical-board evidence actually passes.

## Validation boundary

CI proves software contracts and deterministic/synthetic regressions: GCC/Clang, CTest, static analysis, coverage, ASan/UBSan, libFuzzer, Cortex-A32 cross-build, frontend/decoder contracts, corpus identity, self-training, robustness/FAR regression, runtime-soak/target-evidence schema validation, SDK reproducibility and model supply-chain integrity.

It does **not** claim final commercial acoustic or physical-board qualification. After the non-real-data engineering closure, the only product-evidence phase intentionally left is: (1) real Mandarin through the final microphones/enclosure/AFE, then (2) physical target-board performance/soak and final tuple approval.

See `docs/README.md`, `docs/CUSTOMIZATION.md`, `docs/RELEASE_QUALIFICATION.md`, `docs/TARGET_EVIDENCE.md`, `docs/CORPUS_IDENTITY.md`, `docs/AUDIO_DISCONTINUITY.md` and Issue #2.

## License

Apache-2.0. See `LICENSE`.
