# kws-pipeline

[English](README.md) | [简体中文](README.zh-CN.md)

`kws-pipeline` is a low-compute, always-on keyword spotting engine for embedded Linux/RTOS-class products. It targets Cortex-A32/A7-class CPU budgets, supports configurable Mandarin wake phrases such as **“你好小窝”**, **“小窝”** and **“小窝小窝”**, and is designed to consume mono PCM16 16-kHz audio after a lightweight BF/AEC/RES/NS/AGC chain such as [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline).

Offline training/evaluation tools support Python 3.8+; the canonical locked training container uses
Python 3.12. Device runtime code remains C11 and does not depend on Python.

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

`training/frontend_spec.py` is the dependency-free feature contract. It implements:

- `logmel` (`frontend_kind=0`);
- `pcen-lite` (`frontend_kind=1`).

Both use the same 16-kHz / FFT-512 / mel 80–7600 Hz / 400-sample frame / 320-sample hop geometry. PCEN-lite adds bounded streaming smoothing and gain normalization before the same feature-vector normalization.

The model header records the frontend kind. Checkpoints and export provenance record frontend identity and frontend-spec version, and release qualification cross-checks the runtime frontend against model lineage. CI compares the real C frontend against the reference implementation for both modes.

## Base training and shallow customization

Audit data by decoded PCM identity before training:

```bash
python3 training/audit_dataset.py \
  --split train=data/train.tsv \
  --split calibration=data/calibration.tsv \
  --split test=data/test.tsv \
  --split qualification=data/qualification.tsv \
  --report build/dataset-audit.json \
  --fail-within-split
```

Train and export with the exact token vocabulary and chosen frontend:

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --tokens keywords/tokens.zh.txt \
  --frontend logmel \
  --output build/base.pt

python3 training/export_model.py \
  --checkpoint build/base.pt \
  --tokens keywords/tokens.zh.txt \
  --output build/base.kwm
```

Use `--frontend pcen-lite` when training a PCEN-lite model. The exporter writes `base.kwm.provenance.json`, which binds the model to the checkpoint, token identities, training manifests, frontend identity/spec, hyperparameters and int8 quantization diagnostics.

For L2 shallow customization:

```bash
python3 training/train_ctc.py \
  --manifest data/xiaowo.tsv \
  --manifest build/hard-negatives.tsv \
  --tokens keywords/tokens.zh.txt \
  --warm-start build/base.pt \
  --head-only \
  --epochs 10 \
  --output build/xiaowo.pt
```

Warm starts require compatible vocabulary, geometry and frontend identity.

## Domain-aware synthetic loop

The repository includes a deterministic software-validation loop for near/mid/far acoustic domains and frontend A/B:

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

The example domain config covers nominal **0.3–5.0 m** distances, azimuth, RT60, SNR, white/fan/motor/media noise and optional local playback/AEC residual. Training uses weighted stochastic domains and adaptive curriculum. Calibration/test/qualification positives use deterministic `far -> mid -> near` rotation so a far-field FRR gate can never pass merely because no far positive was sampled.

The loop evaluates complete rendered utterances with the real C runtime, reports per-domain FRR/FAR/latency and keyword confusion, compares `logmel` and `pcen-lite` candidates and freezes the best model/pack/provenance bundle.

This is **synthetic-domain evidence only**. It does not establish real 3–5 m human-speech performance, real robot AFE behavior or target-board acoustic qualification.

## Continuous and long-FAR evaluation

Run a held-out corpus through the real runtime:

```bash
python3 eval/run_corpus.py \
  --runner build/kws_wav \
  --model build/base.kwm \
  --keywords build/xiaowo.kwk \
  --references data/eval/references.jsonl \
  --audio-root data/eval \
  --detections build/detections.jsonl \
  --provenance build/detections.provenance.json

python3 eval/score_events.py \
  --references data/eval/references.jsonl \
  --detections build/detections.jsonl \
  --summary build/summary.json \
  --false-positives build/false-positives.jsonl \
  --false-rejects build/false-rejects.jsonl
```

`eval/domain_metrics.py` adds near/mid/far, angle, RT60, noise, playback and keyword-confusion views when domain metadata is present. `eval/long_far_stream.py` drives the raw streaming C runtime over long continuous background material; `.github/workflows/far-nightly.yml` provides a synthetic/hosted regression watch. Hosted or generated long-FAR results are not a shipping FAR claim.

Never mine the final held-out qualification corpus and then reuse it as unbiased release evidence.

## Artifact-bound release qualification

A green source CI is a software baseline, not a shipping acoustic claim. The qualification bundle re-hashes the exact model, model provenance, source checkpoint, training tokens/manifests, keyword pack, release tokens/config, evaluation runner/references/detections, target benchmark runner/board audio and measured target evidence.

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --model-provenance release/base.kwm.provenance.json \
  --checkpoint release/base.pt \
  --training-tokens release/training-tokens.txt \
  --training-manifest release/train.tsv \
  --keywords release/xiaowo.kwk \
  --tokens release/tokens.txt \
  --config release/runtime.json \
  --eval-runner qualification/kws_wav.eval \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --eval-summary qualification/eval-summary.json \
  --eval-provenance qualification/detections.provenance.json \
  --board-summary qualification/board-summary.json \
  --board-runner qualification/kws_board_bench.target \
  --board-audio qualification/board-audio.wav \
  --evidence qualification/evidence.json \
  --evidence-raw qualification/evidence.raw.jsonl \
  --collector qualification/kws-evidence-collector \
  --attestation-verification qualification/attestation-verification.json \
  --sku pcr02-ssc305 \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json

python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Repeat `--training-manifest` for every manifest used. Product qualification requires an explicitly
`shipping_approved=true` policy bound to the same SKU, distinct builder/DUT identities, and exact hashes
for the collector, raw evidence and attestation-verification result. Example policies fail closed.

## Validation boundary

CI gates GCC/Clang, CTest, ASan/UBSan, libFuzzer parser smoke, Cortex-A32 ARMv7 hard-float cross-build, both frontend parity modes, decoder/prefix contracts, dataset leakage, domain-aware multi-frontend iteration, streaming long-FAR smoke, byte-complete release qualification and clean SDK consumption.

Those results prove software contracts and deterministic synthetic regressions. Shipping qualification still requires a real Mandarin model, independent human/device held-out recordings and physical Cortex-A32 evidence for FAR/hour, FRR, latency, CPU, memory, thermal/power and soak behavior. Repository issue #2 remains open for that evidence.

See `docs/ARCHITECTURE.md`, `docs/CUSTOMIZATION.md`, `docs/EVALUATION.md`, `docs/SYNTHETIC_TRAINING.md`, `docs/PERFORMANCE.md`, `docs/INTEGRATION.md` and `docs/RELEASE_QUALIFICATION.md`.

## License

Apache-2.0. See `LICENSE`.
