# kws-pipeline

[English](README.md) | [简体中文](README.zh-CN.md)

`kws-pipeline` is a low-compute, always-on keyword spotting engine for embedded Linux/RTOS-class products. It targets Cortex-A32/A7-class CPUs and similar budgets, supports configurable Mandarin wake phrases such as **“你好小窝”** and **“小窝小窝”**, and is designed to run after a lightweight front-end such as [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline).

The design is **open-token KWS**, not one binary classifier per wake word:

```text
PCM16 16 kHz
 -> 25 ms / 20 ms-hop log-mel frontend
 -> tiny int8-weight streaming RNN
 -> pinyin-token logits
 -> shared-prefix keyword trie
 -> speech/threshold/refractory gates
 -> wake event
```

A new phrase normally changes only the keyword token path and threshold. If acoustic separation is insufficient, the project supports hard-negative calibration and shallow output-head fine-tuning.

## Product properties

- C11 + libm only on device; PyTorch and `pypinyin` are offline tools.
- No heap, hidden thread, lock, filesystem or text conversion in the real-time library.
- Caller-owned aligned arena and read-only model blob.
- L0 keyword-only update, L1 calibration, L2 `--head-only` shallow customization.
- Default 32-feature / 48-hidden / ~420-token geometry is about **1.2 MMAC/s** for dense acoustic inference and roughly **26 KB** for model weights+biases. These are design estimates, not Cortex-A32 board measurements.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Compile custom wake words

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

Example definitions:

```text
1    你好小窝    0.55    ni3 hao3 xiao3 wo1
2    小窝小窝    0.55    xiao3 wo1 xiao3 wo1
```

The fourth TSV column is explicit tokenization and is recommended for reproducible production releases. During exploration it may be omitted and `pypinyin` can generate tone-aware pinyin.

## Base training and shallow customization

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --vocab-size 420 \
  --output build/base.pt
python3 training/export_model.py \
  --checkpoint build/base.pt \
  --output build/base.kwm
```

For shallow customization:

```bash
python3 training/train_ctc.py \
  --manifest data/xiaowo.tsv \
  --vocab-size 420 \
  --warm-start build/base.pt \
  --head-only \
  --epochs 10 \
  --output build/xiaowo.pt
```

`--head-only` freezes the input/recurrent representation and changes only the acoustic output head.

## Runtime API

```c
kws_model_t model;
kws_engine_t *engine;
kws_config_t cfg = kws_default_config();

kws_model_open(model_blob, model_blob_bytes, &model);
size_t arena_bytes = kws_engine_required_bytes(&model);
kws_engine_init(arena, arena_bytes, &model, &cfg, &engine);
kws_engine_set_keywords(engine, kws_generated_keywords,
                        kws_generated_keyword_count);

int detected = 0;
kws_detection_t hit;
kws_engine_accept_pcm16(engine, pcm, samples, &hit, &detected);
```

## audio-pipeline integration

Preferred product chain:

```text
mic -> BF/AEC/RES/NS/AGC -> mono S16 @ 16 kHz -> kws-pipeline -> ASR/assistant
```

Avoid a second resampler when `audio-pipeline` already emits 16 kHz. Re-evaluate thresholds against the exact AEC/NS/AGC configuration shipped on the device.

## Validation

CI covers strict GCC/Clang builds, CTest, ASan/UBSan, keyword compiler tests, Python syntax and Cortex-A32 ARMv7 hard-float cross-build. Repository tests establish software contracts only; release qualification still requires target-board FAR/hour, FRR, latency, CPU, memory, thermal and long-duration background-noise results using a real trained model.

See `docs/ARCHITECTURE.md`, `docs/CUSTOMIZATION.md`, `docs/PERFORMANCE.md`, `docs/INTEGRATION.md` and `THIRD_PARTY.md`.

## License

Apache-2.0. See `LICENSE`.
