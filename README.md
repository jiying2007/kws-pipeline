# kws-pipeline

[English](README.md) | [简体中文](README.zh-CN.md)

`kws-pipeline` is a low-compute, always-on keyword spotting engine for embedded Linux/RTOS-class products. It targets Cortex-A32/A7-class CPUs and similar budgets, supports configurable Mandarin wake phrases such as **“你好小窝”** and **“小窝小窝”**, and is designed to consume the mono 16-kHz output of a lightweight front-end such as [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline).

The design is **open-token KWS**, not one binary classifier per wake phrase:

```text
PCM16 16 kHz
 -> 25 ms / 20 ms-hop log-mel
 -> tiny int8-weight streaming RNN
 -> pinyin-token logits
 -> shared-prefix keyword trie
 -> speech / threshold / refractory gates
 -> wake event
```

A normal phrase change is an L0 configuration update, not a model retrain. If field data still misses FAR/FRR targets, the repository provides continuous-audio calibration, hard-negative mining and shallow output-head fine-tuning.

## Product properties

- C11 + libm only in the real-time library; PyTorch and `pypinyin` stay offline.
- No heap, hidden thread, lock, filesystem or text/pinyin conversion in the real-time path.
- Caller-owned aligned engine arena; model tensors are zero-copy views into a read-only `.kwm` blob.
- Field-updatable `.kwk` keyword packs; changing a wake phrase does not require relinking firmware.
- ABI v2 binds `.kwm`, `.kwk` and generated C keyword tables to the **same 64-bit vocabulary fingerprint**. Same-sized but differently mapped vocabularies are rejected.
- Keyword updates are validated before the active trie is replaced, so a bad update does not destroy the previous valid configuration.
- L0 keyword-only update, L1 threshold/hard-negative calibration, L2 `--head-only` shallow customization.
- Default 32-feature / 48-hidden / ~420-token geometry is about **1.2 MMAC/s** for dense acoustic inference and about **26 KB** for model weights+biases. These are design estimates, not Cortex-A32 board measurements.

## Build and install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/prefix
```

Installed consumers can use either:

```cmake
find_package(KwsPipeline CONFIG REQUIRED)
target_link_libraries(app PRIVATE KwsPipeline::core)
```

or `pkg-config --cflags --libs kws-pipeline`.

## Compile custom wake phrases

Use explicit pinyin in production so tool updates cannot silently change tokenization:

```text
1    你好小窝    0.55    ni3 hao3 xiao3 wo1
2    小窝小窝    0.55    xiao3 wo1 xiao3 wo1
```

Generate a field-updatable pack, an optional firmware-linked C table and a readable manifest from the same vocabulary:

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-pack build/xiaowo.kwk \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

During exploration the fourth TSV column may be omitted and `pypinyin` can generate tone-aware pinyin. Production releases should keep the explicit column.

## Base training and shallow customization

Train a base CTC model and export ABI-v2 `.kwm` with the exact token vocabulary used by keyword packs:

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --vocab-size 420 \
  --output build/base.pt

python3 training/export_model.py \
  --checkpoint build/base.pt \
  --tokens keywords/tokens.zh.txt \
  --output build/base.kwm
```

For shallow customization, combine normal data with mined negatives and freeze the input/recurrent backbone:

```bash
python3 training/train_ctc.py \
  --manifest data/xiaowo.tsv \
  --manifest build/hard-negatives.tsv \
  --vocab-size 420 \
  --warm-start build/base.pt \
  --head-only \
  --epochs 10 \
  --output build/xiaowo.pt
```

`--head-only` requires a warm start and changes only the acoustic output head. Full-model tuning is possible but needs a broader regression corpus.

## Runtime API

The field-update path is deliberately small:

```c
kws_model_t model;
kws_keyword_pack_t pack;
kws_engine_t *engine;
kws_config_t cfg = kws_default_config();

kws_model_open(model_blob, model_blob_bytes, &model);
kws_keyword_pack_open(pack_blob, pack_blob_bytes, &model, &pack);
size_t arena_bytes = kws_engine_required_bytes(&model);
kws_engine_init(arena, arena_bytes, &model, &cfg, &engine);
kws_engine_set_keyword_pack(engine, &pack);

int detected = 0;
kws_detection_t hit;
kws_engine_accept_pcm16(engine, pcm, samples, &hit, &detected);
```

The parsed keyword pack only has to stay alive until `kws_engine_set_keyword_pack()` returns. The model blob must remain valid for the engine lifetime.

For firmware-linked generated tables, call `kws_engine_set_keywords()` with `kws_generated_vocab_fingerprint`; the same vocabulary identity check still applies.

## Continuous-audio qualification

`kws_wav` runs the **real C runtime** on mono 16-kHz PCM16 WAV files. `run_corpus.py` applies it to a reference corpus and `score_events.py` computes FAR/hour, FRR and wake latency:

```bash
python3 eval/run_corpus.py \
  --runner build/kws_wav \
  --model build/base.kwm \
  --keywords build/xiaowo.kwk \
  --references data/eval/references.jsonl \
  --audio-root data/eval \
  --detections build/detections.jsonl

python3 eval/score_events.py \
  --references data/eval/references.jsonl \
  --detections build/detections.jsonl \
  --summary build/summary.json \
  --false-positives build/false-positives.jsonl \
  --max-far-per-hour 0.2 \
  --max-frr 0.05 \
  --max-p95-latency-ms 500
```

False accepts can be converted to empty-target CTC training clips with `eval/mine_hard_negatives.py`. Never mine from the final held-out qualification corpus and then reuse it as unbiased release evidence.

## audio-pipeline integration

Preferred product chain:

```text
mic -> BF/AEC/RES/NS/AGC -> mono S16 @ 16 kHz -> kws-pipeline -> ASR/assistant
```

Avoid a second resampler when `audio-pipeline` already emits 16 kHz. Re-evaluate thresholds against the exact AEC/NS/AGC configuration shipped on the device, including local speaker playback, AEC residuals and motor/fan/gear noise.

## Validation

CI currently gates strict GCC and Clang builds, CTest, ASan/UBSan, keyword-pack/tool tests, continuous-audio metric scoring, a real default-geometry hosted benchmark, SDK install + pkg-config + clean `find_package` consumer, Python syntax, and Cortex-A32 ARMv7 hard-float cross-build.

Hosted CI numbers are regression signals only. Release qualification still requires the real trained model and target SoC to record FAR/hour, FRR, wake latency, p95/p99 processing time, CPU, memory, thermal/power and long-duration background-noise results.

See `docs/ARCHITECTURE.md`, `docs/CUSTOMIZATION.md`, `docs/PERFORMANCE.md`, `docs/INTEGRATION.md` and `THIRD_PARTY.md`.

## License

Apache-2.0. See `LICENSE`.
