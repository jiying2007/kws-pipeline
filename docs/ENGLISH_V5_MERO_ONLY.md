# English KWS V5.0 — Hey Mero only

**Current active English keyword: Hey Mero** `/heɪ ˈmɛroʊ/` (American English, *Hey MEH-roh*). The prior two-word research profile and all historic model/recording identities remain byte-preserved for reproducibility, **not** as active English wake products. The other historic word is not a new positive, false positive reference, hard negative or runtime event class in V5.0.

Machine profile: [english-v5-mero-only.json](../configs/research/english-v5-mero-only.json).

## Architecture and current evidence

V5 introduces a causal FSMN-CTC research model in [kws-data](https://github.com/jiying2007/kws-data/blob/main/automation/english_v5_mero_fsmn_ctc_once.py), with its own new **4-token vocabulary**: `<blank>`, `HEY`, `MEH`, `ROH`. Original whole-PCM training labels are restricted to independently accepted Hey Mero and explicitly ordinary source rows, grouped by original source-family identity; the training optimizer never uses legacy other-keyword rows, uncertain variants, old calibration, or already exposed locked-test audio. Its initial CPU PyTorch/ONNX output did not alone establish a C-native acoustic runtime; see the later verified C11 evidence below.

The initial CPU PyTorch/ONNX export was **not** proof of a C-native acoustic engine; later [pipeline #840](https://github.com/jiying2007/kws-pipeline/pull/840) delivered the complete PCM16→FFT512/Hann400→40Mel→43,876-param FSMN→CTC streaming C11 implementation. [data #266](https://github.com/jiying2007/kws-data/pull/266) and [#267](https://github.com/jiying2007/kws-data/pull/267) proved original-PCM PyTorch↔C11 (maximum absolute logit difference 0.0001220703125) and x86↔Cortex-A32 QEMU (maximum difference 0.0) on the ten already-exposed synthetic DEV originals. [pipeline #842](https://github.com/jiying2007/kws-pipeline/pull/842) subsequently merged an installable standalone C11 package. **These are software parity and integration results**, not fresh acoustic quality, physical SSC305 performance, natural FAR/h or 3–5 m human FRR.

The bounded decoder `research/english_v5_mero/mero_ctc.c` retains greedy full-sequence CTC behavior, reset/discontinuity, maximum token gap and cooldown. The frozen old `ekws_research_stream` is a different historical runtime and is never relabeled V5. The new [V6 quality evidence contract](../research/english_v5_mero/V6_QUALITY_EVIDENCE.json) and [V6 execution SDD](ENGLISH_V6_MERO_EXECUTION.md) define independent B0/B1 comparison and explicitly prevent treating exposed DEV or synthesized source requests as qualification evidence.

The development output is not independent FAR/FRR. In particular, a success/failure observed in a few synthesized voices, or a low token loss, cannot establish 3–5 m microphone performance.

## Required sequential verification

1. **L1 — research source and actual model:** authentic source/screen ZIP and all WAV/PCM hashes, train-only source groups, source-disjoint dev, actual finite optimizer updates, state/ONNX/NPZ weight hashes, and reproducible *single-keyword* CTC token predictions. If training/export fails, retain exact STOP and do not invent final model weights.
2. **L2 — ARM32 software:** independently implement and validate the exact causal LogMel/STFT, train-only normalization, every FSMN block and CTC score/event sequence against the frozen L1 weight+input tuple. Then measure streaming chunks, reset/clock/gaps, memory allocation bounds, QEMU ARM parity and installable C API. Neither CTC decoder unit tests nor export alone satisfy L2.
3. **L3 — SSC305 and independent acoustic:** obtain real BSP compiler/sysroot and board access, final audio/AFE callbacks, new untouched calibration + locked evaluation, genuine real-distance/room/robot background continuous recordings and repeatable CPU/RAM/power testing. Goals remain Hey Mero FRR ≤5%, combined (now one-keyword) FAR ≤0.1/h, and word-end-to-consumable callback P95 ≤500 ms with one-sided confidence upper bounds.

The existing [SSC305 connection and hardware evidence checklist](ENGLISH_SSC305_ENTRYPOINT.zh-CN.md) may be reused for board identity, BSP and PCM transport **but its previous two-word frozen embedding SDK binary is not the V5 acoustic model**. A new verified V5 model/package must be used; do not substitute names or model IDs.

## Resource and execution restrictions

Research only until independent board+acoustic evidence. Preserve previous tests, data source identities and model hashes. No regeneration of prior approved 64/63/168 clips; no timed patrols, private repositories, Work/Codex allocation, paid model API, or use of other-keyword audio as automatically negative. Event-driven training should run actual numerical work and retain honest failures, with one primary V5 model, one pretrained accuracy benchmark, and at most one lightweight competing model.
