# English KWS V5.0 — Hey Mero only

**Current active English keyword: Hey Mero** `/heɪ ˈmɛroʊ/` (American English, *Hey MEH-roh*). The prior two-word research profile and all historic model/recording identities remain byte-preserved for reproducibility, **not** as active English wake products. The other historic word is not a new positive, false positive reference, hard negative or runtime event class in V5.0.

Machine profile: [english-v5-mero-only.json](../configs/research/english-v5-mero-only.json).

## Architecture and current evidence

V5 introduces a causal FSMN-CTC research model in [kws-data](https://github.com/jiying2007/kws-data/blob/main/automation/english_v5_mero_fsmn_ctc_once.py), with its own new **4-token vocabulary**: `<blank>`, `HEY`, `MEH`, `ROH`. Original whole-PCM training labels are restricted to independently accepted Hey Mero and explicitly ordinary source rows, grouped by original source-family identity; the training optimizer never uses legacy other-keyword rows, uncertain variants, old calibration, or already exposed locked-test audio. Its CPU PyTorch/ONNX output does not make it C-native.

C source in `research/english_v5_mero/mero_ctc.c` is **only the research event-decoding layer**, not a trained acoustic inference engine. It folds repeated tokens according to greedy CTC, requires the complete ordered three-token phrase, enforces monotonic sample clocks and bounded token gaps, and provides reset, discontinuity and cooldown handling. This is entirely distinct from the frozen old `ekws_research_stream`, which retains its historical original class mapping and exact binary requirements; it is not silently relabeled as Hey Mero V5.

The development output is not independent FAR/FRR. In particular, a success/failure observed in a few synthesized voices, or a low token loss, cannot establish 3–5 m microphone performance.

## Required sequential verification

1. **L1 — research source and actual model:** authentic source/screen ZIP and all WAV/PCM hashes, train-only source groups, source-disjoint dev, actual finite optimizer updates, state/ONNX/NPZ weight hashes, and reproducible *single-keyword* CTC token predictions. If training/export fails, retain exact STOP and do not invent final model weights.
2. **L2 — ARM32 software:** independently implement and validate the exact causal LogMel/STFT, train-only normalization, every FSMN block and CTC score/event sequence against the frozen L1 weight+input tuple. Then measure streaming chunks, reset/clock/gaps, memory allocation bounds, QEMU ARM parity and installable C API. Neither CTC decoder unit tests nor export alone satisfy L2.
3. **L3 — SSC305 and independent acoustic:** obtain real BSP compiler/sysroot and board access, final audio/AFE callbacks, new untouched calibration + locked evaluation, genuine real-distance/room/robot background continuous recordings and repeatable CPU/RAM/power testing. Goals remain Hey Mero FRR ≤5%, combined (now one-keyword) FAR ≤0.1/h, and word-end-to-consumable callback P95 ≤500 ms with one-sided confidence upper bounds.

The existing [SSC305 connection and hardware evidence checklist](ENGLISH_SSC305_ENTRYPOINT.zh-CN.md) may be reused for board identity, BSP and PCM transport **but its previous two-word frozen embedding SDK binary is not the V5 acoustic model**. A new verified V5 model/package must be used; do not substitute names or model IDs.

## Resource and execution restrictions

Research only until independent board+acoustic evidence. Preserve previous tests, data source identities and model hashes. No regeneration of prior approved 64/63/168 clips; no timed patrols, private repositories, Work/Codex allocation, paid model API, or use of other-keyword audio as automatically negative. Event-driven training should run actual numerical work and retain honest failures, with one primary V5 model, one pretrained accuracy benchmark, and at most one lightweight competing model.
