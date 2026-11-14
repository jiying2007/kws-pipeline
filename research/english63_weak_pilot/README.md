# English 63: executed weak-label GRU / DS-TCN pilot (2026-11-13)

**Research-only actual CPU training**, not pretrained FSMN, streaming KWS,
phonetic ground truth, verified FAR/FRR, SSC305 qualification or a deployable
model. Both `Hey Vosen` and `Hey Mero` remain candidates. This pilot is
intentionally **not** the formal training-admission path. It does not alter
Mandarin CTC tokens, Chinese models or runtime registration.

## Immutable source and execution

- Reused original Qwen3 TTS run `43242278962`, original artifact
  `13954744587`, ZIP SHA256
  `f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc`.
- Three-source machine evidence join SHA256
  `89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12`.
- Exact offline, PCM16-only, already source-verified research input ZIP SHA256
  `c4c3cdfe363317f0b24fbb948183e5cf7ed433e7c2eef9bb46131651c9350007`.
  This archive contains 63 original WAV copies + a false-admission manifest,
  **not the original native FLOAT companions**. Retain this separate input
  privately. Do not regenerate its original audio.
- 30 five-family synthetic train-role *requested-intent* weak labels; 12
  two-family exposed development-only records; 21 ambiguous nearphone records.
  The latter **21 never enter training**; they are NOT trusted negatives.
- Input 16-kHz mono PCM16; 32-bin 25ms/10ms log-Mel; train-only per-bin
  mean/std. Deterministic single-thread PyTorch CPU, seed `20261113` for GRU,
  `20261214` for DS-TCN; 48 full-batch AdamW steps per model, no sweep,
  threshold or development-driven early stopping.
- Both encoders consume preceding context causally, **but their clip-level
  mean/max pooling is not an event-streaming detector**. A clip prediction
  must never be reported as a streaming wake or false wake.

## Actual observations from frozen one-shot pilot

| Item | GRU (6,531 parameters) | causal DS-TCN encoder (4,803) |
| --- | ---: | ---: |
| Real optimizer updates | 48 | 48 |
| Initial/last whole-clip CE | 1.113650 / 0.264593 | 1.247726 / 0.642680 |
| Weak requested-intent agreement, train only | 29/30 | 29/30 |
| Exposed development weak-intent agreement | 10/12 | 6/12 |
| Quarantined nearphones yielding target-class clip predictions | 17/21 | 14/21 |

**All these metrics are synthetic-source diagnostics, not true recall,
independent test, FAR/hour or evidence to select either model.** The 21
nearphones have unknown acoustic truth, especially across accents. These
models have no independent real-human, long-background or SSC305 evidence.
No champion or best model is selected from these exposed results.

Three separate fresh-start runs (same pinned environment and identical recipe)
produced exact matching safe `.npz` weight SHA256 values:

- GRU: `142a02b4fc0b6c6ec3b404e85bcc5383e17c4551f15341c386c61bf99740d54f`
- DS-TCN: `42ed3f8479b30aec4e639e59075beac60ad84e242dbedbb601d2bc362d403971`

Each saved weight is a base64-encoded **numpy NPZ (non-Pickle)** file. No
raw audio or individual human speaker recording is stored in this Git tree.
No donor/checkpoint weight is imported or published.

## Reproduction

With the exact archived private research input ZIP and CPU Python with
`numpy==2.3.5` and `torch==2.10.0+cpu` (training observed with Python 3.11+;
see `RESULT.json` for exact runtime):

```sh
python research/english63_weak_pilot/english63_weak_train.py \
  --input-zip /restricted/english63-verified-weak-intake.zip \
  --output-dir /restricted/new-english63-weak-replay
```

Compare SHA256 of `gru-weights.npz` and `dsc_tcn-weights.npz` to the above.
Decode the small retained weights for independent forward readback:

```sh
python - <<'PY'
from pathlib import Path
import base64
root=Path('research/english63_weak_pilot')
result=Path('/restricted/english63-saved-result')
result.mkdir(mode=0o700,parents=True,exist_ok=False)
for filename in ('gru-weights.npz','dsc_tcn-weights.npz'):
    (result/filename).write_bytes(base64.b64decode(
        (root/(filename+'.b64')).read_text().strip(),validate=True))
(result/'RESEARCH_RESULT.json').write_bytes((root/'RESULT.json').read_bytes())
PY
python research/english63_weak_pilot/verify_english63_weak_train.py \
  --input-zip /restricted/english63-verified-weak-intake.zip \
  --result-dir /restricted/english63-saved-result
```

`verify_english63_weak_train.py` refuses mismatched source/code/checkpoint
hashes, verifies train-only normalizer and recomputes **all 63** stored
clip-predictions. The source-only regression cases test forged flags,
nearphone promotion, missing/changed inputs and altered weight digests in
normal/-O/-OO modes without invoking training or Torch.

## Research next stage

The result is a **reproducible failure/limitations baseline**, not the
requested strong-pretrained English FSMN. Before legitimate new candidate
training or cross-model selection, independently establish acoustic
pronunciation and voice-family authority, collect verified accent and
background recordings, freeze source-disjoint train/development/test,
then run fixed-budget strong-pretrained FSMN and light alternatives on
**identical authenticated data**, with matching continuous event rules.
Reject nearphones only with trustworthy acoustic labels; never tune a
prototype on the exposed 12/21 records. Finally measure C export parity,
SSC305 A32 CPU/RSS/p99 end-of-word event latency and continuous FAR/hour.


## Executed saved-weight encoder chunk parity (2026-11-13)

The original **unmodified** weak GRU and causal DS-TCN checkpoints also admit
memory-bounded *causal encoder + mean/max accumulator* replay. The separate
`streaming_parity.py` loads only their already saved safe NPZ weights and the
byte-verified 63-clip source archive. It never trains, downloads models,
changes a threshold or rewrites an admitted dataset. No input/source hash,
speaker split or model weight was modified.

On the same 63 synthetic clips, each encoder was compared to its exact
whole-clip forward pass under four input-frame partitions:
`1`, `5`, `31` and `whole`. Recorded in `STREAMING_PARITY.json`:

| Measurement | GRU | DS-TCN |
| --- | ---: | ---: |
| Source clips | 63 | 63 |
| Total chunk/batch comparisons | 252 | 252 |
| Changed final clip class | 0 | 0 |
| Maximum absolute final-logit difference | 9.536743164e-7 | 7.152557373e-7 |

History is explicitly retained across frames/chunks: GRU hidden state,
causal convolution history per dilation, running mean numerator/frame count,
and running elementwise maximum. No full-sequence intermediate encoder
activation needs to be retained. These measurements do **not** benchmark
target RAM or CPU and do not imply real-time / SSC305 qualification.

To reproduce from the original private input ZIP and the safely decoded
saved result directory (as described above):

```sh
python3 research/english63_weak_pilot/streaming_parity.py \
  --input-zip /restricted/english63-verified-weak-intake.zip \
  --result-dir /restricted/english63-saved-result \
  --output /restricted/english63-new-chunk-parity.json
```

The output is create-only, binds the unmodified research source, input
and checkpoint identities, and refuses any partition with changed final
class or absolute logit difference above 2e-5. Existing normal/-O/-OO
CI imports only source/receipt safety checks; no model or audio is loaded
there.

**Boundary:** This is stateful *precomputed-feature encoder* parity, not
streaming PCM frontend parity or a continuous keyword-event detector.
The existing mean/max head still produces a whole-clip classification; it
does not establish trigger timing, wake-event FAR/hour, genuine accent
recall, human label truth or final model suitability. Do not select either
weak classifier as the final English KWS model from this evidence.


## Executed original-PCM streaming frontend parity (2026-11-13)

A **new and separate** source-bound PCM-to-log-Mel research check was executed
against the **unchanged original** synthetic English63 TTS ZIP, not newly
generated audio, and the exact three-evidence source-review JSON:

- Original TTS artifact 13954744587, ZIP SHA256
  f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc
- Three-evidence joined JSON SHA256
  89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12
- Original **63/63 PCM16 WAV + PCM hashes** rechecked against the pinned review.
- Five PCM chunk sizes: **159, 160, 161, 1600, 3199 samples**. Each yielded
  the same 13,010 25-ms/10-ms feature frames across 63 clips.
- Across **315 clip/chunk-plan comparisons** (65,050 compared frames),
  maximum absolute log-Mel difference was **1.9073486328125e-6**,
  below the fixed 1e-5 budget. No truncation or EOF padding.
- The largest observed pending-window + input-chunk sample count was 3,518,
  measured in the frontend implementation. It is **not** total process RAM
  nor SSC305 measured memory.
- The machine result PCM_FRONTEND_PARITY.json binds the source archives,
  execution script SHA256 and explicit non-promotion boundaries. The existing
  hash-bound saved model and encoder-parity sources were **not modified**.

To reproduce with privately retained exact original ZIP and reviewed JSON
(extracted from the original source-bound review artifact):

    python3 research/english63_weak_pilot/pcm_frontend_parity.py \
      --tts-zip /restricted/english63-original-tts.zip \
      --review-json /restricted/english63-three-evidence-review.json \
      --output /restricted/new-pcm-frontend-parity.json

Only the small result, algorithm source and **no-audio/no-numpy/no-Torch**
source-only tests enter Git or existing normal/-O/-OO CI. No raw WAV, new
scheduled job, model inference, TTS, ASR, IPA or optimizer execution is added
to CI. This proves a streaming **frontend's frame/log-Mel parity** against its
existing batch algorithm, **not** PCM-to-model logit parity, continuous wake
event emission, independent FAR/hour, human FRR, acoustic truth, legitimate
English FSMN training, model selection or SSC305 deployability.


## Original PCM to saved GRU/DS-TCN incremental inference

The earlier #797 encoder test began with precomputed features; #798 measured
streaming PCM frame/log-Mel equivalence without a model. The separate
integrated_pcm_encoder.py combines both without editing either hash-bound
baseline or the actual saved model weights.

It consumes immutable original 63-clip TTS ZIP and three-evidence review
JSON, validates original WAV/PCM and role identities and saved RESULT.json +
NPZ digests, recomputes **train-only** frontend normalization, then compares
five PCM chunk plans (159/160/161/1600/3199 samples) against the original
whole-clip forward for both saved research encoders. The incremental path
retains GRU hidden state or DS-TCN causal histories and running mean/max
accumulators; it does not retain the full sequence of encoded frames.

The first full-chain execution uses a source-pinned branch in the public
kws-data repository that owns the existing GitHub Actions artifacts; its own
read-only Actions token accesses only those unmodified original archives. It may
download and run **inference on previously generated bytes only**; it does
not generate new TTS/ASR/IPA/labels, optimize a network or alter admission.
CI's existing normal/-O/-OO entrypoint imports entirely source-only hostile
metadata tests. The one-shot result is a separate **machine-only** research
artifact, never raw WAV. In case of source expiration or mismatch the
run must fail rather than regenerate.

Successful end-logit or class parity is **not** independent English
pronunciation truth, continuous wake event emission, FAR/hour, human FRR,
an accent/generalization result, strong-pretrained FSMN, C/int8 export or
SSC305 board qualification. No champion, formal data admission or candidate
promotion is authorized. Hey Mero and Hey Vosen remain unselected.


### Actual measured full-chain execution and exact receipt

Data-owning repository one-shot Actions run
[43532002008](https://github.com/jiying2007/kws-data/actions/runs/43532002008)
completed **success**, using pinned pipeline research source commit
8c6f34b110bea6d21fcf4659a85f0f6a635b5469. Its run artifact
14087074624 was independently downloaded. Original artifact ZIP SHA256:
f208fc4ab911c67f4bb988671ce67dde927b4f0d6421e5552642c24ab9b2733e.
The extracted machine receipt, now retained **byte-identically** as
INTEGRATED_PCM_ENCODER_PARITY.json, has SHA256
0200368be4813885b3229411cf1821c49ad27ea899b85056d5483238fe292e1a.

Both models consumed the exact original 63 WAVs with the same train-only
normalizer and unchanged saved weights. For each model, five PCM chunk sizes
were compared against a fresh one-clip full-batch forward (315 comparisons):

| Saved research model | Final clip class differences | Largest abs. final-logit difference |
| --- | ---: | ---: |
| GRU | 0/315 | 8.940696716308594e-7 |
| DS-TCN | 0/315 | 5.960464477539062e-7 |

The fixed preregistered 2e-4 difference cap passed without tuning or changing
weights. The experiment does **not** measure keyword event emission at any
intermediate frame: whole-clip mean/max pooling is still a final clip
classifier. Neither original prompt weak intent nor three ASR agreement is
independent acoustic gold. No FAR/hour, FRR, target latency, int8/C export,
real speaker generalization or SSC305 runtime can be inferred.
