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
