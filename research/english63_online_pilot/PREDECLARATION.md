# English63 finite-window weak-MIL pilot — V4 E02, 2026-11-14

## Question and scope

Can the two **already trained** weak GRU / DS-TCN encoders produce bounded,
causal, continuously available research proposals after replacing their
whole-clip readout with a finite-window readout and one fixed weak-MIL
warm-start experiment? This is a **system candidate comparison**, not proof
that one isolated algorithmic change improves acoustic performance.

The new proposals are not authenticated wake events. Zero proposals,
excess proposals and poor weak-intent agreement are valid scientific results,
not execution failures and not reasons to change the threshold or run again.
No winner is selected from the already exposed source records.

## Immutable inputs and roles

- Original TTS run: `jiying2007/kws-data` / `43242278962`.
- Original artifact: `13954744587`; ZIP SHA256
  `f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc`.
- Joined review run: `43346200939`; artifact `14001747172`; ZIP SHA256
  `f83d6a2f399c20af5bbc920ef6c38c42ac1e44b516d2b7291f4510557491cec8`.
- Exact inner review JSON SHA256:
  `89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12`.
- Reuse all 63 original PCM16 WAVs, exactly **2,101,760 PCM samples**.
  No new TTS, ASR, phone inference, source conversion or acoustic labeling.
- Only 30 five-preset-family train-role recordings participate in optimization.
  They keep requested-intent **weak** clip labels, not acoustic truth.
- The 12 exposed development recordings never tune a checkpoint or threshold.
  All 21 ambiguous nearphone recordings are excluded from the loss.
- The existing byte-bound label bridge and original-audio loader are reused
  unchanged. The actual `training/audit_dataset.py` audits the 63-file inventory
  and the observed train/development source-family split before training.
  Synthetic presets are not counted as real humans. Unknown identities remain
  unknown; a clean observed split is not independent-speaker qualification.

Warm starts remain the saved safe NPZ files in `english63_weak_pilot`:

- GRU: `142a02b4fc0b6c6ec3b404e85bcc5383e17c4551f15341c386c61bf99740d54f`.
- DS-TCN: `42ed3f8479b30aec4e639e59075beac60ad84e242dbedbb601d2bc362d403971`.

All old hash-bound Python sources, model bytes, receipts and admission flags
stay unchanged. The Mandarin CTC vocabulary and device runtime are unaffected.

## Fixed recipe, no tuning

- CPU single-thread; `numpy==2.3.5`, `torch==2.10.0+cpu`, MKLDNN disabled,
  deterministic algorithms enabled. Record the actual Python version.
- Preserve the saved 32-bin, 25-ms/10-ms log-Mel and train-only normalizer.
- Preserve encoder topology and all saved weights as the initialization.
- Use the trailing **200 encoded frames** for mean/max readout. The encoder
  itself remains causal: finite convolution carry or a constant-size GRU state.
  The GRU hidden state is bounded in memory, not a claim of a finite acoustic
  receptive field. No whole-stream running maximum is retained.
- Full-batch 30 weak train-role clips; **48 AdamW updates per model**,
  learning rate 0.001, weight decay 0.01, gradient norm cap 1.0.
- GRU seed `20261114`; DS-TCN seed `20261215`. No seed/configuration sweep.
- Weak clip loss: smooth maximum of each target-vs-background frame margin,
  temperature 0.25, normalized by the number of eligible frames, then 3-class
  cross entropy with a fixed zero background bag logit. Frames before 50
  observed frames are masked. This creates **no positive frame labels or word
  endpoints** and does not assert weak ordinary-request audio is truly negative.
- Research proposal policy: softmax target probability at least 0.80,
  same class for three consecutive frames, 100-frame cooldown. These are
  preregistered experiment constants, not calibrated operating thresholds.
- Report every result; no early stopping or development-driven selection.

## Execution and readback

```sh
python3 research/english63_online_pilot/online_pilot.py \
  --tts-zip /restricted/original-english63.zip \
  --review-json /restricted/original-three-evidence-review.json \
  --output-dir /restricted/new-online-pilot

python3 research/english63_online_pilot/online_pilot.py \
  --tts-zip /restricted/original-english63.zip \
  --review-json /restricted/original-three-evidence-review.json \
  --verify-result-dir /restricted/new-online-pilot
```

The output directory must not exist; its parent must exist. Input validation
failure creates no output. Failure after training starts keeps partial files
and `FAILED.json`, exits nonzero and cannot be reported as a completed result.
Successful output contains `SUMMARY.json`, `REPLAY.json`, `events.jsonl`,
`gru-weights.npz`, `dsc_tcn-weights.npz`, never raw audio. NPZ readback uses
`allow_pickle=False` and verifies every tensor shape, finite value and digest.
The verification command is read-only and performs zero optimizer updates.

Before actual training, runner-only synthetic tests check real autograd,
training-vs-incremental logits, prefix causality and finite readout carry.
Source-only tests run in existing normal/-O/-OO CI without Torch or audio.

After training, each model is evaluated with fixed 160-sample and mixed
1/159/161/3199/320/7-sample inputs. Every actual frame is computed separately,
making caller chunk sizes irrelevant to arithmetic order. The score trace,
proposal stream, actual PCM clock and saved-weight readback must agree.

The same 63 WAVs are also concatenated in sorted cell-ID order without resets,
pauses or artificial tail samples; both chunk plans preserve continuous state.
This exceeds the old 1,200-frame limit and tests bounded carry, not natural
conversation or independent negative exposure. Files and reordered/replayed
copies never multiply independent audio hours. EOF emits no frame/proposal;
unconsumed partial windows remain unpadded. Event time is an **availability
sample index**, not a labeled word-end latency or real-time scheduling result.

## Acceptance and stopping

An execution passes only when source/role audits, finite gradients, 48 actual
updates, changed checkpoint bytes, chunk invariance, no-padding clocks and
fresh-NPZ prediction reproduction pass. There is no minimum proposal count,
weak-agreement score or model improvement gate. A genuine numerical or source
failure ends the run and retains its evidence; fix only that demonstrated fault
before considering a new request.

Every receipt keeps `formal_training_admitted=false`, `human_gold=false`,
`model_selection_made=false`, `model_promotion_allowed=false` and
`threshold_selection=false`. Independent English FRR, continuous-negative
FAR/hour, acoustic word-end latency and SSC305 profile remain `NOT_RUN`.
The resulting weights are research candidates, not a deployment SDK.
