# Startup-context learnability (development only)

> Historical retained evidence. The executable diagnostic lane and its self-tests were retired during repository cleanup after the result was captured; this document remains as evidence, not as a current runnable entry point.

This lane responds to issue #242's retained #296 counterexample: the current
model's apparent successes disappear after one second of digital silence.
Cross-CPU training hashes do not explain a same-model inference intervention.
Keep the historical numerical failure; do not use it as a dependency of this lane.

## Fixed scope

Two existing wake phrases, the existing five-token vocabulary, 32-feature/H64
shipping RNN, logmel frontend, C model format and actual C decoder. Keyword
thresholds remain the immutable pool's 0.55. No model/runtime/threshold shipping
change, qualification consumption, new TTS, curriculum or adversarial mining.
The existing immutable speech base is verified and only its train/calibration/
development-test rows enter the portable pool. Train has 128 originals; all
training source/feature/label decisions use train alone. The one-worker paired
job avoids cross-vendor numerical differences as a confound of these trials.

## Declared controls

All three variants start from the same seeded model, use the same 128 training
rows, shuffle order, uniform sample weights, target-length-normalized CTC,
AdamW 0.001, batch 16 and 600 epochs. This new CTC backbone is **not** the old
36-epoch per-frame-normalized multi-loss recipe, so their quality difference
cannot be attributed to one change. Original auxiliary losses are explicitly 0.

1. `plain-ctc`: original clips, full-clip CTC.
2. `context-ctc`: 0/0.2/0.5/1-second deterministic random PCM lead-in and
   0.5-second tail, full-clip CTC.
3. `grounded-ctc`: exactly the same contexts as (2), but CTC restricted to a
   waveform activity envelope plus 0.3-weight blank loss outside the envelope
   with a 40-ms boundary guard. This tests a **combined** treatment, not the
   marginal contribution of span restriction versus blank supervision.

The activity envelope uses the existing 2%-of-peak/min-64 PCM detector for BOTH
wake and nonwake transcripts. It is not forced model alignment, phoneme labels,
or a general-purpose VAD. No test result selects an epoch or threshold. Every
100 epochs a diagnostic weights-only checkpoint is retained; it is not an
optimizer/RNG resume checkpoint. A second grounded seed (2346) is declared in
advance. Scope `positive-only` exists solely for debugging and is never promoted.

## Actual C acceptance

Each model is evaluated on every frozen split, at 0/0.2/1/2/5-second per-clip
lead-ins and in a single state-carrying mixed positive/nonwake stream with
2-second gaps. Standalone 小窝 and incomplete phrases are among original
nonwake transcripts. Also evaluate 1/5/10-second pure silence.

For a separate, explicitly edited counterfactual, blank only the nine posterior
frames entirely inside the added 200-ms silence. Preserve every later logit,
timestamp and speech flag; replay using the unchanged C decoder. Retain both
trace byte identities and per-record detections. Counterfactuals are never
called authentic model output or qualification evidence. Inspect actual C root
admissibility inside activity bounds after a 1-second lead-in.

The **basic train learnability gate** requires all training events, zero training
false accepts for all prefixes and the continuous stream, all training root
support in activity, no lost matches under startup-prefix blanking, and no
silence-only events. Controls are expected to be allowed to fail. CI explicitly
gates this property on both grounded seeds; infrastructure completion alone is
not success. Calibration/test are development feedback, not release holdouts.

## Local preliminary observations (torch 2.10.0+cpu)

| Recipe, seed | Train at every prefix / stream | Cal at 1 s | Test at 1 s |
|---|---|---|---|
| plain-ctc, 1337 | 0/32, 0 FA | 0/16, 0 FA | 0/16, 0 FA |
| context-ctc, 1337 | 0/32, 0 FA | 0/16, 0 FA | 0/16, 0 FA |
| grounded-ctc, 1337 | 32/32, 0 FA | 14/16, 0 FA | 14/16, 0 FA |
| grounded-ctc, 2346 | 32/32, 0 FA | 13/16, 3 FA | 15/16, 4 FA |

The first exploratory positive-only recipe still missed wakes and produced
nonwake false accepts; it was not adopted. An earlier 200-epoch small-positive
probe failed full transcription; preserve that observation. Final local source
changes after preliminary runs added receipts and per-record evidence only;
remote pinned-runtime runs provide separate final-tree evidence, not implied by
these numbers. No claim that calibration/test generalization is solved.

## Remaining product work

Resolve cross-speaker errors using development-only paired controls, expand
public/synthetic non-target speech and background, test non-silent lead-ins and
real stream interruptions, then reintroduce acoustic stress and multi-seed
selection. Frozen real-human, final AFE and target-board evidence remain release
requirements. A direct whole-word classifier is a conditional reference if basic
CTC learning remains inadequate, not an automatic dependency or runtime rewrite.
Model promotion is never authorized by the new experimental recipe.
