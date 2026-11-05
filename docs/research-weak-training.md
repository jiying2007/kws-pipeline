# D0 weak-label consumption diagnostic

`training/research_weak_train.py` is a separate research entry. It consumes the
verified `rnn-four-token` view produced by `active_asr_weak_view.py`; it does not
change product training, a frozen campaign, or candidate promotion rules.

The final D0 view has 40 representable full-text CTC items: 12 complete-keyword
positives and 28 nearphone negatives. Its eight other accepted weak texts are
outside the four-token vocabulary and are excluded, never mapped to blank.
All sources are train-only, all labels are machine weak labels, and there is no
independent development or test set. The 16 uncertain items are not admitted.

## Fixed recipe before the first optimizer update

The two main runs are the existing RNN64 and C1 causal depthwise TCN (hidden 64,
kernels 3/3/3, dilations 1/2/4). They use the same 40 feature tensors and full
targets, seed 20261105, three epochs, four clips per optimizer update, and the
same shuffled order. Each source receives exactly three training exposures.
The optimizer is AdamW with learning rate 0.001, betas 0.9/0.999, epsilon 1e-8,
zero weight decay, and gradient clipping at norm 5. No schedule is tuned using
the observed loss. Architecture parameter counts may differ and are recorded.

The raw ASR weights sum to 3 for keyword 1, 5.5 for keyword 2, and 23.5 for near
negatives. Class multipliers 1, 6/11, and 4/141 yield effective masses 3, 3, and
2/3. Thus both keywords are balanced and near negatives receive 10% of the
total coefficient mass. Every representable clip is still consumed. This is
an explicit diagnostic loss recipe, not a validated sampling policy or a
0/5/10 replacement experiment: D0 has no ordinary-negative pool. Coefficient
mass does not bound observed numerical loss contribution; the latter is
recorded separately by class.

Each sequence CTC loss is divided by full target length, multiplied by its
effective sample weight, then averaged over the batch using the fixed snapshot
mean weight (1/6). The denominator is never a minibatch sum of weights, which
would cancel weights in uniform or one-item batches. Both encoders receive
individual unpadded sequences; gradients are aggregated across four clips.
The existing frontend uses 32 logmel features with complete windows only.
Each clip starts with a fresh encoder state. Both models use the existing
deterministic one-thread CPU contract, below the four-thread budget.

A separate one-epoch probe freezes the final C1 encoder and learns a 64-to-2
keyword head from mean hidden frames. Its binary loss applies each keyword's
mask, ASR weight and the same class multiplier. This probe is excluded from
the main RNN/C1 comparison. All 80 keyword slots in the actual admitted D0 view
are known; synthetic gradient tests additionally exercise unknown masked slots.

## Execution and evidence

Use the pinned core Python 3.12.16 environment. An extracted Actions Python
archive may require its `lib` directory in `LD_LIBRARY_PATH`; the interpreter
and core package versions are checked by the entry.

1. `prepare --view ... --view-manifest-sha256 ... --view-loader ... --output ...`
   runs the verified data adapter, binds WAV and PCM hashes, checks all CTC
   geometry, and freezes the recipe, source code, rows and feature cache.
2. `run --prepared ... --output ... --heavy-lock ... --seconds 600` starts the
   three items sequentially, with a nonblocking shared heavy-model lock and a
   600-second process deadline per item. Existing output directories are
   refused. A failure is preserved and stops the sequence; there is no retry.

Private outputs include the source rows, feature cache, per-sample consumption
ledger, initial and each-epoch checkpoints, optimizer state, RNG state, state
hashes, training-set loss readbacks, timing and peak RSS. Training-set readback
is diagnostic only and never called a generalization score. No quality winner,
production qualification, decoder result or export is emitted. Keep private
audio, text, features and checkpoint bytes outside the public repository.

Run synthetic tests with `python -B tests/test_research_weak_training.py`; add
`--numeric` in the pinned core environment to check actual gradient scaling,
zero masked gradients, unpadded model geometry and checkpoint readback.

## Separate official FSMN2599 consumption diagnostic

`training/research_fsmn_weak.py` extends consumption to all 48 representable
full-text items in the FSMN view. It uses the official
`thuduj12/kws_wenwen_fsmn_ctc` revision
`cd0c18d0dda6742c371fbb72ec66185906a9885d`: `avg_30.pt` SHA256
`d3d89b5dde08c2926d46a97eab6f7179a87b1946ccebd16c02e77042ea8389fe`
and its matching original vocabulary. The old B, A20 and iic/base checkpoints
are not inputs to this new diagnostic. The source lock records every official
URL, revision, byte count and checksum. Source-only WeKWS FSMN/CMVN and the
Torch fbank, context-expansion and frame-skip functions are verified before
loading; downloaded source bytes and licenses remain in the private evidence.
No provider SDK is installed. Public CI never downloads model weights.

Vocabulary IDs are the official raw token IDs minus one; blank is zero and
all full-text characters must have positive in-range IDs. All eight additional
texts are preserved in the 2,599-class output space. The frontend uses unpadded
PCM16 at 16 kHz, 80-dimensional default Povey fbank, left/right context 2 and
frame skip 3. The source-only frontend omits only unused torchaudio MFCC code
and its native import. Fbank mathematics and the three selected processor
functions are unchanged. Diagnostic dither is zero and no SpecAugment is
applied; this is explicitly different from the donor's pretraining recipe.
Two unavailable final context frames are dropped; audio is never extended.

The single fixed sequence has one epoch of 48 clips per stage, batch size four,
seed 20261105 and no selection or quality retry. Stage `head` updates only the
final 140-to-2599 affine with AdamW at 1e-4. Stage `tail` starts from the exact
completed head checkpoint and additionally updates the last FSMN block and
the 250-to-140 output projection at 1e-5; its final affine remains at 1e-4.
Both stages use betas 0.9/0.999, epsilon 1e-8, zero weight decay and clip norm 5.
All other tensors, including CMVN and the first three FSMN blocks, must retain
their exact bytes. The optimizer is fresh for the second stage and that choice
is fixed before training.

The 36 near negatives have raw ASR mass 28.5. Multiplier 4/171 yields mass 2/3;
the keyword masses remain 3 and 3, using the same class balancing as the main
pair. The fixed mean loss coefficient is 5/36. Each stage consumes all 48
sources once, records all full-text targets and source/PCM/label/weight hashes,
and preserves initial/final model and optimizer state. The head/tail stages
form a sequential adaptation diagnostic, not two independent contestants.

Use `fetch --sources ...` to obtain or verify the source pack, then `prepare`
with the fixed view/manifest/loader/output and `--heavy-lock`. `run` takes the
prepared directory, a new output directory, the same lock and `--seconds 600`.
The shared lock supports explicit descriptor inheritance from a local research
queue and remains held across the two stage workers. Original private weights,
audio, weak text, features and checkpoints are excluded from the repository.
