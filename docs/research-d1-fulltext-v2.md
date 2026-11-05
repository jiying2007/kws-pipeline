# D1 full-text v2 foundation

This is a new research candidate and recipe boundary. It does not report a D1
training run, human evaluation, model winner, or target-device qualification.
The historical five-output EncoderSpec v1, D0 40/48-row recipes, official token
bytes, official checkpoint, and saved D0 receipts retain their original meaning.

## Vocabulary and candidate identity

The official FSMN token file is pinned by SHA-256
`5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa`.
Raw IDs 1..2599 map to model outputs 0..2598. `sil` and `<blk>` share output 0;
`<eps>` and the `#0`-onward decoder disambiguation symbols are not outputs.
Consequently, appending lines to the original token file would be incorrect.

`training/research_d1_vocab.py` creates a separate versioned mapping from that
unchanged file and nine predeclared normalized D1 task texts. It appends 沃 at
output 2599 and 灯 at output 2600. All original output IDs stay unchanged. The
mapping and task-text declaration are hashed into every new recipe and model
spec. Observed development text never expands the vocabulary. A future
confirmed transcript containing another unknown character remains a complete,
explicitly unsupported CTC transcript; characters are never deleted or mapped
to blank. Intent texts determine vocabulary capacity, never observed labels.

Fresh RNN64 and C1 reuse their existing encoder equations with 2601-output
heads. `FullTextEncoderSpec` has schema `candidate-encoder-fulltext-spec-v2`;
it does not relax the five-output v1 class. The new FSMN candidate authenticates
the official donor, copies all 2599 old output rows and all other tensors
exactly, and initializes only two appended rows with a separate CPU generator,
seed 20261105 and uniform bounds ±1/sqrt(140). Its receipt records old/new row
hashes. It is called `fsmn-d1-ctc2601-v2`, never the original 2599 checkpoint.
Added softmax classes change probabilities even before an optimizer step;
old-row equality is not a claim of identical recognition or calibration.

## Verified rows and distinct objectives

`training/research_d1_inputs.py` fully exhausts the maintained data loader,
which replays every retained native source, frozen plan, ASR result, PCM
derivative and weak decision before filtering its legacy CTC projection.
The new adapter binds the original manifest/report/input bytes before and
after replay, then projects the already verified report. It neither modifies
the old package nor implements another label decision. This preserves rows
whose old four-character or official-2599 CTC consumer was unsupported.

Each input package has at most 64 rows. Aggregate identity is the pair of
frozen plan SHA and cell/source ID. Package-local `clip-000` IDs can repeat;
global cells and original PCM cannot. All rows, including uncertain rows,
participate in cross-package family, speaker, PCM and reference-ancestor split
checks. Partial aggregate coverage is recorded as partial, not 256 completed
items. The current source-adapter contract requires one common 256-cell plan.

The aggregator explicitly accepts source adapters
`kws-active-d1-frozen-plan-source-v1` and
`kws-active-d1-frozen-plan-source-v2`. The latter distinguishes completed
generation from `failed-native`; it retains the actual failure receipt, its
hash and all recognizer observations. Finite, decodable failed audio may be
retained for audit, but its full-text target is null and every CTC/keyword mask
and weight must be zero. The aggregator and objective projection each reject
an attempted promotion of that failure into training, even when recognizers
agree. A native return that cannot become a legal audio input remains in the
producer's excluded stage ledger; the consumer never fabricates a PCM row or
counts that cell as an observed weak-view input.

Sparse ASR retains the explicit `not_requested` observation and routing hash.
It does not become a failed observation, a negative label, or a third vote.
The maintained data loader still replays every source and requested native
result before deriving labels. The training adapter preserves its original
two-vote confidence weight and all three observation slots without relabeling.
Legacy adapter-v1 rows keep their existing projection; mixed v1/v2 batches
still share the same global cell/PCM uniqueness and family/speaker split checks.

The common CTC cohort contains confirmed training rows whose complete text is
in the fixed 2601 mapping and whose unpadded CTC alignment is feasible for all
three frontends. Actual prepared features must also pass the frame-count
check. Future unknown characters or short audio remain explicit exclusions.
The keyword projection separately retains confirmed per-word masks/weights,
including ordinary negatives unsupported by CTC. Unknown targets stay null
with false masks and zero weights. Development rows never enter either
training cohort. A keyword experiment using additional rows is a different
coverage experiment, not a strict same-data head comparison.

If generation intent conflicts with the confirmed weak keyword labels, the
projection quarantines that row before sampling. It retains `full_text`, the
unchanged generation category and original `label_*` masks, weights and targets;
effective CTC/keyword masks become false, weights zero and effective targets
null. A full-wake intent is never silently converted to a negative, and a near
intent whose observed text is a complete wake is never forced into the negative
pool. Quarantine counts and reasons are retained. Other eligible rows continue
to prepare. A genuinely empty pool blocks only arms requesting it: when no near
pool remains, 0% is available and 5%/10% are explicitly blocked, with no automatic
data filling, relabeling or retry.

## Fixed 0/5/10 exposure comparison

The recipe reuses the existing kws-data sampler, pinned to SHA-256
`5a6f5447b1f4e618f38e9f1bde0b79b1ba3b21db3fbab3fd0e237c622f39e288`.
An unchanged test copy makes the actual algorithm testable offline.

| Near draw percentage | Total draws | Full positives | Near negatives | Ordinary or partial |
|---|---:|---:|---:|---:|
| 0 | 600 | 300 | 0 | 300 |
| 5 | 600 | 300 | 30 | 270 |
| 10 | 600 | 300 | 60 | 240 |

600 is a predeclared draw budget, not `192 × epochs`. All three arms use the
same eligible roster, initial candidate checkpoint, fresh optimizer, positive
source sequence, saved confidence weights, batch size 4 and 150 optimizer
steps. Near draws replace ordinary/partial draws. Family round robin and a
stable within-family interleave spread the available sources and categories;
actual per-word, family, category and source counts are retained. Unequal
confirmed pool composition is reported rather than silently relabeled.
The nominal complete fixture gives 150 draws per wake phrase and 25 per
family/phrase. Real weak evidence can have different eligibility and weights.

The zero arm neither requires a near pool nor exposes near rows. Requested
nonempty pools that are absent fail explicitly. A recipe never requires every
eligible row to appear in every arm. Repeated draws are recorded as repeated
exposures, not additional recordings or independent speakers.

Full-text CTC uses sequence loss divided by full target length, multiplied by
the saved confidence, then averaged over batch item count. The fixed
normalizer is 1.0 in every arm. There is no batch-weight normalization or
ratio-dependent rescaling of positives. Draw percentages are not claims about
confidence coefficient mass or measured numerical loss contribution; the
former is reported by category, the latter requires an actual training run.

## Queue preparation and execution

`training/research_d1_train.py` is a separate v2 entrypoint. It is invoked with
`python -I -B`, the pinned core runtime, and the same shared heavy-model lock
as TTS/ASR. Both commands require a hash-bound supervised queue lease. They
accept a new output directory or the queue's empty output directory; partial
outputs never authorize a second invocation.

The preparation recipe is an immutable JSON declaration:

```json
{
  "schema": "kws-d1-preparation-request-v2",
  "packages": [{"path": "/private/weak-view-batch", "manifest_sha256": "<exact MANIFEST SHA>"}],
  "plan_sha256": "<canonical frozen D1 plan SHA>",
  "view_loader": "/private/kws-data/automation/active_asr_weak_view.py",
  "original_tokens": "/private/official/tokens.txt",
  "official_fsmn_sources": "/private/official"
}
```

Run the queue's preparation job with argv equivalent to:

```text
python -I -B training/research_d1_train.py prepare --recipe {recipe} --output {output_dir} --heavy-lock /private/heavy-model.lock
```

Its declared outputs include `MANIFEST.json`, `preparation.json` and
`arm-availability.json`. The closed manifest binds copied source/audio,
unchanged official donor assets, both native feature geometries, all projected
rows, the common eligible training roster, available exposure recipes and all
three initial checkpoints. Preparation constructs initial models but runs zero
model forwards and zero optimizer steps; its wall time is still charged by the
queue. The preparation receipt reports partial plan coverage and quarantines
without pretending they are full D1 training.

A training job can be declared before preparation completes:

```json
{
  "schema": "kws-d1-training-job-v2",
  "prepared_from_dependency": "d1-prepare",
  "candidate": "rnn64-ctc2601",
  "near_percent": 0,
  "timeout_seconds": 600
}
```

Declare `depends_on: ["d1-prepare"]` in the existing queue request and call
the same entrypoint's `run` command. The worker resolves the actual manifest
SHA from the queue's bound completed dependency receipt; it does not guess an
output hash or modify the job recipe after publication. A separately existing
prepared package may instead be named with absolute `prepared` and exact
`prepared_manifest_sha256` fields. The per-job timeout must be explicit and at
most 999 minutes; the queue's remaining global actual-time budget also applies.

The supported candidates are `rnn64-ctc2601`, `c1-ctc2601` and
`fsmn-head-ctc2601`. The first two train their fresh encoders and heads. The last
trains only the expanded FSMN output affine layer and verifies every other
tensor remains unchanged. D1 tail adaptation is a separate future recipe; this
does not reinterpret D0's already completed 2599 head/tail stages.

Each available candidate/ratio starts from its same frozen initial checkpoint
with a fresh optimizer. The worker checks actual features, loss inputs and
600 exposures, writes a per-step source/PCM/full-text/weight consumption ledger,
records actual weighted loss contribution by category, saves and reopens the
final checkpoint, and reports actual parameter/state and file identities.
Training outputs are `summary.json`, `final.pt` and `consumption.jsonl` (plus a
progress/failure `status.json`). The worker remains in the supervised process
group so the queue's hard deadline covers native operations. On failure,
recorded optimizer steps are a lower bound; an interrupted run never implies
zero work or automatic continuation.

## Scope and verification

The D0 commands still intentionally require their frozen 40/48-row inputs.
These software tests do not launch a real D1 training job. No changes to the global product
vocabulary limit, historical comparison runner, or old runtime receipts are
needed to define this new research path.

`tests/test_research_d1.py` checks the actual official vocabulary bytes, the
actual retained sampler, future-OOV and uncertain masks, duplicate global
cells, repeated local IDs, source-ancestor split checks, and report mutation
during replay. Its replay double only checks aggregation orchestration; real
source/ASR authentication stays with the independently tested data loader.
`--numeric` adds bounded synthetic RNN/C1 outputs and gradients, exact donor
row copying, deterministic appended rows, masked-zero gradients and a four-draw
fixture through the actual optimizer/ledger loop. A real no-model queue fixture
checks completion-driven preparation handoff and rejects a changed dependency
manifest. The numeric fixtures load
no official weights and use no private audio or remote model calls.

The keyword-only projection is available for a separate loss consumer. This
CTC entrypoint records keyword masks as metadata and explicitly reports
`keyword_loss_used: false`; it does not claim that CTC training also optimized
the two-keyword head. The older frozen-C1 clip probe is not a new streaming
event detector. A direct-head D1 recipe and a calibrated decoder remain
separately identifiable follow-ups.

Two synthetic development families support engineering diagnostics. They are
not independent human evaluation, and challenge clips are not natural-audio
FA/hour exposure. The active common-FAR comparison and uncertainty policy
remain necessary for later operating-point comparisons.

The separate [development CTC diagnostic entrypoint](research-d1-development-v2.md)
reads completed training receipts and frozen checkpoints without training or
threshold tuning. It preserves the active plan's `weak_labels_train_only: true`
boundary: machine weak development metrics are predeclared engineering
diagnostics, not formal effectiveness acceptance or a model-winner criterion.

The separate [development evaluator](research-d1-development-v2.md) reads completed
training checkpoints and the same sealed preparation for fixed held-out weak
engineering diagnostics. It does not train, tune thresholds, or choose a winner.
