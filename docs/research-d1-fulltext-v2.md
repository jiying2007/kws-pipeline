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

The common CTC cohort contains confirmed training rows whose complete text is
in the fixed 2601 mapping and whose unpadded CTC alignment is feasible for all
three frontends. Actual prepared features must also pass the frame-count
check. Future unknown characters or short audio remain explicit exclusions.
The keyword projection separately retains confirmed per-word masks/weights,
including ordinary negatives unsupported by CTC. Unknown targets stay null
with false masks and zero weights. Development rows never enter either
training cohort. A keyword experiment using additional rows is a different
coverage experiment, not a strict same-data head comparison.

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

## Scope and verification

The checked-in foundation includes callable vocabulary, input aggregation,
objective projection, recipe, model-construction and loss boundaries. A new
bounded preparation/training command is a separate follow-up; the D0 commands
still intentionally require their frozen 40/48-row inputs. These foundation
tests do not launch a real D1 training job. No changes to the global product
vocabulary limit, historical comparison runner, or old runtime receipts are
needed to define this new research path.

`tests/test_research_d1.py` checks the actual official vocabulary bytes, the
actual retained sampler, future-OOV and uncertain masks, duplicate global
cells, repeated local IDs, source-ancestor split checks, and report mutation
during replay. Its replay double only checks aggregation orchestration; real
source/ASR authentication stays with the independently tested data loader.
`--numeric` adds bounded synthetic RNN/C1 outputs and gradients, exact donor
row copying, deterministic appended rows and masked-zero gradients. It loads
no official weights and uses no private audio or remote model calls.

Two synthetic development families support engineering diagnostics. They are
not independent human evaluation, and challenge clips are not natural-audio
FA/hour exposure. The active common-FAR comparison and uncertainty policy
remain necessary for later operating-point comparisons.
