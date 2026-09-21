# Product model training data contract

The shipping-format model trainer must not silently fall back to the legacy tone
generator.

## Current state

The currently pinned model `model-749187ec1d66` was trained at source
`749187ec1d6662658f06aa9c76d47fde835968db`. At that exact source revision:

- `generator.tts.backend` was `tone`;
- no `generator.external_base_dataset` was bound.

That model remains a valid **synthetic-qualified frozen baseline**, but this fact
explains why later speech-like research did not automatically improve the
deployable `.kwm`: the research corpus and the product training lane were
separate execution paths.

## Governed base for future candidates

Future governed `model-training` runs use the immutable release:

`speech-like-base-5204b798033f`

It is the four-split 384-recording production Stage-A corpus whose independent
384x2 generation run `35318933124` produced the same external-base identity:

`5204b798033fe46e5381c743accf638ab7f16d4e4367f42cba38a6bb44c97b35`

and provider identity:

`d12728562db74951191575e9afe4922adc1260934b02d577dda67d91faa4aff9`

The release is immutable and its corpus archive is additionally pinned by SHA256
in `configs/training/product-speech-like-base-v1.json`.

Before training begins, the workflow:

1. requires the exact current `main` source SHA;
2. downloads the immutable release;
3. verifies release ID, target SHA, archive digest and release checksum manifest;
4. regenerates an effective training config with all four external-base splits;
5. re-hashes every index/summary/WAV through `external_base_dataset.py`;
6. refuses protected evidence and tone fallback;
7. uses that exact effective config for base training, refinement, synthetic
   qualification, robustness and diagnostics.

The effective config SHA is bound across the two training jobs by the existing
base-stage receipt.

## Split-scoped domain rendering

Development rounds render only the splits they consume: train, calibration and
test. Qualification is not rendered during those rounds because it is forbidden
from candidate selection, curriculum feedback and replay mining. After candidate
selection, qualification rendering requests only the qualification split.

Adversarial refinement follows the same boundary: its training dataset contains
train/calibration/test only, while mining and validation qualification cohorts
are rendered separately as qualification-only datasets. The renderer keeps its
historical four-split default for other callers; split scoping is explicit and
recorded in each domain summary.

This removes unused WAV rendering and audit work while making the development /
qualification evidence boundary physical rather than merely conventional.

## Deterministic feature cache

Product RNN training enables the existing deterministic frontend feature cache
with `train.feature_cache_max_items=8192`. Training manifests contain already
rendered PCM WAV files and `Manifest.__getitem__` performs only deterministic
WAV decoding plus frontend feature extraction, so caching does not freeze any
stochastic augmentation or change the training sample stream.

The same cache wrapper is used for base domain rounds and adversarial refinement.
It changes wall-clock work only: epoch count, sample order, optimizer state,
losses, replay composition, thresholds and gates are unchanged. The wrapper is
bound into `training_code_sha256`, and exported model provenance records
`deterministic-feature-cache-v1`, the configured capacity and
`training_math_changed=false`. Promotion rejects future product candidates
that lack this evidence.

## Product development preflight

A versioned model-training request is validated on its pull request before any
full governed training can start. The preflight uses the same immutable
speech-like release, replay provider and effective product config materializer as
the full workflow; it is not allowed to fall back to the tone development lane.

The bounded budget is two base rounds with 12 cold-start and 6 warm-start epochs,
followed by a 6-epoch adversarial refinement. Threshold grid, two coordinate
calibration rounds, bounded two-way calibration execution, replay policies and
strict product gates are retained. Base qualification is deferred and refinement
uses `--stop-after-development-eval`, so the preflight never renders or consumes
qualification, shadow or formal evidence.

The guard is intentionally weaker than product qualification: it does not demand
a strict zero-error candidate from the reduced budget. It does require both
shipping wake words to record at least one correct match in both calibration and
test after refinement. A 100% per-keyword collapse therefore fails the request
PR before the multi-hour run is eligible to start.

On the protected-main push, `model-training` independently resolves the merged
pull request and requires a successful `model-training-preflight` run for the
exact PR head **and** the exact PR base SHA. Successful behavior preflights retain
an artifact named
`xiaowo-product-development-preflight-<base-sha>`; infrastructure-only runs
where the behavior job is skipped retain no such artifact. The full workflow
queries successful runs by head SHA and accepts only one carrying the matching
non-expired base-bound artifact.

This avoids relying on GitHub's optional `workflow_run.pull_requests` metadata,
which may be empty, and prevents an older successful preflight from a previous
base revision being reused after `main` advances. The gate remains fail closed
even if repository UI required-check configuration is incomplete. Manual
`workflow_dispatch` remains an explicit operator path and does not claim a
versioned preflight.

## Auditable training invocation

Governed training keeps the existing manual `workflow_dispatch` entry point and
also accepts a versioned request on protected `main`:

`.github/triggers/model-training-request.json`

The expensive training jobs run on a push only when that exact request file
changes. Ordinary main pushes do not start model training. To request another
governed run, change the request ID and reason through a normal protected-main
pull request. The request PR must first pass the product development preflight.
After merge, the workflow verifies both that preflight binding and that its
checkout SHA is the repository's current `main` before doing any training.

Each run writes `build/model-training/training-invocation.json` into the
retained base-domain state. For a versioned request this binds the exact request
content and SHA256 to the GitHub event SHA/ref; manual dispatches record the same
source identity without claiming a versioned request.

## Promoted model lineage

For new candidates, finalization freezes these next to the selected model:

- `effective-training-config.json`;
- `product-speech-like-base-contract.json`.

`model-promotion` carries them into the immutable model Release. The model
registry mirror then carries the same files into Git with the rest of the model
tuple.

## Product evidence boundary

This speech-like corpus is still offline synthetic speech. Its source model is
8-kHz-bandlimited and normalized to 16 kHz. It is a substantially more realistic
development base than tone synthesis, but it is **not** real-human final-AFE
evidence.

Shipping approval still requires:

1. real-human final-AFE Phase A;
2. physical target-board Phase B.

If Phase A exposes a repeatable model-capacity failure, use a separate
real-human **development** corpus for retraining; never feed the consumed
held-out qualification corpus back into training.

## Replay/resynthesis boundary

The external speech-like base alone is not sufficient: the development loop also
resynthesizes hard negatives and failure cases. Those paths historically read
`generator.tts` directly and could therefore reintroduce tone audio even when
the base corpus was speech-like.

Governed product training now prepares the same hash-bound offline-TTS provider
in provider-only mode and overrides `generator.tts` with its verified command
backend. Replay provider identity must equal the production base-corpus provider
identity.

Replay voice selection is deterministic and restricted to the eight
`train-*` voice slots. Calibration, test and qualification voice identities are
never used for replay.

## Development feedback units

Development loss control compares normalized error rates, not raw event counts.
FRR and FAR/hour have different denominators and units, so comparing false-reject
and false-accept counts directly can reverse the intended recall/precision
decision when corpus duration or wake count changes.

Both RNN and GRU development loops therefore use
`normalized-rates-v1`, with the same policy already exercised by Stage A:
FRR is normalized to 0.10, FAR/hour to 60, and a 10% pressure deadband prevents
small differences from oscillating the controller. Recall pressure adjusts the
exact-wake multiplier upward and ordered-token pressure downward; precision
pressure moves them in the opposite direction.

RNN failure replay remains latched after a failure until stability confirmation.
The normalized controller honors that latch even after the instantaneous error
rates return to the low-pressure region; GRU retains its non-latched policy.

## Development execution budget

RNN and GRU development rounds render only train, calibration and test. Formal
qualification remains outside the development loop and is not rendered as unused
work. Development calibration also reuses the product configuration's bounded
two-way threshold-trial parallelism.

These changes preserve the development dataset members, threshold grid, runtime
decoder, scoring functions and controller inputs. They reduce feedback latency
only, so development evidence remains comparable while the quick loop better
serves as a preflight before governed product training.

## Bounded product round budget

The product loop distinguishes cold-start fitting from warm-start refinement.
Round 0 keeps the full 36-epoch budget. Later warm-start rounds use 12 epochs,
matching the bounded fine-tune scale already used by adversarial refinement,
and decay the learning rate by 0.85 per round. The trainer shuffle seed advances
by 1009 per round so a warm-started model is not repeatedly optimized against
the same batch permutation.

The loop is bounded to two through four rounds. A strict calibration/test pass
may stop the loop once the two-round minimum has been reached, and two stale
objective rounds may also stop it. Persistent improvement may still consume all
four rounds. Therefore the worst-case base-training epoch budget falls from
144 to 72 without reducing the cold-start budget or relaxing any gate; successful
or stale runs can terminate earlier.

Each record retains the actual epochs, learning rate and training seed used for
that round, and checkpoint/model provenance continues to retain the same values.

## Calibration execution budget

Threshold trials for one keyword/coordinate are independent: they use the same
frozen model and references but separate keyword packs and output directories.
Product calibration therefore evaluates up to two threshold trials concurrently.
Results are collected in threshold order before the existing selection logic is
applied, so this changes wall-clock execution only.

With two shipping keywords, seven thresholds and two coordinate rounds, the
historical upper bound is 28 trial corpus evaluations plus one final calibration
evaluation per candidate. Two-way execution reduces the trial wall-clock path
without changing the threshold grid, decoder, metrics, gates or selected result.
Base-domain and adversarial-refinement calibration use the same bounded setting.

## Calibration fallback

The shipping gates remain strict zero-error. Threshold calibration first prefers
any threshold that satisfies those gates, and equivalent strict operating points
continue to use the lower-median plateau rule.

When no threshold is strict, calibration must not use FAR-first lexicographic
fallback. Under 0-FAR/0-FRR gates that policy can prefer an all-reject operating
point solely because it has fewer false accepts, even when recall is unusable.
Non-strict threshold candidates therefore use the same balanced development
`objective()` used for model candidate ranking. This changes only development
fallback selection; it does not relax the promotion, qualification, robustness
or continuous-FAR gates.

## Deferred base-stage qualification

The staged product workflow stops the base iterator after development candidate
selection. It does not render or evaluate the intermediate base candidate on a
qualification cohort because adversarial refinement may immediately replace that
candidate.

The base manifest records `qualification_deferred=true`,
`qualification_qualified=null` and a development-only evidence class. The base
job exit code therefore represents development strict-candidate status only.
Standalone `iterate_domain.py` keeps its historical immediate qualification
behavior unless `--defer-qualification` is explicitly supplied.

Refinement/qualification stages remain responsible for qualification after the
candidate has been refined/frozen, and finalization still re-runs the untouched
qualification cohort before promotion. No qualification gate is removed or
relaxed; only an intermediate duplicate evaluation is eliminated.

## Development refinement source

Adversarial refinement is a development-only optimization stage and does not
consume formal qualification evidence. If base iteration already has a strict
calibration/test candidate, refinement keeps using that strict candidate.

If base iteration has no strict candidate, refinement may start from a
development-only fallback selected without qualification feedback. The fallback
is recall-first: it minimizes worst-case FRR across calibration, test, their
3-5 m far-distance slices, and every shipping keyword; then total FRR; then FAR.
This prevents either a near all-reject checkpoint or a checkpoint that collapses
one wake word from winning refinement source selection merely because the frozen
zero-error gate heavily penalizes FAR.

Formal qualification remains unchanged: it still requires a strict
calibration/test development candidate after refinement, and the guarded formal
renderer still rejects any non-strict candidate.

Model-training #339 (run `35501188138`) proved that a single global wake
multiplier is not a sufficient refinement actuator. Its v1 row-weight
bookkeeping matched 432 exact-wake rows from base mass 864 to non-wake mass
2,976 with multiplier 3.4444. That is not an exact statement about gradient
mass: the trainer currently normalizes sample weights independently inside each
mini-batch. Even with the bookkeeping balance, calibration/test FRR ended at
0.875/0.9375 while test FAR fell to 493.73 h^-1; keyword 1 (`你好小窝`)
remained 32/32 false rejects. The run is diagnostic-only and is not eligible
for promotion.

The first per-keyword policy also exposed a second authority problem. Product
hard-negative replay already records `focus_keyword_id`, but v2 discarded that
semantic evidence and inferred pressure only from token edit distance. For the
fixed replay set, configured focus pressure is 352 examples for keyword 1 and
224 for keyword 2, while edit-distance attribution changes it to roughly
296/280. In particular, the 64-example `hao wo xiao wo ni` negative is
explicitly focused on keyword 1 but is closer by token edit distance to keyword
2.

Refinement therefore uses
`per-keyword-provenance-pressure-balance-v3`. Replay sidecars are the primary
pressure authority: fixed hard-negative/positive-stress evidence,
model-mined-adversarial evidence, and development/qualification-repair failure
evidence are expanded in renderer row order and their explicit focus keyword(s)
receive the non-wake row mass. Multiple explicit focus IDs split that row's
mass. Token edit distance remains a fallback only for rows without semantic
focus metadata, such as ordinary base-dataset negatives; ties still split
equally. Evidence records explicit-focus and fallback row counts and promotion
requires those counts to account for every non-wake row.

Each keyword receives its own bounded [1, 12] exact-wake multiplier from that
row-pressure accounting. The trainer keeps a default wake multiplier of 1.0 and
receives the per-keyword map explicitly. The map and assigned masses are
retained in refinement evidence and model provenance, and model promotion
verifies both copies are identical.

## Dataset-mean sample-weight normalization

The earlier trainer normalized target-sensitive sample weights by the sum of
weights inside each mini-batch. That makes absolute multipliers partially
self-cancelling: a homogeneous batch of 4x wake examples has the same normalized
per-sample coefficients as a homogeneous batch of 1x examples. Dataset-level
wake-mass accounting therefore did not map cleanly onto optimization pressure.

The trainer now computes one deterministic weight profile from the complete
training manifests before the first epoch. CTC, sequence-margin and
prefix-completion losses divide each batch's weighted sum by
`batch_size * dataset_mean_weight`. Ordered-token loss uses the equivalent
mean over all non-empty targets because empty targets do not participate in that
objective. This preserves the overall loss scale while making a sample's
multiplier independent of which other samples happened to share its mini-batch.

The policy is `dataset-mean-sample-weight-v1`. Checkpoint/model provenance
records total rows, non-empty rows, exact-wake rows, total effective weights and
both fixed means. Model promotion rejects product candidates that do not prove
this normalization policy. Recurrent-release loss remains unweighted because it
models post-utterance blank release rather than target identity.

The resulting sample weights still apply consistently to all target-sensitive
objectives: per-frame CTC, sequence margin, strict-prefix completion, and the
ordered-token loss added in #158. Ordered-token loss preserves its legacy value
when all participating samples have equal weights; non-uniform per-keyword wake
weights change its gradient in the same way as the other target-sensitive
objectives. Recurrent-release loss remains unweighted because it models
post-utterance blank release rather than wake/non-wake target identity.

Base training, replay counts, loss coefficients, epochs, learning rate,
thresholds, strict gates and formal qualification remain unchanged.

Model-training run `35439346929` was cancelled after this hidden tone replay
path was identified. It is diagnostic-only and is not eligible for promotion.
