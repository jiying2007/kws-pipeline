# KWS Training Reset v1

## Purpose

This lane exists to answer model-learning questions quickly and causally before a candidate enters the governed development, freeze, shadow, formal-qualification, real-human, or target-board paths.

The previous pipeline mixed research feedback with qualification-style gates. In particular, a development point could collapse to zero FAR by rejecting every wake, while the trainer simultaneously changed replay, curriculum, loss weights, learning rate, and acoustic realizations. Training Reset v1 separates those concerns.

## Three lanes

### Research

Research is diagnostic-only.

- one acoustic render per experiment;
- one declared model/loss variable at a time;
- no adaptive curriculum;
- no failure replay;
- no fixed hard-negative replay;
- no warm start;
- no adaptive loss controller;
- no fresh/shadow/formal/qualification consumption;
- no candidate freeze or promotion;
- acoustic quality never changes the workflow exit status;
- software, data, provenance and contract errors remain fail-closed.

A research run emits a scorecard. A bad model is a valid research result.

### Confirm

Confirm is used only after a research hypothesis exists.

Use paired A/B or predeclared multiseed cohorts. Keep corpus, acoustic realization, decoder, threshold policy, training budget and runtime identity fixed except for the declared experimental variable.

A single paired result is evidence for direction, not promotion.

### Qualification

Qualification starts only from a frozen candidate.

Fresh validation, independent shadow, formal qualification, target-board evidence and real-human/product acoustic evidence stay protected and may not feed back into research training.

## M0-M4 loss ladder

The research loss ladder is defined by
`configs/training/kws-v2-research-reset-v1.json`.

| Profile | CTC | Ordered token | Sequence margin | Prefix completion | Recurrent release |
| --- | --- | --- | --- | --- | --- |
| M0 | on | 0 | 0 | 0 | 0 |
| M1 | on | on | 0 | 0 | 0 |
| M2 | on | on | on | 0 | 0 |
| M3 | on | on | on | on | 0 |
| M4 | on | on | on | on | on |

Only one auxiliary term is introduced at each step. Existing trainer defaults remain unchanged for legacy/governed paths.

Do not advance from M0 merely because M0 misses a strict zero-error gate. Advance only when the M0 diagnostic identifies a loss-related failure mode and the next term has a testable hypothesis.


## Sample-weight semantics

The trainers now expose two distinct sample-weight actuators:

- `positive_example_weight` is retained for backward compatibility but means **non-empty-target weight**. It is not a wake-positive weight.
- `wake_example_weight` applies only when the complete CTC target sequence exactly equals one configured wake keyword.

This distinction matters because all canonical Stage-A positive, confusable and negative speech rows carry non-empty token targets. On the historical canonical-only Stage-A corpus, changing `positive_example_weight` scales every sample equally and therefore cancels in the normalized weighted loss.

Training Reset derives two fixed balancing factors before training:

```
target-bearing weight = clamp(empty-target / target-bearing, configured bounds)
wake weight           = clamp(tokenized-nonwake / exact-wake, configured bounds)
```

The same two factors are held constant from M0 through M4. They are baseline sampling semantics, not auxiliary losses.

The normalized development controller uses `wake_example_weight` as its recall actuator. It no longer changes the legacy target-bearing weight when reacting to FRR pressure.

## Operating-point diagnostics

Research does not use the strict production gate to choose a threshold.

Every fixed checkpoint is swept across a broad common threshold grid. The scorecard retains:

- the calibration/test operating curve;
- Pareto thresholds;
- FRR at several FAR/hour budgets;
- test behavior at those calibration-selected research operating points;
- continuous ordinary-negative exposure at a research-only threshold.

The research threshold is not a shipping threshold and must never be written back into training.

A checkpoint whose curve contains a useful operating region but whose strict development calibration chooses an all-reject point is classified as a calibration/gate problem, not an acoustic-learning failure.

## Ordinary-speech negative sidecar

The canonical Stage-A corpus remains immutable. Its 384-recording identity is not changed by Training Reset.

The sidecar defined by
`configs/training/kws-v2-research-negative-utterances-v1.json`
adds 32 ordinary/near-confusable utterances over disjoint Stage-A voice slots:

- 8 train voices -> 256 recordings;
- 4 calibration voices -> 128 recordings;
- 4 test voices -> 128 recordings;
- no freeze/qualification voices.

The sidecar is research-only and portable. It is generated through the same pinned offline TTS provider, verified runtime assets, license evidence and voice inventory as Stage A.

Because the current canonical KWS vocabulary contains only the four wake-word pinyin tokens, ordinary Mandarin cannot be represented honestly as normal CTC targets. The sidecar therefore uses an explicit
`empty-target-nonwake`
policy.

That policy is valid for diagnosing a fixed-wake-word system. It is **not** evidence that the current four-token acoustic model supports shallow-customizable KWS.

## Classifier separability baseline

`training/research_keyword_classifier.py` runs a small clip-level GRU classifier on the same frontend and split identities.

Its role is diagnostic:

- if both the classifier and CTC overlap badly, suspect data/frontend/encoder capacity;
- if the classifier separates well and CTC does not, suspect CTC tokenization, decoder behavior or the four-token vocabulary;
- if the CTC operating curve separates but the selected operating point collapses, suspect calibration/threshold policy;
- if one seed is good but predeclared independent seeds are unstable, suspect optimization/training stability.

Classifier accuracy is never a shipping KWS metric.

## Current vocabulary boundary and v2 trigger

The current canonical vocabulary is:

- `<blk>`
- `ni3`
- `hao3`
- `xiao3`
- `wo1`

This is acceptable only as a fixed-wake experimental acoustic vocabulary.

A vocabulary/encoder v2 investigation becomes mandatory when either condition is observed:

1. the clip classifier separates ordinary speech from wakes while M0-M4 CTC cannot produce a useful operating region; or
2. product requirements require genuine shallow customization beyond a small predefined keyword set.

The v2 line must be a separate architecture experiment. It must not silently change the current model ABI. Candidate directions include a broader Mandarin phone/pinyin inventory with a streaming encoder and keyword graph, or a dedicated fixed-keyword classifier if shallow customization is no longer required.

## Normalized development controller

The governed Stage-A controller no longer compares raw false-reject and false-accept counts.

For `normalized-rates-v1` it uses:

```
FR pressure  = FRR / normalization_frr
FA pressure  = FAR_per_hour / normalization_far_per_hour
```

A deadband prevents small differences from causing oscillation. Replay repetition is derived from normalized severity rather than dataset-size-dependent failure counts.

This fixes cases such as:

```
8 false rejects / 1000 wakes  -> FRR 0.8%
3 false accepts / 1 minute    -> FAR 180/h
```

where raw counts incorrectly suggest recall is the larger problem.

Every completed development round records the controller signal mode, FRR, FAR/hour, severity, next weights and next replay repetition.

Legacy policies without `signal_mode` retain the old count-based behavior for compatibility.

## Negative exposure

Training Reset builds continuous research-only negative WAV exposure from:

- canonical test negatives;
- ordinary-speech sidecar test negatives;
- white/fan/motor/media background mixtures.

Every negative clip is injected at least once. The resulting FAR/hour is useful for relative research diagnostics only. It is not a shipping FAR claim.

The default research exposure is 600 seconds. Confirm uses at least 3600 seconds. Product FAR still requires long real continuous recordings through final microphones, enclosure and shipping AFE.

## Test and evidence rules

A Training Reset change is ready for research use only when:

1. reset policy contract passes;
2. trainer default-loss compatibility passes;
3. normalized-controller regression passes;
4. Python syntax passes;
5. ordinary-negative sidecar generation is reproducible and portable;
6. M0 produces a complete research scorecard;
7. research scorecard explicitly states no protected evidence and no promotion permission.

A model metric never turns the research workflow red. Missing data, invalid provenance, split leakage, non-portable artifacts or code failures do.

## Experiment order

Use this order unless a prior result falsifies the premise:

1. GRU64 logmel M0 with ordinary-speech sidecar;
2. RNN64 logmel M0 only if architecture comparison remains useful;
3. PCEN-lite M0 against the same corpus and budget;
4. M1 through M4 one term at a time;
5. capacity search only after frontend/loss baseline is understood;
6. paired or multiseed confirmation;
7. frozen candidate;
8. protected qualification.

Failure replay is not a baseline mechanism. Re-enable it only after a stable baseline exists and a paired experiment shows incremental benefit.

## Main commands

Manual research execution uses
`.github/workflows/kws-v2-research-reset.yml`.

Ordinary-speech sidecar generation plus the automatic GRU-logmel M0 diagnostic uses
`.github/workflows/kws-v2-research-negative-sidecar.yml`.

The automatic M0 chain is:

```
verified offline TTS
  -> 512 ordinary/near-confusable negatives
  -> portable empty-target sidecar
  -> immutable 384-recording Stage-A corpus + sidecar
  -> GRU64/logmel M0 CTC-only
  -> threshold operating curve
  -> clip-classifier separability baseline
  -> continuous negative exposure
  -> research-scorecard.json
```

This is the new entry point for model iteration.

## Architecture investigation closure through 2026-09-19

Training Reset has now falsified several earlier explanations for the inability to
produce a useful KWS operating region. These results are retained as research
evidence and must not be reinterpreted as shipping qualification.

### What is no longer an authorized tuning direction

The following hypotheses have been tested sufficiently to stop blind tuning:

- threshold-only recovery: the retained threshold sweeps reach low/zero false
  accepts only after wake recall collapses;
- replay as the primary baseline mechanism: replay remains disabled until a
  stable baseline exists and is not used to explain the current separation
  failure;
- stronger four-token margin tuning: M2 margin 0.1 was the best observed point;
  0.2 and 0.3 raised confidence broadly without creating a transferable
  wake/non-wake separation band;
- lightweight verifier-only repair: a frozen-M2 hidden-state linear verifier did
  not separate canonical token-sharing near misses;
- "phonetic-v2 only needs more epochs": the 72-token tone-pinyin experiment
  became worse at 24 epochs, with higher non-wake confidence;
- initialization, learning-rate schedule, or exact class-count sampling as a
  sufficient stabilization fix: the pre-registered multiseed experiments did
  not pass their stability gates.

These negative results are valuable closure. Repeating the same tuning axes with
new arbitrary constants is not an authorized next action.

### Research classifier semantics and deterministic runtime

The clip classifier is diagnostic only. It now uses the same
`TinyStreamingGRU` recurrent-cell semantics as the GRU research family, and
positive score separation is computed from the **true configured keyword
class**, not the maximum wake-class probability.

Architecture decisions use calibration-selected operating points transferred to
test. Test metrics never choose a threshold or a start.

Research classifier training has a separate cross-CPU deterministic contract:

- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`;
- `MKL_CBWR=COMPATIBLE`;
- `OPENBLAS_NUM_THREADS=1`;
- `ATEN_CPU_CAPABILITY=default`;
- MKLDNN disabled;
- deterministic torch algorithms;
- model initial/final state SHA256 retained.

Run `35427343767` proved bit-exact initial state, final state, training history,
calibration output and test output for B0/FC1 replicas on the selected seeds.
One FC1 pair was bit-exact across different hosted AMD EPYC CPU models. Research
architecture comparisons are not authoritative unless this numerical contract
is preserved.

### Multiseed stabilization results

The direct fixed-keyword classifier remains an upper-bound diagnostic rather
than a product model.

The main stabilization experiments closed as follows:

1. paired independent-seed replication: failed the pre-registered replication
   gate;
2. variance decomposition: model initialization was the dominant observed
   random source;
3. orthogonal/Xavier GRU initialization: failed to reduce variance and increased
   the observed recall-delta spread;
4. warmup + cosine schedule on the reproducible runtime: reduced spread only
   modestly while materially reducing the mean recall gain;
5. exact-balanced per-epoch sampling: did not reduce the spread versus the
   replacement-based balanced sampler;
6. calibration-only multi-start for PCEN-lite + GRU128: produced strong absolute
   results but failed the pre-registered relative gate.

For the last experiment, run `35428804016` selected starts using calibration
only over five independent three-start cohorts. The selected PCEN128 target had
strict-10 test wake recall mean `0.834375`, population standard deviation
`0.044852`, and mean negative FP rate `0.06375`. It achieved four
absolute-strong cohorts, but only three of five primary directional passes
against independently selected B0; the pre-registered requirement was four.
Therefore `multistart_pass=false`. The gate is not relaxed after observing the
result.

### Runtime and deployability boundary

The capacity result is a **mechanism upper bound**, not a deployable candidate.

The current C/export contract is:

- maximum feature dimension: 40;
- maximum recurrent hidden dimension: 64;
- maximum vocabulary size: 512.

The canonical static resource estimator gives the current five-token,
32-feature, 50-step/s shapes approximately:

| Research shape | Serialized GRU estimate | Dense work | Current static runtime fit |
| --- | ---: | ---: | --- |
| logmel + GRU64 | 20,388 bytes | 0.9376 MMAC/s | yes |
| PCEN-lite + GRU64 | 20,388 bytes | 0.9376 MMAC/s | yes |
| PCEN-lite + GRU128 | 65,252 bytes | 3.104 MMAC/s | **no** |

`kws-v2-research-architecture-fit-v1` fails closed: hosted arithmetic estimates
are never target CPU, RAM, thermal or power evidence. Even a runtime-fit research
shape has `shipping_candidate_allowed=false` until an approved physical target
resource budget and the required target-board measurements are bound.

The next low-risk architecture question is therefore PCEN-lite + GRU64 under the
same calibration-only fresh multi-start method. It uses a new seed namespace and
does not reuse the already observed FC1 multistart seeds. A positive classifier
result authorizes a streaming CTC/objective confirmation at the existing hidden
dimension; a negative result authorizes an efficient encoder-v2 investigation.

### Research closure versus product closure

The repository may merge a completed research infrastructure/negative-result
chain without real-human or target-board data. Missing product evidence must not
force research experiments to pretend to be failed software runs.

That does **not** make a model production-ready. Shipping promotion remains
blocked until a frozen runtime-fit candidate independently satisfies the
protected path, including:

- fresh/shadow/formal qualification;
- real human speech and near-confusable speech;
- final microphones, enclosure and shipping AFE;
- 3-5 m / azimuth / reverberation / playback and mechanical-noise behavior;
- long continuous real-audio FAR exposure with confidence bounds;
- exact target-board CPU/RTF/headroom, RSS/stack, thermal and power evidence;
- required soak, discontinuity/XRUN and suspend/resume evidence;
- exact source/model/pack/raw-evidence/attestation identity.

Research evidence informs what to build next. It never substitutes for those
shipping gates.

### Runtime-fit PCEN64 confirmation result

The follow-up runtime-fit experiment is closed.

Run `35429252239` used five new three-start cohorts that did not reuse the
previous multistart seeds. Both arms stayed at GRU64; the only architecture-side
difference was logmel versus PCEN-lite. Start selection remained calibration
only.

The pre-registered result was negative:

- primary directional passes: `1 / 5` (required `4 / 5`);
- secondary directional passes: `0 / 5` (required `3 / 5`);
- absolute-strong passes: `0 / 5`;
- selected PCEN64 strict-10 test wake recall mean: `0.6875`;
- population standard deviation: `0.083268`;
- selected strict-10 test negative FP mean: `0.09375`.

Therefore PCEN-lite is not a reliable drop-in recovery for the existing GRU64
capacity. The gate is not relaxed after observing the result.

At this point both low-risk explanations are closed:

1. more capacity helps in the direct-classifier upper bound, but GRU128 is
   outside the current runtime/export hidden-dimension contract; and
2. changing only the frontend while preserving GRU64 does not reproduce that
   gain.

The authorized next line is `efficient-encoder-architecture-v2`. It should be a
separate research change set with an explicit resource envelope, fresh seeds and
the same calibration-only/test-after-selection discipline. New runtime/model
format work is not authorized until an efficient research architecture first
demonstrates a repeatable gain.

