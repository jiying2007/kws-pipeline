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
