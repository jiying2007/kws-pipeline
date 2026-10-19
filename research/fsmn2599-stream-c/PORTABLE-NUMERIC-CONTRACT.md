# Portable numerical contract: proposal, not a replacement pass gate

Status: DRAFT_NOT_REGISTERED. No new numerical tolerances, qualification result,
reference implementation or model identity is adopted by this document.

## Keep three claims separate

1. Frozen Torch compatibility: retain the existing atol=1e-5, rtol=1e-5,
   reference-scaled elementwise rule, exact structure/shape and finite checks.
   Historical frontend/network failures remain failures. Profiling does not
   loosen or replace this gate.
2. Portable arithmetic conformance: a future implementation may be evaluated
   against an explicitly specified independent numerical contract. A pass here
   must never be relabelled as Torch parity. Pre-register the oracle/error model,
   stage thresholds and target/compiler set before evaluating held-out data.
3. Decision/quality conformance: evaluate full posterior competition, decoder
   boundaries and independently sourced false-accept/false-reject behaviour.
   Matching a few observed events does not prove elementwise equivalence or FAR.

## What must be specified before a portable gate can run

- PCM integer scaling,400/160 framing, zero dither, DC,0.97 preemphasis, Povey,
  FFT512,80 mel bins, epsilon floor, context splice/stride, CMVN, cache reset and
  synthetic-EOF policy remain pinned; no hidden alternate frontend.
- Pinned payload and alphabet/token mapping; full2599 normalization/top3/ties;
  floating-point format, rounding, subnormal handling, contraction and compiler
  reassociation rules; libm, architecture and build identities.
- Exact oracle scope: high-precision/interval arithmetic where feasible, its own
  approximation bounds, and whether it models the mathematical transform or a
  particular floating-point operation sequence. Float64 alone is not an exact
  gold, and a more accurate FFT need not be closer to frozen Torch float32.
- Per-stage test classes: exact integer and indexing invariants; window/preemphasis;
  complex spectrum; linear power/mel energy; logfbank; CMVN/body/cache; logits;
  all2599 posterior entries and normalization; decoder/event structure.
- Error metrics include absolute, reference-relative, ULP where meaningful,
  signed bias, distribution/tail and worst location. For near-zero spectrum or
  energy, relative error alone is ill-conditioned. Report linear energy and
  proximity to epsilon alongside log error; do not silently exempt such rows.
- For recurrent state, quantify error over declared horizons, resets, gaps and
  varied signal energy. A short isolated-frame bound cannot certify long streams.

## Discontinuous decisions need margin tests

Track perturbation margins around full-class top3 membership/ties, the strict
0.05 token cutoff, keyword score threshold, duration/window boundaries and
refractory/dedup decisions. A tiny posterior difference can flip a discrete
branch. Require targeted synthetic boundary cases plus held-out real streams.
Do not invent a universal epsilon guaranteeing identical decisions. If a chosen
contract permits differences, predeclare how they are counted rather than
filtering disagreeing samples after seeing results.

## Dataset separation and promotion

Use synthetic and exposed development material to diagnose and select candidate
arithmetic only. Freeze the portable contract before independently sourced
acceptance trials. Report speaker/session/source lineage and uncertainty for
false-accept rates and recall. The observed frame2 FFT/log sensitivity is a
localisation clue, not authority to choose a larger tolerance after the fact.
No benchmark result alone promotes a model/runtime to production.

## Minimum next artefact

A registered JSON contract should contain schema/version, model/source hashes,
oracle identity and error bounds, per-stage metrics and tolerances, accepted
platform/compiler flags, test-set identity, recurrence horizons, decision-margin
policy and explicit separate fields for frozen_Torch_gate and portable_gate.
Until those fields are justified and registered, portable_gate stays NOT_RUN.
This preparation deliberately supplies no guessed numeric limits or new pass.
