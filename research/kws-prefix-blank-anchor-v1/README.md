# Fixed-prefix blank anchor research v1 (consumed; execution disabled)

The unique 600-update campaign completed with evidence PASS and scientific FAIL: control exactly reproduced, both treatment prefixes became blank, but KW1 omitted ni and the full-pair joint endpoint failed. See [closure](closure.json) and [original observations](observations/2026-10-15-completed-scientific-fail/README.md). The budget is fully consumed; no retry or restart is authorized. Current plan changes only execution_enabled back to false.

## Historical reviewed source and preregistration

The following record describes the enabled execution source c9c85ab0471c71ea9dc6fc105c7d59e87e845c36 before its unique run; it does not describe current execution permission.

# Fixed-prefix blank anchor research v1

Reviewed source is ENABLED for the separately approved single 600-update campaign.
No numerical experiment, model forward/backward, optimizer call, or weights export
was performed during implementation or stdlib integration testing. The disabled
reviewed plan SHA256 was ba5aad0da2d9bdb821aa8c6842657c8e9b36186a95b866421773e37401b0cd51;
only execution_enabled changed in the plan. Exact protected-main source verification
and one separately frozen launcher remain mandatory before the unique invocation.

This is a new, independent 600-update campaign, not continuation of the closed
balanced-fit experiment. C and T each have at most 300 updates; no retry/resume,
seed search, coefficient search, or budget rollover. An exclusive campaign claim
and new output directory are mandatory. Publication does not authorize execution.

## Scientific contract

Both arms use the frozen two original PCM identities, seed 20261015, original
32-logmel/hidden48/5-class unidirectional RNN, Adam settings and original per-frame
CTC reduction. C preserves `((nll/lengths)*w).sum()` exactly, with float32 uniform
weights. Its detached anchor diagnostic does not enter backward. T adds the mean
per-sample prefix blank negative-log probability sum divided by the ORIGINAL
frame count, with coefficient exactly one. No audio cropping, augmentation, VAD,
internal quiet-region labels, decoding mask, or product-path edit is permitted.

KW1 frames 0..22 and KW2 frames 0..9 are fixed. The source PCM hashes are checked
against frozen energy evidence; original unwindowed PCM RMS at 400 samples/320 hop
and threshold 1e-4 must reproduce the fixed first crossing. They are not general
speech boundaries. The read-only registry verifier is the closure-corrected version;
original model/frontend/runtime are unchanged.

C must reproduce the original initial weight SHA, all 300 pre-step losses and
unclipped gradient norms, and every original sample observable at steps
0/50/100/200/300 EXACTLY (no tolerance). The independent control-baseline fixture
contains public archived observations/receipts and its source archive hash; it is
not new experimental output. No runtime access to the old plan, claim, or output
directory occurs. C matching the old failed pair gate correctly admits T. Any
mismatch stops without running T, tuning, retrying, or extending the experiment.

## Timing, outputs, and budget

- `intent.json` is durably written before model preparation.
- `call-NNNN.json` with `step_reserved` is durably written before optimizer.step.
  A crash consumes this slot forever; there is no restart/resume path.
- `step-NNNN.json` is written only after a finite, changed weight update. Its loss,
  sample CTC/anchor components and unclipped gradient norm refer to PRE-step values.
- `{C,T}-0000.json` is before all arm optimizer calls. 0050/0100/0200/0300 observations
  are POST-step. Each is one no-grad model forward, preserving RNG/parameters.
  Each sample contains all effective-frame log_probs, raw argmax/collapse, original
  target CTC NLL, KW1/KW2/empty NLLs, fixed-prefix blank statistics, and first raw
  nonblank index. Argmax path indexes map to 20 ms hops with 25 ms windows.
- Five observation forwards plus 300 training forwards per arm: maximum 610 model
  forwards and 600 optimizer updates. Alternate CTC scores reuse the SAME observation
  lp, not additional forwards. No additional preflight model calls are authorized.
- `{C,T}-initialization.json` binds identical reset weight SHA and seed. No weights
  or reusable model checkpoint are exported.
- `failure.json` retains stage/update/count and numerical context; no retry. A
  complete run writes `terminal.json`. The primary endpoint is T step 300 only:
  both full raw sequences correct AND both fixed prefixes entirely blank argmax.
  Intermediate checkpoints are diagnostics and cannot be selected as endpoints.

This exposed two-sample machine-label mechanism experiment is not human gold,
independent evaluation, product qualification, real recall, FA/h, or generalization.

## Safe static tests

`python -B test_contract.py` and `python -B numeric_preflight.py` use stdlib only.
The latter verifies pins; despite its inherited name, it does NOT run synthetic
model tests. Run again under `-O` and `-OO` to ensure no safety relies on asserts.
`run_anchor.py` uses the same CLI names as the original diagnostic but requires the
new externally reviewed enabled-plan hash. Do not invoke it during preparation.
