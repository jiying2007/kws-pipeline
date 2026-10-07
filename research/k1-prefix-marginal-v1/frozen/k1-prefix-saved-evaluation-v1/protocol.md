# First-prefix saved-logit EOF diagnostic v1

Prepared only. Numerical evaluation of real saved rows is forbidden until an
independent source review and the parent-owned release. This is one diagnostic,
not model inference, decoding, audio processing, or a live KWS replacement.

## Frozen question and observations

Evaluate the six whole observations in manifest order M1–M5, N1, with one fresh
DP instance per observation. Consume all 537 saved six-value model rows in all
57 original callbacks, including rows the historical beam decoder skipped.
Carry state across callback boundaries; reset only between observations. Use
the existing 1500 ms leading / 300 ms trailing configuration unchanged. M6 is
excluded. No crops, alignment selection, row selection, terminal slots,
thresholds, alternate decisions, or rescue run are allowed.

Human whole-clip first-prefix labels are M1=A (你好小窝), M2=R (小窝小窝),
M3=B (你好小屋), M4=R (小屋小屋), M5=R (你好你好), N1=A
(你好小窝，屋里有人). K2 is outside the candidate. The primary question is
M3 rejection with N1 preservation. The minimum gate requires both A examples
to accept and all four non-A examples to reject with numerical certification.
A numerical unresolved outcome fails the gate even if the emitted action is
no-accept. Any failure terminates this candidate as a proposed repair; no tuning.
All six predeclared observations are reported once, including failures.

## Input meaning and arithmetic

The pinned A20 model header declares six raw logits. Both collectors serialize
the six contiguous FP32 values with C printf %.9g. Recover each decimal JSON
number by struct.pack/unpack of IEEE-754 float32, promote to Python float, then
compute stable ordinary softmax over all six values at unit temperature:
e_i=exp(x_i-max(x)); p_i=e_i/math.fsum(e). All six p_i are passed as literal
weights to the reviewed module, which normalizes their exact binary64 ratios.
There is no selected-token renormalization, pruning, blank scale, prior,
temperature choice, or floor. Nonfinite values, FP32 overflow, and a positive
softmax term underflowing to zero are fatal input/numerical failures.

The module's certified decision is A>1/2, with ties rejecting. Certified lower
A>1/2 means ACCEPT_K1; certified upper A<=1/2 means REJECT_K1. If its fixed
numerical bounds straddle the boundary, record NUMERICALLY_UNRESOLVED and fail
the gate. No epsilon is fitted to these observations. Certification covers the
exact ratios of the supplied binary64 softmax weights; it is not a proof of
exact real-valued exp or acoustic calibration. The module's independent mock
and oracle tests and source review must pass before release.

## One-shot release and bounds

The release file pins the bytes of manifest.json, protocol.md, runner.py,
test_runner.py, and the mathematical module. A preparation file is deliberately
NOT_RELEASED. The parent must make a separate RELEASED file after review, with
the same hashes, nonempty review/authorization references, and an explicit
request hash on the command line. No numerical execution is authorized by the
existence of these source files. The runner requires --execute, --release,
--release-sha256, --module, --melo5-raw, and --n1-raw. Source paths may be any
local paths; immutable size/hash pins decide whether they are the allowed files.

The runner uses stdlib only and imports no model/frontend/decoder. It has no
network or subprocess operations. It applies 5 CPU seconds, 15 wall seconds,
64 MiB address space, and a 1 MiB total output limit. The fixed local
attempt.jsonl is created with O_EXCL before any saved input is read. It is never
removed or retried, including after errors, timeout, or a killed process.
results.json is also exclusive. An incomplete ledger is evidence of failure,
not permission to rerun. Failures retain their diagnostic and consumed attempt.

The only semantic outputs are the six EOF masses/log-masses, numerical bounds,
certificate status and decision, fixed-label gate, and declared EOF availability
in samples/ms. Availability is the supplied observation EOF; it is not a word
endpoint or word latency. Resource consumption is bookkeeping only.

These are already exposed observations from one stock voice. A complete pass
establishes only compatibility with these six labels. It is neither fresh nor
held-out evidence and establishes no generalization, accuracy, calibration,
streaming latency, duplicate-event behavior, or readiness for deployment.
