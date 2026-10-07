# First-prefix saved-logit EOF diagnostic: numerical recovery v2

Prepared only. This is cumulative attempt 2; attempt 1 failed before its first
EOF and remains preserved. Numerical evaluation of real saved rows is forbidden until an
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
temperature choice, or floor. Nonfinite values and FP32 overflow are fatal.
An exp term or final divided weight that actually rounds to zero is permitted
only when Fraction(recovered_i)-Fraction(max_recovered) <= -512. This guard is
an exact comparison of the recovered binary32 values, not an assumption that
the ordinary binary64 subtraction used by math.exp is exact. All computed
positive terms and weights remain unchanged even below -512. No value is
pruned because it crosses -512. Every row still contains six weights.

Record exponential zeros, additional division-only zeros, converted rows, and
rows with zero weights separately. Conversion failures name the observation,
zero-based callback index and row index without printing input vectors. A
failed observation records committed DP rows separately from converted rows,
since the current callback is transactional.

For the ideal softmax alone, e>2 bounds each permitted omitted exponential
term by 2^-512. Its denominator is at least one, so ideal row mass removed is
less than 6*2^-512. Removing and renormalizing that tail has total variation
equal to the removed mass. Over T independent rows the product-law and any
bucket probability difference is at most 6*T*2^-512 (L1 distance at most
12*T*2^-512). This bounds only removal of those ideal tail terms, not all
math.exp/subtraction/fsum/division rounding. No bound is used to tune the gate.

The module's certified decision is A>1/2, with ties rejecting. Certified lower
A>1/2 means ACCEPT_K1; certified upper A<=1/2 means REJECT_K1. If its fixed
numerical bounds straddle the boundary, record NUMERICALLY_UNRESOLVED and fail
the gate. No epsilon is fitted to these observations. Certification covers the
exact ratios of the supplied binary64 softmax weights; it is not a proof of
exact real-valued exp or acoustic calibration. The module's independent mock
and oracle tests and source review must pass before release.

## One-shot release and bounds

The release file pins the bytes of manifest.json (unchanged from v1),
protocol.md, causal-note.md, runner.py, test_runner.py, and the unchanged
mathematical module. It must identify cumulative attempt 2 and bind the original
failed attempt ledger SHA256
cf9c92a0e8db14cd187d8bbbd606a4dc0739fc62781e0f5dae69404c0313dbc2.
The runner checks this identity in the release; preservation of the old ledger
is verified separately by the coordinator, not by reopening it at runtime.
A preparation file is deliberately
NOT_RELEASED. The parent must make a separate RELEASED file after review, with
the same hashes, nonempty review/authorization references, and an explicit
request hash on the command line. No numerical execution is authorized by the
existence of these source files. The runner requires --execute, --release,
--release-sha256, --module, --melo5-raw, and --n1-raw. Source paths may be any
local paths; immutable size/hash pins decide whether they are the allowed files.

The runner uses stdlib only and imports no model/frontend/decoder. It has no
network or subprocess operations. It applies 5 CPU seconds, 15 wall seconds,
64 MiB address space, and a 1 MiB total output limit. The new directory's fixed
local attempt.jsonl is created with O_EXCL before any saved input is read. It is never
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
