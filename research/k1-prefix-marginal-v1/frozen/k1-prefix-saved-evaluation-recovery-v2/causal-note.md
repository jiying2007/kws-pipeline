# Preserved first attempt and proposed numerical recovery

Attempt 1 ran once. Its frozen guard rejected a computed softmax exponential
zero before M1 finalized. The preserved ledger contains attempt_started and
attempt_failed, with ValueError: softmax exponential underflow; it contains no
observation_eof event, and no results.json exists. Partial real-row DP work did
occur. Absence of an EOF result does not mean zero scientific rows were used.

The coordinator's subsequent static inspection identified M1 callback index 6,
row index 9 (zero-based), with a recovered-FP32 logit gap of 979.3129272460938.
This finite logit span can legitimately round an ordinary binary64 exponential
to zero. The first adapter forbade all such zeros, although the mathematical
module explicitly supports zero literal weights. This is an adapter input-range
failure, not evidence of a semantic candidate pass or fail.

Static accounting from that coordinator-reported failure location and the
frozen program order: both complete sources, including all 537 rows, had been
read and parsed before evaluation. M1 reached 69 softmax row attempts and 68
computed DP transitions. The interrupted callback rolled back, leaving 59
committed rows (9 + 5*10) and zero EOF decisions. These counts are reconstructed
from source order and callback geometry; the original ledger did not log them.

Preserved identities:

- Attempt-1 ledger SHA256: cf9c92a0e8db14cd187d8bbbd606a4dc0739fc62781e0f5dae69404c0313dbc2
- Attempt-1 release SHA256: 049f9785ad7106e16de08e9ca9f468429a441595dfe9dd3a73c944935df8dd42
- Attempt-1 runner SHA256: 5125ed3f3237ee38611b628bc9f3d18dc34f50c036341381fd6991f310e8af8c
- Unchanged observation manifest SHA256: 07970965185e45d6383bbe33f96ebc7740875dcc9ad68523b124a6ba4aa24578
- Unchanged mathematical module SHA256: e9edc93ffde5569cf7d931342b959738e3a40085957d3d6e1b275664d6729963

The independent numerical reviewer recommends the narrow guard described in
protocol.md: retain ordinary FP32 recovery and full-six binary64 softmax;
permit only actually computed zero terms/weights whose exact recovered-value
delta is at most -512; leave every positive result unchanged. The bound in the
protocol concerns ideal removed-tail mass only. The certificate still concerns
the exact normalized law of the six rounded binary64 weights, not ideal real
exp, every floating-point rounding error, or semantic calibration.

The new adapter records underflow counts and observation/callback/row error
context. Its new release and exclusive ledger identify cumulative attempt 2
and the preserved failed first attempt. The six observations, all 537 rows,
57 callbacks, six DP instances, semantic labels, A>1/2 certificate rule,
all-six gate, no-tuning stop rule, and resource caps are unchanged. No original
file is edited, no original attempt is retried, and no recovery evaluation is
authorized until the new source review and parent-owned release.

The prior ledger hash is a required release identity. The recovery runner does
not reopen the old ledger; its preservation is independently coordinator-verified.
