# Full-alphabet three-window C research decoder

This isolated C11 module accepts complete 2,599-class posterior rows; it does not
load an acoustic model, decode audio, modify the shipping engine, register a
model, or establish a deployment threshold. The two retained sibling directories
preserve the reviewed source bytes and Python-reference pins.

## Validation boundary

Historical source comments saying NOT RUN or unrun describe initial preparation.
Subsequent bounded validation passed eight synthetic C smoke groups and 28 Python
differential methods, including 25 C-backed methods. The other three methods are
integer geometry, a Python cache test and retained-source identity. This is not
all 41 mapped C targets. Historical ASan/UBSan smoke passed; LeakSanitizer was not
verified on the original ptrace-constrained host and is not claimed there.

A separate prior production-ABI replay matched 864 saved research clips exactly
(38,353 posterior rows, 40,768 beam updates, 947 windows and 41 candidate events),
including binary64 scores and summaries. Only these aggregate historical counts
are recorded here: no posterior arrays, per-clip identities, audio, weights or
private result bodies are included. That replay is not an independent accuracy,
FAR, board-performance, acoustic-network numerical or product-admission result.
The previously failed FP32 acoustic-network result remains failed.

## Reproducible artificial-fixture check

From the repository root:

    python3 -S -B tests/test_research_three_window_c.py --compiler gcc --sanitizers

The existing hosted gcc/clang matrix builds a temporary ASan/UBSan/LSan smoke
executable and a strict floating-point shared test library, then runs only
artificial fixtures against the fixed standard-library Python reference.
Sanitizer failures are fatal; no automatic fallback disables LeakSanitizer.
Without --sanitizers the same bounded ordinary build is available for hosts that
cannot run sanitizer runtimes; passing it never proves sanitizer coverage.
Two serial compilations and two test processes have 45/60-second wall bounds.
No downloads, model calls, real-posterior input, or permanent binary output.

## API and limits

See [kws3.h](kws-fsmn2599-three-window-c-preparation-20261018/kws3.h) for create,
reset, double/float32 step, chunk processing and finish. Exact 126-row windows,
42-row hop, full-class top3, stable beam ordering, local consume-reset, exact-node
dedup and refractory are research semantics. Errors latch; reset is a real
session boundary. Approximate measured host storage was 532,792 bytes, not a
board memory or throughput qualification. Strict binary64/binary32 assumptions
and rounding/subnormal checks are required; fast-math is prohibited.

[Fixture mapping](kws-fsmn2599-three-window-c-preparation-20261018/FIXTURE-MAP.md)
retains coverage targets, not a claim that every target passed. Historical source
hashes include the earlier Python synthetic summary identity; that summary is
not needed or distributed for the current differential tests.

## Attribution

[NOTICE](kws-fsmn2599-three-window-c-preparation-20261018/NOTICE) and the retained
[Apache-2.0 license](kws-fsmn2599-three-window-c-preparation-20261018/LICENSE)
describe WeKws attribution and modifications. Original vendor source is read as
AST by the stdlib reference; its Torch imports are never executed in these tests.
