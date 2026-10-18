# Full-alphabet three-window C research decoder

This isolated C11 module accepts complete 2,599-class posterior rows; it does not
load an acoustic model, decode audio, modify the shipping engine, register a
model, or establish a deployment threshold. The existing C implementation now
supports explicit threshold configuration; retained Python-reference bytes and
pins are unchanged.

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

## Explicit threshold interface

`kws3_create_with_threshold(id, threshold, &decoder)` accepts a finite binary64
threshold in [0, 1]. The old constructor is exactly threshold 0. The original
11 exported symbols and public event/summary layouts remain compatible; the two
new symbols bring the production total to 13. The status getter returns the
fixed threshold and committed threshold-rejection count. Reset preserves the
threshold and clears the count.

The ordering matches the retained Python threshold-before-refractory adapter:
local detection/reset, exact-node seen registration, threshold rejection, then
shared refractory. A score equal to the threshold is accepted. A low-score
rejection keeps its seen key and does not extend refractory. This cannot be
implemented by filtering already emitted threshold-0 events.

The exact updated C/header/fixture bytes passed a separate bounded run of eight
threshold fixtures and a 68-clip saved-posterior replay at thresholds 0, 0.1 and
0.7. All 204 saved event/summary comparisons, including binary64 score bytes,
and all 68 historical posterior hashes matched. The replay generated 4,921
float64 posterior rows once and consumed 14,763 C rows, 22,008 beam updates and
453 windows. Emitted event counts were 41/40/24; threshold rejections were
0/1/17. These are decoder compatibility counts, not accuracy or FAR results.
The anonymous [result record](kws-fsmn2599-three-window-c-preparation-20261018/THRESHOLD-INTERFACE-RESULTS.json)
contains source and independent-review identities. The private saved posterior
inputs are excluded, so that replay cannot be reproduced from this repository.

Comments in the retained C/fixture source and NOTICE saying NOT RUN or unrun
describe their original preparation-time state, not the later validation above.
The updated-source legacy suite and sanitizer matrix are checked by integration
CI; they are not claimed as rerun by the separate bounded host execution.
The prior frontend/acoustic-network atol=rtol=1e-5 failures remain unchanged.
This change supplies the decoder threshold interface only; it does not add a C
event-head/OR branch or qualify a complete C chain or an SSC305 deployment.

## Reproducible artificial-fixture check

From the repository root:

    python3 -S -B tests/test_research_three_window_c.py --compiler gcc --sanitizers

The existing hosted gcc/clang matrix builds temporary ASan/UBSan/LSan legacy and
threshold-fixture executables, a strict floating-point shared test library, and
a separate production shared library. It retains all eight legacy smoke groups
and 28 differential methods and adds eight threshold groups (569 completed rows,
572 step attempts including three expected failures, 949 beam updates, 21 windows
and 11 direct merge calls). Production symbols must be exactly the 13 public
symbols, without test hooks. All CI data are artificial fixtures; the fixed
standard-library Python reference is used only by the legacy differential suite.
Sanitizer failures are fatal; no automatic fallback disables LeakSanitizer.
Without --sanitizers the same bounded ordinary build is available for hosts that
cannot run sanitizer runtimes; passing it never proves sanitizer coverage.
Four serial compilations each have a 45-second bound; three test processes each
have a 60-second bound, followed by a 10-second production symbol audit.
No downloads, model calls, real-posterior input, or permanent binary output.

## API and limits

See [kws3.h](kws-fsmn2599-three-window-c-preparation-20261018/kws3.h) for create,
reset, double/float32 step, chunk processing and finish. Exact 126-row windows,
42-row hop, full-class top3, stable beam ordering, local consume-reset, exact-node
dedup and refractory are research semantics. Errors latch; reset is a real
session boundary. The historical host storage observation was 532,792 bytes
before the internal threshold fields were added. Query `kws3_storage_bytes()`
for the current opaque allocation size; it is not a fixed external ABI size or a
board memory/throughput qualification. Strict binary64/binary32 assumptions
and rounding/subnormal checks are required; fast-math is prohibited.

[Fixture mapping](kws-fsmn2599-three-window-c-preparation-20261018/FIXTURE-MAP.md)
retains coverage targets, not a claim that every target passed. Historical source
hashes include the earlier Python synthetic summary identity; that summary is
not needed or distributed for the current differential tests. The appended
current source hashes cover the threshold C/header/fixture and anonymous result.

## Attribution

[NOTICE](kws-fsmn2599-three-window-c-preparation-20261018/NOTICE) and the retained
[Apache-2.0 license](kws-fsmn2599-three-window-c-preparation-20261018/LICENSE)
describe WeKws attribution and modifications. Original vendor source is read as
AST by the stdlib reference; its Torch imports are never executed in these tests.
