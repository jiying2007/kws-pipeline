# Opt-in retention observation, not acoustic validation

This separate `bounded-retention-trace-v1` target exports eleven `kt_*` functions.
It does not change the existing ten-export aggregate `kr_*` target, candidate.patch,
threshold 0.7, retention duration five rows, decoder source or production defaults.
One trace library supports exactly one live decoder and one owning pthread. This is
an intentional research boundary, not a concurrent production API.

## Source-only checks

From a complete repository, run:

    python research/bounded-local-retention-2026-10-20/trace/run_checks.py --repo-root . --cc gcc
    python -m unittest discover -s tests -p test_link_retention_events.py -v
    python -O -m unittest discover -s tests -p test_link_retention_events.py -v
    python -OO -m unittest discover -s tests -p test_link_retention_events.py -v

The native check uses invented blank posteriors, compares 130 rows with capture
on/off, checks lifecycle failures, and verifies C serialization against the Python
linker. An additional invented token-posterior sequence exercises a native low-score
retention start at row 3, strong close at row 4, and exact returned-event 0 linkage.
This single path is not complete native retention branch coverage. Invented
sidecar fixtures test the retention lifecycle, merge rejection and event linkage.
Neither test type proves real utterance behavior or an acoustic-model improvement.
`--cc clang` is supported but must actually be run on a host with clang installed.

## New target and caller sequence

Build explicitly with `build_trace.py --mode bounded-retention-trace-v1 --repo-root
. --output <new-directory>`. Reconstructs and verifies the existing pinned candidate
source, then links only the separate trace facade. It does not load weights.

1. `kt_create("bounded-retention-trace-v1", recording_id, &decoder)`.
2. `kt_begin(decoder)` immediately, before any row. Otherwise it is a noncapturing
   decoder invocation, not a valid trace. Begin never resets decoder state.
3. For each real caller-provided row, call `kt_step_f32`; only on success call
   `kt_json_row` from `trace_json.h` or `kt_drain` with a bounded serialization sink.
   A subsequent step is forbidden until pending ledger records have been drained.
4. Require `kt_status.valid`, no drops and zero pending; require `kt_end` success.
5. The controller must separately flush/close its file, collect process exit0,
   final native summary/retention counters and active windows, and bind the exact
   frozen inputs, library, harness and plan identities before trace admission.
6. `kt_destroy` returns invalid for unfinished/failed capture even though the owning
   thread releases the decoder. Calling after successful end is the only successful
   capturing destruction. Wrong threads/handles and sink reentry invalidate capture.

`kt_end` marks observation ended; it does not call decoder finish, reset, pad, or
expire an active episode. Further stepping is rejected. Summary getters are snapshots,
not admission flags. Always inspect trace status and the external execution receipt.
The underlying diagnostic ledger has no disable API and can remain internally active
after end; the facade prevents further steps on that ended handle. Each new capture
begins a fresh ledger. Do not infer trace-off performance from a reused library.

`trace_json.h` emits row JSON only; it intentionally cannot generate a successful
header/footer. It limits serialized bytes and marks failure sticky. On partial write
or uncertain append, preserve the partial evidence and stop: never retry the append.
A successful buffered write is not durable file completion. Header/footer budgets
and actual sink close/exit verification belong to the still-unreleased controller.

## Admission and claim boundary

`tools/link_retention_events.py` consumes binary JSONL header, successful row
transactions and footer. Header requires run/recording identity, candidate arm,
PCM/plan/candidate/library/harness/ABI SHA256s, fixed threshold/retention rows and
caller-bounded byte/record/row ceilings. Its test fixtures document the exact schema.
All records include sequence/kind/row; commit/trial markers include status0.
Detect/merge records and returned events include window, keyword, four nodes and
16 lowercase hexadecimal binary64 score bits (never decimal score joins). Returned
events also have their actual event_index. A candidate_accepted trace record does
not yet have that index; matching is by the complete candidate key.

Parser memory is bounded by a row, at most three active episodes and at most three
candidates/events. It emits episode callbacks provisionally; consumers must stage
these privately and publish none until full footer admission succeeds. A missing
footer, nonzero exit, invalid capture, drops, pending records, sequence gaps,
ambiguous joins or inconsistent final counters reject admission. It checks declared
consistency; cryptographic identity and declared completion do not independently
attest that real inference or correct labels actually occurred.

A strong close can still be deduplicated or rejected by refractory. A linked emitted
event is a retained-path event, not a true wake or rescue. Truth comes only from
eligible canonical reference matching. Candidate-only correct matching plus a linked
retained episode may be described as retention-associated paired recovery under a
frozen strategy-only comparison. Individual causal necessity is not established.
Uncertain labels and excluded diagnostics cannot be transformed into successes.

Trace writes perturb timing. The fixed approximately6MB TLS ledger existed already;
no claim of zero overhead, unbiased latency, portable full runtime dependency closure,
scientific release or production qualification is made.

## Integration not performed by this source-only patch

The existing public-clip scientific harness and its frozen artifacts are untouched.
A reviewed controller must adopt this separate library/row serializer, freeze all
new build/source identities and bounded resources, and verify actual file closure.
No old aggregate-only run can be retroactively converted into detailed episode data.
Repository publication must register the new Python tool in its navigation index and
new tests in the existing source/test inventory, add explicit opt-in trace CI checks,
and update source-retention metadata for actual changes. Do not silently relax the
existing aggregate ABI allowlist or imply these checks already ran on remote CI.

Detected candidate/event keys enforce window_start <= row, node frames within
[3*window_start, 3*row], and duration 5..250 inclusive, matching the pinned detection
filter. These checks are not a complete independent reconstruction of window scheduling.
