# Installed SDK numeric trace

This is a diagnostic consumer for one bounded numeric comparison. It reports no
performance numbers. The stream lifecycle, PCM feed, score callback and event
callback use the installed public research ABI. GNU ld `--wrap=ehead_step`
observes each original 96-float embedding before calling the original installed
head exactly once. It does not rebuild or replace the installed stream wrapper.

The original archive was inspected with `nm`, without executing it:

```text
ekws_research_stream.c.o: U ehead_step
english_causal_head.c.o: T ehead_step
```

Original archive SHA-256:
`040089f781d87366a3d8a27933043aa96e891849f2bf9ce6bbd2321d0d70152a`.
The private header is needed only for the exact linker-hook argument types:
`source/research/english_final_embedding/english_causal_head.h`, SHA-256
`557c528c7f2511810816a29768f6003b6a4a723b2fe7d60a2c1fdd353ba51de9`.
The public installed header SHA-256 is
`b804e564fea7afb6dc53966de0225eb770f2f73d124768f6abac4d3e22268343`.

There is no `instrument_stream_trace.py`: the link hook makes a source-copy
transformer unnecessary and keeps the baseline consumer on the installed `.a`.
The numerical head and frontend assets remain the caller's pinned artifacts.

## Build and invocation

Using existing SDK and output directories:

```sh
cc -std=c11 -O2 -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror \
  -I"$SDK_ROOT/sdk-native/include" \
  -I"$SDK_ROOT/source/research/english_final_embedding" \
  -I"$TRACE_SOURCE_ROOT" \
  "$TRACE_SOURCE_ROOT/ekws_stream_trace.c" \
  "$SDK_ROOT/sdk-native/lib/libekws_research_stream.a" \
  -Wl,--wrap=ehead_step -lm -lrt -o "$TRACE_OUTPUT_DIR/trace-baseline"

"$TRACE_OUTPUT_DIR/trace-baseline" \
  "$SDK_ROOT/sdk-native/models/causal-head.bin" \
  "$INPUT_PCM" "$NEW_OUTPUT_TRACE"
```

The candidate uses the same consumer, headers, options and `--wrap` option, with
the candidate archive substituted for the baseline archive. The performance
consumer must never use this wrapper or these observer callbacks. Inspect the
candidate archive's undefined `ehead_step` reference as well; a bypassed hook
also fails at the first score callback because the embedding count is missing.

CLI: `ekws_stream_trace HEAD.bin INPUT.pcm OUTPUT.trace`.
`OUTPUT.trace` must not exist. Input is a regular raw mono, signed 16-bit
little-endian PCM file interpreted at 16,000 Hz. The consumer cannot infer a raw
file's actual acquisition sample rate or channel layout; the input manifest
must authenticate those properties and the WAV-to-PCM provenance.

The input length must be even and contain 35,200 through 4,800,000 samples
(2.2 through 300 seconds). The pinned head file must contain exactly 197,908
bytes. Feed calls are 320 samples, with a final shorter block if necessary.
There is no reset, padding, repetition, resampling or inferred word-end label.

## Binary format

The authoritative field order is documented in `ekws_stream_trace.h`. Every
integer is unsigned little-endian; every floating-point value is the original
32-bit IEEE754 bit pattern, encoded as a little-endian `uint32`. No native C
struct or padding bytes are serialized. The consumer verifies the float
representation, including negative zero.

| Part | Bytes | Layout |
|---|---:|---|
| File header | 40 | 8-byte `EKWSTRC1`, six `u32` fields, one `u64` input count |
| Record prefix | 5 | `u8 type`, `u32 payload_bytes` |
| Embedding payload, type 1 | 400 | `u64 ordinal`, `u64 sample`, 96 float32 bit patterns |
| Score payload, type 2 | 48 | `u64 ordinal`, `u32 ready`, `u32 keyword`, `u64 sample`, 3 logits, 3 probabilities |
| Event payload, type 3 | 32 | `u64 event ordinal`, `u64 embedding ordinal`, `u32 keyword`, `u64 sample`, confidence |
| End payload, type 255 | 48 | Six `u64`: accepted samples, embeddings, scores, ready, events, feed calls |

Record order is embedding, score, optional event, repeated, followed by one END.
Each score includes all six numerical outputs, including the first 15 not-ready
scores. Each embedding contains all 96 original components before head
normalization. Sample positions are the original absolute observed-sample
counts, not word-end annotations. Ordinals are zero-based.

For each planned 16-second input, successful metadata must report:

```json
{
  "input_samples": 256000,
  "input_bytes": 512000,
  "feed_calls": 800,
  "embedding_records": 188,
  "real_head_calls": 188,
  "score_records": 188,
  "ready_outputs": 173,
  "input_repetitions": 1,
  "resets": 0,
  "eof_padding_samples": 0,
  "trace_completed": true,
  "end_record_written": true,
  "score_observer_enabled": true,
  "original_head_link_wrap": true,
  "performance_measured": false
}
```

With zero events, there are 377 records including END and 86,197 file bytes.
Each actual event adds one 37-byte record. No timestamp, file path, source-build
identity or timing number enters the binary trace, so equal numerical streams
from equal inputs can be compared byte for byte.

## Failure and comparison requirements

Output is created exclusively and written unbuffered. Short writes, stream
errors, flush, sync, close or final file-size errors fail the invocation. A hook
write failure returns nonzero instead of calling the original head; the
installed stream propagates its head error and poisons that stream. Public
score/event failures propagate the installed callback failure path. No failed
stream is retried. Failure removes the newly created partial output when the
filesystem permits; the nonzero process exit and `trace_completed=false` remain
mandatory rejection signals even if cleanup fails.

The consumer checks that original head calls, hook records, score records and
SDK embeddings agree, that ready/event counts and sample clocks agree, and that
the trace ends after exact input consumption. Input size, mtime and ctime are
checked again after the feed. The worker must additionally hash the pinned input
and head before/after each invocation; their cryptographic identity is not
inferred from a path or from the trace header.

For the planned comparison, run each of the two existing 16-second PCM inputs
once with the baseline and once with the candidate. These are four diagnostic
stream executions over 64 seconds of replay, representing 32 seconds of distinct
previously exposed audio. A successful pair requires zero exit status, complete
JSON metadata with exact expected counts, valid header/record lengths/order/END,
and byte-identical traces. Report both file hashes and full-byte equality.
Do not replace numeric equality with event-digest or threshold-decision equality.
Do not turn this observation into independent FAR/FRR, latency, real-time, board
or model-admission evidence.

Local verification is restricted to source syntax, read-only symbol inspection
and format-layout checks. No model, kernel, linked trace executable or input
replay was run while preparing these files. Actual baseline/candidate execution
belongs to the bounded public Actions run.
