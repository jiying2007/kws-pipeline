# Observer-free installed English SDK performance

This adds an independent consumer of the existing English research SDK ABI 1.
It does not change the SDK, frozen frontend/model bytes, thresholds, or past runs.
The caller must authenticate the source SDK archive before extracting it. The
driver records the exact installed header/library/head and consumer identities.

## Run on the authorized host

```sh
python3 research/english_stream_performance/run_benchmark.py \
  --sdk-root /absolute/path/to/sdk-native \
  --manifest /absolute/path/to/inputs.json \
  --output /absolute/path/to/new-performance-result
```

The output directory must not exist. The driver compiles the consumer against the
installed static library, then launches one process per input, once, sequentially.
It has no download, synthesis, ASR, training, or retry mechanism. Use a separate
approved execution identity for a new experiment; do not re-run consumed one-shot
launches. Initial total input is bounded to 300 seconds and 16 cases.

Example manifest (replace the example hash and provenance with verified inputs):

```json
{
  "schema": "english-stream-performance-inputs-v1",
  "cases": [{
    "id": "natural-background-01",
    "path": "natural-background-01.pcm",
    "sha256": "EXACT_LOWERCASE_SHA256",
    "samples": 256000,
    "sample_rate_hz": 16000,
    "encoding": "s16le_mono",
    "speech_kind": "natural_speech",
    "continuity": "single_recording",
    "source": {"repository": "jiying2007/kws-data", "run_id": 0,
               "artifact_id": 0, "member": "REPLACE_WITH_OBSERVED_MEMBER"}
  }]
}
```

Every PCM is decoded and held in memory before `accept` timing. The callback only
counts events and computes an FNV-1a checksum of keyword/sample/confidence bits;
it performs no I/O. The diagnostic score observer is NULL. No reset, discontinuity,
loop, synthesized context, or EOF padding occurs inside a case.

Each input must contain at least 51,200 samples (3.2 seconds): the first 35,200
real samples are reported as cold preparation, and at least one second remains
for steady processing. This boundary comes from this SDK's fixed 1-second frontend
and 16-embedding head history. A single 16-second natural speech recording is a
useful initial input. If existing short TTS files are concatenated for a performance
probe, declare `synthetic_speech` / `concatenated_recordings`, retain the ordered
source members and conversion recipe, and do not call it a natural continuous
recording or independent FAR evaluation. Artificial zero/sine/LCG verification
signals are intentionally not accepted as speech inputs by this driver.

## Measurements and limits

- `passed` / `measurement_completed` mean all declared inputs were measured and
  the stream integrity checks passed. No CPU, memory or acoustic budget is
  evaluated here (`performance_budget_evaluated=false`).
- `init_*` covers SDK init only, excluding preloading and allocation.
- `warmup`, `steady`, and `whole_stream` each contain sample count, actual duration,
  sum of monotonic accept wall time, process CPU time and their ratios.
- Per-20-ms-call p50/p95/p99/max and counts over the provided input duration show
  service-time bursts. The final partial call is included. These are not word-end
  latency, real-time microphone deadlines, or an application scheduling test.
- Linux RSS/PSS and process lifetime peak RSS are recorded before allocating the
  stream, after init/head release, after warm-up, and after the stream. `/proc`
  memory reads are outside timed calls. Missing observations are null. Process
  values include the consumer, input buffer and timing arrays; buffer sizes are
  also reported, and no isolated KWS peak is inferred from their sum/difference.
- There is no sleep or real-time pacing. Continuous states are preserved within
  each recording. Repeated/concatenated input never adds independent FAR hours.
- These research weights remain unqualified. No pronunciation, FAR/FRR, word-end
  latency, SSC305, or model promotion conclusion is produced.

The existing installed static library was not built with `-pg`, so attaching gprof
to this consumer would not produce reliable SDK function accounting. `perf` on
hosted runners may be unavailable.

`instrument_graph.py` creates a separate copy of the exact frozen frontend with
node wall/process CPU accounting. Compile that copy with `ekws_graph_profile.c`
and set `EKWS_GRAPH_PROFILE_PATH` to a new result path. Its graph validation and
node totals diagnose the distribution of work; they exclude ring copies,
normalization, the head and event callback, and include clock overhead. The
original source and installed library stay unchanged. Every diagnostic result
must be valid, have expected per-node call counts, and match the baseline event
digest and stream counters for the same PCM. Diagnostic replay is another pass
over exposed input, not additional independent audio. Do not mix instrumented
component timings with the uninstrumented headline measurement.

## BSP use

The C consumer uses the public installed ABI, libc and POSIX clocks. Build it with
the real BSP compiler/sysroot and an SDK built for that same ABI. Generic ARM/QEMU
timings are not SSC305 measurements. The Python driver is host orchestration and
is not required on the board; the C executable accepts `HEAD.bin INPUT.pcm` and
prints one JSON result. A real board's result still requires separate device
identity, frequency/load/temperature, microphone and application timing evidence.
