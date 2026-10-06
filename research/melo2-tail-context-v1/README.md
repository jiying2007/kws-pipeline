# M1/M2: one fixed 300 ms appended-context condition

Appending exactly 4,800 digital-zero samples to each original PCM16 stream
produced one K1 event on M1 and no event on M2. M1's event became available with
14,400 input samples (0.9 s), which is 2,325 samples or 145.3125 ms after its
original file EOF. This is callback input availability, not acoustic word-end
latency. The original five-clip diagnostic remains 0/2 expected-keyword hits and
0/3 nonwake clips with events. Its controls were reused, not rerun.

Only the two human-word positive clips M1/M2 received the 300 ms context. No matched-continuation negative controls were run. Original M3–M5 zero events are short-EOF controls and cannot establish false-activation behavior under this appended-context condition.

All 50 original model rows (300 FP32 scalars) whose frontend support is inside
original EOF remain bit-identical. M1's preserved centers end at 69 and M2's at
75, using `(center + 2) * 160 + 400 <= original EOF`. Original third callbacks
were finishes with 5/7 rows; the modified third callbacks were full feeds with
10 rows and 14,400 samples available. That change in grouping is expected and is
separate from logit drift. M1's third callback computed 10 rows, searched nine
through center 81, activated, and skipped center 84 in decoder search. All logits
were retained: 70 model rows and 69 searched decoder rows is the correct total.

This single synthetic context condition demonstrates M1 sensitivity to added
context; it does not establish acoustic tail truncation, general improvement,
population FAR/FRR, independent-source qualification or a production tail policy.
M2 remains a miss. Human labels, the earlier machine FAIL, original source
quarantine and licensing limitations are unchanged. No duration, threshold or
decoder sweep is part of this result; M3–M6 were not run.

## Reproduce the saved comparison

This annex depends on the unchanged source core at
`research/melo5-a20-diagnostic-v1` and data core at
`research/2026-10-06-melo5-a20-diagnostic`. With both source folders in a pipeline
checkout and both data folders in a data checkout:

```sh
python3 -B research/melo2-tail-context-v1/verify_context.py \
  --data-dir /path/to/kws-data/research/2026-10-06-melo2-tail-context
python3 -B research/melo2-tail-context-v1/tests/test_context.py \
  --data-dir /path/to/kws-data/research/2026-10-06-melo2-tail-context -v
```

Use `--core-source` and `--core-data` if the core folders are elsewhere. The
verifier first validates the core using its original verifier, then validates
this closed four-file annex and reproduces all 30,514 bytes of context
`observations.json`. Eight focused tests cover preserved versus added context,
missing/duplicate centers, changed callback grouping, EOF-relative availability,
wrong tail length, early activation accounting and a changed source dependency.
Only Python's standard library is used. No audio, compiler, network, model,
frontend or native decoder is invoked.

## Exact evidence and reuse

The context raw projection removes only `run_start.pid`: 13,408 public bytes,
SHA256 `90196e0730db17fbae25d604d7e57d51dec4bb1b6ef1192b3c36d7f2fae29cf2`.
All other bytes, including the other 24 JSONL records, are unchanged. Original
raw was 13,416 bytes, SHA256
`bda14751fe0d9bf9a03d02d3e99ca84cfe4af79a4c8688dacbab4f08a311a698`.
The observations remain exact historical bytes, SHA256
`797bef4955144642f2c8c92c41afc7034bfd118dc4bd1db7081ac76789bcad18`.

Those observations name the acquired context raw and original-control raw
hashes, both before PID removal. The core already supplies the original-to-public
control mapping. A temporary control receipt changes only its expected raw hash
to permit the frozen comparator to read the core public trace. The verifier then
normalizes the two declared provenance hashes back to their acquired identities
solely for exact comparison with historical observations. No PID is recovered,
and no scientific value changes. Resource projection removes absolute process
clocks and release bookkeeping, retaining measurements and their limitations.

`metadata/bindings.json` records exact dependencies, transformations and hashes.
Its small, checked source substitutions reconstruct the exact frozen context
collector, scorer and plan from the immutable core files. The context input
header, comparator, manifest and geometry are copied byte-for-byte. Temporary
source reconstruction is hash-checked and removed afterward; the core is never
modified. Model, native library, decoder and runtime source bindings are inherited
unchanged from the core. This avoids another runtime or complete scorer copy.
The reconstructed plan is its historical preparation snapshot, not a claim that
the completed condition was never executed. Collector executable identity and
its one historical compile command are recorded separately from source hashes.

The input recipe preserves each original 44-byte-header WAV's PCM bytes, appends
9,600 zero bytes, and updates only RIFF/data length fields. Original, modified,
prefix and zero-tail hashes are in the manifest. No modified WAV, original WAV,
model payload or acquisition launcher is included. Saved-output reproduction is
complete; this is not a turnkey acquisition package.

Whole-child CPU was 0.031605999999999995 s and launch-to-reap wall time was
0.032162644 s. Lifetime peak RSS 12,520 KiB may include inherited Python memory
before exec. Maximum sampled RSS was 2,868 KiB; it is not an exact native peak.
Native timers retain the core's post-GO, post-dlopen to pre-final-record scope,
including initialization and preceding trace output; feed/finish times include
callback output. These are one local run's observations with no continuous
thread/child-absence or embedded-board qualification. Research evidence only;
there is no new commercial or generated-output license grant.
