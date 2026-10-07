# Original A20 on five exposed Melo clips

One completed original-A20 diagnostic produced **0/2 expected-keyword hits and
0/3 nonwake clips with activation events**. M1 is human-word K1 (你好小窝), M2 is
K2 (小窝小窝), and M3–M5 have nonwake word labels. M6 is excluded because its
current human transcript is partial. These are descriptive counts on five
previously exposed clips from one Melo source. They are not population FRR/FAR,
fresh-speaker evaluation, evidence of general KWS improvement or training admission.

Human hearing used native 44.1 kHz float audio after prior ASR hypothesis
exposure. The replay used exact bound 16 kHz PCM16 derivatives that were not
independently heard. Acoustic tail completeness and word-end alignment are
unknown; no true-word-end latency is available. The original source quarantine
and licensing uncertainty are not changed by this diagnostic.

## Reproduce the saved result

The companion data is in
`kws-data/research/2026-10-06-melo5-a20-diagnostic`. From any directory, run:

```sh
python3 -B /path/to/kws-pipeline/research/melo5-a20-diagnostic-v1/verify_saved.py \
  --data-dir /path/to/kws-data/research/2026-10-06-melo5-a20-diagnostic
python3 -B /path/to/kws-pipeline/research/melo5-a20-diagnostic-v1/tests/test_verify_saved.py \
  --data-dir /path/to/kws-data/research/2026-10-06-melo5-a20-diagnostic -v
```

Python's standard library is sufficient. Verification checks the closed four-file
data archive and frozen source/metadata hashes, validates all 50 JSONL records,
128 saved six-class logit rows and exact feed/finish geometry, and reproduces
`observations.json` byte-for-byte. It does not load native libraries, execute a
model/frontend/native decoder, play audio, compile code or access the network.
The 12 tests include malformed-trace and archive-corruption failures; missing or
malformed evidence cannot become a no-trigger result.

`original_A20.raw.jsonl` is the acquired trace with only the trailing PID member
removed from its first `run_start` record. Every other byte, including all numeric
spellings and the other 49 records, is unchanged. Its public SHA256 is
`b93d9068eba66a8a744d9e706b0d16a763240df566942714029fa3f053b6f4ff`.
The original acquisition was 25,615 bytes with SHA256
`fc901e842278a9b2bf05cb1052f8c1a4b17a6633828e78ae1be008d6853a7636`.

`observations.json` preserves all 10,763 original bytes, SHA256
`7fa85a94cacc3a4a8353496d8075432798752390de8b27ae60b4fbb31f0f8381`.
Its raw hash intentionally identifies the acquired trace. The verifier explicitly
normalizes only this provenance hash when comparing the score of the public
projection. This is a declared original-to-public mapping, not a claim that the
two raw files have the same hash. `resources.json` removes absolute monotonic
clock readings and release-file inventory, preserves all retained measurements,
and states the reviewed memory caveat. All transformations and original/public
identities are in `metadata/bindings.json`.

## Frozen execution sources and dependencies

The final collector, generated five-input header, strict scorer and metadata are
copied unchanged. The scorer's original relative `metadata/` lookup is preserved.
`metadata/bindings.json` pins every copied file and binds runtime source by exact
path, byte count and SHA256 to pipeline commit
[5abf8de13f0ed097c223c82a98428bec3aab7e7f](https://github.com/jiying2007/kws-pipeline/tree/5abf8de13f0ed097c223c82a98428bec3aab7e7f/research/cosy49-results-v1/historical/numerical/vendor/runtime).
Those runtime files are not duplicated. For controlled source reconstruction,
copy the listed files to their `runtime/` destinations beside this package's
`src/`, then use the recorded GCC compile/link argument lists. Compilation was
not repeated during archive preparation. A different toolchain or system is not
promised to reproduce identical binaries.

The original native library SHA256 is
`68457311f93540405dfa0bf5fdc1cd74421fea22b44d5920fec465be5de4d912`;
the acquisition collector executable is
`e1914ce681dae0d776ba44276a1ab1f05839d58c7c8a6d6ecd6878700b65ad04`.
The original A20 payload remains an external dependency, 1,565,280 bytes, SHA256
`a80b233d3c958d25f49085f487ef5cde839bd69f803ed057b59bbd20adc6e43d`.
Canonical state and retained checkpoint provenance identities are also recorded;
the checkpoint was not reparsed here. Existing audio is identified by its public
archive commit, ZIP member names, WAV hashes and PCM-region hashes. No weights or
WAVs are duplicated, and the acquisition launcher is not distributed here.
Saved-output reproduction is complete; this is not a turnkey acquisition package.

The historical provenance's `collector_source_sha256` and
`scorer_source_sha256` fields identify retained endpoint98 helper inputs before
the Melo5 adaptations, despite their unqualified names. The bindings retain
those historical hashes and separately identify the final Melo5 C source
`d941b9db947fb6674b124ea49ac2292e230db236859b7db95afa1cbfa65c070b`
and scorer
`6d3ee242f432aba0cbe3eebc792e411334c52508840efe27e7d3d87b20c04c2e`.
Both final files match the reviewed revision-2 source freeze. These source hashes
are distinct from the compiled collector executable hash. The exact
`metadata/plan.json` is a pre-execution snapshot: its preparation/no-execution
fields describe that earlier stage, while the data `SUMMARY.json` records the
completed result.

## Timing, memory and interpretation

The 64,090 samples (4.005625 s) used five cold resets, 16 feeds, five finishes,
16 callbacks, 390 fbank rows and 128 native model steps. The retained FP64
accumulation/FFT64 runtime and original decoder configuration were unchanged:
six competing tokens, pruning strictly above 0.05, threshold 0, beams 3/20,
duration gates 5–250 frames and 50-frame cooldown. Actual short tails were used;
there was no padded EOF flush.

Native post-GO timing was 0.051394324 s wall and 0.052333467 s process CPU.
These timers start after dynamic loading, include payload initialization and
preceding trace output, and stop before the final record, deallocation and
`dlclose`. Feed/finish timing includes callback output. Audio was unpaced.
The separate supervisor launch-to-reap envelope was 0.054479546 s, and owned-child
user plus system CPU was 0.053152 s; validation adds a distinct envelope of
0.106362213 s. These are one local execution's observations, not a benchmark or
board measurement.

The 12,496 KiB `ru_maxrss` lifetime high-water mark may include the child's
inherited Python memory before `exec`; the inherited portion is unknown.
The maximum of six sampled native-process RSS/HWM observations was 2,872 KiB
(VmSize 5,436 KiB). Samples do not establish continuous or exact native peaks,
continuous thread compliance or child absence. Concurrent supervisor usage is
excluded. No resource qualification or optimization is claimed.

The saved greedy strings (M1 你好小, M2 你窝, M3 你好小, M4 屋, M5 你好) are
summaries of existing logits. They do not prove an acoustic cutoff or identify
the cause of missing activation. Research distribution only; existing component
provenance applies. Generated-voice/output rights and commercial clearance remain
unestablished, and this addition makes no new license grant.
