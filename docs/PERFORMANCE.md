# Performance and release gates

Hosted results are regression signals only. Shipping claims require the exact SoC, compiler flags, DVFS governor, thermal policy, audio route, model artifact, keyword pack and held-out corpus.

## Default design envelope

- mono PCM16, 16 kHz;
- 25-ms window, 20-ms hop (50 acoustic steps/s);
- 32 log-mel features;
- 48 recurrent units;
- target Mandarin full-pinyin vocabulary roughly 350-450 tokens;
- dense acoustic work about 1.2 MMAC/s at 420 tokens;
- ABI-v2 default-geometry `.kwm` is roughly 26 KB;
- engine memory is fixed and queryable with `kws_engine_required_bytes()`.

These are arithmetic/format estimates, not Cortex-A32 measurements.

## Hosted regression signal

`kws_bench` runs the actual C frontend + model + decoder for synthetic 16-kHz audio using the default 32-feature / 48-hidden / 420-token geometry. CI executes it under both GCC and Clang so model/arena byte accounting and gross real-time regressions remain visible.

The resulting x86 RTF is intentionally **not a gate for target Cortex-A32/A7 performance**. Runner type, clock frequency, cache hierarchy and compiler differ too much to convert hosted utilization into target-device CPU percentages.

## Acoustic release gate

Release qualification runs the real runtime, model and keyword pack over continuous audio:

```text
references.jsonl + WAV corpus
 -> eval/run_corpus.py + kws_wav
 -> detections.jsonl + detections.provenance.json
 -> eval/score_events.py
 -> eval-summary.json
 -> FAR/hour, FRR, p50/p95 latency, per-keyword metrics
```

`run_corpus.py --provenance` SHA256-binds the runner binary, model, keyword pack, reference annotations and generated detections. `score_events.py` stores reference/detection SHA256 values in its summary, so the release manifest can detect a report copied from a different corpus run.

False accepts may be converted to hard-negative clips. The final held-out qualification set must remain isolated from hard-negative mining and model tuning.

## Real-artifact target-board benchmark

Build `kws_board_bench` for the target Linux toolchain and execute the **shipping `.kwm` and `.kwk`** on representative post-AEC/post-NS 16-kHz PCM16 audio:

```bash
./kws_board_bench release/base.kwm release/xiaowo.kwk board-audio.wav 10 \
  > board-summary.json
```

The tool measures one ABI-v2 hop (320 samples / 20 ms) per call using `CLOCK_MONOTONIC` and emits:

- model, keyword-pack and engine bytes;
- audio duration, repeat count and block count;
- mean/p50/p95/p99/max process time;
- real-time factor (RTF);
- p99 headroom relative to the 20-ms hop deadline.

For low-end Cortex-A devices, a useful scheduling objective is at least 4x p99 headroom when the product's complete audio thread architecture permits it. The actual shipping policy is SKU-specific and belongs in the approved qualification policy, not in source code.

The Cortex-A32 CI cross-build compiles this tool together with the core, which proves compiler/ISA compatibility. It does not generate target-board timing evidence.

## Artifact-bound qualification manifest

`tools/release_manifest.py` combines:

- exact `.kwm`, `.kwk`, token vocabulary and runtime config hashes;
- evaluation summary + evaluation provenance;
- target-board benchmark summary;
- target/board/toolchain/governor/audio-front-end identity;
- soak duration, CPU, RSS, stack high-water, temperature and power measurements.

It rejects mismatched vocabulary fingerprints, evaluation provenance from different model/pack artifacts, summary/reference/detection hash mismatches, board reports with different artifact sizes, malformed/non-finite values and missing target evidence.

`tools/qualification_gate.py` then applies a separate SKU policy. This separation keeps evidence immutable while allowing requirements to vary by product. See `docs/RELEASE_QUALIFICATION.md`.

## Target-board certification checklist

For every shipping SKU retain:

- exact `.kwm`, `.kwk`, token vocabulary and runtime-config SHA256 plus ABI/fingerprint;
- source commit SHA and corpus identity/version;
- compiler/toolchain, CPU flags and optimization flags;
- CPU topology, affinity, governor/DVFS and thermal policy;
- mean/p50/p95/p99/max process time and RTF from `kws_board_bench`;
- CPU percentage in a sustained always-on run;
- RSS/private memory and stack high-water mark;
- wake latency from keyword end;
- thermal/power impact in an extended soak;
- FRR by speaker, distance, angle, SPL/SNR and acoustic bucket;
- false accepts/hour on long continuous negative audio;
- TV/music/speech playback, near-homophones and partial-phrase negatives;
- AEC/NS/AGC configuration and local-speaker playback conditions;
- motor/fan/gear/mechanical-noise scenarios relevant to the product;
- audio XRUN/backpressure evidence from the complete product pipeline;
- qualification manifest, approved policy and gate result.

Hosted execution, synthetic models, cross-build and QEMU-style signals must never be presented as real-board latency, acoustic quality, CPU, thermal or power data.
