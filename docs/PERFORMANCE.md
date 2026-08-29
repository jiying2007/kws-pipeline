# Performance and release gates

Hosted results are regression signals only. Shipping claims require the exact SoC, compiler flags, DVFS governor, thermal policy, audio route, model artifact, keyword pack, held-out corpus and machine/raw target evidence.

## Default design envelope

- mono PCM16, 16 kHz;
- 25-ms window, 20-ms hop (50 acoustic steps/s);
- 32 log-mel or PCEN-lite features;
- 48 recurrent units;
- target Mandarin full-pinyin vocabulary roughly 350–450 tokens;
- dense acoustic work about 1.2 MMAC/s at 420 tokens;
- KWSP model ABI v2 default-geometry `.kwm` roughly 26 KB;
- fixed/queryable engine memory through `kws_engine_required_bytes()`.

The `.kwm` stores int8 matrices while v0.3 continues to use float activations/accumulation. Cortex-A32 NEON accelerates int8-weight × float-activation dot products without changing model/training semantics. Full-int8 activation/requantization is a separate model-format/accuracy decision and must not be introduced without profiling, retraining and parity qualification.

The numbers above are arithmetic/format estimates, not Cortex-A32 measurements.

## Hosted regression matrix

`kws_bench` runs the actual C frontend + model + decoder and covers deterministic envelopes including logmel/PCEN and normal/max keyword counts. It reports model/engine bytes, Trie size, dense MAC estimates, RTF and CPU microseconds per audio second.

CI runs the matrix under GCC and Clang and cross-builds the Cortex-A32 NEON path. Hosted x86 timing must **not** be converted into a target-device CPU percentage.

## Frontend cost and optimization policy

PCEN-lite contains stateful normalization and floating-point nonlinear operations. The frontend may dominate runtime on a small Cortex-A device even when the dense RNN MAC count is low. Do not optimize based on MAC count alone.

The required sequence is:

1. measure the exact shipping model/frontend on the physical target;
2. profile the full KWS path including FFT/nonlinear/frontend work;
3. optimize the measured hotspot;
4. preserve C/reference/training parity and rerun acoustic qualification.

Any LUT/polynomial/approximation or future full-int8 conversion must be gated by the same model/frontend lineage and real acoustic evidence.

## Acoustic release gate

Release evaluation is continuous-audio based:

```text
references.jsonl + real WAV corpus
 -> eval/run_corpus.py + exact kws_wav
 -> detections.jsonl + evaluation provenance schema v2
 -> eval/score_events.py
 -> FAR/hour, FRR, p50/p95 latency, per-keyword metrics
```

In v0.3, evaluation provenance binds **every original WAV** by file SHA256 + decoded PCM SHA256 + frame count. `references.duration_s` must equal real WAV duration. FAR exposure therefore comes from the actual audio bytes rather than a self-declared duration.

The final held-out qualification set must remain isolated from replay/model/threshold tuning.

### Statistical qualification

Policy schema v2 gates point estimates and one-sided confidence bounds:

- FRR: one-sided Wilson binomial upper bound from false rejects / expected wakes;
- FAR: one-sided exact Poisson rate upper bound from false accepts / negative exposure hours.

Zero observed false accepts in finite time is not proof of zero underlying FAR. Evidence hours/events must be sufficient for the approved confidence-bound requirement.

## Frozen-model long-FAR regression

The nightly workflow freezes one runner/model/pack identity and streams multiple independent synthetic negative shards. `eval/aggregate_far.py` refuses to aggregate exposure from different identities.

This is **synthetic streaming FAR regression**, not real-room acoustic qualification and not additional target-board evidence.

## Real-artifact target-board benchmark

Build `kws_board_bench` with the shipping target toolchain and execute the exact `.kwm/.kwk` on representative post-AEC/post-NS audio:

```bash
./kws_board_bench release/base.kwm release/xiaowo.kwk board-audio.wav 10 \
  > board-summary.json
```

The tool measures one 320-sample/20-ms hop per call with `CLOCK_MONOTONIC` and emits:

- model/pack/engine bytes;
- audio duration/repeats/block count;
- mean/p50/p95/p99/max process time;
- RTF;
- p99 headroom against the 20-ms deadline.

A useful low-end scheduling objective is at least 4× p99 headroom when the product's full audio-thread architecture permits it. The actual gate is SKU-specific and belongs in the approved policy.

Cross-build success proves compiler/ISA compatibility only; it does not generate physical-board timing evidence.

## Machine-bound target evidence

Use `tools/collect_target_evidence.py` on the physical target. The collector records/binds available machine state such as:

- target/board/SOC/toolchain/compiler flags;
- kernel/machine/CPU identity;
- online CPUs and governor;
- RSS;
- thermal observations;
- soak duration;
- externally supplied CPU/stack/power measurements;
- audio-frontend identity;
- raw measurement files with SHA256;
- external instrument/calibration identity where applicable.

The qualification manifest requires the exact evidence collector and raw evidence files. A manually typed resource JSON with no retained raw evidence is not sufficient for v0.3 final qualification.

## Audio continuity and soak

A realistic always-on performance test also needs the complete audio path:

- audio-pipeline XRUN/backpressure counts;
- KWS discontinuity counters;
- sustained CPU/load under expected concurrency;
- thermal/power stabilization;
- memory/RSS and stack high-water;
- route/suspend/resume behavior where the product supports them.

When capture continuity breaks, integration must call `kws_engine_notify_discontinuity()` so acoustic state does not cross a sample gap.

## Artifact-bound qualification manifest

`tools/qualification_manifest.py` v0.3 combines and independently rechecks:

- exact `.kwm`, `.kwk`, vocabulary and runtime config;
- model provenance schema v3, source checkpoint and canonical real training-corpus identity;
- dataset audit schema v3;
- evaluation summary/provenance plus **actual held-out WAV corpus identity**;
- target-board benchmark summary and exact board runner/audio;
- target evidence schema v2, exact collector and every raw evidence file.

`tools/qualification_gate.py` applies the separate SKU policy after these integrity checks. See `docs/RELEASE_QUALIFICATION.md`.

## Target-board certification checklist

For every shipping SKU retain:

- exact `.kwm`, `.kwk`, token vocabulary/runtime-config identity;
- source commit and released software version;
- model provenance + checkpoint + training image digest + real training corpus identity;
- clean dataset audit;
- compiler/toolchain, CPU flags, topology, affinity, governor/DVFS and thermal policy;
- mean/p50/p95/p99/max process time and RTF from `kws_board_bench`;
- sustained CPU percentage, RSS and stack high-water;
- wake latency from keyword end;
- extended soak duration/thermal/power evidence and raw measurement bytes;
- FRR by speaker, distance, angle, SPL/SNR and acoustic bucket;
- real continuous false accepts/hour plus confidence bound;
- TV/music/speech playback, near-homophones and partial phrases;
- final AEC/NS/AGC configuration and local-speaker playback conditions;
- motor/fan/gear/mechanical-noise scenarios;
- audio XRUN/backpressure + KWS discontinuity evidence;
- measured dual-mic RIR identity when used;
- original held-out qualification WAV bytes or immutable storage identities;
- qualification manifest schema v2, approved policy and gate result schema v3.

Hosted execution, synthetic models, simulated RIR, cross-build or QEMU-style signals must never be presented as real-board latency, acoustic quality, CPU, thermal or power data.
