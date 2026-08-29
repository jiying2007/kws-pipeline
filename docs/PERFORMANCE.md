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

These are arithmetic/format estimates, not Cortex-A32 measurements.

## Hosted regression matrix

`kws_bench` runs the actual C frontend + model + decoder and covers deterministic logmel/PCEN and normal/max-keyword envelopes. CI runs the matrix under GCC and Clang and cross-builds the Cortex-A32 NEON path. Hosted x86 timing must **not** be converted into target-device CPU percentage.

## Frontend cost and optimization policy

PCEN-lite contains stateful normalization and floating-point nonlinear operations. The frontend may dominate runtime on a small Cortex-A device even when dense RNN MAC count is low. Do not optimize from MAC count alone.

Required sequence:

1. measure the exact shipping model/frontend on the physical target;
2. profile the complete KWS path including FFT/nonlinear/frontend work;
3. optimize the measured hotspot;
4. preserve C/reference/training parity and rerun acoustic qualification.

Any LUT/polynomial/approximation or future full-int8 conversion must be justified by measured target data.

## Acoustic release gate

```text
references.jsonl + real WAV corpus
 -> eval/run_corpus.py + exact kws_wav
 -> detections + evaluation provenance schema v2
 -> eval/score_events.py
 -> FAR/hour, FRR, p50/p95 latency, per-keyword metrics
```

v0.3 evaluation provenance binds every original WAV by file SHA256 + decoded PCM SHA256 + frame count. `references.duration_s` must equal real WAV duration, so FAR exposure comes from actual audio bytes rather than a self-declared duration.

Final held-out data must remain isolated from replay/model/threshold tuning.

### Statistical qualification

Policy schema v2 gates point estimates and one-sided confidence bounds:

- FRR: one-sided Wilson binomial upper bound;
- FAR: one-sided exact Poisson rate upper bound.

Zero observed false accepts over finite exposure is not proof of zero underlying FAR. Evidence duration/event count must support the approved confidence bound.

## Frozen-model long-FAR regression

Nightly regression freezes one runner/model/pack identity and streams independent synthetic negative shards. `eval/aggregate_far.py` rejects mixed identities.

This is **synthetic streaming FAR regression**, not real-room acoustic qualification or physical-board evidence.

## Real-artifact target-board benchmark

Build `kws_board_bench` with the shipping target toolchain and execute the exact `.kwm/.kwk` on representative post-AEC/post-NS audio:

```bash
./kws_board_bench release/base.kwm release/xiaowo.kwk board-audio.wav 10 \
  > board-summary.json
```

The tool measures one 320-sample/20-ms hop with `CLOCK_MONOTONIC` and reports model/pack/engine bytes, mean/p50/p95/p99/max processing time, RTF and p99 headroom.

A useful low-end scheduling objective is at least 4× p99 headroom when the complete product audio architecture permits it. The actual gate is SKU-specific.

Cross-build success proves compiler/ISA compatibility only; it does not generate physical-board timing evidence.

## Sustained process evidence

Per-hop benchmark timing and sustained product-process resource behavior are different measurements. Use `tools/collect_runtime_soak.py` to supervise the **actual process under test**:

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

Runtime-soak schema v2 records actual/requested duration and samples the child process:

- max RSS from `/proc/<pid>/status`;
- average CPU percentage from `/proc/<pid>/stat` CPU-time deltas;
- thermal-zone observations/max temperature when available;
- raw time-series samples;
- early-exit/completion state.

The collector fails if the child exits before the requested soak duration. This prevents the evidence collector’s own RSS/CPU from being mistaken for the product under test.

## Target evidence assembly

`tools/collect_target_evidence.py` consumes the retained runtime-soak JSON and derives `soak_hours`, CPU, RSS and max temperature from it. These fields are no longer final command-line declarations.

Stack high-water is platform/harness specific and must be measured by the product harness with retained raw evidence. External power evidence requires the original trace plus instrument/calibration identity.

Example:

```bash
python3 tools/collect_target_evidence.py \
  --output qualification/evidence.json \
  --target product-sku-a \
  --board-revision A \
  --toolchain arm-linux-gnueabihf-gcc-... \
  --compiler-flags '-O3 -mcpu=cortex-a32 ...' \
  --audio-frontend audio-pipeline-vX \
  --audio-frontend-sha256 <sha256> \
  --runtime-soak qualification/runtime-soak.json \
  --stack-high-water-bytes <measured> \
  --average-power-mw <measured> \
  --raw-evidence qualification/stack-watermark.txt \
  --power-raw qualification/power.csv \
  --instrument-id <meter-id> \
  --calibration-id <calibration-id>
```

The qualification manifest binds the exact evidence collector plus runtime-soak, power and any additional raw evidence files.

## Audio continuity and soak

A realistic always-on performance test also needs the complete audio path:

- audio-pipeline XRUN/backpressure counts;
- KWS discontinuity counters;
- sustained load under expected concurrency;
- thermal/power stabilization;
- RSS and stack high-water;
- route/suspend/resume behavior where supported.

When capture continuity breaks, integration must call `kws_engine_notify_discontinuity()` so acoustic state does not cross a sample gap.

## Artifact-bound qualification manifest

`tools/qualification_manifest.py` v0.3 independently rechecks:

- exact `.kwm`, `.kwk`, vocabulary and runtime config;
- model provenance schema v3, checkpoint and canonical real training-corpus identity;
- exact clean dataset audit coverage;
- evaluation summary/provenance plus actual held-out WAV corpus identity;
- target-board benchmark summary and exact runner/audio;
- target evidence schema v2, exact collector and every raw evidence file.

`tools/qualification_gate.py` applies the separate SKU policy after these integrity checks. See `docs/RELEASE_QUALIFICATION.md`.

## Target-board certification checklist

For every shipping SKU retain:

- exact model/pack/token/runtime identities and release source/tag;
- model provenance + checkpoint + training image digest + real training-corpus identity;
- clean dataset audit covering exact final `references.jsonl`;
- compiler/toolchain, CPU flags/topology/affinity/governor/DVFS/thermal policy;
- mean/p50/p95/p99/max process time and RTF from `kws_board_bench`;
- runtime-soak JSON with sustained CPU/RSS/thermal/elapsed evidence;
- product-harness stack high-water raw evidence;
- power raw trace plus instrument/calibration identity;
- wake latency from keyword end;
- FRR by speaker/distance/angle/SPL/SNR/domain;
- real continuous false accepts/hour plus confidence bound;
- playback/near-homophone/partial phrase/AEC residual/mechanical-noise scenarios;
- audio XRUN/backpressure + KWS discontinuity evidence;
- measured dual-mic RIR identity when used;
- original held-out qualification WAV identity;
- qualification manifest schema v2, policy schema v2 and gate result schema v3.

Hosted execution, synthetic models, simulated RIR, cross-build or QEMU-style signals must never be presented as real-board latency, acoustic quality, CPU, thermal or power data.
