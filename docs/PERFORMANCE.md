# Performance and release gates

Hosted results are regression signals only. Shipping claims require the exact SoC, compiler flags, DVFS governor, thermal policy, audio route, model/keyword artifacts, held-out corpus and trusted product-board evidence.

## Default design envelope

- mono PCM16, 16 kHz;
- 25-ms window, 20-ms hop (50 acoustic steps/s);
- 32 log-mel or PCEN-lite features;
- 48 recurrent units;
- target Mandarin full-pinyin vocabulary roughly 350–450 tokens;
- dense acoustic work about 1.2 MMAC/s at 420 tokens;
- KWSP ABI v2 default `.kwm` roughly 26 KB;
- fixed/queryable engine memory through `kws_engine_required_bytes()`.

The `.kwm` stores int8 matrices while v0.3 uses float activations/accumulation. Cortex-A32 NEON accelerates int8-weight × float-activation dot products without changing model/training semantics. These are arithmetic/format estimates, not physical Cortex-A32 measurements.

## Hosted regression matrix

`kws_bench` runs the actual C frontend + model + decoder. CI runs GCC/Clang, static analysis, C coverage, sanitizers, fuzzing, a Cortex-A32 cross-build whose CTest suite executes under `qemu-arm-static`, and a hosted-vs-Cortex-A32 numerical parity gate on the int8 kernel. Hosted x86 timing must never be converted into target-device CPU percentage.

## Optimization policy

PCEN-lite and frontend nonlinear work may dominate runtime on a small Cortex-A device even when dense RNN MAC count is low. Optimize in this order:

1. measure the exact shipping model/frontend on the physical target;
2. profile the complete KWS path including FFT/nonlinear/frontend work;
3. optimize the measured hotspot;
4. preserve C/reference/training parity;
5. rerun the SKU acoustic/resource qualification.

Any LUT/polynomial approximation or future full-int8 conversion requires measured target motivation and requalification.

## Acoustic gate

```text
references.jsonl + real WAV corpus
 -> eval/run_corpus.py + exact kws_wav
 -> detections + evaluation provenance schema v2
 -> eval/score_events.py
 -> FAR/hour, FRR, latency and per-keyword metrics
```

v0.3 binds every original WAV by file SHA256 + decoded PCM SHA256 + frame count. `references.duration_s` must equal the real WAV duration, so FAR exposure comes from actual audio bytes.

Policy schema v3 gates point estimates and one-sided confidence bounds:

- FRR: one-sided Wilson binomial upper bound;
- FAR: one-sided exact Poisson rate upper bound.

Zero observed false accepts over finite exposure is not proof of zero underlying FAR.

## Synthetic long-FAR regression

Nightly regression freezes one runner/model/pack identity and streams independent synthetic negative shards. This is **synthetic streaming FAR regression**, not real-room acoustic qualification or physical-board evidence.

## Target-board benchmark

Build `kws_board_bench` with the shipping target toolchain and execute the exact `.kwm/.kwk` on representative post-AFE audio:

```bash
./kws_board_bench release/base.kwm release/xiaowo.kwk qualification/board-audio.wav 10 \
  > qualification/board-summary.json
```

The schema-v2 summary binds runtime source/config identity, board runner, model, pack and audio, and reports model/pack/engine bytes, mean/p50/p95/p99/max processing time, RTF and p99 headroom. Cross-build success proves compiler/ISA compatibility only; it is not target timing evidence.

## Sustained process evidence

Per-hop timing and sustained product-process behavior are different measurements. Supervise the actual process:

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

Runtime-soak schema v3 retains actual/requested duration, early-exit state, child-process CPU-time/RSS/thermal/thread samples and summary fields. CPU gates use `process-cpu-one-core-v1`: aggregate process CPU seconds / measured wall seconds × 100, with no core-count division or 100% clamp. `max_thread_count` is the observed sample maximum; CPU/audio-second is null because soak audio exposure is not measured. Board-benchmark RTF has its own known input-audio duration and is a separate metric. Later verification independently recomputes the retained summaries. Older capacity-normalized evidence is rejected. See [the CPU unit contract](TARGET_EVIDENCE.md#cpu-units-and-unavailable-audio-exposure).

## Canonical raw measurement identity

Freeze runtime-soak, stack, power and any other target measurement files into `qualification/evidence-raw.jsonl` using exact `{name, sha256, bytes}` rows. The row set must exactly equal runtime-soak + repeated `--raw-evidence` + `--power-raw`.

Power requires the original instrument export plus instrument/calibration identity. Stack high-water remains product-harness specific and must have retained raw evidence.

The controlled product trust layer must also produce `qualification/attestation-verification.json`, schema v1, verifying the canonical raw manifest plus exact collector/board-runner/model/keyword-pack identities.

Benchmark input must contain at least one complete 400-sample model frame per reset repeat. Schema v2 records actual processed samples/frames and binds 16-kHz / 400-window / 320-hop geometry, per-repeat blocks and final-call length. Input-call timings include the unpadded tail and use a nominal 20 ms deadline with nearest-rank percentiles. Release and Phase-B gates independently verify the WAV and require the exact board summary in the externally attested raw-evidence tuple.

## Product-board evidence assembly

```bash
python3 tools/collect_target_evidence.py \
  --output qualification/evidence.json \
  --target product-sku-a \
  --board-revision A \
  --soc cortex-a32 \
  --toolchain arm-linux-gnueabihf-gcc-... \
  --compiler-flags '-O3 -mcpu=cortex-a32 ...' \
  --audio-frontend audio-pipeline-vX \
  --audio-frontend-sha256 <sha256> \
  --runtime-soak qualification/runtime-soak.json \
  --stack-high-water-bytes <measured> \
  --average-power-mw <measured> \
  --raw-evidence qualification/stack-watermark.txt \
  --raw-evidence qualification/board-summary.json \
  --power-raw qualification/power.csv \
  --evidence-raw qualification/evidence-raw.jsonl \
  --attestation-verification qualification/attestation-verification.json \
  --board-runner qualification/kws_board_bench.target \
  --model release/base.kwm \
  --keyword-pack release/xiaowo.kwk \
  --board-audio qualification/board-audio.wav \
  --sku product-sku-a \
  --source-sha "$(git rev-parse HEAD)" \
  --builder-id qualification-builder-01 \
  --dut-id product-dut-01 \
  --collector-id qualification-station-01 \
  --instrument-id <meter-id> \
  --calibration-id <calibration-id>
```

`builder-id` and `dut-id` must be distinct. Accepted shipping resource evidence is schema v3 with `evidence_class=product-board` and exact SKU/source/artifact/raw/attestation identity.

## Audio continuity and soak

A realistic always-on performance qualification also retains:

- audio-pipeline XRUN/backpressure counts;
- KWS discontinuity counters;
- sustained load under expected concurrency;
- thermal/power stabilization;
- RSS and stack high-water;
- route/suspend/resume behavior where supported.

When capture continuity breaks, integration must call `kws_engine_notify_discontinuity()` so acoustic state does not cross a sample gap.

## Artifact-bound qualification

`qualification_manifest.py` independently verifies:

- exact `.kwm`, `.kwk`, vocabulary/config/checkpoint/model provenance;
- actual training WAV corpus identity;
- clean dataset-audit coverage;
- evaluation summary/provenance plus actual held-out WAV identity/duration;
- target board runner/audio/summary;
- product-board evidence schema v3;
- exact collector, canonical `--evidence-raw`, `--attestation-verification` and every raw evidence file;
- exact `--sku` and `--source-sha`.

`qualification_gate.py` requires the selected `--references`, `--detections` and `--board-audio` files, independently replays canonical event scoring and verifies effective benchmark workload, then applies the matching `shipping_approved=true` SKU policy.

## Target-board certification checklist

For every shipping SKU retain:

- exact source/tag/runtime build identity and model/pack/token/config artifacts;
- model provenance + checkpoint + training image digest + training-corpus identity;
- clean dataset audit covering exact final references;
- compiler/toolchain, CPU topology/affinity/governor/DVFS/thermal policy;
- target benchmark timing/RTF/headroom;
- runtime-soak raw CPU/RSS/thermal/elapsed samples;
- product-harness stack raw evidence;
- power raw trace + instrument/calibration identity;
- canonical raw-evidence manifest and external attestation-verification result;
- SKU/source/builder/DUT/collector identities;
- wake latency, FRR by domain, real continuous FAR/hour + confidence bounds;
- playback/near-homophone/AEC residual/mechanical-noise scenarios;
- XRUN/backpressure/discontinuity evidence;
- original held-out WAV identity;
- qualification manifest v3, policy v3 and gate result v4.

Hosted execution, synthetic models, simulated RIR, cross-build or QEMU-style signals must never be presented as real-board latency, acoustic quality, CPU, thermal or power data.
