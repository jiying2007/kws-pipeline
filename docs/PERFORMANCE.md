# Performance and release gates

Hosted results are regression signals only. Shipping claims require the exact SoC, compiler flags, DVFS governor, thermal policy, audio route, model artifact, keyword pack and held-out corpus.

## Default design envelope

- mono PCM16, 16 kHz;
- 25-ms window, 20-ms hop (50 acoustic steps/s);
- 32 log-mel or PCEN-lite features;
- 48 recurrent units;
- target Mandarin full-pinyin vocabulary roughly 350-450 tokens;
- dense acoustic work about 1.2 MMAC/s at 420 tokens;
- KWSP model ABI v2 default-geometry `.kwm` is roughly 26 KB;
- engine memory is fixed and queryable with `kws_engine_required_bytes()`.

The `.kwm` stores int8 matrices, but v0.2.x intentionally keeps float activations and accumulation. Cortex-A32 NEON accelerates the int8-weight × float-activation dot products without changing model/training semantics. Full-int8 activation/requantization remains a separate model-format/accuracy decision and must not be introduced without retraining and parity qualification.

These are arithmetic/format estimates, not Cortex-A32 measurements.

## Hosted regression matrix

`kws_bench` runs the actual C frontend + model + decoder for synthetic 16-kHz audio using the default 32-feature / 48-hidden / 420-token geometry. It now exercises four deterministic envelopes:

| Case | Frontend | Keywords | Purpose |
| --- | --- | ---: | --- |
| baseline | logmel | 1 | minimum decoder load |
| product | logmel | 4 | normal multi-keyword load |
| pcen-product | pcen-lite | 4 | PCEN frontend cost |
| worst-case | pcen-lite | 16 | maximum keyword count + shared-prefix Trie |

Each line reports model/engine bytes, Trie nodes, estimated dense MACs/frame, RTF and CPU microseconds per audio second. CI executes the matrix under GCC and Clang. The Cortex-A32 cross-build separately proves the NEON path compiles for the shipping ISA family.

The resulting x86 RTF is intentionally **not a gate for target Cortex-A32/A7 performance**. Runner type, clock frequency, cache hierarchy and compiler differ too much to convert hosted utilization into target-device CPU percentages.

## PCEN cost control

PCEN-lite keeps its trained alpha normalization but avoids generic powers where the exponent is fixed: square root uses `sqrtf()` and `sqrt(2)` is a compile-time constant. Frontend C/Python/Torch parity remains the authority; any further approximation/LUT optimization must first preserve that parity before it can enter the target kernel.

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

`run_corpus.py --provenance` SHA256-binds the runner binary, model, keyword pack, reference annotations and generated detections. `score_events.py` stores reference/detection SHA256 values in its summary, so the qualification manifest can detect a report copied from a different corpus run.

False accepts may be converted to hard-negative clips. The final held-out qualification set must remain isolated from hard-negative mining and model tuning.

### Statistical qualification

Point estimates alone are not a shipping gate. Qualification policy schema v2 additionally sets a confidence level plus one-sided maximum bounds for FAR and FRR:

- FRR: one-sided Wilson binomial upper bound from false rejects / expected wakes;
- FAR: one-sided exact Poisson rate upper bound from false accepts / negative exposure hours.

For example, observing zero false accepts in 24 hours still has a non-zero 95% upper rate bound (about 0.125 FA/hour). Increasing evidence hours/events is therefore part of passing the gate, rather than declaring a zero point estimate to be proof of zero underlying error rate.

## Frozen-model long-FAR regression

The nightly workflow trains one model per frontend, freezes that exact model/pack SHA tuple, then streams four independent two-hour negative shards through the same tuple. `eval/aggregate_far.py` refuses to add exposure from different runner/model/pack identities and computes aggregate point and confidence-bound FAR.

This remains **synthetic streaming FAR regression evidence**, not real-room acoustic qualification. Training-seed robustness is a different experiment and must not be counted as additional FAR hours.

## Real-artifact target-board benchmark

Build `kws_board_bench` for the target Linux toolchain and execute the **shipping `.kwm` and `.kwk`** on representative post-AEC/post-NS 16-kHz PCM16 audio:

```bash
./kws_board_bench release/base.kwm release/xiaowo.kwk board-audio.wav 10 \
  > board-summary.json
```

The tool measures one KWSP-v2 hop (320 samples / 20 ms) per call using `CLOCK_MONOTONIC` and emits:

- model, keyword-pack and engine bytes;
- audio duration, repeat count and block count;
- mean/p50/p95/p99/max process time;
- real-time factor (RTF);
- p99 headroom relative to the 20-ms hop deadline.

For low-end Cortex-A devices, a useful scheduling objective is at least 4x p99 headroom when the product's complete audio thread architecture permits it. The actual shipping policy is SKU-specific and belongs in the approved qualification policy, not in source code.

The Cortex-A32 CI cross-build compiles this tool together with the core, which proves compiler/ISA compatibility. It does not generate target-board timing evidence.

## Artifact-bound qualification manifest

`tools/qualification_manifest.py` combines:

- exact `.kwm`, `.kwk`, token vocabulary and runtime config hashes;
- model-export provenance, source checkpoint, training vocabulary/manifests;
- evaluation summary + evaluation provenance;
- target-board benchmark summary;
- target/board/toolchain/governor/audio-front-end identity;
- soak duration, CPU, RSS, stack high-water, temperature and power measurements.

It rejects mismatched vocabulary fingerprints, evaluation provenance from different model/pack artifacts, summary/reference/detection hash mismatches, board reports with different artifact sizes, malformed/non-finite values and missing target evidence.

`tools/qualification_gate.py` then applies a separate SKU policy, including statistical FAR/FRR bounds. This separation keeps evidence immutable while allowing requirements to vary by product. See `docs/RELEASE_QUALIFICATION.md`.

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
- false accepts/hour on long continuous negative audio plus the policy confidence bound;
- TV/music/speech playback, near-homophones and partial-phrase negatives;
- AEC/NS/AGC configuration and local-speaker playback conditions;
- motor/fan/gear/mechanical-noise scenarios relevant to the product;
- audio XRUN/backpressure evidence from the complete product pipeline;
- measured dual-mic RIR manifest/hash when RIR augmentation is part of the candidate lineage;
- qualification manifest, approved policy and gate result.

Hosted execution, synthetic models, simulated RIR, cross-build and QEMU-style signals must never be presented as real-board latency, acoustic quality, CPU, thermal or power data.
