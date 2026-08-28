# Performance and release gates

Hosted results are regression signals only. Shipping claims require the exact SoC, compiler flags, DVFS governor, thermal policy, audio route and model artifact.

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

The CI benchmark runs the actual C frontend + model + decoder for 30 seconds of synthetic 16-kHz audio using 32 features, 48 hidden units and 420 output tokens. On one GitHub Ubuntu runner for commit `61d9c4bf6903512fd57323204cdcec60889538b4`, GCC reported:

```text
model_bytes=25944
engine_bytes=15464
audio_s=30
cpu_s=0.044281
rtf=0.001476
```

Clang on another hosted runner reported RTF `0.001573`. These values prove the benchmark and size accounting are live and provide a regression baseline; they are **not** Cortex-A32 performance claims and should not be converted into target-device CPU percentages.

## Acoustic release gate

Release qualification uses the real runtime, model and keyword pack over continuous audio:

```text
references.jsonl + WAV corpus
 -> eval/run_corpus.py + kws_wav
 -> detections.jsonl
 -> eval/score_events.py
 -> FAR/hour, FRR, p50/p95 latency, per-keyword metrics
```

The scorer can fail CI/qualification runs through `--max-far-per-hour`, `--max-frr` and `--max-p95-latency-ms`. Threshold values are product requirements and must not be copied blindly from examples.

False accepts are exported to `false-positives.jsonl` and may be converted to hard-negative clips. The final held-out qualification set must remain isolated from hard-negative mining and model tuning.

## Target-board certification

For every shipping SKU record:

- exact `.kwm`, `.kwk` and token-vocabulary SHA-256 plus ABI version/fingerprint;
- compiler/toolchain, CPU flags and optimization flags;
- CPU topology, affinity, governor/DVFS and thermal policy;
- mean/p95/p99 process time per 20-ms hop;
- at least 4x p99 scheduling headroom against the 20-ms hop where practical;
- CPU percentage in a 30-minute always-on run;
- resident/private memory and thread-stack high-water mark;
- wake latency from keyword end;
- thermal/power impact in an 8-hour soak;
- FRR by speaker, distance, angle, SPL/SNR and acoustic bucket;
- false accepts/hour on long continuous negative audio;
- TV/music/speech playback, near-homophones and partial-phrase negatives;
- AEC/NS/AGC configuration and local-speaker playback conditions;
- motor/fan/gear/mechanical-noise scenarios relevant to the product;
- audio XRUN/backpressure evidence from the complete product pipeline.

Cross-build proves compiler/ISA compatibility only. Hosted execution and QEMU-style signals must never be presented as real-board latency, CPU, thermal or power data.
