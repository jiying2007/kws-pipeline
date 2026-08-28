# Performance and release gates

Hosted results are regression signals only. Shipping claims require the exact SoC, compiler flags, DVFS governor, thermal policy, audio route and model artifact.

## Default design envelope

- mono PCM16, 16 kHz;
- 25-ms window, 20-ms hop (50 acoustic steps/s);
- 32 log-mel features;
- 48 recurrent units;
- target Mandarin full-pinyin vocabulary roughly 350-450 tokens;
- dense acoustic work about 1.2 MMAC/s at 420 tokens;
- model weight+bias size about 26 KB at 420 tokens;
- engine memory is fixed and queryable with `kws_engine_required_bytes()`.

These numbers are arithmetic/format estimates, not Cortex-A32 measurements.

## Target-board certification

For every shipping SKU record:

- `.kwm`, token file and keyword-manifest SHA-256;
- compiler/toolchain and CPU flags;
- mean/p95/p99 process time per 20-ms hop;
- CPU percentage in a 30-minute always-on run;
- RSS/private dirty and stack high-water mark;
- wake latency from keyword end;
- thermal/power impact in an 8-hour soak;
- FRR by speaker/acoustic bucket;
- false accepts/hour on continuous negative audio;
- AEC/NS/AGC configuration used during evaluation.

A useful scheduling target for low-end Cortex-A devices is at least 4x p99 headroom relative to the 20-ms hop deadline. Do not convert hosted x86 utilization into a target-board KPI.
