# Performance and research evidence

**Scope: technical research only.** This page defines measurable DSP, acoustic and target-resource experiments. No shipping SKU, Release, commercial model promotion or product-certification approval is a prerequisite. Retired product procedures remain in Git history; see [research-only execution](RESEARCH_ONLY_EXECUTION.md).

## Reference design envelope (estimates)

- Mono PCM16 at 16 kHz, 25-ms analysis window, 20-ms hop (50 acoustic steps/s).
- 32 log-mel/PCEN-lite features, exploratory 48-unit recurrent model, roughly 350–450 Mandarin pinyin tokens.
- Estimated 1.2 MMAC/s at 420 tokens; historical KWSP ABI v2 `.kwm` around 26 KB.
- Query runtime arena through `kws_engine_required_bytes()`. Int8 model matrices and float accumulation are not a full-int8 runtime.

These are configuration/arithmetic estimates, **not** physical SSC305 latency, CPU, memory or power measurements.

## Hosted correctness and optimization

`kws_bench` covers C frontend/model/decoder. Required CI includes GCC/Clang, parity, sanitizer/fuzz coverage, ARM cross-build and QEMU; hosted x86 timing cannot be converted into Cortex-A32 CPU percentage.

Optimize only after freezing the exact code SHA, model/keyword pack, frontend/AFE configuration and input WAV hash. Profile FFT, nonlinear features, decoder and I/O; change one variable; preserve training/C parity and independent holdout. When a real board is unavailable, target performance remains `NOT_RUN`.

## Acoustic evaluation and honest uncertainty

```text
source-verified WAV + references.jsonl
  -> eval/run_corpus.py + exact kws_wav
  -> retained detections and provenance
  -> eval/score_events.py
  -> per-word FAR/hour, FRR, word-end latency, confidence bounds
```

Bind each original WAV by file SHA256, decoded PCM SHA256, frame count and actual audio duration. Report one-sided Wilson upper confidence bounds for FRR and exact Poisson upper bounds for FAR/hour. Zero observed false accepts does **not** prove zero true FAR. Synthetic/TTS/cloned-voice negative shards are development regressions, not real-human acoustic truth. Retain accent, near-homophone, loudspeaker and domain-shift results by speaker/source split.

## Physical target benchmark (optional)

`kws_board_bench` measures an explicitly chosen `.kwm/.kwk` pair on permissioned post-AFE WAV audio. Its historical command arguments are model, keyword pack, WAV and reset repetitions. Record input hashes, exact binary, compiler, board revision, CPU/governor/thermal conditions, measured samples/frames, p50/p95/p99/max, RTF and deadline misses. At least one complete 400-sample model frame is required per repeat; use 16-kHz / 400-window / 320-hop geometry. QEMU and cross-build results prove compatibility, not real-device timing.

## Long-running resource observation (optional)

`tools/collect_runtime_soak.py` may observe an **audited research process** using `--hours`, `--sample-seconds`, `--output` and `--command`. Keep actual/requested duration, early exit, CPU-time, RSS, thermal and thread samples. CPU percentage follows `process-cpu-one-core-v1`: aggregate process CPU seconds divided by actual wall seconds × 100, with no core normalization or clamp. CPU/audio-second is unavailable without measured audio exposure. Benchmark RTF is a separate measurement. See [target evidence](TARGET_EVIDENCE.md#cpu-units-and-unavailable-audio-exposure). A 24/72-hour duration is **not** an admission or publication gate.

For measured physical power/stack/thermal data retain unmodified sensor export, instrument/calibration IDs, source/model/pack hashes and exact board manifest. Missing data is `NOT_RUN`, never zero or a pass.

## Evidence identity and audio continuity

Bind model, keyword pack, source SHA, toolchain, corpus rights, train/calibration/test splits, references, detections, target traces and source hashes. An exact `{name, sha256, bytes}` manifest may verify raw evidence bytes; independent test evidence cannot be replaced by training replays. Old `qualification/` filenames and product-policy validators describe **historical formats**, not current scientific authority; do not run old SKU approval commands.

When capture breaks, call `kws_engine_notify_discontinuity()`. Retain audio-pipeline XRUN/backpressure, KWS gap counters, actual sustained load, RSS/stack and sensor trace separately. A passing host test, synthetic FAR experiment and physical measurement are different evidence classes and cannot be promoted into field-quality claims without independent data.
