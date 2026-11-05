# Physical-target research evidence collection

**Technical research only.** Real SSC305/DUT studies require source-bound, independently retained observations, **not product Shipping or certification authority**. Historical Phase-B/product receipts, filenames and signing schemes remain in the immutable Git history; they are not current model/algorithm/data admission gates. Hosted fixtures and QEMU cannot substitute for physical measurements. Missing hardware, rights or actual captured evidence remains `NOT_RUN` or `BLOCKED`.

## Baseline runtime identity

Build the C runner from the exact audited commit, bind executable and configuration SHA256, and record actual board, OS, governor, affinity, audio I/O path and sample rate. The preserved research comparator assets are:

- `models/registry/model-749187ec1d66/xiaowo-model.kwm`
- `models/registry/model-749187ec1d66/xiaowo-keywords.kwk`

Verify both through `tools/verify_model_registry.py` before a diagnostic. These bytes represent the **historical synthetic comparator**, not independently demonstrated real-speech performance or an SSC305 result. Fresh research models require their own source/manifest/consumed-data hashes and cannot inherit this identity.

## Runtime timing and audio exposure

Use `kws_board_bench` with an appropriately licensed and source-hashed *actual post-AFE WAV* to measure per-hop latency, real-time factor, headroom and the exact workload. Preserve measured frames, wall time, runner/model/pack/AFE versions and input WAV/decoded PCM identity. Cross-compilation and synthetic benchmark scores remain software diagnostics, not target acoustic claims.

`tools/collect_runtime_soak.py` independently observes a **real, long-running research KWS process** using `--hours`, `--sample-seconds`, `--output` and `--command`. Pass an actual installed process path only after reviewing the executable and its source SHA; this repository does **not** supply a fake `product-kws-soak` process. The observer fails on early exit and retains requested/actual duration, child PID, process CPU and RSS, threads, temperature and actual samples. Duration is not proof that audio was processed.

## CPU units and unavailable audio exposure

CPU measurements retain `measurement_contract_id=process-cpu-one-core-v1` with `cpu_percent_semantics="process_cpu_seconds / wall_seconds * 100"`. `process_cpu_seconds` is measured from the child process's `/proc/<pid>/stat` user and system CPU deltas, aggregated over that process's threads. A launcher that forks away the workload is not an acceptable observation target. Descendants are outside the sampler's scope.

`wall_seconds` equals measured elapsed observation time, excluding cleanup. **100% means one fully occupied core; 150% means 1.5 cores.** Do not divide by the number of available cores, affinity width or thread count. On a dual-core device, 20 CPU-seconds in 100 wall-seconds means 20%, not 10%. Capture the actual thread-count sample series and maximum observed RSS; samples do not rule out between-sample peaks.

This soak collector does **not** count actual audio samples. Therefore `audio_seconds` and `cpu_seconds_per_audio_second` remain null; wall time must not be substituted for audio duration. `kws_board_bench` can report per-audio-second RTF only for its own known WAV/repeat execution. Legacy units in older archived receipts cannot simply be relabelled as fresh research evidence.

## Independent source, board, stack and power evidence

For any physical study, retain an exact manifest of actual raw traces, hashes and byte lengths, including the source checkout/build, runner binary, model/keyword pack, captured WAV/PCM, process CPU/RSS/thermal series, board/clock/route and any real external instrument output. Recompute hashes independently at readback. Missing files, changed source identity, inconsistent durations, non-finite metrics and unmeasured dimensions fail or remain `NOT_RUN`; summaries cannot substitute for the raw observations.

Generic Linux does not expose a portable and trustworthy stack high-water mark for arbitrary threads; report `stack_high_water_bytes` only with separately documented instrumentation and retain its raw measurements. Power measurements require actual instrument/CSV/calibration identity, not a guessed milliwatt average. Keep process and algorithm resource measurements separate from acoustic FAR/FRR. If downstream audio-pipeline exposes XRUN/backpressure counters, preserve the actual snapshots and `kws_engine_get_stats()` discontinuity telemetry alongside the tested AFE identity.

## Independent scientific interpretation

Reserved independent human speech, far-field 3–5 m, challenging near-homophones and continuous negative streams are measured under their own identities and may not be fed back to training while still called held-out. A successful hosted test, model-hash verification, source-only DUT-contract fixture or synthetic replay is **not** a claim that the SSC305 board, real microphones, true audio exposure, physical thermal/power budget or wake-up error rate was measured.

The retired product certificate/Shipping approval generators and `release/base.kwm` examples are not present-day commands. Historical policy/attestation schemas may be audited offline with explicitly guarded replay tools; do not restore old `qualification_manifest.py` or `qualification_gate.py` as research permission. Do not fabricate hardware, source rights, measurement records or model-quality claims.
