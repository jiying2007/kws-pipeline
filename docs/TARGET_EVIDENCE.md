# Target evidence collection

Target-board qualification must retain raw, machine-collected evidence alongside summarized SKU metrics. v0.3 separates **runtime soak acquisition** from **final target evidence assembly** so the evidence collector cannot accidentally measure itself and report that as product RSS/CPU.

## Runtime timing

Cross-build and retain `kws_board_bench`, then run the exact shipping `.kwm/.kwk` and representative post-AFE WAV. Preserve the emitted JSON and the benchmark executable/audio hashes.

`kws_board_bench` measures per-hop processing time/RTF/headroom. It does not replace sustained product-process CPU/RSS/thermal/soak evidence.

## Runtime soak

Run the actual product/KWS qualification process under `tools/collect_runtime_soak.py`:

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

The soak collector supervises the child process and samples that **child process**, not the collector itself. Schema v2 records:

- requested and actual elapsed hours;
- whether the requested duration completed without early process exit;
- child PID/command identity for engineering traceability;
- max child-process RSS;
- average child-process CPU percentage from `/proc/<pid>/stat` CPU-time deltas;
- thermal-zone samples and max temperature when available;
- raw time-series samples.

If the monitored process exits before the requested duration, the collector fails rather than manufacturing a successful soak summary.

Also retain audio-pipeline XRUN/backpressure counters and `kws_engine_get_stats()` snapshots/discontinuity counters from the complete product integration.

## Stack evidence

Generic Linux does not provide a reliable portable stack high-water metric for an arbitrary product thread. `stack_high_water_bytes` remains a product-harness measurement. It must come from an approved target harness/debug mechanism and its raw evidence should be retained with `--raw-evidence`.

Do not use `VmStk` allocation size as stack high-water unless the SKU policy explicitly defines that weaker metric.

## Power evidence

Power normally comes from an external instrument. Retain the original CSV/trace, instrument ID and calibration ID. The summarized average-power value is not accepted without the raw power file identity.

## Assemble target evidence

After runtime soak and power acquisition, run the exact retained repository collector:

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

The target-evidence collector derives `soak_hours`, `cpu_percent`, `rss_kib` and `max_temp_c` from the retained runtime-soak JSON rather than command-line declarations. It additionally records target/board/SoC/toolchain identity, kernel/machine/governor/online-CPU state, uptime, audio-front-end identity and SHA256 for every raw evidence file.

`runtime-soak.json` and `power.csv` are automatically part of the raw-evidence tuple; additional stack/audio/thermal traces can be repeated through `--raw-evidence`.

## Qualification binding

`qualification_manifest.py` requires the exact `collect_target_evidence.py` file and every raw artifact declared by the evidence JSON. Final manifest construction must therefore pass all collector-bound raw files, for example:

```bash
--evidence qualification/evidence.json \
--evidence-collector qualification/collect_target_evidence.py \
--raw-evidence qualification/runtime-soak.json \
--raw-evidence qualification/stack-watermark.txt \
--raw-evidence qualification/power.csv
```

Copy/retain the exact collector from the release source tree in the qualification bundle instead of relying on an unversioned system copy.

## Trust boundary

Machine/raw binding prevents accidental/self-reported evidence substitution, but SHA256 alone is not an external trust root. Production authenticity also depends on controlled qualification infrastructure plus release/signing/OTA trust roots.
