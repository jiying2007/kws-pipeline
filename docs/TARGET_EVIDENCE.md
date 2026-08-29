# Target evidence collection

Target-board qualification must retain raw, machine-collected evidence alongside the summarized SKU metrics.

## Runtime timing

Cross-build and retain `kws_board_bench`, then run the exact shipping `.kwm/.kwk` and representative post-AFE WAV. Preserve the emitted JSON and the benchmark executable/audio hashes.

## System evidence

Use `tools/collect_target_evidence.py` on the target after the sustained run. The collector records target/board/SoC/toolchain identity, kernel/machine/governor/online CPU state, uptime, RSS, thermal state and supplied soak/CPU/stack/power measurements. When power comes from an external instrument, pass the raw export plus instrument and calibration identifiers so the raw bytes are SHA256-bound.

The CPU and stack measurements remain platform-specific inputs because generic Linux cannot infer a product thread's sustained CPU percentage or stack watermark reliably. They must come from the approved product harness, not an arbitrary hand-written JSON file.

## Soak

`tools/collect_runtime_soak.py` can supervise a long-running qualification command and retain liveness/RSS samples. Products should additionally retain audio XRUN/backpressure counters, KWS telemetry snapshots and platform thermal/power traces.

## Trust boundary

SHA256 binding proves that retained files have not been substituted inside the bundle. Production release authenticity still depends on the platform signing/OTA trust root and on controlled qualification infrastructure.
