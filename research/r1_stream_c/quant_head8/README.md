# R1 strong FSMN optional per-row INT8 head + exact critical rows

**Research-only, opt-in.** The frozen baseline `R1-DEV-FSMN-20261107` remains unchanged. The original FP32-vs-Torch/MKL gate is FAIL and is not superseded by this independent quantization route. The R1 reference64 SDK remains the control.

The 140→2601 final classifier consumes 364,140 float32 parameters. The initial per-row symmetric `int8` weights use a double-precision scale per output row and preserve five full FP32 rows (blank and the four distinct complete-keyword tokens). The remaining prefix and biases remain float32. Frozen package size: 1,960,140 bytes versus 3,028,860 original (**1,068,720 bytes lower**, 35.28% model payload decrease). This is a weight-only hybrid; not a full-network INT8 or NEON-optimized integer GEMM.

The packer reads the existing **exact owner-held model** and rejects another source hash. The source materializer first verifies and reads the source-only archive already present in the public repository (no external download). It validates exact retained standalone SDK source identities. It produces a distinct SDK tree; never mutates the baseline SDK, the original checkpoint, or frozen historical observations. Only source code and aggregate result metadata may enter public GitHub; do not publish the quantized weight payload or raw speech.

```
python3 research/r1_stream_c/quant_head8/materialize.py \
  --repo . --output /fresh/quant-sdk --kernel par4 \
  --model /private/model.f32le --frontend /private/frontend.f32le
cmake -S /fresh/quant-sdk -B /fresh/quant-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build /fresh/quant-build --parallel 2
/fresh/quant-build/fsmn_run --research /fresh/quant-sdk/weights/model.r1hq8 /fresh/quant-sdk/weights/frontend.f32le --pcm /private/clip.pcm
```

The default `--kernel scalar` is the experimental numerical control, and `--kernel par4` is the four-lane accumulator challenger. The same quantized bytes are used for both; the reduction order is different. The retained x86 observations show matching frame argmax and event clocks for all 134 observed inputs but do not establish equivalence for all future audio.

Missing model weights: omit `--model/--frontend` to compile source-only mock contracts; this does not prove inference. The core asserts the compact package hash and source model identity. It refuses unknown payloads and does not attempt to quantize or download during model inference. No float32 head is reconstructed in runtime memory. It retains all original 2601 competitors and all original streaming invariants.

Comparison protocol: on **exactly 134 saved PCM recordings** (100 train-role, 34 exposed weak-label development), compare every frame argmax, every complete-word proposal and timing. Never multiply independent sample counts with replay runs. Profile wall clock and model-weight bytes separately; report host-only effects without attributing to SSC305. Before considering promotion, require independent real-human and naturally continuous negative streams and a target-board profile. The repo CI only builds the source-only variant and exercises mocked transport; it neither calls the model nor measures its real-resource effect.

### Same-host observations (2026-11-10; **not target hardware**)

The owner-held 134 PCM records yield 12,534 model frames and 88 greedy complete-word proposals per variant. The FP32 reference, per-row INT8 scalar and per-row INT8 par4 have exactly identical **12,534 argmax IDs, source-dependency clocks and all event records**. The three independently generated private frame-trace binaries have identical SHA256 `3d115b436bba73d62da1dbd94be9a10e63b3e22dd7e7248745c8a0d54899756a`.

All 402 real PCM CLI invocations were performed as separate low-RSS processes, with variant order rotated among recordings. Median peak RSS: FP32 6912 KiB, INT8 4864 KiB (2048 KiB / 29.6% decrease on this host). Sum of in-run compute-and-I/O durations: FP32 8.499824 s, INT8 scalar 8.877988 s (+4.45%), INT8 par4 7.865590 s (-7.46%). Timing includes file I/O but excludes model creation; it is a single sample of this host, not a guaranteed speedup or ARM/SSC305 evidence. Independent FAR/FRR and the historical FP32 Torch/MKL numeric gate remain unqualified/failed, respectively. Record all numerical failures, including changed competitor logits, even when argmax is stable.

## Source and rights

The software incorporates the byte-identical, attributed Apache-2.0 / torchaudio source retained in the original SDK. The source license does not grant public model or original audio redistribution. Packing the owner-held checkpoint is an authorized noncommercial research transformation, not a new rights grant.
