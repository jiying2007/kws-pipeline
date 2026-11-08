# R1 Binary64 cross-architecture replay

## Purpose and scope

This research lane verifies that the fixed R1-DEV-FSMN Binary64 C reference behaves consistently on x86-64 and ARMv7-A. It does not change the trained checkpoint, default FP32 runtime, old2599 port, wake thresholds, or model-selection rules.

The capture executables are built from the same source on x86-64 and Cortex-A32/NEON-VFPv4/hard-float. The build is source-only: it does not load private weights or audio. The owner runs the resulting executables locally with the retained model package and source PCM.

## Fixed numerical contract

- Features: atol=1e-8, rtol=1e-8, maximum absolute difference=1e-8.
- Logits: atol=1e-8, rtol=1e-8, maximum absolute difference=1e-6.
- Stable-softmax posterior: atol=1e-8, rtol=1e-8, maximum absolute difference=1e-8.
- Source/dependency/delivery clocks and frame argmax must match exactly.
- The collapsed greedy sequence must match exactly.
- EOF adds no synthetic samples or output frames.

These are the already-used Binary64 reference bounds. They do not replace or relax the original FP32-vs-Binary64 gate, which remains FAIL.

## Actual retained replay, 2026-11-08

The owner-held input capsule and model package were verified by exact SHA256. The 134 source WAVs were mapped to the existing training-feature-bindings.json rows by source ID and PCM hash; all134 PCM hashes, feature-frame counts and input clocks were checked before replay.

| Measurement | Result |
|---|---:|
| Retained inputs | 134 |
| Original audio duration | 380.0800625 s |
| Output frames per architecture | 12,534 |
| Source clocks equal | 134/134 |
| Frame argmax equal | 134/134 |
| Complete greedy sequence equal | 134/134 |
| Also equal to retained FP32 C argmax/sequence | 134/134 |
| Bitwise-identical x86/ARM captures | 118/134 |

The other16 captures differ only by tiny floating-point rounding and pass the fixed contract: maximum feature difference3.552713678800501e-15, maximum logit difference5.258016244624741e-13, maximum posterior difference9.547918011776346e-15.

This is ARM user-mode emulation, not a physical SSC305 run. It is not a calibrated event FAR/FRR result, INT8 qualification, multi-seed generalization result, or release approval.

## Reproduce the build

Install the repository's bounded ARM cross-build tools, then build into a new absolute directory:

~~~sh
sudo python3 -I -S -B tools/install_arm_dependencies.py
bash research/r1_stream_c/build_cross_arch.sh /tmp/r1-reference64-build-001
~~~

The script materializes the source-only Binary64 reference, builds x86-64 and ARMv7-A capture executables, verifies the ELF machine/ABI identities, copies the QEMU user-mode executor, and writes BUILD.json with SHA256/byte receipts. It refuses to overwrite an existing output directory. It does not include model weights, source audio, labels, or numeric matrices.

## Recover and replay original inputs

1. Recover the owner-held R1 model package and original input capsule from their Library paths.
2. Recover exact PCM locally (the script does not publish it):

~~~sh
python3 research/r1_stream_c/cross_arch_recover_inputs.py --capsule /owner/checkpoint-99399e8c75808d4b4f74b15d.zip --model-package /owner/KWS_R1_DEV_FSMN_20261107 --output /private/r1-inputs
~~~

3. Run the fixed streaming validator with a fresh output directory:

~~~sh
python3 research/r1_stream_c/cross_arch_validate.py --input-manifest /private/r1-inputs/INPUT_MANIFEST.json --pcm-dir /private/r1-inputs/pcm --model /owner/weights/model.f32le --tables /owner/weights/frontend.f32le --host-binary /tmp/r1-reference64-build-001/host/r1d_capture_x86_64 --arm-binary /tmp/r1-reference64-build-001/arm/r1d_capture_armhf_static --qemu /tmp/r1-reference64-build-001/executor/qemu-arm-static --build-manifest /tmp/r1-reference64-build-001/BUILD.json --output /private/r1-cross-arch-run
~~~

The default validator uses only the Python standard library and compares one frame at a time, keeping memory bounded. Optional saved-NPZ comparison (--numeric-root plus --final-index) imports NumPy lazily; it verifies retained FP32 clocks, argmax and greedy sequences. These are separate evidence checks and do not convert the FP32 numerical gate to PASS.

The output contains per-record x86/ARM capture files and a closed CROSS_ARCH_REPLAY.json with exact hashes. Use a new output path for each run; the validator refuses to overwrite existing evidence.

## Privacy and failure handling

The public repository contains source, the local builder, and the validator only. The owner-held Library retains the model package and source capsule. Raw audio, private weights, labels and numeric matrices are never included in the public build output.

The initial filename-collision batch and the incomplete first ARM attempt for record028 are retained as invalid/incomplete evidence in the owner package. They are excluded from the accepted134-record result. The corrected runner uses distinct architecture-specific output paths and a fresh output directory.
