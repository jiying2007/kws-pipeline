# R1 Binary64 cross-architecture replay

## Purpose and scope

This research lane verifies that the fixed R1-DEV-FSMN Binary64 C reference behaves consistently on x86-64 and ARMv7-A. It does not change the trained checkpoint, default FP32 runtime, old2599 port, wake thresholds, or model-selection rules.

The build workflow is manual (workflow_dispatch) and source-only. It creates the same capture driver for x86-64 and Cortex-A32/NEON-VFPv4/hard-float. The workflow never loads private weights or audio. The owner runs the resulting executables locally with the retained model package and source PCM.

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

## Reproduce

1. Recover the owner-held R1 model package and original input capsule from the Library paths recorded in the handoff.
2. Recover exact PCM locally (the script does not publish it):

~~~sh
python3 research/r1_stream_c/cross_arch_recover_inputs.py \
  --capsule /owner/checkpoint-99399e8c75808d4b4f74b15d.zip \
  --model-package /owner/KWS_R1_DEV_FSMN_20261107 \
  --output /private/r1-inputs
~~~

3. Download the source-only x86/ARM capture artifact from the manual R1 Binary64 x86 and ARM cross-architecture capture build workflow. Verify its BUILD.json hashes.
4. Run the fixed validator with a fresh output directory:

~~~sh
python3 research/r1_stream_c/cross_arch_validate.py \
  --input-manifest /private/r1-inputs/INPUT_MANIFEST.json \
  --pcm-dir /private/r1-inputs/pcm \
  --model /owner/weights/model.f32le \
  --tables /owner/weights/frontend.f32le \
  --host-binary /artifact/host/r1d_capture_x86_64 \
  --arm-binary /artifact/arm/r1d_capture_armhf_static \
  --qemu /artifact/executor/qemu-arm-static \
  --build-manifest /artifact/BUILD.json \
  --output /private/r1-cross-arch-run
~~~

The output contains per-record x86/ARM feature/logit captures and a closed CROSS_ARCH_REPLAY.json with exact hashes. Use a new output path for each run; the validator refuses to overwrite existing evidence.

If the original FP32 numeric archives and final-numeric-index.json are available, pass --numeric-root and --final-index to additionally verify retained FP32 source clocks, argmax and greedy sequences. These are separate evidence checks and do not convert the FP32 numerical gate to PASS.

## Privacy and failure handling

The build artifact and public repository contain source, compiler outputs, the QEMU user-mode executor and hashes only. They contain no raw audio, private weights, labels, or numeric matrices. The owner-held Library retains the model package and source capsule.

The initial filename-collision batch and the incomplete first ARM attempt for record028 are retained as invalid/incomplete evidence in the owner package. They are excluded from the accepted134-record result. The corrected runner uses distinct architecture-specific output paths and a fresh output directory.
