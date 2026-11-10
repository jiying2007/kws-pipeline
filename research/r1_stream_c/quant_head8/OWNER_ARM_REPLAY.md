# Owner-local ARM execution of the pretrained R1 head-INT8 models

This is a **source-only** opt-in procedure. It does not download, publish, or
embed the original checkpoint, quantized checkpoint, natural recordings, private
PCM source binding, voices, logits, or ASR transcripts. Run it on an authorized
host with all original assets already stored outside the public checkout.

## Purpose and exact inputs

- Frozen strong FSMN FP32 model SHA256:
  `e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb`.
- Frozen 80-band frontend SHA256:
  `49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49`.
- Owner-only original PCM index `INPUT_MANIFEST.json` SHA256:
  `63d2f75e5e93e454bdf23624277fa728bba0cef6eee9cce311f037612198d6ab`.
- Pinned original public **source** ZIP:
  `research/r1_stream_c/deploy/archive-20261110/KWS_FSMN_R1_SOURCE_20261110.zip`.
  The existing qhead8 source materializer verifies its ZIP SHA and all original
  source identities; it packs the known owner checkpoint in a fresh private tree.

The runner builds *four* real native libraries/executables: x86 host FP32
reference64, ARM FP32 reference64, ARM head8 scalar, ARM head8 par4. The same
C frame/event observer is linked to all four builds. For each checked PCM input,
it compares **every argmax frame ID**, frame source/dependency and delivery clocks,
and all complete-keyword event identities. This is actual model inference, not a
source compilation smoke test. A disagreement stops the run; partial receipts
and traces are preserved. No tolerance is silently relaxed.

## Command on an ARM/QEMU-capable, authorized Linux host

```bash
python3 -B research/r1_stream_c/quant_head8/owner_arm_replay.py \
  --repo . \
  --reference-sdk /private/KWS_FSMN_R1_SDK_20261110 \
  --model /private/KWS_FSMN_R1_SDK_20261110/weights/model.f32le \
  --frontend /private/KWS_FSMN_R1_SDK_20261110/weights/frontend.f32le \
  --pcm-dir /private/KWS_FSMN_R1_EVIDENCE_20261110/inputs/pcm \
  --manifest /private/KWS_FSMN_R1_EVIDENCE_20261110/inputs/INPUT_MANIFEST.json \
  --arm-gcc /usr/bin/arm-linux-gnueabihf-gcc \
  --readelf /usr/bin/readelf \
  --emulator /usr/bin/qemu-arm-static \
  --sysroot /usr/arm-linux-gnueabihf \
  --output /private/kws-r1-arm-smoke-20261110 \
  --indices 0,100,133 --timeout 180
```

Only after the bounded three-row smoke passes, use **another fresh private
output directory** and `--indices all` for the 134-record exact PCM corpus.
Compiler/target/ABI checks run before inference. The emulator and cross compiler
must be installed by the owner using an approved toolchain, not downloaded or
installed by this script. The real SSC305 firmware might require a different
sysroot/ABI; passing generic ARMv7 QEMU does not qualify that firmware.

The report and individual frame traces stay under `--output` with owner-only
permissions. Never commit this private directory or place it in a Git checkout.
The output status `SOURCE_ARCH_BOUND_REPLAY_PASS_NOT_BOARD_QUALIFICATION` only
means these exact saved inputs matched their host reference. It is **not** a
continuous real-negative FAR/hour, a new held-out real-human accuracy result,
a keyword-end latency, a direct Torch-MKL-vs-FP32 repair, or a measurement of
physical SSC305 CPU/RSS/power/thermal/soak. The original FP32/Torch numerical
failure remains `FAIL_UNCHANGED` and full logits tolerance remains unmeasured.

## Automated CI scope

`test_owner_arm.py` uses **invented fixtures only**. It tests index rejection,
malformed frame/event traces, hash fail-closed behavior, ELF target validation,
and compilation of the observer using the pinned public header under GCC/Clang.
The repository CI MUST NOT run the real owner-local script because required
model weights and audio are intentionally absent from public GitHub.

No new GitHub Actions workflow, recurring monitor, ASR/TTS job, training run,
private repository, or Codex/ChatGPT Work quota is required.

## Bounded subprocess recovery (2026-11-10)

The owner replay driver now reads stdout and stderr concurrently in bounded
chunks, rejecting either stream at **4 MiB** before accumulating more bytes.
This fixes the previous `subprocess.run(capture_output=True)` behavior, which
buffered unbounded output before checking its size. On a deadline or output
limit, it kills the isolated child process group and retains the failing
owner-local attempt; normal nonzero exit codes and stderr are not suppressed.
A quiet hung runner and both-pipe backpressure are covered by invented-only
tests. Mandatory build artifacts are checked explicitly even under `python -O`.

No model, TTS, ASR, original PCM or owner checkpoint runs in this change. This
does not change inference mathematics, observed ARM parity, model ranking,
natural FAR/FRR, or actual SSC305 qualification.
