# Capture contract and build hardening

This additive change preserves the B ABI, all 36 DSO exports, payload identity,
frontend/network arithmetic, softmax, decoder and the existing numerical FAIL.
It is not a new model or a production/SSC305 admission.

## Reusable capture integration

Include `bsa_capture.h`. On a capture-reported XRUN/route change/clock reset:

1. Stop feeding the affected generation. Capture code must detect sequence loss;
   this helper cannot infer device loss from raw PCM bytes.
2. Discard a partial PCM16 sample and regain a confirmed PCM16LE sample boundary.
3. Set a `bsa_capture_anchor` from actual acquisition metadata. Keep unknown loss
   unknown; use a new clock epoch when clocks become incomparable.
4. Call `bsa_capture_reset(handle, id, callback, user, &clock, &anchor, reason)`.
   It invokes the real reset, checks status, and binds coordinates to generation.
   Resume only on success. Do not finish or add silence at the gap.
5. Map real relative endpoints with `bsa_capture_map_real`. It rejects stale
   generations and overflow. For EOF rows, only available_real_sample denotes a
   real endpoint; required_padded_sample may belong to synthetic tail.

The helper is header-only and does not add DSO symbols. It is capture integration
code, not a hardware driver. It has no autonomous XRUN detection, byte framing,
clock synchronisation, rate estimation or wall-clock alert-latency measurement.
Application code must copy callback data it needs to retain and honour the
existing single-threaded/non-reentrant contract.

## Host checks

`python3 -S -B tests/test_research_fsmn_stream.py --execute`

The bounded driver now performs five compiles/links within the existing combined
60-second CPU/wall compile budget: complete C DSO, original mock fixture, original
C example, capture mock fixture and a C++ link-only client. Original eight mock
groups and new two capture groups each execute once; production binaries, the
C example and C++ client are never executed. No payload is read. The new capture
fixture calls the reusable helper and verifies XRUN/reset with a pending odd byte,
no synthetic gap tail, fresh clock epoch, unknown sample loss, stale generation,
integer overflow, repeated finish and continuous arbitrary-byte chunk invariance.
These short synthetic checks are not a soak or acoustic test.

## Full B ARM cross-build CI integration

`python3 -S -B tests/run_research_fsmn_stream_arm_build.py --execute`

Run in the existing Linux ARM cross-toolchain job after installing official Ubuntu
packages `gcc-arm-linux-gnueabihf`, `g++-arm-linux-gnueabihf` and
`binutils-arm-linux-gnueabihf`. The script fails if tools are missing. It compiles
and links frontend, PCM transport, FSMN, SHA256, softmax, decoder and adapter into
one ARM hard-float DSO, checks all 36 symbols and links a C++ client. It targets
cortex-a32/neon-vfpv4/hard, with strict no-fast-math/no-contraction flags and the
same bounded source verification. It does not invoke QEMU or execute a model.
This is complementary to existing ARM decoder smoke, not a replacement.

Suggested existing-job step:

    - name: Full B ARM C and C++ link checks
      run: python3 -S -B tests/run_research_fsmn_stream_arm_build.py --execute

Cross-build success is not proof of SSC305 runtime, libm/fenv compatibility,
realtime speed, memory, audio capture correctness or FAR. Those need separate
board and acoustic evidence. Do not relabel this check as full-chain ARM parity.
