# R1 numerical continuation — 2026-11-07

This change does not alter the trained checkpoint, default FP32 port, historical2599 implementation, decoder thresholds or existing acceptance rules. It adds two explicitly selected arithmetic implementations through `numerics.py`; neither is automatically a qualified detector.

## Opt-in build

```sh
python3 -I -S -B research/r1_stream_c/numerics.py \
  --mode reference64 --output /new/reference-source
bash /new/reference-source/build.sh
```

Source generation and compilation do not load weights or execute a model. To export the existing exact owner-held model, omit `-S`, use the pinned CPython3.12.16/core runtime, and add `--model-package /owner/KWS_R1_DEV_FSMN_20261107`. The unchanged trusted exporter authenticates the model card, checkpoint and tensor identities before loading. Public source contains no private voices or weight bytes.

The reference executable is `bin/r1d_pcm --research weights/model.f32le weights/frontend.f32le input.pcm`; input remains mono16000Hz signed16 little-endian raw PCM, not WAV. Output remains an explicitly uncalibrated greedy proposal, not a calibrated wake event. The binary64 API, backend symbols and state magic values are distinct from FP32. A build emits `NOT_RUN_FOR_THIS_BUILD`, not a borrowed PASS from another binary.

## Fixed wider-reduction candidate: useful, but rejected for the existing gate

`--mode wide32` uses binary64 FFT scratch and affine/memory accumulation, then retains FP32 activation/storage boundaries. The independently registered implementation was evaluated on the same134 retained recordings, without model training or label changes.

Full PCM logit RMSE falls from2.9179516721e-5 to2.0007211436e-5 (31.43% reduction), and elements exceeding the unchanged atol1e-5/rtol1e-5 rule fall from260255 to189815. Maximum logit difference is0.0069694519 and maximum posterior difference5.3450423276e-5. The feature maximum becomes0.0140054226, worse than the original0.013548255. Network-only RMSE against the double upstream oracle drops by69.64%, but the original strict and engineering gates still fail. All observed argmax and greedy sequences remain unchanged. This is not a passed FP32 runtime and is not made the default.

Plan SHA256:4558a1eabe3a793334655ab59863b4c9ad329cb5f36b7bf05ff65f9795a6fc71.
Implementation SHA256:ab7c2b0f250a28b3c371645852eb2f39bdbac309328a80f156228796c272933f.

## Separate all-binary64 reference: actual conformance PASS

Before executing this second experiment, its mathematical contract and tighter reference bounds were frozen separately. All PCM preprocessing, FFT, magnitude, mel/log, CMVN, network activations/cache/reductions and logits use binary64. Original stored FP32 weights and frontend table bytes are losslessly widened, not retrained. Preemphasis uses binary64 `0.97`; the original2^-23 log floor and frozen table values are explicit. This is a different arithmetic contract, not a retroactive widening of the old FP32 tolerance.

The independent oracle uses PyTorch binary64 FFT/vectorized frontend and the unmodified upstream model converted to double. It does not call the C implementation to create its expected outputs. Limits frozen before execution: atol1e-8/rtol1e-8 plus max-feature1e-8, max-logit1e-6, max-posterior1e-8, matching argmax/greedy, source clocks and chunk invariance.

| Measurement over all134 records /12534 output frames | Actual |
| --- | ---: |
| Maximum feature difference | 2.7303936889e-11 |
| Maximum logit difference | 2.5586643915e-11 |
| Maximum posterior difference | 1.1590728377e-13 |
| Out-of-bound features /5013600 | 0 |
| Out-of-bound logits /32600934 | 0 |
| Argmax, greedy, dependency-clock and selected chunk checks | all pass |

All observed argmax and collapsed sequences also match the original FP32 reference. However, comparing binary64 values to the original FP32 tensors still fails the old rule on219379 elements, maximum0.0041192316. The old FP32 gate is therefore explicitly **FAIL_UNCHANGED**, not replaced by this PASS.

The new reference has249224 bytes of x86 state and the same3028860-byte weight payload; this is not a process-RSS or SSC305 performance claim. Main reference execution constructed one independent double oracle model, ran134 oracle forwards and138 C streams (134 inputs plus four predetermined chunk variants). No optimizer/TTS/ASR calls occurred.

Reference plan SHA256:17657f161894e9b80203f44ff08c28d1d457c0a3fb52ecfcf3a1cc5eaede5f7f.
Evaluated implementation SHA256:0e11dbc15b59b2e188fb47ccb834c89ae8560c2417ca9b51f10eb87f0aef5cce.

## Integration and runtime validation actually performed

The final public generator additionally separates precision symbols and magic values. Its complete134-record replay plus four selected chunk variants is bit-identical to the evaluated reference C feature/logit tensors and clocks. That integration readback uses saved arrays and performs zero new Torch forwards. GCC and Clang each pass18 real-API tests with fictional PCM. Four fixed real CLI inputs pass GCC and ASan+UBSan/leak checking and agree with saved proposals; the unknown input remains unscored. Mixed-precision handles are rejected rather than misinterpreted. Thirteen source-only contracts pass normal/-O/-OO on CPython3.12.16 and are attached to the existing stream test entry; no new recurring workflow is introduced.

The wider-reduction worker was interrupted after36 complete records. Its partial attempt, completed-call ledger and explicit recovery declaration are retained. Records0..35 were reused, not rerun; the one unretained interrupted record was explicitly accounted for before the remaining work. Two mechanical reference-source compilation errors occurred before model invocation and were repaired in retained attempts. An initially corrupted inline-constant publication was replaced by the exact bounded external constant file before PR validation. These are not erased or counted as successful numeric evidence.

## What this establishes, and what it does not

There is now a numerically conformant streaming C **reference implementation** for the fixed model under an independently specified binary64 contract. It can be used to diagnose and bound later FP32/INT8 implementations instead of treating one platform's batched FP32 reduction as exact real arithmetic. It does not turn exposed weak clips into independent data, calibrate a live detector, complete multi-seed research, qualify INT8, or measure a physical SSC305. The original134 roles and all historical failures remain unchanged. No threshold sweep, new voice generation, ASR, retraining or unknown external-call replay belongs to this continuation.
