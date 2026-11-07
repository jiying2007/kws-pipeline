# R1-DEV FSMN C streaming research port

This is a distinct2601-class port of the owner-held R1-DEV-FSMN-20261107 checkpoint. It is an executable research prototype, **not a numerically qualified or calibrated wake detector**. The historical2599 B model, payload, sources and numerical FAIL are unchanged.

## Build and export

The repository contains source and fixed identities only. No raw voices or trained weights are published here.

```sh
python3 -I -S -B research/r1_stream_c/port.py --output /new/source-only-directory
bash /new/source-only-directory/build.sh
```

The source-only build can compile without loading any model. To export the exact owner-held R1 package, use the existing pinned core environment (CPython3.12.16/Torch2.13.0+cpu) and a fresh output:

```sh
python3 -I -B research/r1_stream_c/port.py \
  --model-package /owner/KWS_R1_DEV_FSMN_20261107 \
  --output /new/r1-stream
bash /new/r1-stream/build.sh
/new/r1-stream/bin/r1_pcm --research \
  /new/r1-stream/weights/model.f32le \
  /new/r1-stream/weights/frontend.f32le /owner/input.pcm
```

Input is raw mono16000Hz signed16 little-endian PCM, **not a WAV container**. `-` accepts stdin. The compiled executable needs no Python, Torch, ASR or TTS. The output is explicitly `uncalibrated_greedy_proposal`: collapsed token suffix matching,150-frame span and50-frame per-word cooldown, no threshold search. It must not be presented as an independently calibrated event detector.

The fixed checkpoint hash is `99ac2239807ce5a93c8723a01df2e53c446f868245caccb2e7809deb6e943571`. Export produces3028860-byte float weights, SHA256 `e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb`, and83840-byte frontend tables, SHA256 `49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49`. Both are authenticated before C inference. An exact externally pinned model card is required before Python source imports during export.

## Streaming semantics

Reuse the existing network's delayed memory equations; do not add another layer of future-frame delay. The frontend retains400-sample windows/160-sample hops, five-frame context, skip3 and edge replication. Output row m requires exactly720+480*m real input samples. This is an input-dependency clock, **not an annotated word end or measured detection latency**. Delivery time additionally includes transport chunking.

`r1_create/feed/finish/reset/destroy` uses bounded state and accepts arbitrary byte boundaries. EOF adds **zero** samples/rows; the old B adapter's4800-sample synthetic tail is not used. Odd-byte EOF is rejected. Callbacks are synchronous; the handle is single-threaded/non-reentrant and must not be mutated or destroyed inside a callback. Feature-only differential testing and PCM mode cannot be mixed without reset. Errors preserve the original finite-state contracts. No change to default deployed runtime or the old B ABI is made.

## Actual execution,2026-11-07

The complete134 retained records (96 train,34 exposed weak development,4 unknown) were replayed against their saved R1 feature/logit tensors, without retraining or rerunning the old C1. One Python model construction performed142 feature streams including the fixed chunk variants. C performed272 full PCM streams plus134 feature-only streams. Eight later native CLI streams (four GCC and four ASan+UBSan cases) brought total real full C streams to the predeclared280 maximum.

- All12534 unique output-frame argmax classes and every collapsed sequence match the original reference.
- All134 recordings have bit-identical C features/logits across640-byte and irregular chunking; four fixed records also pass one-byte chunks. Source clocks are valid and EOF adds no synthetic rows.
- A saved-array audit independently establishes each observed frame's argmax margin exceeds twice its maximum C logit perturbation. This is a finite-record argmax result, not a bound on arbitrary future audio, probabilities, thresholds or FAR.
- On34 exposed weak development clips, the fixed CLI proposal rule retains K1/K2 hits10/10 each and zero proposal-positive clips among each word's24 known weak negatives. Unknown rows are not counted as negatives. No natural negative hours or independent accuracy are created.
- GCC and Clang both compile;18 real C API tests with fictional PCM pass for each. Four real CLI cases pass both GCC and ASan+UBSan (leak detection enabled), matching proposals computed from saved C outputs.
- Regenerating source/weights from the pinned source and owner package reproduces both GCC binaries byte-for-byte. This is a host/toolchain reproduction, not cross-platform equivalence.
- Observed x86 state object125160bytes; separate weights3028860bytes. RTF0.024799085 was measured over380.0800625s of these recordings, including Python capture overhead. It is not SSC305 performance or a worst-case real-time guarantee.

## Numerical gate remains FAIL

Frozen reference rule is `abs(actual-reference) <= 1e-5 + 1e-5*abs(reference)`. Full C logits fail on all134 records:260255/32600934 elements exceed that bound. Network-only C on saved Torch features also fails (105338 elements), so frontend drift is not the only issue. Python framewise processing itself has different floating reduction than the saved whole-sequence Torch path.

Measured maxima:

| Scope | Logit absolute difference | Posterior absolute difference |
| --- | ---: | ---: |
| Python framewise vs saved whole | 0.0007019043 | 0.0000411836 |
| C network on saved features | 0.0007934570 | 0.0000667004 |
| C full PCM pipeline | 0.0092086792 | 0.0000649097 |

Frontend maximum absolute difference is0.013548255. The separate preregistered engineering gates (logit2e-4,posterior1e-5,feature2e-4) also fail. No tolerance was widened after observing these results. Hard-decision agreement is reported alongside, not substituted for, numerical failure. C/int8 qualification, calibrated live events, multi-seed quality, independent human/negative-stream performance and physical SSC305 tests remain incomplete.

## Tests and retained evidence

Seven new source-only contracts are appended to the existing `tests/test_research_fsmn_stream.py --execute` entry; both historical tests remain unchanged. No new recurring workflow, dataset catalog or model job is added. The seven new tests passed locally in canonical3.12.16. Attempting the old two-test portion in the partial offline source snapshot failed because its sanitized child lacked the relocated libpython and the snapshot omitted old auxiliary sources; those failures are retained. Complete-checkout hosted CI must verify the integrated entry and is not inferred from local partial testing.

Private delivery keeps the immutable preregistration,134 numerical outputs,per-record comparison,independent saved-array audit,CLI/sanitizer logs and actual execution counters. It contains no new voice synthesis or ASR observations. Rights and attribution of original code and weights remain unchanged. Use this port for further research; do not replace an independently qualified detector based solely on these exposed clip results.
