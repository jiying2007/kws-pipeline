# Native C API English Hey Mero original-audio research replay

Research-only follow-up to English C event mapping ID3 and real 63-Qwen
unpadded streaming events. The optional C ABI adapter remains a distinct
experiment from the default Mandarin runtime. No SSC305 BSP or real board
is available here.

## Frozen source and runtime

- Original English63 Qwen WAVs ZIP SHA256:
  f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc
- Original three-evidence joined JSON SHA256:
  89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12
- Official Zh-En Zipformer 3M model SHA256:
  68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6
- Previously observed **unpadded** 63-original-PCM event artifact SHA256:
  4f006138eb9dcc6f856ef0e61da8dcf10034904e4e683255db302775a19a62f6
  and original JSON SHA256:
  d5cfbd295299ff8c0ee7524538e44c9755ad25a943f5178455816bc691dbe521
- Exact official Sherpa-ONNX v1.13.8 x86_64 shared release archive
  28,156,791 bytes, SHA256
  c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4.
  Compiled C adapter validates separately pinned existing host C API
  and ONNX Runtime binary hashes via dependencies.lock.json.
- No model adaptation, backend change, threshold change, new ASR/TTS,
  source regeneration or artificially appended silence.

## Execution and comparison

Compile the existing C/C++ PCM adapter with the actual official runtime
library, using research/sherpa_pcm/build.py --library (not --stub).
Use the **INT8 chunk16 official encoder + joiner and FP32 decoder**,
with original official HEY MERO en.phone mapping and alias HEY_MERO.

The C ABI takes signed 16 kHz mono PCM16 and returns event string/ID3,
input-available sample count, and EOF flag. Compare real playback under
three caller input patterns: fixed 320, fixed 1600 and irregular
1/159/321/17/640 samples. For each original clip, create a fresh
logical stream, feed only original PCM, and call finish without adding
zeros. Compare count per original source cell against historical Python
unpadded data: five events/63 original clips on the same published INT8
model. Differences, including earlier detection times or missing events,
must be reported, not hidden or tuned away.

The exact build source, dynamic library content digests and per-cell
outcomes are written to a create-only machine receipt. The model and
recorded audio are NOT uploaded or copied into this Git tree.

## Qualification limits

Even exact host C/Python agreement is *not* actual Cortex-A32/SigmaStar
SSC305 compatibility. The inspected generic Linux ARM32 runtime requires
GLIBC_2.34, GLIBCXX_3.4.29 and CXXABI_1.3.13; the real SSC305 sysroot,
compiler, kernel, loader and available RAM/CPU budgets remain unknown.
Official generic runtime also includes third-party components such as
eSpeak; licensing for any future distribution requires its own review.

More importantly, five wake events from 14 requested synthesized Hey
Mero prompts are not genuine human FRR or complete keyword recall, while
61 minutes of segmented LibriSpeech with no false events is not household
continuous FAR/hour. End-of-word latency, accents, 3–5m microphones,
continuous environmental speech and real target power/resource budgets
remain NOT_RUN. Do not promote a model based on C ABI parity alone.
