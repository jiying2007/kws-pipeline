# Isolated Linux host KWS research

This optional source recipe reconstructs the tested B host candidate: a Sherpa
KWS-root whole-translation-unit subset, 14-function C API, fixed bounded-Viterbi
decoder policy and a file-based CLI. It does not enter the default C11 product
build, model registry or the sibling `sherpa_pcm` adapter. It is an engineering
reference, with no improved acoustic-quality, latency or SSC305 qualification.

The repository contains source text, one decoder patch, exact dependency/source
locks and offline helper tests. Upstream sources, ORT, models and audio are
external inputs. See [BUILDING.md](BUILDING.md), [RESULTS.md](RESULTS.md),
[PROVENANCE.md](PROVENANCE.md) and [中文说明](README.zh-CN.md).

## Offline checks

From the repository root, with Python 3 and GCC C++17 available:

```sh
python3 research/sherpa_host/tests/run_tests.py --sanitize
```

This checks real CLI parser/serializer/session helpers against a fake C API,
launcher argument preservation, relocation and integrity failures, the 19
original wrapper spans and 14 exports, compile/source/license identities, and
negative materializer inputs. Waveforms are numeric temporary fixtures. It
does not compile the wrapper or real decoder, load ORT, run a model, or test
Viterbi/Hypothesis/context-graph behavior. Those need the explicitly locked
external source and ORT assets; ordinary CI neither fetches nor executes them.
The root six CI contexts remain separate and unchanged.

## CLI contract

After assembling an explicitly verified runtime with the documented staging
entry, keep its relative layout and execute its `run-kws`:

```sh
./run-kws --wav -- "relative directory/audio.wav"
./run-kws --pcm-s16le-16000-mono -- "-audio.pcm"
```

One invocation accepts one ordinary file, up to 1 GiB; WAV is the default.
WAV must be little-endian RIFF/WAVE PCM format 1, mono, 16 kHz, signed 16-bit,
with exactly one `fmt ` and one `data` chunk. `fmt ` is 16 bytes or 18 with zero
extension length. RIFF length must equal file length; chunks and odd-size
padding must be complete. Empty audio, odd PCM bytes, duplicate/truncated
chunks, trailing bytes, compressed/float/RF64/RIFX input, devices and pipes
are rejected. Raw PCM requires the explicit format option. No conversion,
resampling, channel mixing or automatic silence padding occurs.

Input paths stay relative to the caller's working directory. Quoted spaces,
relocated kits and `--` before dash-leading filenames are supported. Keep the
input unchanged while processing: length/truncation checks do not detect
same-length concurrent edits. Never redirect output onto the input file.

The fixed profile uses CPU, one thread, 80 features, 320-sample feeds,
max_active_paths=4, trailing_blanks=1, score=1 and threshold=0.25, with the two
research keywords in `cli/config/keywords.txt`. A short final feed is unpadded;
one InputFinished call drains results. Each event is emitted before resetting
the stream. Result, stream and spotter ownership follows the C API contract.

stdout contains NDJSON event rows and one final successful complete row.
Events include keyword, tokens, timestamps, start_time, available_samples and
eof. Completion includes input_bytes, audio_samples, feed_calls, events and eof.
Time values retain the API values; available_samples records supplied audio,
not acoustic latency. Invalid UTF-8, nonfinite values, oversized results and
output failures produce nonzero exit status; earlier output may be partial.
Always check the status and completion row. Errors go to stderr.

The launcher checks all entries in its generated checksum manifest and sets a
minimal single-thread environment before executing the CLI. It is an integrity
check against accidental mixing, not a signature or hostile-loader sandbox.
Changing models, configuration, source or binaries requires new identities and
validation, not just regenerating a green checksum file.

## Platform and ABI limits

The tested binaries are Linux x86-64. Their dependencies require, among other
versions, GLIBC_2.38, GLIBCXX_3.4.32 and CXXABI_1.3.15. The full dependency and
symbol sets matter; these maxima alone are insufficient. B and ORT have no
declared minimum ISA, so their CPU floor is unknown. CLI baseline metadata
does not establish a baseline for the whole runtime. GNU coreutils and
`/proc/self/exe` are required. This is not a universal Linux/Windows binary.

The subset SONAME is `libsherpa-onnx-kws-c-api.so`; only the 14 exports in
`cmake/kws-exports.lds` exist. The shared full upstream header declares more
APIs. Never exchange opaque handles with the full Sherpa C API library. Model
factories and incidental sentencepiece/FasterDecoder/FST/Eigen paths remain.
SSC305 Cortex-A32 requires its real SDK, sysroot, loader and board acceptance.
