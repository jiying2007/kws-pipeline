# FSMN2599 B continuous C research adapter

A bounded-state PCM16LE transport adapter for the exact registered B model
identity. It composes the canonical frontend, unchanged four-lane FSMN2599 C
network, research C softmax, and existing threshold-aware C decoder.

**Research only. The frontend/network reference numerical gate at 1e-5 remains
FAIL. The softmax has its own research contract. OR composition is not implemented.
Production claims and qualification are false.** Loading a matching B payload
is supported; trained weights and recordings are not distributed here.

## What is reusable

`b_stream_adapter.h` exposes create/begin/feed/finish/reset/get/destroy. Feed accepts
any byte chunk, including odd boundaries and zero bytes. It does not assume a
fixed number of recordings, a known recording duration, or 640-byte calls.
Synchronous events can be consumed immediately during feed: an always-on stream
need not finish to deliver a wakeup. Every row reports its generation, validity
so far, source requirement, transport availability, and synthetic-EOF flag.

The threshold is explicit and immutable per handle. Historical real-input
transport checks used 0.7; this is a research test configuration, not a recommended
product threshold. Model identity is registered and checked, not caller-invented.

- [API and lifecycle](API-CONTRACT.md)
- [Anonymous historical observations](RESULTS.json)
- [Public source bindings](source-binding.json)
- [Attribution](NOTICE.md)

## Build and bounded artificial fixtures

From the repository root, after execution is authorized:

    python3 -S -B tests/test_research_fsmn_stream.py --execute

The entrypoint uses a temporary build directory, strict floating-point flags,
source hashes, a production symbol audit, and one isolated eight-group synthetic
fixture run. It does not load a production model or read recordings. The real
PCM transport is tested against mock acoustic/decoder components. Mock behavior
is not an acoustic oracle. The existing threshold decoder and FFT constants are
referenced in their original repository paths rather than copied.

The current corrected public-layout revision passed three compiles, the exact 36-symbol check, and
one eight-group mock run through build.py directly. The test-entry wrapper above
and hosted CI have not run. CI wiring belongs to the repository's existing
hosted-build job; this module adds no workflow or model-download job.

The declared candidate validation budget is three serial compilers sharing 60s
wall/CPU, then one fixture process limited to 30s wall/CPU, each with 512MiB address
space and bounded output. No retry, threshold grid, training, Torch, or real model
inference belongs to this public synthetic check. Only the already-verified GCC
configuration is proposed here; no new sanitizer or Clang coverage is claimed.

## Limits and interpretation

A prior bounded same-C two-lane diagnostic compared fixed-size and arbitrary-byte
transport on the same three existing recordings. Posterior byte-stream hashes
and event structure/binary64 scores agreed. Transport delivery endpoints were
checked separately and can differ between chunkings. This says nothing about
reference-framework agreement or acoustic improvements.

The API requires the documented IEEE floating-point environment and masks/restores
exception traps per numerical call. Unsupported modes fail; compiler flags are
part of the contract. The decoder's `numerical_support_ok` means only that no
zero-mass or empty-beam window rows occurred; it is not a reference-parity gate.

No independent FAR/FRR hours, target-board timing/memory, or long-running always-on
qualification is established. EOF events use a fixed synthetic tail and must not
be merged into claims about real-input wakeups. The private diagnostic runner,
per-recording identities, recordings, payload, posterior arrays and native
binaries are deliberately absent.

## Load your exact private B payload and raw PCM

The source distribution contains model identity metadata only. Obtain the matching
research B payload through your authorized private artifact delivery; there is no
public download or automatic fetch. The provided private manifest must name
checkpoint SHA256 `a366d9f5ebe7027a496e3f119debcf1e19d2f7a79226c96b31a4ddb3a2757f77`
and payload SHA256 `2175df1a7f8621a6b6566a9729b63f021c007d93f2c8c17cb54e7c0f496bbfad`.
The payload length is 3,027,732 bytes. Check it with `sha256sum`; `bsa_create` also
checks the registered payload hash before inference. Do not rename another model
or substitute a shape-compatible file. The private bundle includes its original
manifest and model_identity.h; it does not contain recordings.

`example_pcm.c` accepts only user-supplied filenames, recording ID and threshold.
Build from the repository root with the same fixed arithmetic flags:

    m=research/fsmn2599-stream-c
    d=research/fsmn2599-three-window-c/kws-fsmn2599-three-window-c-preparation-20261018
    tmp=$(mktemp -d)
    mkdir -p "$tmp/include/vendor"
    ln -s "$PWD/research/donor_fbank/fft_twiddles.h" "$tmp/include/vendor/fft_twiddles.h"
    gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fno-fast-math -ffp-contract=off -fexcess-precision=standard \
      -I"$m" -I"$d" -I"$tmp/include" "$m/example_pcm.c" "$m/b_stream_adapter.c" \
      "$m/canonical_pcm.c" "$m/canonical_fbank.c" "$m/fsmn2599.c" "$m/sha256.c" \
      "$m/softmax2599.c" "$d/kws3.c" -lm -o example_pcm
    rm -r "$tmp"
    ./example_pcm /your/model.f32le /your/mono16k-signed16le.pcm my-recording 0.7

Those paths are placeholders supplied by you. Input must be raw PCM, not a WAV
container; the example performs no conversion. It prints events immediately with
source and availability clocks, hex binary64 scores and EOF flags. True file EOF
adds the documented synthetic tail; an odd final byte is rejected. Empty input
still exercises that tail. Errors stop without retry. The current corrected public-layout revision passed three compiles (library, mock fixture,
and this CLI), the exact 36-symbol check, and all eight mock groups. The CLI
was only compiled, not executed; the test-entry wrapper and hosted CI remain
unrun. The earlier real B two-transport result used the original bound runtime,
not these newly built binaries. No SSC305 execution or production qualification
is established.

## Current contract correction status
The current source separates decoder validity from support diagnostics as
described in API-CONTRACT.md. Its small source/example change passed the fixed three-compile, 36-symbol
and eight-group mock validation, including the injected support-diagnostic case. The summary counters are retained and the PCM example
prints a warning without rejecting a valid recording. No real negative-flow
failure has been observed or established by this contract correction.

## Current single-DSO same26 diagnostic

The corrected single-DSO runtime was subsequently exercised once on the same
26 already-exposed development recordings at fixed thresholds0.7 and0.1, sharing
one backbone/posterior pass with two independent decoders. The independently
reviewed anonymous evidence is [SAME26-RESULTS.json](SAME26-RESULTS.json). All26
recordings were retained, with no support warnings.

At0.7 the full/strict-live result is8/11 target hits with zero observed false
detections; at0.1 full EOF gives11/11 with keyword false detections1/2, while
strict-live gives10/11. Compared with the saved old reference there are no gained
or lost per-recording detections in these fixed modes, and all22 event structures
and source grids match. Scores are not bit-exact (maximum absolute differences
approximately6.74e-6 and1.27e-5); the existing1e-5 numerical FAIL remains. Old
body scoring uses an analytic source clock without EOF-origin labels, whereas
new strict-live additionally excludes EOF-derived events. These exposed-sample
observations do not establish independent accuracy, FAR, generalization, SSC305
performance or production readiness, and do not recommend either threshold.

## Official pretrained lineage

B is a head-adapted version of the official pretrained
[WeKws FSMN CTC model](https://modelscope.cn/models/thuduj12/kws_wenwen_fsmn_ctc),
not a model trained from scratch. The original `avg_30.pt` is pinned to revision
`cd0c18d0dda6742c371fbb72ec66185906a9885d` and SHA256
`d3d89b5dde08c2926d46a97eab6f7179a87b1946ccebd16c02e77042ea8389fe`.
The [fixed upstream implementation](https://github.com/wenet-e2e/wekws/blob/6a45aeb994dd81c0969ff877a5a7c46d60ed0c86/wekws/model/fsmn.py)
and original attribution are retained in NOTICE.md.

The model has756,133 parameters. Its800 CMVN buffer values are not parameters;
the FP32 payload therefore has756,933 elements. B used220 head-only optimization
steps on the existing offline feature cache, changing only
`backbone.out_linear2.linear.weight` and `.bias` (366,459 parameters). The other28
of30 exported tensors remained byte-identical to the donor, and the2599-class
alphabet was retained. This lineage is supported by the existing donor/run tensor
identity records and original export manifest; no duplicate model manifest or
new model was created for this delivery. The B checkpoint/payload hashes above
remain the loading identity. Pretrained provenance does not remove the current
C reference numerical FAIL or imply production qualification.
