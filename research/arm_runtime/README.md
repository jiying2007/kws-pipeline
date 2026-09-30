# ARM32 runtime audit and BSP preflight

[简体中文报告](report/ARM32_RUNTIME_REVIEW.zh-CN.md)

This isolated research folder preserves a **static inspection**, read-only probes,
and an **unexecuted** KWS-oriented build recipe. It changes neither the product
runtime nor the existing sherpa PCM adapter. No model, runtime binary, toolchain,
audio, ASR result, or target-board measurement is included.

## Result and blockers

The official sherpa-onnx 1.13.8 ARM32 shared release was downloaded and its
32,556,415-byte archive matched SHA256
`832c200eb361e361c587812f116436760e7728d2ee93ea4759ad5d59ab5c2e6c`.
The corresponding source is `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`.
Both selected libraries declare ELF32 little-endian ARM EABI5 hard-float,
ARMv7/VFPv3/NEONv1 and VFP register arguments. ELF attributes are not execution tests.

- Both libraries require GLIBC_2.34 and GLIBCXX_3.4.29; the C API also requires
  CXXABI_1.3.13. SSC305's actual compiler, sysroot, loader and libc remain unknown.
  Generic Ubuntu ARM CI does not establish compatibility with that BSP
- The C API's defined local symbols include `espeak_Initialize`, `espeak_Synth`
  and related implementation functions. The pinned eSpeak source has GPLv3
  COPYING; upstream builds TTS by default and statically includes this component
- Removing TTS CLI files does not remove statically linked code. Do not describe
  this generic bundle as an Apache-only KWS SDK. GPL permits commercial use;
  applicable obligations depend on the intended distribution and integration.
  This report does not provide a final legal determination
- No ARM adapter compilation, linking, loading, emulation or execution occurred.
  No new acoustic experiment occurred. The next build input is the actual SSC305
  SDK/BSP and C/C++ compiler/sysroot, not a substitute generic toolchain

## Footprint and dependency boundary

`libsherpa-onnx-c-api.so` is 5,101,612 bytes and `libonnxruntime.so` is
20,617,061 bytes: **25,718,673 bytes** total. Adding the previously pinned four
model/token files (5,737,545 bytes) and keyword text (75 bytes) gives
31,456,293 bytes. This arithmetic excludes the unbuilt ARM adapter, standard
libraries, license/notice files, packaging and filesystem overhead; it is not
RSS, a completed SDK size, or a stripped KWS-only size.

The inspected direct DT_NEEDED closure needs the C API, ORT, `libm.so.6`,
`libstdc++.so.6`, `libgcc_s.so.1`, `libc.so.6` and `ld-linux-armhf.so.3`.
It does not require the bundled CLI programs, ALSA, cargs or C++ wrapper.
This does not prove the absence of static components or possible dynamic loading.

The upstream-selected maintainer ORT 1.28.2 ARM zip is 8,616,811 bytes, SHA256
`a8b4a3f1c338798c52d15c6b10b5eed0174cc3f485059948ac3ef456e824d342`.
Its ORT library is byte-identical to the sherpa release's library. The zip includes
Microsoft MIT LICENSE and a 325,054-byte ThirdPartyNotices.txt; their hashes and
its self-reported source commit are in `report/ort-provenance.json`. It is not a
Microsoft official binary-release guarantee or an independently reproduced build.
The sherpa release tar itself contains no LICENSE/NOTICE or C API header.
Full final-package license completeness remains a separate review.

## Offline tests, no downloads

Run from this folder:

```sh
python3 -m unittest discover -s tests -v
sh -n scripts/probe_board_readonly.sh
sh -n scripts/probe_sdk_readonly.sh
```

The dedicated workflow uses only standard-library fixtures and shell syntax
checks. It does **not** download the 32 MB runtime, install a cross compiler,
load a library, or run a model. Retained real-library inspection results are
historical evidence, not rerun by these fixture tests.

With the exact reviewed runtime already obtained separately:

```sh
python3 scripts/audit_arm_runtime.py --library-dir /path/to/reviewed/runtime/lib
```

The audit pins exactly two filenames, hashes and byte counts in code as well as
the manifest. Empty/partial/duplicate/path-traversal locks and changed runtime
bytes fail before inspection. It calls readelf, never ldd or a target executable,
and reads requirements only from the Version needs section. Changing a JSON lock
does not authorize a different runtime. A rebuilt BSP package needs its own review.

## Read-only SDK and optional board inventory

```sh
sh scripts/probe_sdk_readonly.sh /absolute/trusted/SDK/bin/target-gcc /copied/target/busybox
sh scripts/probe_board_readonly.sh
```

Run the first on the SDK host, the second only if the actual board is available.
The SDK probe executes only query options of the explicitly supplied trusted
compiler and readelf; it never executes the copied target ELF. The board probe
reads kernel/CPU features, libc declaration, loader/library names and existing ELF
headers if readelf exists. Both print to stdout without network, installation,
configuration changes, audio capture or candidate-library loading. Review returned
paths/device metadata before publishing. Also provide the SDK/BSP release/source.

## Unexecuted build and acceptance plan

`report/kws-only-build-recipe.json` records the exact source, explicit BSP ORT
include/library directories, actual SDK/sysroot placeholders, CMake options
(TTS/diarization/PortAudio/WebSocket/Python/JNI/examples/binaries off), and commands.
It is a KWS-oriented pruning recipe, not proof that all non-KWS code is absent or
that dependencies have been prefetched. If the BSP has an older or different libc,
rebuild ORT for it rather than silently installing another libc on the target.
Do not weaken the existing PCM adapter's x86 dependency guards to admit ARM files.

`scripts/check_native_gates.py` checks an unstripped future build against explicit
GLIBC/GLIBCXX/CXXABI ceilings and defined eSpeak/phonemize symbols. It refuses a
missing `.symtab`; stripped symbol absence cannot prove component removal.
The recorded official bundle **fails** the no-eSpeak gate as expected.
`older-sdk-threshold-example.json` uses illustrative ceilings, not known SSC305
versions. These gates are necessary checks only: full provider-symbol matching,
architecture/loader compatibility, link maps, build-input and license review,
link/load/lifecycle tests and execution still remain.

The existing adapter contract is mono 16 kHz PCM16 with 320-sample staging, one
inference thread and unchanged frozen decoder settings. Create loads once;
reset replaces stream state, finish flushes partial samples and EOF without
adding silence. Calls are single-threaded/non-reentrant; callback text expires
on callback return. No explicit feed filesystem I/O does not mean allocation-free
or page-fault-free execution.

After BSP compatibility: cross-build/run model-free lifecycle tests; test real
library loading without a model; replay the previously authorized frozen PCM
corpus separately; then measure startup I/O, faults, steady I/O/RSS/allocations,
single-core-normalized CPU/audio ratio, decode tail latency, ingress deadlines,
overruns, concurrent video/ISP/audio-DMA load, memory pressure and long soak.
Product budgets must be supplied rather than invented. No x86 speed/RSS figures
are extrapolated to A32, and no synthetic result establishes production FAR.

Exact sources, hashes, preserved static outputs, expected-failure gates and model
fetch metadata are under `report/` and `evidence/`. Model licensing is independent;
the metadata includes no weights and grants no redistribution rights.

Readelf text snapshots have trailing whitespace removed for repository storage;
the parsed fields and runtime hashes are unchanged. The current archive suite has
13 offline tests; the initial standalone audit suite had 8.
