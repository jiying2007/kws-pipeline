# Exact local-input build

No script downloads dependencies or installs tools. Work outside the Git
checkout. Python 3.9+ is required. The actual reference used GCC/G++ 14.2.0,
CMake 3.31.10, GNU Make 4.4 and GNU ld 2.44, Linux x86-64, Release and Unix
Makefiles. These do not define a complete reproducible sysroot/toolchain.

## Source preparation

`dependencies.lock.json` identifies upstream archives/commits and runtime/model
bytes. Obtain and inspect these inputs separately. `sources.lock.json` fixes
every selected file, original hash, output hash and narrow derivative patch.
Create a local JSON file mapping each of the following component names to its
extracted root (the directory immediately containing its upstream files):

- sherpa-onnx
- kaldi-native-fbank
- kissfft
- kaldi-decoder
- kaldifst
- simple-sentencepiece
- nlohmann-json
- openfst-top-level
- eigen-top-level
- onnxruntime-headers

The last root must contain the locked `include/` headers, LICENSE and complete
ThirdPartyNotices. Its recorded source is a matching-version maintainer ARM
archive, headers/notices only; no ARM library may substitute for the x86 ORT.
The materializer verifies selected file bytes; archive hashes are acquisition
provenance, not evidence that this script validated your entire archive. Extra
unselected upstream files are ignored and never copied or compiled.
The original Sherpa `sherpa-onnx/c-api/c-api.cc` is additionally read by its full
hash to verify all 19 extracted wrapper spans, but is not copied or compiled.

```sh
python3 research/sherpa_host/materialize.py \
  --inputs /absolute/path/local-inputs.json \
  --output /absolute/path/new-host-source
python3 research/sherpa_host/build.py \
  --source /absolute/path/new-host-source \
  --build /absolute/path/new-host-build \
  --ort /absolute/path/libonnxruntime.so
```

Both output directories must be new, outside the checkout and input tree.
Preparation rejects invalid component sets, unsafe relative paths, symlinks,
changed files, patch context/offset drift, existing destinations and output
contamination. It copies only 491 source/build/config/license files, including
the unchanged full API header once, and a deterministic identity receipt.
The single complete 9,366-byte decoder patch applies without offsets or fuzz.

The 491 production files reconstruct the tested candidate byte for byte;
historical candidate documentation, redundant metadata and duplicate CLI header
are omitted. The repository CLI tests reuse the identical header already at
`research/sherpa_pcm/vendor/sherpa-onnx/c-api/c-api.h`.

`build.py` rechecks the complete materialized tree and pinned ORT before CMake,
then builds target `kws` with `--parallel 1`. It records tool executable hashes,
version strings, compile-command hash and artifact hashes. It neither runs a
model nor treats a successful build as runtime acceptance. `--cmake`, `--cc`
and `--cxx` select already-installed tools; changed tools need new validation.

The CMake and LockedTargets bytes are identical to the tested B build, including
historical preparation comments. The current results are in RESULTS.md. They
retain 37 Sherpa TUs, 14 vendor TUs, one wrapper TU and one CLI TU; exact per-TU
options, definitions, include order and SYSTEM attributes are recorded in
`compile-contracts.json`. Only the two original KISS C TUs use fast-math. There
is no new LTO, GC or CPU optimization. The exact project-root macro-prefix-map
affects diagnostic paths only; it is not a claim about all upstream strings.

Output paths are `bin/kws` and `lib/libsherpa-onnx-kws-c-api.so`. CLI RUNPATH is
`$ORIGIN/../lib`, subset RUNPATH is `$ORIGIN`, and ORT stays unchanged. A rebuild
on another host need not yield historical binary hashes; inspect and validate
its actual ABI, loading, behavior and dependency requirements separately.

## Optional audited-byte staging

`stage.py` accepts only the historical runtime/model hashes in the lock. Supply
a local root with `bin/`, `lib/`, `models/` and the exact model notice files.
It copies the materialized corresponding source (including all retained third-
party notices), project LICENSE, usage docs, model notices and fixed profile,
then generates MANIFEST.sha256. It does not execute, archive or upload anything.

```sh
python3 research/sherpa_host/stage.py \
  --source /absolute/path/new-host-source \
  --runtime /absolute/path/verified-runtime-inputs \
  --model-notices /absolute/path/model-notices \
  --output /absolute/path/new-host-kit
```

Model notices are README.upstream.md, standard Apache-2.0.txt and ORIGIN.json
with identities in the dependency lock. See PROVENANCE.md. A staged kit has its
own manifest identity and is not automatically the same archive as an earlier
delivery. The launcher requires the generated manifest. Hashes are not signatures.
