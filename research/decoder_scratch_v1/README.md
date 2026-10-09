# Decoder scratch lifetime v1

Opt-in host research optimization derived from the already public native A20
baseline. Historical baseline bytes, source manifests, numerical conclusions,
beams, thresholds and shipping defaults remain unchanged. This is not a reissue
of a historical model/runtime bundle and contains no weights, recordings or saved
private tensors.

## Reproduce

From the repository root, with Python 3, a C11 compiler, GNU patch and libm:

```
python3 -B research/decoder_scratch_v1/build.py --output /tmp/decoder-scratch-v1
```

The output must be new. The command first pins the original SOURCE_MANIFEST.json hash, then verifies the closed source inventory (excluding Python bytecode caches) and every file hash before executing the now-verified checker. It
verifies pinned decoder hashes and patch hash, applies the patch with zero fuzz,
verifies derived decoder hashes and writes a new derived manifest/provenance.
It compiles a complete research shared runtime, executes original and optimized
resource-size queries, then runs decoder-only invented-input tests normally,
under ASan (leak detection disabled), under UBSan, and in an exact semantic-state
differential test against the unchanged original decoder. No acoustic model
forward, audio processing, training, original once-run replay, or threshold
search occurs. The shared runtime is built but never loaded/executed by this
command. Build receipts include compiler, commands, outputs and binary hash.

## Why the change is safe within this scope

The next-hypothesis array is only read until the retained hypotheses are copied.
The compaction buffer is only written after that point. The patch first copies
all retained hypotheses, then switches a naturally aligned union from `next`
to `compact`; neither inactive member is read. Stable sorting, node order,
shallow node aliasing, arithmetic order, all capacity bounds, error codes and
result/clock behavior are unchanged. The 70,000-byte transactional state copy
is preserved: late capacity failures still leave caller state/result untouched.
The final state is committed only after a successful whole-chunk operation.

Workspace layout/ABI changes. Rebuild every caller and allocate scratch using
the new header or size query; do not mix objects built against different header
versions. No reduction to path beam, score beam or prefix/node capacities is made.

## Measured host resources

On the recorded GCC 14 x86-64 ABI:

- Decoder state: 70,000 → 70,000 bytes
- Workspace: 170,400 → 136,800 bytes (33,600 saved, 19.72%)
- Full caller stream arena: 333,704 → 300,104 bytes (10.07%)
- Weights: 1,565,280 → 1,565,280 bytes
- Arena plus weights: 1,898,984 → 1,865,384 bytes

The union shares 33,600 bytes of next hypotheses with the 61,440-byte compact
buffer. It does not remove the 5,360-byte remap or the transactional copy.
Host `sizeof` is measured, not a board result. Stack, allocator alignment,
linker layout, latency/energy and the SSC305 ABI still need separate measurement.
This optimization does not fix the saved D20 raw-logit FAIL, qualify D90, or
explain the separate shipping model's false accepts.

## Native execution guard and compiler selection

All compiles, including both differential-test libraries, use the selected
`--cc`. Before any generated executable runs or shared library is loaded, the
module checks its ELF class, machine, byte order, OSABI/version and relevant ARM
ABI flags against the running Python process's actual ELF. Foreign targets are
rejected before execution even when binfmt emulation is installed; this module
has no cross-target execution mode. The parser/comparison functions are reused
from the unchanged public native build driver, with their source hash recorded
in DERIVATION.json. This checks ELF compatibility, not full CPU-instruction or
vendor-loader qualification. Every compiler/test subprocess has a 120-second
timeout. A failed guard or timeout produces no passing receipt.

The build runs 12 file-only ELF positive/negative tests. Additional end-to-end
checks use a mock foreign compiler and a logging wrapper around the chosen host
compiler, without installing or executing a cross toolchain:

```
python3 -B research/decoder_scratch_v1/test_driver_guards.py
```

These verify foreign runtime blocking before the first resource/test execution,
foreign differential-library blocking before CDLL, and compiler propagation to
both differential libraries. Python assertion optimization remains rejected.

Source admission additionally rejects symlinks, self-consistent rewrites of the
source manifest, modified checker bytes before execution, and unlisted files.
The six end-to-end guard tests include those three source-admission negatives.
