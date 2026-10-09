# Reproducibility levels

`kws-pipeline` distinguishes three claims:

1. **Traceable training**: the checkpoint records the exact training environment and corpus identity actually used.
2. **Rebuildable training**: the base container image and all Python dependencies are digest/hash pinned so the environment can be reconstructed from retained inputs.
3. **Bit-reproducible SDK**: two independent release builds produce byte-identical installed SDK trees.

Deterministic tar metadata alone satisfies none of the compiled-binary claims. Release documentation must use the strongest claim actually proved by the current gates.


## Runtime build configuration identity

`kws_build_info()->config_digest` is a versioned SHA-256 fingerprint of the
CMake configuration inputs, separate from `source_revision`. It includes the
selected configuration's C and executable/static linker flags in addition to
the global flags; compiler ID/version/target, target system/ABI/sysroot settings;
the top-level toolchain file's SHA-256; build options; and declared compile/link
properties of the core and hosted targets. Target generator expressions are
retained as declarations and bound to the selected configuration, rather than
claimed to be a complete expansion of every compiler command. Included toolchain
files are represented by their resulting settings, not recursively hashed.

Values are individually hashed in a fixed field order. Source/build directory
prefixes are normalized and the compiler/toolchain installation filename is not
used as an identity. No timestamp, output directory, or install destination is
added. A relocated, byte-identical top-level toolchain has the same fingerprint;
changing its contents deliberately changes the fingerprint. The digest is not a
hash of the compiler executable, source tree, linked dependencies or final
binary, and equal configuration digests alone do not prove equal binaries. Exact
runner SHA-256 binding and the independent installed-SDK byte comparison remain
required evidence gates.

Single-configuration generators retain `generated/kws_build_config.h` for
standalone analysis tools. Multi-configuration generators use
`generated/<Config>/kws_build_config.h`; the core's include path selects the
actual `--config`, independently of `CMAKE_BUILD_TYPE`. Both include the common
`generated/kws_source_revision.h`, which is refreshed before every core build.

### Incremental source identity

Without an explicit `KWS_SOURCE_REVISION`, a normal `cmake --build` reads the
current Git HEAD and tracked-file status at the project's source root. It
refreshes `source_revision` for dirty-to-clean transitions and new commits even
when CMake was not reconfigured or no compiled source changed. Linked worktrees,
detached HEADs and packed refs are resolved by Git itself. The generated source
identity header is written only when its contents change; unchanged builds do
not recompile or relink the library. All generated files stay in the build tree.
The refresh target uses CMake 3.16-compatible features and does not change the
configuration fingerprint's input set.

A `.git` directory or gitfile must exist at the project root, and Git must
successfully resolve that root, HEAD and tracked status. Otherwise the revision
is `unknown`; an exported source copy inside some other Git repository must not
inherit that repository's commit. Ordinary vendored copies used through
`add_subdirectory` also report `unknown` when they have no own Git metadata,
even if an enclosing repository tracks them; use an explicit revision label for
that integration. Git submodules with their own gitfile are supported. A nonempty
`-DKWS_SOURCE_REVISION=...` remains an authoritative caller-supplied label,
including for exported source trees; it is not independently verified as a
source commit. Untracked files are excluded
from dirty detection. `HEAD-dirty` is a state label, not a hash of modified
contents, so exact SDK/runner byte hashes are still required. As with compiler
inputs generally, keep the source tree unchanged while a build is running.

### Relocatable SDK discovery

The installed `kws-pipeline.pc` computes its prefix from the actual relative
`CMAKE_INSTALL_LIBDIR` depth, including `lib`, `lib64` and multiarch layouts such
as `lib/x86_64-linux-gnu`. Both pkg-config and the CMake package support
`cmake --install --prefix`, `DESTDIR` staging, and subsequently moving the whole
installed prefix. `CMAKE_INSTALL_LIBDIR` and `CMAKE_INSTALL_INCLUDEDIR` must be
nonempty prefix-relative paths without `..` components. Absolute directories
and paths escaping the prefix are rejected during configuration, because they
cannot provide this relocation contract. This restriction also applies to
absolute directories that happen to be under the initial prefix.

`python3 tests/test_reproducible_sdk.py` builds real Release `-O0` and `-O3`
variants, checks that both the frontend object and configuration digest differ,
and compares two installed `-O3` SDKs from different build directories. It also
checks reconfiguration, empty build type, toolchain location/content identity,
and relocated source/build trees with debug information and compiler prefix maps.
Temporary local Git fixtures exercise clean/dirty/staged/clean transitions,
new HEADs with and without source edits, packed refs, linked worktrees/detached
HEADs, overrides, and exported sources inside an unrelated enclosing repository.
An executable linked against each tested archive reads `kws_build_info()`;
no-op builds are checked for unchanged header/object/archive timestamps.

Real pkg-config and CMake consumers compile, link and run before and after
prefix relocation for `lib`, `lib64`, a multilevel custom libdir/include layout,
and the platform's GNUInstallDirs `/usr` default. The `/usr` case uses `DESTDIR`;
the others use `--install --prefix`. Negative tests reject absolute and
parent-traversing install directories. A successful `pkg-config --modversion`
alone is not evidence of a usable SDK.

With Ninja and CMake >= 3.17 the same test builds and repeats Debug/Release using
Ninja Multi-Config, checks selected archive identities, changes only Release
flags to verify per-configuration binding, and exercises incremental source
refresh in both configurations. It reports an explicit skip if Ninja
Multi-Config is unavailable; it does not establish Visual Studio or Xcode
coverage. These hosted regressions require Git, pkg-config, a GNU-compatible C
compiler and Unix Makefiles.

## CPU training reproducibility boundary

The bounded RNN micro-training fixture records exact tensor bytes on independent
hosted runners. Input identity, training code, Torch version, thread topology and
numeric trace completeness are always fail-closed.

Exact model-byte equality is required when the pair is not an observed
AMD-versus-Intel cross-vendor pair. For an AMD/Intel pair, the repository retains
and reports the first numeric divergence but does not require universal bit
equality. Retained #298/#146 evidence already established that cross-vendor
trajectories can diverge from sub-nanoscopic optimizer differences even under the
same pinned CPU training contract.

This exception is diagnostic only. It does not relax dataset identity, objective
math, product-development preflight, formal qualification, robustness,
continuous-FAR, model promotion, or shipping evidence gates. A random matching
hosted CPU pair must not be used to erase a retained cross-vendor failure.
