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
actual `--config`, independently of `CMAKE_BUILD_TYPE`.

`python3 tests/test_reproducible_sdk.py` builds real Release `-O0` and `-O3`
variants, checks that both the frontend object and configuration digest differ,
and compares two installed `-O3` SDKs from different build directories. It also
checks reconfiguration, empty build type, toolchain location/content identity,
and relocated source/build trees with debug information and compiler prefix maps.
With Ninja and CMake >= 3.17 it builds and repeats Debug/Release using Ninja
Multi-Config, checks selected archive identities, and changes only Release flags
to verify per-configuration binding. That test reports an explicit skip if Ninja
Multi-Config is unavailable; it does not establish Visual Studio or Xcode
coverage. These hosted optimization regressions require a GNU-compatible C
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
