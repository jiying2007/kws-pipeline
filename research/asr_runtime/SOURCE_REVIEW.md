# Static source/build-tool review for the no-speech-model qualification

All ten official artifacts were read as bytes, never imported or executed. Exact
URLs, sizes and SHA256 are in `locks/inventory.json`; the body-verification receipt
and streamed archive summaries are in `locks/source-artifacts-audit.json`.

- Five source archives: 19,783,626 bytes
- Five build-tool wheels: 1,267,200 bytes
- Total preparation intake: 21,050,826 bytes, below the separate 32 MiB bound
- Source hooks/builds/installs: zero
- Container image layer bodies or target speech weights: zero

`locks/source-members.json` binds each source member's actual decompressed length
and SHA256. Paths, duplicate names, symlinks, hardlinks, devices, archive counts,
actual expansion and wheel RECORD identity were checked without extraction.
Vendored dist-info inside setuptools is payload, not a second top-level wheel
identity. Only a top-level dist-info/METADATA, WHEEL and RECORD identify the wheel.

## Build hooks and runtime imports

| Source | Statically checked behavior | Build dependencies |
| --- | --- | --- |
| antlr4-python3-runtime 4.9.3 | setup.py imports setuptools only; explicit package list, source directory and pygrun script | Pinned setuptools/backend tools |
| crcmod 1.7 | distutils setup selects python3; defines Extension `crcmod._crcfunext` from `python3/src/_crcfunext.c`; reads README as text | Pinned setuptools distutils shim, Python headers and C compiler in pinned image |
| jieba 0.42.1 | distutils setup has literal metadata and package-data globs; it does not import jieba | Pinned setuptools distutils shim |
| oss2 2.19.1 | reads oss2/__init__.py as text and regex-extracts version; reads README; never imports oss2 for build | Pinned setuptools/backend tools |
| sox 1.5.0 | setup.py imports setuptools; setup.cfg references `sox.version.version`; that file contains a literal 1.5.0 | Pinned setuptools static attribute reader/backend tools |

The exact setuptools 81 wheel's StaticModule uses AST/literal evaluation first.
Its `read_attr` fallback would import only if static reading failed. The pinned
SoX version file is a simple literal assignment, so no SoX runtime/numpy import
is needed for the declared build path. A dynamic missing hook/dependency at run
time still fails offline; the controller never installs an unlocked fallback.

SoX's runtime `__init__.py` performs a fixed `sox -h` child probe. The container
boundary deliberately permits such child processes within the same namespace,
network and resource limits. Absence of the executable may be reported; it is not
silently installed and does not establish audio-processing capability.

## Setuptools .pth and native fallback

The only `.pth` among these five build wheels is the exact locked
`distutils-precedence.pth`. It conditionally invokes `_distutils_hack.add_shim()`.
The handler inserts the pinned distutils finder into Python's import machinery.
Its exact member hash is the sole `.pth` allowlist in `locks/build-plan.json`.
This provides the distutils path needed by legacy source scripts on Python 3.12.

crcmod catches compilation errors and retries without the native extension.
That upstream behavior is preserved, but a pure wheel **cannot qualify**: the
plan requires the exact CPython 3.12 x86_64 extension member, ELF64 machine 62,
non-pure wheel metadata and actual `crcmod._crcfunext` import. The two independently
built wheel bytes and complete payload-manifest hashes must match.

## Reproducible construction and payload audit

Sources are copied unchanged from read-only artifact inputs into a fresh fixed
`/work/sources/<name-version>` tree inside each of two isolated containers.
Extraction accepts only the exact reviewed regular members, never follows links,
and recomputes the source member manifest. Source file times and SOURCE_DATE_EPOCH
are fixed. Pip receives this fixed directory, avoiding its random archive-unpack
path appearing in compiler debug information. No source patch is applied.

Every built wheel is checked for exact name/version, source-declared dependency
metadata (normalizing whitespace/quotes and package-name separators), all static
payload hashes, declared generated metadata/script members, native ELF identity,
and RECORD membership/hash/size equality. Unexpected files are rejected. Three
sources include full LICENSE files; their exact bytes must survive in the wheel.
Jieba and ANTLR sdists do not include standalone complete license files. Preserve
all upstream headers/metadata; **this job does not redistribute package or built
wheel bodies**. Separate license completeness work is required before any future
redistribution.

Jieba's sdist contains bundled dictionaries, HMM data and lac_small statistical
assets. They were hashed as ordinary package members, not loaded. Other installed
dependencies may initialize package-bundled tokenizers/statistical state on import.
The scope is therefore precise: no target ASR/TTS/KWS weight retrieval/loading,
no user audio and no speech-model forward calls. It is not a claim that all
third-party packages contain no models or have no import side effects.

## Limits of this review

The other 133 upstream wheels remain metadata-only candidates until the approved
runner downloads and audits their exact bodies. The five source builds and every
real dependency import remain unperformed. Static source review, a closed graph,
or passing synthetic fixtures must not be presented as real runtime readiness.
