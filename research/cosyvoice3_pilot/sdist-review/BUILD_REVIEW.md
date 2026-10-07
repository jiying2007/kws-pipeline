# Exact source-distribution build review

Status: exact official PyPI input archives SHA256-verified and statically reviewed. No package build, candidate import, installation, model download or GitHub write occurred locally. These are source-input qualifications; native build/runtime compatibility is deliberately pending the admitted runner gate.

## Inputs

The machine-readable `sdist-receipts.json` gives exact official PyPI URLs, compressed byte counts and SHA256 values. `source-member-manifests.json` records each regular member's path, size and SHA256. Archives contain no absolute/traversal paths or links. Setup/config/version files were read as text; none was executed. Combined compressed size is 1,183,072 bytes.

- pyworld 0.3.4: 251,990 bytes, SHA256 106c70ee7a7d8c9ba488d8022f2033706929a40f0264256b8e87da5aab98303a
- openai-whisper 20250625: 803,191 bytes, SHA256 37a91a3921809d9f44748ffc73c0a55c9f366c85a3ef5c2ae0cc09540432eb96
- wget 3.2: 10,857 bytes, SHA256 35e630eca2aa50ce998b9b1a127bb26b30dfee573702782aa982f875e3f16061
- antlr4-python3-runtime 4.9.3: 117,034 bytes, SHA256 f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b

## Locked build environment

CPython 3.12.14, Linux x86_64, fresh non-system-site-packages venv. pip 26.2, setuptools 81.0.0, wheel 0.46.2, Cython 3.0.12, NumPy 1.26.4 and packaging 24.2 are exact official wheel inputs in runtime-lock.json. No extra setuptools groups are selected. Runtime includes these pinned tools, allowing exact closed-environment accounting; builder venv is separate and discarded after successful builds. NumPy/Cython never fall back to source builds.

Use pip wheel --no-index --no-cache-dir --no-deps --no-build-isolation. NumPy1.26.4 is both build and runtime ABI baseline; SETUPTOOLS_USE_DISTUTILS=local is required for Python3.12 legacy setup scripts. An inherited C++ compiler is required for pyworld; the installer does not install system packages. No full model assets are present during dependency staging.

## pyworld 0.3.4

pyproject.toml declares wheel, setuptools, numpy and cython, without version bounds or backend; PEP517 defaults to setuptools.build_meta:__legacy__. setup.py first imports setuptools, then distutils.version.LooseVersion (unused), NumPy and setuptools.command.build_ext. It reads README, discovers packages and globs bundled WORLD C++ sources. Extension pyworld.pyworld is language=c++, built from pyworld/pyworld.pyx plus 11 lib/World/src/*.cpp files and bundled headers. The archive contains that complete declared source set. No setup network/subprocess call or external link-library flag was found. Runtime metadata is numpy and cython>=0.24. Source distribution extras are not selected. Build success and deterministic dio/stonemask native execution remain mandatory remote checks.

## Whisper 20250625

Backend setuptools.build_meta; requires setuptools>=61.2. Actual archived whisper/version.py is the literal `__version__ = "20250625"`, so setuptools can extract it statically. The package includes feature/tokenizer assets and is pure Python. Runtime dependencies are more-itertools, numba, numpy, tiktoken, torch, tqdm and triton>=2 on Linux x86_64 or linux2; the locked Torch requirement triton3.7.1 satisfies this. No FFmpeg/Rust/Torch/model download is part of building the wheel. Runtime feature test uses a synthetic in-memory PCM signal, avoiding FFmpeg and model assets.

## wget 3.2

No pyproject; legacy setuptools PEP517 fallback. setup.py imports distutils.core.setup, reads the version from wget.py text without importing it, and reads README.txt. A single pure-Python module; no declared install requirements, custom build command, network call or subprocess in setup. Vendored distutils is explicitly enabled.

## antlr4-python3-runtime 4.9.3

No pyproject; setuptools legacy PEP517 fallback. setup.py selects six explicit packages under src and the pygrun script. Its only dependency is typing; python_version<'3.5', inactive on Python3.12. No network/subprocess/custom build behavior in setup. This exact 4.9.3 satisfies Hydra/OmegaConf's ==4.9.* requirement.

## Execution safeguards and limits

The installer acquires one official PyPI artifact at a time, checks expected SHA256 and length, checks wheel member paths/expansion and METADATA identities, installs without resolving, then removes only its own staging copy. It builds the four exact sdists in a separate locked build venv, records each realized wheel hash and metadata, runs builder/runtime pip check, and removes its own builder/staging directories only. Build-time Python network calls are blocked by an audit hook; pip index/network resolution is disabled. This is defense-in-depth, not a kernel network namespace.

Metadata consistency cannot establish API/operator compatibility. The separate offline runtime qualifier checks the exact installed package set and active metadata constraints, imports the direct roots, and exercises synthetic CPU operations. Generation source, adapter binding, strict model loading, memory peak and model inference are separate gates. The runner must stop if any qualification fails. No retry with alternative versions is silently permitted.

## Review scope

This is a bounded review of build entry points, declared dependencies, archived source completeness and traversal/link safety, not a comprehensive security audit of every runtime module or bundled C++ function. The earlier approval-cancelled read-only fetch was retried once after the parent supplied the user's exact approval evidence; that retry succeeded.
