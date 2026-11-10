#!/usr/bin/env python3
"""Compile/link both source-only R1 qhead8 kernels for ARM; no model inference."""
from __future__ import annotations
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def checked_run(argv, timeout=120):
    environment = os.environ.copy()
    environment.pop('LD_LIBRARY_PATH', None)
    result = subprocess.run(argv, env=environment, text=True, capture_output=True,
                            timeout=timeout, check=True)
    return result.stdout


def verify_elf(readelf, path):
    if not path.is_file() or path.is_symlink():
        raise ValueError('missing regular ARM output: ' + str(path))
    header = checked_run([readelf, '-h', '-W', str(path)], timeout=10)
    if not (re.search(r'Class:\s+ELF32', header)
            and re.search(r'Machine:\s+ARM\b', header)
            and 'hard-float ABI' in header):
        raise ValueError('not verified 32-bit ARM hard-float ELF: ' + str(path))
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def execute(root, compiler, readelf):
    """Use the existing public source ZIP; never request weights or audio."""
    module_path = root / 'research/r1_stream_c/quant_head8/materialize.py'
    spec = importlib.util.spec_from_file_location('r1_qhead8_arm_source', module_path)
    if spec is None or spec.loader is None:
        raise ValueError('missing R1 qhead8 source generator')
    materializer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(materializer)
    cmake = shutil.which('cmake')
    if not cmake or not compiler or not readelf:
        raise RuntimeError('CMake, ARM hard-float GCC and readelf are mandatory')
    result = {}
    with tempfile.TemporaryDirectory(prefix='r1-qhead8-arm-') as tmp:
        folder = Path(tmp)
        for kernel in ('scalar', 'par4'):
            source = folder / ('source-' + kernel)
            build = folder / ('build-' + kernel)
            receipt = materializer.materialize_from_repository(root, source,
                                                                kernel=kernel)
            if (receipt.get('private_model_packed') is not False
                    or receipt.get('kernel') != kernel
                    or receipt.get('original_fp32_numeric_gate') != 'FAIL_UNCHANGED'
                    or (source / 'weights').exists()):
                raise ValueError('unexpected model/weight/precision scope')
            toolchain = source / 'cmake/armv7-hardfloat.cmake'
            if not toolchain.is_file():
                raise ValueError('missing pinned hard-float toolchain')
            checked_run([cmake, '-S', str(source), '-B', str(build),
                         '-DCMAKE_BUILD_TYPE=Release',
                         '-DCMAKE_TOOLCHAIN_FILE=' + str(toolchain),
                         '-DCMAKE_C_COMPILER=' + compiler,
                         '-DBUILD_TESTING=OFF'], timeout=90)
            checked_run([cmake, '--build', str(build), '--parallel', '2'],
                        timeout=120)
            executable = build / 'fsmn_run'
            shared = build / 'libfsmn_sdk.so.0.1.0'
            compiled_exe = verify_elf(readelf, executable)
            compiled_shared = verify_elf(readelf, shared)
            symbols = checked_run([readelf, '-W', '--dyn-syms', str(shared)],
                                  timeout=10)
            for symbol in ('fsmn_sdk_create', 'fsmn_sdk_feed',
                           'fsmn_sdk_finish', 'fsmn_sdk_model_sha256'):
                if not re.search(r'\b' + symbol + r'\b', symbols):
                    raise ValueError('missing public ARM SDK export: ' + symbol)
            result[kernel] = {'executable': compiled_exe, 'shared': compiled_shared,
                              'required_sdk_exports': 4, 'weights_published': False,
                              'model_forwards': 0}
    return {'schema': 'r1-qhead8-arm-source-build-v1',
            'target': 'cortex-a32-neon-vfpv4-hardfloat',
            'source_variants': result, 'model_executions': 0,
            'physical_ssc305': False, 'qualification_allowed': False}
