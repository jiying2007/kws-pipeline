"""Bounded ARM hard-float decoder smoke; no ctypes or model execution."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import tempfile

from test_research_three_window_c import (
    PUBLIC_SYMBOLS, check_threshold_completion, run,
)

COMPILER = 'arm-linux-gnueabihf-gcc'
READELF = 'readelf'
FLAGS = ['-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
         '-fno-fast-math', '-ffp-contract=off', '-fexcess-precision=standard',
         '-mcpu=cortex-a32', '-mfpu=neon-vfpv4', '-mfloat-abi=hard']



def check_arm_dynamic_symbols(text):
    """Fail closed on ARM ELF32 dynamic exports; preserve the 13-symbol ABI gate."""
    for pattern in (r'^\s*Class:\s+ELF32\s*$',
                    r'^\s*Machine:\s+ARM\s*$',
                    r'^\s*Type:\s+DYN\s+\(',
                    r"^\s*Data:\s+2's complement, little endian\s*$"):
        if re.search(pattern, text, re.MULTILINE) is None:
            raise RuntimeError('production ARM ELF identity mismatch')
    header = re.search(r"^Symbol table '\.dynsym' contains (\d+) entries:\s*$",
                       text, re.MULTILINE)
    if header is None:
        raise RuntimeError('ARM ELF dynamic symbol table missing')
    names, indices = [], []
    for line in text[header.end():].splitlines():
        if line.startswith('Symbol table '):
            break
        columns = line.split()
        if not columns or not re.fullmatch(r'\d+:', columns[0]):
            continue
        if (len(columns) < 7 or not re.fullmatch(r'[0-9a-fA-F]+', columns[1])
                or not columns[2].isdigit()):
            raise RuntimeError('malformed ARM ELF dynamic symbol row')
        indices.append(int(columns[0][:-1]))
        kind, binding, visibility, section = columns[3:7]
        raw_name = columns[7] if len(columns) >= 8 else ''
        name = raw_name.split('@', 1)[0]
        if section != 'UND' and ('kws3_test' in name or name.startswith('kws3_')):
            if (not name.startswith('kws3_') or kind != 'FUNC'
                    or binding != 'GLOBAL' or visibility != 'DEFAULT'):
                raise RuntimeError('invalid ARM ELF production symbol attributes')
            names.append(name)
    if indices != list(range(int(header.group(1)))):
        raise RuntimeError('incomplete ARM ELF dynamic symbol table')
    if len(names) != len(PUBLIC_SYMBOLS) or set(names) != PUBLIC_SYMBOLS:
        raise RuntimeError('production ARM ELF exports differ from the 13-symbol ABI')


def check_smoke_completion(text):
    if not re.search(r'^synthetic groups=8 storage_bytes=[1-9][0-9]*; no acoustic admission$',
                     text, re.MULTILINE):
        raise RuntimeError('eight-group ARM C smoke completion missing')


def run_arm(output, emulator, sysroot):
    """Three serial builds, two bounded QEMU runs, one cross-nm audit."""
    common = [COMPILER] + FLAGS
    prefix = [emulator, '-L', str(sysroot)]
    for source, name, check in (
            ('test_synthetic.c', 'smoke', check_smoke_completion),
            ('test_threshold.c', 'threshold', check_threshold_completion)):
        executable = output / name
        run(common + ['-O1', '-g', source, '-lm', '-o', str(executable)], timeout=45)
        check(run(prefix + [str(executable)], timeout=60))
    # No KWS3_TESTING: this library is inspected, never loaded into host Python.
    production = output / 'libkws3.so'
    run(common + ['-O2', '-fPIC', '-shared', 'kws3.c', '-lm', '-o', str(production)], timeout=45)
    # Cross-nm timed out in original hosted failures. Read identical dynamic
    # exports with GNU readelf, retaining the 10-second limit and no fallback.
    symbols = run([READELF, '-h', '-W', '--dyn-syms', str(production)],
                  env=dict(os.environ, LC_ALL='C'), timeout=10)
    check_arm_dynamic_symbols(symbols)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emulator', default='/usr/bin/qemu-arm-static')
    parser.add_argument('--emulator-sysroot', required=True)
    args = parser.parse_args(argv)
    sysroot = Path(args.emulator_sysroot).resolve()
    if not (sysroot / 'lib/ld-linux-armhf.so.3').is_file():
        parser.error('ARM hard-float loader missing from emulator sysroot')
    with tempfile.TemporaryDirectory(prefix='research-kws3-arm-') as temporary:
        run_arm(Path(temporary), args.emulator, sysroot)
    print('PASS: ARM hard-float decoder smoke; three builds; eight legacy groups; '
          'eight threshold groups (569 rows, 572 attempts, 949 beam updates, '
          '21 windows, 11 direct merges); 13 production symbols without test hooks; '
          'no host ctypes load, model/audio/real-posterior inputs, or sanitizer claim; '
          'not SSC305 SDK, board performance, frontend/network numerical qualification')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
