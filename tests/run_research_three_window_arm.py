"""Bounded ARM hard-float decoder smoke; no ctypes or model execution."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import tempfile

from test_research_three_window_c import (
    check_production_symbols, check_threshold_completion, run,
)

COMPILER = 'arm-linux-gnueabihf-gcc'
NM = 'arm-linux-gnueabihf-nm'
FLAGS = ['-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
         '-fno-fast-math', '-ffp-contract=off', '-fexcess-precision=standard',
         '-mcpu=cortex-a32', '-mfpu=neon-vfpv4', '-mfloat-abi=hard']


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
    check_production_symbols(run([NM, '-D', '--defined-only', str(production)], timeout=10))


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
