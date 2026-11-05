#!/usr/bin/env python3
"""Enter preserved research drivers after isolating the host Python loader path.

setup-python may need its own lib directory to start a shared CPython build.
Once Python has started, that one verified path must not reach a C compiler,
emulator, or preserved driver whose environment forbids loader overrides.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
import sysconfig


def clear_interpreter_library_path(environment=None):
    """Remove only this interpreter's sole lib path; reject every other override.

    Validation precedes mutation. Presence of LD_PRELOAD/LD_AUDIT, even empty,
    remains an error. Empty, relative, additional and repeated library search
    paths are also errors, not paths that this adapter silently sanitizes.
    """
    environment = os.environ if environment is None else environment
    if any(name in environment for name in ('LD_PRELOAD', 'LD_AUDIT')):
        raise RuntimeError('loader overrides are forbidden')
    if 'LD_LIBRARY_PATH' not in environment:
        return None
    value = environment['LD_LIBRARY_PATH']
    directory = Path(sys.base_prefix) / 'lib'
    soname = sysconfig.get_config_var('INSTSONAME')
    if (value != str(directory) or not directory.is_absolute()
            or not isinstance(soname, str) or Path(soname).name != soname
            or not soname.startswith(f'libpython{sys.version_info.major}.{sys.version_info.minor}.so')
            or not (directory / soname).is_file()
            or (directory / soname).resolve().parent != directory.resolve()):
        raise RuntimeError('loader overrides are forbidden: expected only the interpreter lib directory')
    del environment['LD_LIBRARY_PATH']
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    clear_interpreter_library_path()
    script = args.script.resolve(strict=True)
    # Match direct script execution, including sibling imports and its argv.
    sys.path[0] = str(script.parent)
    sys.argv = [str(script), *args.arguments]
    runpy.run_path(str(script), run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
