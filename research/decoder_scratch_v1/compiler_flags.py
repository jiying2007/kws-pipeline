"""Authoritative C11/strict floating-point flags for every scratch-v1 compile.

Unsupported flags are fatal under -Werror. Never silently drop an option for a
compiler: runtime and synthetic differential checks must share the same policy.
Sanitizer builds may override optimization level, not floating-point semantics.
"""
STRICT_FP_FLAGS = (
    '-fno-fast-math',
    '-ffp-contract=off',
    '-frounding-math',
    '-fexcess-precision=standard',
)
CFLAGS = ('-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', *STRICT_FP_FLAGS, '-fPIC')
