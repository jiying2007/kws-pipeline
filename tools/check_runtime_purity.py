#!/usr/bin/env python3
"""Fail-closed dependency contract for the real-time library.

The README claims the real-time path has "no heap, hidden thread, lock,
filesystem or text/pinyin conversion". Nothing enforced that claim: it held
only for as long as a reviewer happened to notice. A single ``malloc`` or
``pthread_mutex_lock`` added to ``src/`` would have passed every other gate in
this repository.

The check is at the symbol level rather than the source level, so a macro, an
alias or a wrapper in another translation unit cannot hide it: if the archive
references ``malloc``, ``malloc`` appears as an undefined symbol.

The contract is an allowlist, not a denylist. A denylist can only reject the
dependencies somebody thought of, and the interesting failure is always the one
nobody listed; an allowlist makes every new external dependency a deliberate
act that has to be argued for in this file.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

INTERNAL_PREFIX = "kws_"

# libc entry points the runtime may call. Every one of these is pure
# computation over caller-owned memory: no allocation, no descriptor, no lock,
# no locale, no clock. The __*_chk entries are the _FORTIFY_SOURCE variants of
# the mem* family; __stack_chk_* is the toolchain's stack protector.
ALLOWED_LIBC = frozenset(
    """
    memcpy memmove memset memcmp memchr
    __memcpy_chk __memmove_chk __memset_chk __memcmp_chk __memchr_chk
    strlen strnlen strcmp strncmp strchr strrchr strstr
    abs labs llabs
    __stack_chk_fail __stack_chk_fail_local __stack_chk_guard
    """.split()
)

# Compiler-generated EABI runtime helpers. ARMv7 has no 64-bit integer divide
# instruction, so the compiler emits calls to these for `uint64_t` division and
# modulo -- both of which the engine performs on its sample counters. They are
# pure arithmetic: no heap, no lock, no descriptor. Omitting them would make the
# gate false-fail on the one architecture that actually ships.
#
# Add a helper only when a real build needs it. `__aeabi_idiv0` in particular
# must stay out: it is the divide-by-zero handler and calls raise(SIGFPE), which
# is a signal dependency the runtime contract excludes.
ALLOWED_ABI_HELPERS = frozenset(
    """
    __aeabi_ldivmod __aeabi_uldivmod
    """.split()
)

# ISO C <math.h> names. The float (f) and long-double (l) variants are derived
# by suffix rather than listed, so `sqrtf` and `sqrtl` both resolve to `sqrt`.
MATH_FUNCTIONS = frozenset(
    """
    acos asin atan atan2 cos sin tan
    acosh asinh atanh cosh sinh tanh
    exp exp2 expm1 frexp ilogb ldexp log log10 log1p log2 logb modf scalbn scalbln
    cbrt fabs hypot pow sqrt
    erf erfc lgamma tgamma
    ceil floor nearbyint rint lrint llrint round lround llround trunc
    fmod remainder remquo copysign nan nextafter nexttoward fdim fmax fmin fma
    """.split()
)

NM = "nm"
# nm prints undefined symbols with a U (or a lowercase w/v for weak ones) in the
# type column, under a header line naming the archive member.
NM_LINE = re.compile(r"^\s*[Uwv]\s+(\S+)\s*$")
# Anti-vacuity sentinel: the archive must reference at least one symbol of the
# project's own. An archive that references none is the wrong input, not a
# clean result, and must not be reported as a pass.
#
# The sentinel is an internal symbol rather than a libc one on purpose. An
# earlier version required memcpy/memmove/memset, which held under gcc but not
# under clang: at -O2 clang inlines fixed-size memory ops instead of calling
# them, so the archive legitimately references no mem* at all and the sentinel
# rejected a correct build. Whether a call is emitted is a code-generation
# detail, not part of the runtime contract; whether the library calls into its
# own translation units is.


def allowed(symbol: str) -> bool:
    if symbol.startswith(INTERNAL_PREFIX):
        return True
    if symbol in ALLOWED_LIBC:
        return True
    if symbol in ALLOWED_ABI_HELPERS:
        return True
    if symbol in MATH_FUNCTIONS:
        return True
    if len(symbol) > 1 and symbol[-1] in "fl" and symbol[:-1] in MATH_FUNCTIONS:
        return True
    return False


def parse_nm(text: str) -> list[str]:
    symbols = set()
    for raw in text.splitlines():
        match = NM_LINE.match(raw)
        if match:
            symbols.add(match.group(1))
    return sorted(symbols)


def read_symbols(library: pathlib.Path | None, nm_output: pathlib.Path | None) -> list[str]:
    if nm_output is not None:
        return parse_nm(nm_output.read_text(encoding="utf-8"))
    assert library is not None
    completed = subprocess.run(
        [NM, "--undefined-only", "-g", str(library)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"{NM} failed on {library}: {completed.stderr.strip()}")
    return parse_nm(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the real-time library's undefined symbols against the purity allowlist."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--library", type=pathlib.Path, help="static library to inspect with nm")
    source.add_argument(
        "--nm-output",
        type=pathlib.Path,
        help="read a saved nm listing instead of running nm (for self-tests)",
    )
    args = parser.parse_args()

    if args.library is not None and not args.library.is_file():
        raise ValueError(f"library not found: {args.library}")
    symbols = read_symbols(args.library, args.nm_output)
    if not symbols:
        raise ValueError("no undefined symbols parsed: refusing to pass vacuously")
    if not any(symbol.startswith(INTERNAL_PREFIX) for symbol in symbols):
        raise ValueError(
            "no " + INTERNAL_PREFIX + "* symbol is undefined: wrong archive?"
        )

    violations = [symbol for symbol in symbols if not allowed(symbol)]
    if violations:
        # Print the whole set, not just the offenders: on an unseen toolchain
        # the useful question is usually "what else is in here", and a bare
        # violation list makes that a second round trip.
        print("runtime purity violations: " + ", ".join(violations), file=sys.stderr)
        print("all undefined symbols: " + ", ".join(symbols), file=sys.stderr)
        return 1
    print(f"check_runtime_purity: ok undefined_symbols={len(symbols)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
