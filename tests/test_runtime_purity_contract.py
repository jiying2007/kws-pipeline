from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_runtime_purity.py"

# The undefined-symbol set actually produced by the hosted Release build of
# libkws_pipeline.a. It is pinned here on purpose: this fixture is the contract,
# so a new external dependency has to be argued for twice -- once in the
# checker's allowlist and once here.
OBSERVED = [
    "__stack_chk_fail",
    "cosf",
    "expf",
    "kws_decoder_init",
    "kws_decoder_reset",
    "kws_decoder_set_keywords",
    "kws_decoder_step",
    "kws_frontend_init",
    "kws_frontend_last_dbfs",
    "kws_frontend_push",
    "kws_frontend_reset",
    "log10f",
    "log1pf",
    "logf",
    "memcmp",
    "memcpy",
    "memmove",
    "memset",
    "powf",
    "sqrtf",
    "tanhf",
]

# The same sources built for ARMv7 hard-float (Sigmastar toolchain). ARMv7 has
# no 64-bit integer divide instruction, so the compiler emits EABI helpers for
# the sample-counter division. They are pure arithmetic, and without them in the
# allowlist the gate would false-fail on the architecture that actually ships.
OBSERVED_ARM = sorted([*OBSERVED, "__aeabi_ldivmod", "__aeabi_uldivmod"])

# One representative per category the README claim enumerates, plus a few that
# are only reachable through an include nobody thinks about.
DENIED_SAMPLES = {
    "heap": ["malloc", "calloc", "realloc", "free", "strdup", "posix_memalign", "mmap"],
    "thread": ["pthread_create", "pthread_mutex_lock", "thrd_create", "mtx_lock"],
    "filesystem": ["fopen", "open", "openat", "read", "write", "close", "fread"],
    "stdio": ["printf", "snprintf", "fprintf", "puts", "perror"],
    "text_conversion": ["iconv_open", "setlocale", "mbstowcs", "nl_langinfo"],
    "dynamic_loading": ["dlopen", "dlsym"],
    "time_random_env": ["clock_gettime", "gettimeofday", "rand", "getenv", "time"],
    "process_signal": ["system", "fork", "execve", "abort", "exit", "signal", "raise"],
    "assert": ["__assert_fail", "assert"],
    # The EABI divide-by-zero handler is the one __aeabi_* symbol that must stay
    # out: it calls raise(SIGFPE). Allowing it would mean the ABI allowlist had
    # been widened by prefix rather than argued for per symbol.
    "abi_trap": ["__aeabi_idiv0"],
}


def nm_text(symbols: list[str]) -> str:
    """Render a listing in the shape `nm --undefined-only -g` produces."""
    body = "".join(f"                 U {symbol}\n" for symbol in symbols)
    return f"\nkws.c.o:\n{body}"


def run(root: pathlib.Path, symbols: list[str]) -> int:
    listing = root / "nm.txt"
    listing.write_text(nm_text(symbols), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(CHECKER), "--nm-output", str(listing)],
        check=False,
        capture_output=True,
        text=True,
    ).returncode


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)

        # The measured production symbol set is the passing case.
        assert run(root, OBSERVED) == 0, "the shipped dependency set must satisfy the contract"

        # So is the ARMv7-hard-float build of the same sources.
        assert run(root, OBSERVED_ARM) == 0, "the ARMv7 dependency set must satisfy the contract"

        # Every enumerated category must be rejected. Without this the checker
        # could be broken into always passing and the run above would not notice.
        for category, samples in sorted(DENIED_SAMPLES.items()):
            for symbol in samples:
                code = run(root, [*OBSERVED, symbol])
                assert code == 1, f"{category}: {symbol} must be rejected, got exit {code}"

        # A rejected symbol must fail even when it is the only dependency.
        assert run(root, ["memcpy", "malloc"]) == 1

        # An empty or unreadable listing is not a pass: a vacuous result must be
        # an error rather than a green check.
        assert run(root, []) == 2, "an empty symbol set must not pass"

        # Neither is an archive that references no memory primitive at all --
        # that is a wrong input, not a clean library.
        assert run(root, ["cosf", "sqrtf"]) == 2, "a listing without mem* must not pass"

        # And a missing library must be an error, not a silent skip.
        assert (
            subprocess.run(
                [sys.executable, str(CHECKER), "--library", str(root / "absent.a")],
                check=False,
                capture_output=True,
                text=True,
            ).returncode
            == 2
        )

    print("test_runtime_purity_contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
