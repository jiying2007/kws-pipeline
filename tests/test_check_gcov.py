from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_gcov  # noqa: E402

TOOL = ROOT / "tools" / "check_gcov.py"

# Two workflows call this and nothing asserted it. It parses gcov output and
# turns it into a pass/fail number, so a parser that quietly counts the wrong
# lines shifts the threshold without anyone seeing why. The cases below pin
# what counts as an executable line and what counts as covered.

HEADER = (
    "        -:    0:Source:src/kws_engine.c\n"
    "        -:    0:Graph:build/kws.gcno\n"
)


def write(home: pathlib.Path, name: str, body: str) -> pathlib.Path:
    path = home / name
    path.write_text(body, encoding="utf-8")
    return path


def run(home: pathlib.Path, minimum: float):
    return subprocess.run(
        [sys.executable, str(TOOL), "--root", str(home), "--minimum", str(minimum)],
        check=False, capture_output=True, text=True,
    )


def check_parsing() -> None:
    with tempfile.TemporaryDirectory() as td:
        home = pathlib.Path(td)

        path = write(home, "header.c.gcov", HEADER)
        assert check_gcov.coverage([path]) == (0, 0), "gcov headers are not code"

        body = "\n".join(
            [
                HEADER,
                "        -:    1:static int unused(void)",
                "    #####:    2:    return 1;",
                "    =====:    3:    return 2;",
                "        0:    4:    return 3;",
                "        7:    5:    return 4;",
                "       12*:    6:    return 5;",
                "     bogus:    7:    return 6;",
                "",
            ]
        )
        path = write(home, "mixed.c.gcov", body)
        covered, total = check_gcov.coverage([path])
        # Lines 2..7 are executable; only 5 and 6 ran. A zero count, an
        # unreached branch and an unparsable count all count against you.
        assert total == 6, total
        assert covered == 2, covered


def check_cli() -> None:
    with tempfile.TemporaryDirectory() as td:
        home = pathlib.Path(td)

        # No evidence at all is an error, not a pass: a coverage gate that
        # silently succeeds when nothing was measured is worse than no gate.
        done = run(home, 50)
        assert done.returncode != 0, done.stdout
        assert "no .gcov files found" in done.stderr, done.stderr

        write(home, "header.c.gcov", HEADER)
        done = run(home, 50)
        assert done.returncode != 0, done.stdout
        assert "no executable lines" in done.stderr, done.stderr

        write(
            home,
            "half.c.gcov",
            "\n".join([HEADER, "    #####:    1:    a();", "        3:    2:    b();", ""]),
        )

        # Exactly at the threshold must pass: the comparison carries an
        # epsilon, so binary floating point must not decide the result.
        done = run(home, 50)
        assert done.returncode == 0, done.stderr
        assert "50.00%" in done.stdout, done.stdout

        done = run(home, 50.01)
        assert done.returncode == 1, done.stdout
        assert "is below required" in done.stderr, done.stderr


def main() -> int:
    check_parsing()
    check_cli()
    print("test_check_gcov: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
