from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    docs = [
        ROOT / "docs" / "TERMINAL_HARDENING.md",
        ROOT / "docs" / "CORPUS_IDENTITY.md",
        ROOT / "docs" / "TARGET_EVIDENCE.md",
        ROOT / "docs" / "REPRODUCIBILITY.md",
        ROOT / "docs" / "GOVERNANCE_TARGET.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert text.strip(), path
    boundary = (ROOT / "docs" / "TERMINAL_HARDENING.md").read_text(encoding="utf-8")
    assert "does not claim real Mandarin/device qualification" in boundary
    print("test_terminal_docs: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
