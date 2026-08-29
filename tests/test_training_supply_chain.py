from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    dockerfile = (ROOT / "training" / "Dockerfile").read_text(encoding="utf-8")
    lock = (ROOT / "training" / "requirements.lock").read_text(encoding="utf-8")
    assert "@sha256:" in dockerfile, "training base image must be digest pinned"
    assert "--require-hashes" in dockerfile, "pip install must enforce hashes"
    assert "--hash=sha256:" in lock, "training dependencies must carry hashes"
    print("test_training_supply_chain: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
