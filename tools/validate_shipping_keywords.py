#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

EXPECTED = (
    ("1", "你好小窝", Decimal("0.55"), "ni3 hao3 xiao3 wo1"),
    ("2", "小窝小窝", Decimal("0.55"), "xiao3 wo1 xiao3 wo1"),
)


def validate_shipping_keywords(path: Path) -> list[dict[str, str]]:
    rows = [
        raw.split("\t")
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]
    if len(rows) != len(EXPECTED):
        raise ValueError(f"expected {len(EXPECTED)} shipping keywords, got {len(rows)}: {rows!r}")

    canonical: list[dict[str, str]] = []
    for index, (row, expected) in enumerate(zip(rows, EXPECTED, strict=True), start=1):
        if len(row) != 4:
            raise ValueError(f"shipping keyword row {index} must contain exactly 4 TSV fields: {row!r}")
        kid, phrase, threshold_text, pronunciation = row
        expected_id, expected_phrase, expected_threshold, expected_pronunciation = expected
        if kid != expected_id or phrase != expected_phrase or pronunciation != expected_pronunciation:
            raise ValueError(f"shipping keyword identity drifted at row {index}: {row!r}")
        try:
            threshold = Decimal(threshold_text)
        except InvalidOperation as exc:
            raise ValueError(f"shipping keyword threshold is not numeric at row {index}: {threshold_text!r}") from exc
        if not threshold.is_finite() or threshold != expected_threshold:
            raise ValueError(
                f"shipping keyword threshold drifted at row {index}: "
                f"expected {expected_threshold}, got {threshold_text!r}"
            )
        canonical.append(
            {
                "id": kid,
                "phrase": phrase,
                "threshold": str(expected_threshold),
                "pronunciation": pronunciation,
            }
        )
    return canonical


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the immutable two-keyword shipping TSV contract using numeric threshold semantics."
    )
    parser.add_argument("keywords_tsv", type=Path)
    args = parser.parse_args()
    try:
        canonical = validate_shipping_keywords(args.keywords_tsv)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"shipping_keywords": canonical}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
