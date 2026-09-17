#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

KINDS = {"positive", "confusable", "negative", "background"}
MANIFEST_CLASS = "speech-like-synthetic-recording-v1"
LABEL_CLASS = "speech-like-training-label-v1"


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def key(row: dict, label: str) -> tuple[str, str, str]:
    voice = row.get("voice_id")
    source = row.get("source_id")
    file_sha = row.get("file_sha256")
    if not all(isinstance(value, str) and value.strip() for value in (voice, source, file_sha)):
        raise ValueError(f"{label}: voice_id/source_id/file_sha256 are required")
    return voice.strip(), source.strip(), file_sha.strip()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_label(row: dict, label: str) -> dict:
    if int(row.get("schema_version", 0)) != 1 or row.get("evidence_class") != LABEL_CLASS:
        raise ValueError(f"{label}: label identity mismatch")
    kind = str(row.get("kind", ""))
    if kind not in KINDS:
        raise ValueError(f"{label}: unsupported kind {kind!r}")
    keyword = row.get("keyword_id")
    if kind == "positive":
        if isinstance(keyword, bool) or not isinstance(keyword, int) or keyword < 0:
            raise ValueError(f"{label}: positive requires non-negative integer keyword_id")
    elif keyword is not None:
        raise ValueError(f"{label}: non-positive rows must use keyword_id=null")
    return {
        "kind": kind,
        "keyword_id": keyword,
        "label_source": str(row.get("label_source", "development-corpus-plan-v1")),
    }


def attach(manifest: pathlib.Path, labels: pathlib.Path) -> list[dict]:
    manifest_rows = load_jsonl(manifest)
    label_rows = load_jsonl(labels)
    label_map: dict[tuple[str, str, str], dict] = {}
    for index, row in enumerate(label_rows, 1):
        item_key = key(row, f"{labels}:{index}")
        if item_key in label_map:
            raise ValueError(f"{labels}:{index}: duplicate label identity")
        label_map[item_key] = normalize_label(row, f"{labels}:{index}")

    output: list[dict] = []
    used: set[tuple[str, str, str]] = set()
    for index, row in enumerate(manifest_rows, 1):
        if int(row.get("schema_version", 0)) != 1 or row.get("evidence_class") != MANIFEST_CLASS:
            raise ValueError(f"{manifest}:{index}: manifest identity mismatch")
        item_key = key(row, f"{manifest}:{index}")
        if item_key not in label_map:
            raise ValueError(f"{manifest}:{index}: missing training label")
        label_value = label_map[item_key]
        used.add(item_key)
        label_identity = {
            "voice_id": item_key[0],
            "source_id": item_key[1],
            "file_sha256": item_key[2],
            **label_value,
        }
        output.append({
            **row,
            "kind": label_value["kind"],
            "keyword_id": label_value["keyword_id"],
            "label_source": label_value["label_source"],
            "label_provenance_sha256": canonical_sha256(label_identity),
        })
    unused = sorted(set(label_map) - used)
    if unused:
        raise ValueError(f"labels contain {len(unused)} identity/identities absent from manifest")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach immutable training labels to audited speech-like manifests.")
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--labels", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    rows = attach(args.manifest.resolve(), args.labels.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(f"speech-like labels attached: recordings={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
