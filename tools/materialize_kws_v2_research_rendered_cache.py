#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import wave

SPLITS = ("train", "calibration", "test")
EVIDENCE_CLASS = "kws-v2-research-rendered-dataset-cache-v1"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def source_audio(path_text: str, source_root: pathlib.Path, label: str) -> tuple[pathlib.Path, pathlib.Path]:
    raw = pathlib.Path(path_text)
    path = raw.resolve() if raw.is_absolute() else (source_root / raw).resolve()
    if not path.is_file():
        raise ValueError(f"{label}: missing WAV {path}")
    try:
        relative = path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"{label}: audio escapes rendered dataset root: {path}") from exc
    if not relative.parts or relative.parts[0] != "clips":
        raise ValueError(f"{label}: expected rendered audio under clips/: {relative}")
    return path, relative


def inspect_wav(path: pathlib.Path) -> None:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != 16000
            or reader.getcomptype() != "NONE"
            or reader.getnframes() <= 0
        ):
            raise ValueError(f"invalid rendered WAV: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a portable research rendered-dataset cache."
    )
    parser.add_argument("--source-dataset", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    source = args.source_dataset.resolve()
    output = args.output_dir.resolve()
    if not source.is_dir():
        raise ValueError(f"source dataset missing: {source}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output-dir must be empty")
    output.mkdir(parents=True, exist_ok=True)

    summary_path = source / "domain-summary.json"
    domain_index = source / "domain-index.jsonl"
    if not summary_path.is_file() or not domain_index.is_file():
        raise ValueError("rendered dataset is missing domain summary/index")
    summary = load_object(summary_path)
    config_sha256 = str(summary.get("config_sha256", ""))
    if len(config_sha256) != 64:
        raise ValueError("rendered domain summary lacks config_sha256")
    expected_index_sha = str(summary.get("domain_index_sha256", ""))
    actual_index_sha = sha256_file(domain_index)
    if expected_index_sha != actual_index_sha:
        raise ValueError("rendered domain index hash does not match domain summary")

    copied: dict[pathlib.Path, str] = {}
    split_receipts: dict[str, dict] = {}
    all_wav_hashes: set[str] = set()

    def ensure_copied(src: pathlib.Path, relative: pathlib.Path) -> str:
        digest = sha256_file(src)
        prior = copied.get(relative)
        if prior is not None:
            if prior != digest:
                raise ValueError(f"portable path collision with different bytes: {relative}")
            return digest
        inspect_wav(src)
        dst = output / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha256_file(dst) != digest:
            raise ValueError(f"WAV changed while materializing cache: {relative}")
        copied[relative] = digest
        all_wav_hashes.add(digest)
        return digest

    for split in SPLITS:
        source_tsv = source / f"{split}.tsv"
        if not source_tsv.is_file():
            raise ValueError(f"rendered split manifest missing: {source_tsv}")
        portable_rows: list[str] = []
        recordings = 0
        for line_no, raw in enumerate(source_tsv.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            if "\t" not in raw:
                raise ValueError(f"{source_tsv}:{line_no}: expected WAV<TAB>targets")
            audio_text, targets = raw.split("\t", 1)
            src, relative = source_audio(audio_text.strip(), source, f"{source_tsv}:{line_no}")
            ensure_copied(src, relative)
            portable_rows.append(f"{relative.as_posix()}\t{targets}\n")
            recordings += 1
        if recordings <= 0:
            raise ValueError(f"rendered split is empty: {split}")
        portable_tsv = output / f"{split}.tsv"
        portable_tsv.write_text("".join(portable_rows), encoding="utf-8")

        receipt = {
            "recordings": recordings,
            "manifest": portable_tsv.name,
            "manifest_sha256": sha256_file(portable_tsv),
        }

        source_refs = source / f"{split}.references.jsonl"
        if source_refs.is_file():
            rewritten: list[dict] = []
            seen_recordings: set[str] = set()
            for index, row in enumerate(read_jsonl(source_refs), 1):
                recording = str(row.get("recording", ""))
                path_text = row.get("path", row.get("audio_path"))
                if not recording or recording in seen_recordings or not isinstance(path_text, str):
                    raise ValueError(f"{source_refs}:{index}: invalid reference identity")
                src, relative = source_audio(path_text, source, f"{source_refs}:{index}")
                ensure_copied(src, relative)
                value = dict(row)
                value["path"] = relative.as_posix()
                value.pop("audio_path", None)
                rewritten.append(value)
                seen_recordings.add(recording)
            portable_refs = output / f"{split}.references.jsonl"
            portable_refs.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in rewritten
                ),
                encoding="utf-8",
            )
            receipt["references"] = portable_refs.name
            receipt["references_sha256"] = sha256_file(portable_refs)
            receipt["reference_recordings"] = len(rewritten)
        split_receipts[split] = receipt

    receipt = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "protected_evidence_used": False,
        "qualification_split_consumed": False,
        "source_config_sha256": config_sha256,
        "source_domain_index_sha256": actual_index_sha,
        "source_domain_summary_sha256": sha256_file(summary_path),
        "splits": split_receipts,
        "unique_wav_files": len(copied),
        "unique_wav_sha256": len(all_wav_hashes),
        "portable_paths_only": True,
    }
    target = output / "cache-receipt.json"
    target.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"research-rendered-cache: config={config_sha256} "
        f"wav={len(copied)} unique={len(all_wav_hashes)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, wave.Error, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
