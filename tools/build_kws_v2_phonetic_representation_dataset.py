#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from kws_vocab import load_tokens  # noqa: E402

EVIDENCE_CLASS = "kws-v2-phonetic-representation-dataset-v1"
SPLITS = ("train", "calibration", "test")


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


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def parse_tsv(path: pathlib.Path) -> list[tuple[str, list[int]]]:
    rows: list[tuple[str, list[int]]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" not in raw:
            raise ValueError(f"{path}:{line_no}: expected WAV<TAB>token ids")
        audio, tokens = raw.split("\t", 1)
        if not audio.strip():
            raise ValueError(f"{path}:{line_no}: empty audio path")
        try:
            ids = [int(value) for value in tokens.split()] if tokens.strip() else []
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid token id") from exc
        rows.append((audio.strip(), ids))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def resolve_audio(root: pathlib.Path, raw: str, label: str) -> pathlib.Path:
    path = pathlib.Path(raw)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label}: missing audio {resolved}")
    return resolved


def inverse_tokens(mapping: dict[str, int]) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, value in mapping.items():
        if value in result:
            raise ValueError("token inventory contains duplicate ids")
        result[int(value)] = str(name)
    return result


def canonical_rows(
    *,
    split: str,
    cache_root: pathlib.Path,
    source_tokens: dict[str, int],
    target_tokens: dict[str, int],
    wake_targets: set[tuple[int, ...]],
) -> tuple[list[dict], dict]:
    source_inverse = inverse_tokens(source_tokens)
    manifest = cache_root / f"{split}.tsv"
    rows: list[dict] = []
    wake = 0
    nonwake = 0
    for index, (audio_raw, source_ids) in enumerate(parse_tsv(manifest), 1):
        names: list[str] = []
        for token_id in source_ids:
            name = source_inverse.get(token_id)
            if name is None or name == "<blk>":
                raise ValueError(f"{manifest}:{index}: invalid source token id {token_id}")
            if name not in target_tokens:
                raise ValueError(f"{manifest}:{index}: target vocab missing token {name}")
            names.append(name)
        ids = [int(target_tokens[name]) for name in names]
        audio = resolve_audio(manifest.parent, audio_raw, f"{manifest}:{index}")
        is_wake = tuple(ids) in wake_targets
        wake += int(is_wake)
        nonwake += int(not is_wake)
        rows.append(
            {
                "audio": str(audio),
                "tokens": names,
                "target_ids": ids,
                "source": "canonical",
                "utterance_id": None,
                "wake": is_wake,
            }
        )
    return rows, {
        "recordings": len(rows),
        "wake_recordings": wake,
        "nonwake_recordings": nonwake,
        "manifest_sha256": sha256_file(manifest),
    }


def sidecar_rows(
    *,
    split: str,
    sidecar_root: pathlib.Path,
    annotations: dict,
    target_tokens: dict[str, int],
    wake_targets: set[tuple[int, ...]],
) -> tuple[list[dict], dict]:
    split_root = sidecar_root / split
    index_path = split_root / "index.jsonl"
    rows: list[dict] = []
    wake = 0
    nonwake = 0
    seen_sources: set[str] = set()
    for ordinal, source in enumerate(load_jsonl(index_path), 1):
        utterance_id = str(source.get("utterance_id", "")).strip()
        source_id = str(source.get("source_id", "")).strip()
        if not utterance_id or not source_id or source_id in seen_sources:
            raise ValueError(f"{index_path}:{ordinal}: invalid sidecar identity")
        seen_sources.add(source_id)
        raw_tokens = annotations.get(utterance_id)
        if not isinstance(raw_tokens, list) or not raw_tokens:
            raise ValueError(f"{index_path}:{ordinal}: missing phonetic label for {utterance_id}")
        names = [str(value) for value in raw_tokens]
        missing = [name for name in names if name not in target_tokens]
        if missing:
            raise ValueError(
                f"{index_path}:{ordinal}: target vocab missing {sorted(set(missing))}"
            )
        ids = [int(target_tokens[name]) for name in names]
        audio_raw = source.get("path")
        expected_sha = str(source.get("file_sha256", ""))
        if not isinstance(audio_raw, str) or not audio_raw:
            raise ValueError(f"{index_path}:{ordinal}: missing audio path")
        audio = resolve_audio(split_root, audio_raw, f"{index_path}:{ordinal}")
        if sha256_file(audio) != expected_sha:
            raise ValueError(f"{index_path}:{ordinal}: WAV hash mismatch")
        is_wake = tuple(ids) in wake_targets
        wake += int(is_wake)
        nonwake += int(not is_wake)
        rows.append(
            {
                "audio": str(audio),
                "tokens": names,
                "target_ids": ids,
                "source": "ordinary-sidecar",
                "utterance_id": utterance_id,
                "source_id": source_id,
                "wake": is_wake,
            }
        )
    return rows, {
        "recordings": len(rows),
        "wake_recordings": wake,
        "nonwake_recordings": nonwake,
        "index_sha256": sha256_file(index_path),
    }


def rewrite_references(
    *,
    split: str,
    canonical_root: pathlib.Path,
    sidecar_root: pathlib.Path,
    output: pathlib.Path,
) -> dict:
    sources = [
        ("canonical", canonical_root, canonical_root / f"{split}.references.jsonl"),
        (
            "ordinary-sidecar",
            sidecar_root / split,
            sidecar_root / split / "negative.references.jsonl",
        ),
    ]
    rows: list[dict] = []
    seen_recordings: set[str] = set()
    source_counts: dict[str, int] = {}
    for source_name, root, path in sources:
        if not path.is_file():
            raise ValueError(f"reference source missing: {path}")
        count = 0
        for line_no, row in enumerate(load_jsonl(path), 1):
            recording = str(row.get("recording", "")).strip()
            raw_audio = row.get("audio_path") or row.get("path")
            if not recording or recording in seen_recordings or not isinstance(raw_audio, str):
                raise ValueError(f"{path}:{line_no}: invalid reference identity")
            audio = resolve_audio(root, raw_audio, f"{path}:{line_no}")
            value = dict(row)
            value["path"] = str(audio)
            value.pop("audio_path", None)
            domain = value.get("domain")
            if not isinstance(domain, dict):
                domain = {}
            domain = dict(domain)
            domain["representation_source"] = source_name
            value["domain"] = domain
            rows.append(value)
            seen_recordings.add(recording)
            count += 1
        source_counts[source_name] = count
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return {
        "recordings": len(rows),
        "source_recordings": source_counts,
        "sha256": sha256_file(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Relabel frozen research audio with a broader tone-pinyin vocabulary."
    )
    parser.add_argument("--canonical-cache", required=True, type=pathlib.Path)
    parser.add_argument("--ordinary-sidecar", required=True, type=pathlib.Path)
    parser.add_argument("--source-tokens", required=True, type=pathlib.Path)
    parser.add_argument("--target-tokens", required=True, type=pathlib.Path)
    parser.add_argument("--target-keywords", required=True, type=pathlib.Path)
    parser.add_argument("--labels", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    canonical_root = args.canonical_cache.resolve()
    sidecar_root = args.ordinary_sidecar.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output-dir must be empty")
    output.mkdir(parents=True, exist_ok=True)

    cache_receipt = load_object(canonical_root / "cache-receipt.json")
    sidecar_receipt = load_object(sidecar_root / "sidecar-receipt.json")
    labels_doc = load_object(args.labels.resolve())
    if (
        cache_receipt.get("evidence_class")
        != "kws-v2-research-rendered-dataset-cache-v1"
        or cache_receipt.get("evidence_scope") != "research-only"
        or cache_receipt.get("qualification_split_consumed") is not False
    ):
        raise ValueError("canonical cache evidence contract mismatch")
    if (
        sidecar_receipt.get("evidence_class")
        != "kws-v2-research-negative-sidecar-v1"
        or sidecar_receipt.get("evidence_scope") != "research-only"
        or sidecar_receipt.get("qualification_split_consumed") is not False
    ):
        raise ValueError("ordinary sidecar evidence contract mismatch")
    if (
        labels_doc.get("policy") != "kws-v2-phonetic-labels-v1"
        or labels_doc.get("evidence_scope") != "research-only"
    ):
        raise ValueError("phonetic labels policy mismatch")
    annotations = labels_doc.get("ordinary_utterances")
    if not isinstance(annotations, dict) or len(annotations) != 32:
        raise ValueError("phonetic labels must cover exactly 32 ordinary utterances")

    source_tokens = load_tokens(args.source_tokens.resolve())
    target_tokens = load_tokens(args.target_tokens.resolve())
    if len(target_tokens) != 72:
        raise ValueError(f"phonetic-v2 vocabulary must contain 72 tokens, got {len(target_tokens)}")
    for name, expected in (("ni3", 1), ("hao3", 2), ("xiao3", 3), ("wo1", 4)):
        if int(target_tokens.get(name, -1)) != expected:
            raise ValueError(f"phonetic-v2 must preserve {name} as id {expected}")

    keyword_rows: list[tuple[int, ...]] = []
    for line_no, raw in enumerate(
        args.target_keywords.resolve().read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) < 4:
            raise ValueError(f"{args.target_keywords}:{line_no}: malformed keyword row")
        names = cols[3].split()
        keyword_rows.append(tuple(int(target_tokens[name]) for name in names))
    wake_targets = set(keyword_rows)
    if len(wake_targets) != 2:
        raise ValueError("phonetic-v2 experiment requires exactly two wake sequences")

    split_receipts: dict[str, dict] = {}
    total_wake = total_nonwake = 0
    for split in SPLITS:
        canonical, canonical_receipt = canonical_rows(
            split=split,
            cache_root=canonical_root,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            wake_targets=wake_targets,
        )
        sidecar, sidecar_split_receipt = sidecar_rows(
            split=split,
            sidecar_root=sidecar_root,
            annotations=annotations,
            target_tokens=target_tokens,
            wake_targets=wake_targets,
        )
        rows = canonical + sidecar
        manifest = output / f"{split}.tsv"
        manifest.write_text(
            "".join(
                f"{row['audio']}\t{' '.join(str(value) for value in row['target_ids'])}\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        wake = sum(int(row["wake"]) for row in rows)
        nonwake = len(rows) - wake
        total_wake += wake
        total_nonwake += nonwake
        split_receipt = {
            "recordings": len(rows),
            "wake_recordings": wake,
            "nonwake_recordings": nonwake,
            "canonical": canonical_receipt,
            "ordinary_sidecar": sidecar_split_receipt,
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
        }
        if split != "train":
            refs = output / f"{split}.references.jsonl"
            split_receipt["references"] = rewrite_references(
                split=split,
                canonical_root=canonical_root,
                sidecar_root=sidecar_root,
                output=refs,
            )
        split_receipts[split] = split_receipt

    receipt = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "protected_evidence_used": False,
        "qualification_split_consumed": False,
        "shipping_metric": False,
        "source_cache_receipt_sha256": sha256_file(
            canonical_root / "cache-receipt.json"
        ),
        "source_sidecar_receipt_sha256": sha256_file(
            sidecar_root / "sidecar-receipt.json"
        ),
        "labels_sha256": sha256_file(args.labels.resolve()),
        "source_tokens_sha256": sha256_file(args.source_tokens.resolve()),
        "target_tokens_sha256": sha256_file(args.target_tokens.resolve()),
        "target_keywords_sha256": sha256_file(args.target_keywords.resolve()),
        "target_vocab_size": len(target_tokens),
        "wake_token_ids_preserved": {
            "ni3": 1,
            "hao3": 2,
            "xiao3": 3,
            "wo1": 4,
        },
        "splits": split_receipts,
        "total_wake_recordings": total_wake,
        "total_nonwake_recordings": total_nonwake,
    }
    target = output / "representation-receipt.json"
    target.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "phonetic-representation-dataset: "
        f"vocab={receipt['target_vocab_size']} "
        f"train={split_receipts['train']['recordings']} "
        f"cal={split_receipts['calibration']['recordings']} "
        f"test={split_receipts['test']['recordings']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
