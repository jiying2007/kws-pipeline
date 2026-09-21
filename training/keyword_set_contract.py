from __future__ import annotations

import hashlib
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = "kws-keyword-set-contract-v1"
UINT32_MAX = 0xFFFFFFFF


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def resolve_repo_path(value: object, *, root: pathlib.Path, label: str) -> pathlib.Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} path is missing")
    path = pathlib.Path(text)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes repository root: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} file is missing: {resolved}")
    return resolved


def parse_tokens(path: pathlib.Path) -> dict[str, int]:
    result: dict[str, int] = {}
    ids: set[int] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        cols = value.split()
        if len(cols) == 1:
            token = cols[0]
            token_id = len(result)
        elif len(cols) == 2:
            token, raw_id = cols
            try:
                token_id = int(raw_id)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: token id is invalid") from exc
        else:
            raise ValueError(f"{path}:{line_no}: expected TOKEN [ID]")
        if not token or token in result or token_id < 0 or token_id in ids:
            raise ValueError(f"{path}:{line_no}: duplicate/invalid token")
        result[token] = token_id
        ids.add(token_id)
    if not result:
        raise ValueError("token vocabulary is empty")
    return result


def parse_keyword_tsv(path: pathlib.Path, token_map: dict[str, int]) -> list[dict]:
    result: list[dict] = []
    seen_ids: set[int] = set()
    seen_sequences: set[tuple[int, ...]] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) < 4 or len(cols) > 8:
            raise ValueError(f"{path}:{line_no}: expected 4..8 TSV columns")
        try:
            keyword_id = int(cols[0])
            threshold = float(cols[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid keyword id/threshold") from exc
        text = cols[1].strip()
        tokens = cols[3].split()
        if keyword_id < 0 or keyword_id > UINT32_MAX or keyword_id in seen_ids:
            raise ValueError(f"{path}:{line_no}: keyword id must be unique uint32")
        if not text or not tokens:
            raise ValueError(f"{path}:{line_no}: keyword text/tokens are required")
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(f"{path}:{line_no}: threshold must be in (0,1)")
        missing = [token for token in tokens if token not in token_map]
        if missing:
            raise ValueError(
                f"{path}:{line_no}: unknown token(s): {', '.join(missing)}"
            )
        sequence = tuple(token_map[token] for token in tokens)
        if any(token_id == 0 for token_id in sequence):
            raise ValueError(f"{path}:{line_no}: blank token cannot appear in keyword")
        if sequence in seen_sequences:
            raise ValueError(f"{path}:{line_no}: duplicate acoustic keyword path")
        result.append(
            {
                "id": keyword_id,
                "text": text,
                "threshold": threshold,
                "tokens": tokens,
            }
        )
        seen_ids.add(keyword_id)
        seen_sequences.add(sequence)
    if not result:
        raise ValueError("keyword TSV contains no keywords")
    return result


def normalize_contract_keyword(row: object, index: int) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"keyword contract entry {index} must be an object")
    keyword_id = row.get("id")
    if isinstance(keyword_id, bool) or not isinstance(keyword_id, int):
        raise ValueError(f"keyword contract entry {index}.id must be integer")
    if keyword_id < 0 or keyword_id > UINT32_MAX:
        raise ValueError(f"keyword contract entry {index}.id must fit uint32")
    text = str(row.get("text") or "").strip()
    if not text:
        raise ValueError(f"keyword contract entry {index}.text is required")
    threshold_raw = row.get("threshold")
    if isinstance(threshold_raw, bool) or not isinstance(threshold_raw, (int, float)):
        raise ValueError(f"keyword contract entry {index}.threshold must be numeric")
    threshold = float(threshold_raw)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError(f"keyword contract entry {index}.threshold must be in (0,1)")
    raw_tokens = row.get("tokens")
    if (
        not isinstance(raw_tokens, list)
        or not raw_tokens
        or any(not isinstance(token, str) or not token for token in raw_tokens)
    ):
        raise ValueError(f"keyword contract entry {index}.tokens must be strings")
    return {
        "id": keyword_id,
        "text": text,
        "threshold": threshold,
        "tokens": list(raw_tokens),
    }


def verify_keyword_set_contract(
    contract_path: pathlib.Path,
    *,
    root: pathlib.Path = ROOT,
    expected_tokens_path: pathlib.Path | None = None,
    expected_keywords_path: pathlib.Path | None = None,
) -> dict:
    root = root.resolve()
    contract_path = contract_path.resolve()
    value = load_object(contract_path)
    if int(value.get("schema_version", 0)) != 1 or value.get("policy") != POLICY:
        raise ValueError("keyword-set contract identity mismatch")
    contract_id = str(value.get("contract_id") or "").strip()
    locale = str(value.get("locale") or "").strip()
    if not contract_id or not locale:
        raise ValueError("keyword-set contract id/locale is missing")
    tokens_path = resolve_repo_path(
        value.get("tokens_path"),
        root=root,
        label="keyword-set tokens",
    )
    keywords_path = resolve_repo_path(
        value.get("keywords_path"),
        root=root,
        label="keyword-set keywords",
    )
    if expected_tokens_path is not None and tokens_path != expected_tokens_path.resolve():
        raise ValueError("training config tokens path differs from keyword-set contract")
    if expected_keywords_path is not None and keywords_path != expected_keywords_path.resolve():
        raise ValueError("training config keywords path differs from keyword-set contract")

    token_map = parse_tokens(tokens_path)
    actual_keywords = parse_keyword_tsv(keywords_path, token_map)
    raw_expected = value.get("keywords")
    if not isinstance(raw_expected, list) or not raw_expected:
        raise ValueError("keyword-set contract keywords must be non-empty")
    expected_keywords = [
        normalize_contract_keyword(row, index)
        for index, row in enumerate(raw_expected)
    ]
    if actual_keywords != expected_keywords:
        raise ValueError(
            "keyword TSV semantic content differs from keyword-set contract"
        )
    ids = [int(row["id"]) for row in expected_keywords]
    if len(set(ids)) != len(ids):
        raise ValueError("keyword-set contract contains duplicate keyword ids")

    semantic_identity = {
        "policy": POLICY,
        "contract_id": contract_id,
        "locale": locale,
        "keywords": expected_keywords,
    }
    return {
        "schema_version": 1,
        "policy": POLICY,
        "contract_id": contract_id,
        "locale": locale,
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "tokens_path": str(tokens_path),
        "tokens_sha256": sha256_file(tokens_path),
        "keywords_path": str(keywords_path),
        "keywords_sha256": sha256_file(keywords_path),
        "keyword_count": len(expected_keywords),
        "keyword_ids": ids,
        "semantic_sha256": canonical_sha256(semantic_identity),
        "keywords": expected_keywords,
    }
