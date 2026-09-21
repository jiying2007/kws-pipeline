from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from kws_vocab import load_tokens  # noqa: E402

IDENTITY_POLICY = "kws-keyword-set-identity-v1"
CONTRACT_POLICY = "kws-keyword-set-contract-v1"
UINT32_MAX = 0xFFFFFFFF


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _resolve(path: pathlib.Path, *, root: pathlib.Path = ROOT) -> pathlib.Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def semantic_keyword_rows(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[int] = set()
    seen_paths: set[tuple[str, ...]] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if not 4 <= len(cols) <= 8:
            raise ValueError(f"{path}:{line_no}: expected 4..8 TSV columns")
        keyword_id = int(cols[0])
        if keyword_id < 0 or keyword_id > UINT32_MAX:
            raise ValueError(f"{path}:{line_no}: keyword id must fit uint32")
        text = cols[1].strip()
        tokens = tuple(cols[3].split())
        if not text or not tokens:
            raise ValueError(f"{path}:{line_no}: keyword text/tokens must be non-empty")
        if keyword_id in seen_ids:
            raise ValueError(f"{path}:{line_no}: duplicate keyword id {keyword_id}")
        if tokens in seen_paths:
            raise ValueError(f"{path}:{line_no}: duplicate acoustic keyword path")
        seen_ids.add(keyword_id)
        seen_paths.add(tokens)
        rows.append({"id": keyword_id, "text": text, "tokens": list(tokens)})
    if not rows:
        raise ValueError(f"{path}: keyword set is empty")
    return sorted(rows, key=lambda row: int(row["id"]))


def build_keyword_set_identity(
    tokens_path: pathlib.Path,
    keywords_path: pathlib.Path,
) -> dict:
    tokens_path = tokens_path.resolve()
    keywords_path = keywords_path.resolve()
    token_map = load_tokens(tokens_path)
    token_rows = [
        {"id": int(token_id), "token": str(token)}
        for token, token_id in sorted(
            token_map.items(),
            key=lambda item: (int(item[1]), str(item[0])),
        )
    ]
    keyword_rows = semantic_keyword_rows(keywords_path)
    for row in keyword_rows:
        missing = [token for token in row["tokens"] if token not in token_map]
        if missing:
            raise ValueError(
                f"{keywords_path}: keyword {row['id']} uses unknown token(s): "
                + ", ".join(missing)
            )
        if any(int(token_map[token]) == 0 for token in row["tokens"]):
            raise ValueError(f"{keywords_path}: blank token cannot appear in a keyword")
    identity = {
        "schema_version": 1,
        "policy": IDENTITY_POLICY,
        "tokens": token_rows,
        "keywords": keyword_rows,
    }
    return {
        **identity,
        "keyword_set_sha256": _canonical_sha256(identity),
        "keyword_count": len(keyword_rows),
    }


def verify_keyword_set_contract(
    contract_path: pathlib.Path,
    *,
    tokens_path: pathlib.Path | None = None,
    keywords_path: pathlib.Path | None = None,
    root: pathlib.Path = ROOT,
) -> dict:
    contract_path = _resolve(contract_path, root=root)
    contract = load_object(contract_path)
    if (
        int(contract.get("schema_version", 0)) != 1
        or contract.get("policy") != CONTRACT_POLICY
    ):
        raise ValueError("keyword-set contract identity mismatch")
    contract_tokens = _resolve(
        pathlib.Path(str(contract.get("tokens_path") or "")),
        root=root,
    )
    contract_keywords = _resolve(
        pathlib.Path(str(contract.get("keywords_path") or "")),
        root=root,
    )
    actual_tokens = contract_tokens if tokens_path is None else tokens_path.resolve()
    actual_keywords = (
        contract_keywords if keywords_path is None else keywords_path.resolve()
    )
    if actual_tokens != contract_tokens:
        raise ValueError("keyword-set tokens path differs from contract")
    if actual_keywords != contract_keywords:
        raise ValueError("keyword-set keywords path differs from contract")
    identity = build_keyword_set_identity(actual_tokens, actual_keywords)
    expected = str(contract.get("keyword_set_sha256") or "")
    if identity["keyword_set_sha256"] != expected:
        raise ValueError(
            "keyword-set semantic identity mismatch: "
            f"{identity['keyword_set_sha256']} != {expected}"
        )
    expected_count = int(contract.get("keyword_count", -1))
    if expected_count != int(identity["keyword_count"]):
        raise ValueError("keyword-set keyword_count differs from contract")
    return {
        **identity,
        "contract_path": str(contract_path),
        "contract_id": str(contract.get("contract_id") or ""),
        "locale": str(contract.get("locale") or ""),
    }
