#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))

from keyword_set_contract import verify_keyword_set_contract  # noqa: E402
from speech_like_corpus_plan import normalize_plan  # noqa: E402

POLICY = "keyword-set-derived-speech-like-plan-v1"


def contains_subsequence(sequence: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    if not pattern:
        return True
    index = 0
    for token in sequence:
        if token == pattern[index]:
            index += 1
            if index == len(pattern):
                return True
    return False


def safe_nonwake(
    tokens: tuple[str, ...],
    wake_paths: list[tuple[str, ...]],
) -> bool:
    return bool(tokens) and not any(
        contains_subsequence(tokens, wake) for wake in wake_paths
    )


def deterministic_pause(text: str, separator: str) -> str | None:
    if len(text) < 2:
        return None
    split = max(1, len(text) // 2)
    if split >= len(text):
        return None
    return text[:split] + separator + text[split:]


def lexical_variants(
    *,
    text: str,
    tokens: list[str],
    wake_paths: list[tuple[str, ...]],
    limit: int,
) -> list[tuple[str, str, list[str]]]:
    if len(text) != len(tokens):
        raise ValueError(
            "automatic corpus-plan v1 requires one text codepoint per explicit token"
        )
    if limit <= 0:
        raise ValueError("lexical variant limit must be positive")
    raw: list[tuple[str, str, list[str]]] = []
    if len(tokens) > 1:
        raw.append(("prefix", text[:-1], tokens[:-1]))
        raw.append(("suffix", text[1:], tokens[1:]))
    if len(tokens) > 2:
        middle = len(tokens) // 2
        raw.append(
            (
                f"drop-{middle}",
                text[:middle] + text[middle + 1 :],
                tokens[:middle] + tokens[middle + 1 :],
            )
        )
    for index in range(max(0, len(tokens) - 1)):
        swapped_tokens = list(tokens)
        swapped_tokens[index], swapped_tokens[index + 1] = (
            swapped_tokens[index + 1],
            swapped_tokens[index],
        )
        chars = list(text)
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
        raw.append((f"swap-{index}", "".join(chars), swapped_tokens))
    if len(tokens) > 1:
        raw.append(("reverse", text[::-1], list(reversed(tokens))))

    result: list[tuple[str, str, list[str]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for reason, variant_text, variant_tokens in raw:
        target = tuple(variant_tokens)
        key = (variant_text, target)
        if (
            not variant_text
            or not safe_nonwake(target, wake_paths)
            or key in seen
        ):
            continue
        seen.add(key)
        result.append((reason, variant_text, variant_tokens))
        if len(result) >= limit:
            break
    return result


def generate_plan(
    *,
    keyword_contract: pathlib.Path,
    template_plan: pathlib.Path,
    output: pathlib.Path,
    lexical_variants_per_keyword: int = 6,
) -> dict:
    identity = verify_keyword_set_contract(keyword_contract)
    template = json.loads(template_plan.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError("corpus-plan template must be an object")
    normalized_template = normalize_plan(template_plan)
    separator = normalized_template["deterministic_pause_separator"]
    wake_paths = [
        tuple(str(token) for token in row["tokens"])
        for row in identity["keywords"]
    ]

    utterances: list[dict] = []
    seen_ids: set[str] = set()
    seen_nonwake: set[tuple[str, tuple[str, ...]]] = set()

    def append(row: dict) -> None:
        utterance_id = str(row["id"])
        if utterance_id in seen_ids:
            raise ValueError(f"duplicate generated utterance id: {utterance_id}")
        seen_ids.add(utterance_id)
        utterances.append(row)

    for keyword in identity["keywords"]:
        keyword_id = int(keyword["id"])
        text = str(keyword["text"])
        tokens = [str(token) for token in keyword["tokens"]]
        if len(text) != len(tokens):
            raise ValueError(
                f"keyword {keyword_id}: automatic corpus-plan v1 requires "
                "one text codepoint per explicit token"
            )
        append(
            {
                "id": f"kw{keyword_id}-exact",
                "kind": "positive",
                "keyword_id": keyword_id,
                "text": text,
                "tokens": tokens,
            }
        )
        paused = deterministic_pause(text, separator)
        if paused is not None:
            append(
                {
                    "id": f"kw{keyword_id}-pause",
                    "kind": "positive",
                    "keyword_id": keyword_id,
                    "text": paused,
                    "tokens": tokens,
                }
            )

        variants = lexical_variants(
            text=text,
            tokens=tokens,
            wake_paths=wake_paths,
            limit=lexical_variants_per_keyword,
        )
        for ordinal, (reason, variant_text, variant_tokens) in enumerate(variants):
            key = (variant_text, tuple(variant_tokens))
            if key in seen_nonwake:
                continue
            seen_nonwake.add(key)
            kind = "confusable" if reason in {"prefix", "suffix"} else "negative"
            append(
                {
                    "id": f"kw{keyword_id}-{kind}-{ordinal:02d}-{reason}",
                    "kind": kind,
                    "keyword_id": None,
                    "text": variant_text,
                    "tokens": variant_tokens,
                }
            )

    if not utterances:
        raise ValueError("generated corpus plan has no utterances")

    value = {
        "schema_version": 1,
        "policy": "speech-like-corpus-plan-v1",
        "locale": identity["locale"],
        "keyword_set_contract_id": identity["contract_id"],
        "keyword_set_semantic_sha256": identity["semantic_sha256"],
        "generation_policy": POLICY,
        "split_roles": copy.deepcopy(template["split_roles"]),
        "utterances": utterances,
        "constraints": copy.deepcopy(template["constraints"]),
        "notes": [
            "Generated deterministically from a versioned keyword-set contract.",
            "Non-positive lexical variants are rejected if they contain any configured wake path.",
            "Background/noise and acoustic-domain expansion remain the responsibility of the domain renderer.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    normalized = normalize_plan(output)
    return {
        "schema_version": 1,
        "policy": POLICY,
        "keyword_set_contract_id": identity["contract_id"],
        "keyword_set_semantic_sha256": identity["semantic_sha256"],
        "keyword_count": len(identity["keywords"]),
        "utterances": len(normalized["utterances"]),
        "positive_utterances": sum(
            row["kind"] == "positive" for row in normalized["utterances"]
        ),
        "nonpositive_utterances": sum(
            row["kind"] != "positive" for row in normalized["utterances"]
        ),
        "output": str(output.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword-set-contract", required=True, type=pathlib.Path)
    parser.add_argument(
        "--template-plan",
        type=pathlib.Path,
        default=ROOT / "configs/training/speech-like-corpus-plan-v1.json",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--lexical-variants-per-keyword", type=int, default=6)
    args = parser.parse_args()
    result = generate_plan(
        keyword_contract=args.keyword_set_contract.resolve(),
        template_plan=args.template_plan.resolve(),
        output=args.output.resolve(),
        lexical_variants_per_keyword=args.lexical_variants_per_keyword,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
