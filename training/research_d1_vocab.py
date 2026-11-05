"""D1's predeclared 2601-output vocabulary; no model or data inspection.

The original token file is retained byte for byte. Its decoder-only symbols
are not output classes: raw IDs 1..2599 map to outputs 0..2598, including the
sil/<blk> alias at zero. New characters use a separate derived output map.
Observed development text never participates in vocabulary construction.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA = "kws-d1-fulltext-vocabulary-v2"
BASE_TOKENS_SHA256 = "5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa"
BASE_OUTPUTS = 2599
OUTPUTS = 2601
APPENDED = ("沃", "灯")
TASK_TEXTS = (
    "你好小窝", "小窝小窝", "小窝", "今天晚上吃什么", "请把灯打开",
    "我想听一首音乐", "现在外面下雨了", "你好小屋", "小沃小沃",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def make_vocabulary(original_tokens):
    require(type(original_tokens) is bytes and
            hashlib.sha256(original_tokens).hexdigest() == BASE_TOKENS_SHA256,
            "D1_OFFICIAL_TOKEN_BYTES_CHANGED")
    base = {}
    for line in original_tokens.decode("utf-8").splitlines():
        token, number = line.rsplit(maxsplit=1)
        output = int(number) - 1
        if 0 <= output < BASE_OUTPUTS:
            require(token not in base, "D1_DUPLICATE_BASE_SYMBOL")
            base[token] = output
    require(set(base.values()) == set(range(BASE_OUTPUTS)) and
            base.get("<blk>") == base.get("sil") == 0,
            "D1_BASE_OUTPUT_IDS_CHANGED")
    missing = tuple(sorted({char for text in TASK_TEXTS for char in text
                            if char not in base or base[char] <= 0}))
    require(missing == APPENDED, "D1_DECLARED_CHARACTER_EXTENSION_CHANGED")
    tokens = {**base, **{char: BASE_OUTPUTS + index for index, char in enumerate(missing)}}
    return {
        "schema": SCHEMA, "base_tokens_sha256": BASE_TOKENS_SHA256,
        "base_output_count": BASE_OUTPUTS, "output_count": OUTPUTS, "blank_id": 0,
        "base_mapping": "raw ID minus one, restricted to raw IDs 1..2599",
        "base_aliases": {"sil": 0, "<blk>": 0},
        "excluded_base_symbols": "raw ID 0 and raw IDs >=2600 are not model outputs",
        "basis": "predeclared-D1-task-text-only", "task_texts": list(TASK_TEXTS),
        "appended": [{"text": char, "output_id": tokens[char]} for char in missing],
        "token_to_id": tokens, "observed_development_text_used": False,
        "future_oov_policy": "unsupported-whole-transcript; never delete or map to blank",
    }


def validate_vocabulary(value, original_tokens):
    require(type(value) is dict and value == make_vocabulary(original_tokens),
            "D1_DERIVED_VOCABULARY_CHANGED")
    return value


def fulltext_targets(text, vocabulary):
    """Return the complete sequence or an explicit unsupported result."""
    require(vocabulary.get("schema") == SCHEMA and vocabulary.get("output_count") == OUTPUTS,
            "D1_VOCABULARY_SCHEMA")
    if not isinstance(text, str) or not text:
        return {"supported": False, "reason": "no_confirmed_full_transcript", "target_ids": None,
                "oov_characters": []}
    mapping = vocabulary["token_to_id"]
    missing = sorted({char for char in text if char not in mapping or mapping[char] <= 0})
    if missing:
        return {"supported": False, "reason": "full_transcript_out_of_vocabulary", "target_ids": None,
                "oov_characters": missing}
    ids = [mapping[char] for char in text]
    require(all(type(value) is int and 0 < value < OUTPUTS for value in ids),
            "D1_NONBLANK_TARGET_REQUIRED")
    return {"supported": True, "reason": "complete_full_transcript", "target_ids": ids,
            "oov_characters": []}
