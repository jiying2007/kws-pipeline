#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))

from keyword_set_contract import verify_keyword_set_contract  # noqa: E402
from speech_like_corpus_plan import normalize_plan  # noqa: E402


def main() -> int:
    current_contract = ROOT / "configs/training/xiaowo-keyword-set-v1.json"
    current_plan = ROOT / "configs/training/speech-like-corpus-plan-v1.json"
    identity = verify_keyword_set_contract(current_contract)
    plan = normalize_plan(current_plan)
    positives = [
        row for row in plan["utterances"] if row["kind"] == "positive"
    ]
    assert {int(row["keyword_id"]) for row in positives} == set(
        identity["keyword_ids"]
    )
    for keyword in identity["keywords"]:
        assert any(
            int(row["keyword_id"]) == int(keyword["id"])
            and row["text"] == keyword["text"]
            and row["tokens"] == keyword["tokens"]
            for row in positives
        )

    with tempfile.TemporaryDirectory(prefix="keyword-set-contract-") as tmp:
        root = pathlib.Path(tmp)
        (root / "keywords").mkdir()
        tokens = root / "keywords/tokens.txt"
        keywords = root / "keywords/single.tsv"
        contract = root / "single-contract.json"
        plan_path = root / "single-plan.json"
        tokens.write_text("<blk> 0\na 1\nb 2\n", encoding="utf-8")
        keywords.write_text(
            "0\t小窝\t0.55\ta b\n",
            encoding="utf-8",
        )
        contract.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "policy": "kws-keyword-set-contract-v1",
                    "contract_id": "single-wake-id-zero-v1",
                    "locale": "zh-CN",
                    "tokens_path": "keywords/tokens.txt",
                    "keywords_path": "keywords/single.tsv",
                    "keywords": [
                        {
                            "id": 0,
                            "text": "小窝",
                            "threshold": 0.55,
                            "tokens": ["a", "b"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        single = verify_keyword_set_contract(
            contract,
            root=root,
            expected_tokens_path=tokens,
            expected_keywords_path=keywords,
        )
        assert single["keyword_ids"] == [0]
        assert single["keyword_count"] == 1

        plan_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "policy": "speech-like-corpus-plan-v1",
                    "locale": "zh-CN",
                    "split_roles": {
                        "train": {
                            "provider_group": "train",
                            "voice_slots": ["train-00"],
                        },
                        "calibration": {
                            "provider_group": "generalization-search",
                            "voice_slots": ["cal-00"],
                        },
                        "test": {
                            "provider_group": "generalization-search",
                            "voice_slots": ["test-00"],
                        },
                        "qualification": {
                            "provider_group": "generalization-freeze",
                            "voice_slots": ["qual-00"],
                        },
                    },
                    "utterances": [
                        {
                            "id": "kw0-exact",
                            "kind": "positive",
                            "keyword_id": 0,
                            "text": "小窝",
                            "tokens": ["a", "b"],
                        },
                        {
                            "id": "kw0-pause",
                            "kind": "positive",
                            "keyword_id": 0,
                            "text": "小。窝",
                            "tokens": ["a", "b"],
                        },
                    ],
                    "constraints": {
                        "require_unique_voice_id_across_splits": True,
                        "require_unique_source_id_per_request": True,
                        "tone_backend_allowed": False,
                        "protected_evidence_allowed": False,
                        "deterministic_pause_separator": "。",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        normalized = normalize_plan(plan_path)
        assert {
            int(row["keyword_id"])
            for row in normalized["utterances"]
            if row["kind"] == "positive"
        } == {0}

        tampered = json.loads(contract.read_text(encoding="utf-8"))
        tampered["keywords"][0]["text"] = "不是小窝"
        contract.write_text(
            json.dumps(tampered, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            verify_keyword_set_contract(
                contract,
                root=root,
                expected_tokens_path=tokens,
                expected_keywords_path=keywords,
            )
        except ValueError as exc:
            assert "semantic content differs" in str(exc)
        else:
            raise AssertionError("keyword contract semantic drift was accepted")

    print("keyword-set contract and corpus-plan binding: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
