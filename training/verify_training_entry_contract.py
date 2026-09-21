from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from keyword_set_identity import verify_keyword_set_contract  # noqa: E402


def read_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify(
    config_path: pathlib.Path,
    shipping_path: pathlib.Path,
    keywords_path: pathlib.Path,
    *,
    require_product_speech_like_base: bool = False,
) -> dict:
    config = read_json(config_path)
    shipping = read_json(shipping_path)
    active = int(config["qualification_holdout_seed"])
    retired = [int(value) for value in config["retired_qualification_holdout_seeds"]]
    model = shipping["model"]
    policy = shipping["nightly_policy"]
    frozen = int(model["qualification_seed"])
    state = str(model["qualification_seed_state"])
    reserved = int(policy["next_formal_candidate_seed_reserved"])
    if state != "consumed-and-frozen":
        raise ValueError(f"unexpected shipping qualification seed state: {state!r}")
    if active == frozen:
        raise ValueError(f"formal qualification seed {active} is already consumed/frozen")
    if frozen not in retired:
        raise ValueError(f"consumed qualification seed {frozen} must be retired")
    if active != reserved:
        raise ValueError(f"formal qualification seed must equal reserved seed {reserved}; got {active}")
    if active in retired:
        raise ValueError(f"active qualification seed {active} is already retired")

    keyword_contract_raw = str(config.get("keyword_set_contract") or "")
    if not keyword_contract_raw:
        raise ValueError("training config keyword_set_contract is required")
    tokens_raw = str(config.get("tokens") or "")
    if not tokens_raw:
        raise ValueError("training config tokens path is required")
    tokens_path = pathlib.Path(tokens_raw)
    if not tokens_path.is_absolute():
        tokens_path = (ROOT / tokens_path).resolve()
    keyword_contract_path = pathlib.Path(keyword_contract_raw)
    if not keyword_contract_path.is_absolute():
        keyword_contract_path = (ROOT / keyword_contract_path).resolve()
    keyword_identity = verify_keyword_set_contract(
        keyword_contract_path,
        tokens_path=tokens_path,
        keywords_path=keywords_path.resolve(),
    )

    shipping_rows = shipping.get("shipping_wake_words")
    if not isinstance(shipping_rows, list) or not shipping_rows:
        raise ValueError("shipping_wake_words must be a non-empty list")
    shipping_semantic = sorted(
        [
            {
                "id": int(row["id"]),
                "text": str(row["text"]),
                "tokens": [str(value) for value in row["tokens"]],
            }
            for row in shipping_rows
            if isinstance(row, dict)
        ],
        key=lambda row: int(row["id"]),
    )
    if len(shipping_semantic) != len(shipping_rows):
        raise ValueError("shipping_wake_words entries must be objects")
    if shipping_semantic != keyword_identity["keywords"]:
        raise ValueError(
            "shipping wake-word semantics differ from keyword-set contract"
        )

    product_data = None
    if require_product_speech_like_base:
        product_data = config.get("product_candidate_data")
        if not isinstance(product_data, dict):
            raise ValueError("governed product training requires product_candidate_data")
        if product_data.get("policy") != "external-speech-like-product-base-v1":
            raise ValueError("product candidate data policy mismatch")
        if product_data.get("tone_fallback_allowed") is not False:
            raise ValueError("tone fallback is forbidden for governed product candidate training")
        if product_data.get("protected_evidence_used") is not False:
            raise ValueError("protected evidence may not feed governed candidate training")
        for field in ("external_base_bundle_sha256", "provider_identity_sha256"):
            value = str(product_data.get(field) or "")
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"product_candidate_data.{field} must be lowercase SHA256")
        generator = config.get("generator")
        if not isinstance(generator, dict):
            raise ValueError("governed product training generator must be an object")
        tts = generator.get("tts")
        if not isinstance(tts, dict) or tts.get("backend") != "command":
            raise ValueError("governed product training replay TTS must use command backend")
        command = tts.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError("governed product training replay TTS command is missing")
        profiles = tts.get("speaker_profiles")
        if not isinstance(profiles, list) or len(profiles) != int(
            product_data.get("replay_train_voice_slots", -1)
        ):
            raise ValueError("governed product training replay speaker profiles drifted")
        if tts.get("replay_voice_scope") != "train-only":
            raise ValueError("governed product replay may use only train voice slots")
        provider_identity = str(tts.get("provider_identity_sha256") or "")
        if provider_identity != str(product_data.get("replay_provider_identity_sha256") or ""):
            raise ValueError("governed product replay provider identity drifted")
        if provider_identity != str(product_data.get("provider_identity_sha256") or ""):
            raise ValueError("replay provider must match product base provider identity")
        if product_data.get("replay_tone_allowed") is not False:
            raise ValueError("tone replay is forbidden for governed product candidate training")
        if str(product_data.get("keyword_set_contract") or "") != keyword_contract_raw:
            raise ValueError("product candidate keyword-set contract drifted")
        if str(product_data.get("keyword_set_sha256") or "") != str(
            keyword_identity["keyword_set_sha256"]
        ):
            raise ValueError("product candidate keyword-set identity drifted")
        if int(product_data.get("keyword_count", -1)) != int(
            keyword_identity["keyword_count"]
        ):
            raise ValueError("product candidate keyword_count drifted")
        external = generator.get("external_base_dataset")
        required_splits = {"train", "calibration", "test", "qualification"}
        if not isinstance(external, dict) or set(external) != required_splits:
            raise ValueError("governed product training requires all four external-base splits")
        for split in sorted(required_splits):
            row = external[split]
            if not isinstance(row, dict):
                raise ValueError(f"external base {split} must be an object")
            for field in ("index", "summary"):
                if not str(row.get(field) or ""):
                    raise ValueError(f"external base {split}.{field} is required")
            for field in ("index_sha256", "summary_sha256"):
                value = str(row.get(field) or "")
                if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                    raise ValueError(f"external base {split}.{field} must be lowercase SHA256")

    return {
        "active_formal_seed": active,
        "frozen_formal_seed": frozen,
        "reserved_formal_seed": reserved,
        "retired_seed_count": len(retired),
        "shipping_wake_words": [
            str(row["text"]) for row in keyword_identity["keywords"]
        ],
        "keyword_set_sha256": str(keyword_identity["keyword_set_sha256"]),
        "keyword_count": int(keyword_identity["keyword_count"]),
        "product_speech_like_base_required": require_product_speech_like_base,
        "product_external_base_bundle_sha256": (
            str(product_data["external_base_bundle_sha256"]) if product_data else None
        ),
    }


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=root / "configs/training/xiaowo.torch-domain.json",
    )
    parser.add_argument(
        "--shipping",
        type=pathlib.Path,
        default=root / "configs/shipping.xiaowo.json",
    )
    parser.add_argument(
        "--keywords",
        type=pathlib.Path,
        default=root / "keywords/zh_cn_example.tsv",
    )
    parser.add_argument("--require-product-speech-like-base", action="store_true")
    args = parser.parse_args()
    result = verify(
        args.config.resolve(),
        args.shipping.resolve(),
        args.keywords.resolve(),
        require_product_speech_like_base=args.require_product_speech_like_base,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
