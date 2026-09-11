from __future__ import annotations

import json
import pathlib


def read_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify(config_path: pathlib.Path, shipping_path: pathlib.Path, keywords_path: pathlib.Path) -> dict:
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

    rows = [
        raw.split("\t")
        for raw in keywords_path.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]
    expected = [
        ["1", "你好小窝", "0.55", "ni3 hao3 xiao3 wo1"],
        ["2", "小窝小窝", "0.55", "xiao3 wo1 xiao3 wo1"],
    ]
    if rows != expected:
        raise ValueError(f"shipping wake-word contract drifted: {rows!r}")
    if any(len(row[1]) != 4 for row in rows) or any(row[1] == "小窝" for row in rows):
        raise ValueError("shipping wake words must remain exactly the two four-character phrases")
    return {
        "active_formal_seed": active,
        "frozen_formal_seed": frozen,
        "reserved_formal_seed": reserved,
        "retired_seed_count": len(retired),
        "shipping_wake_words": [row[1] for row in rows],
    }


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    result = verify(
        root / "configs/training/xiaowo.torch-domain.json",
        root / "configs/shipping.xiaowo.json",
        root / "keywords/zh_cn_example.tsv",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
