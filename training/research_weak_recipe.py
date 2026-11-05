"""Fixed train-only weak-label consumption recipe; no optimizer or ML imports."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import random

CHAR_IDS = {"你": 1, "好": 2, "小": 3, "窝": 4}
TOKEN_NAMES = ["<blk>", "ni3", "hao3", "xiao3", "wo1"]
SEED = 20261105
EPOCHS = 3
BATCH_SIZE = 4
HEAD_EPOCHS = 1
OPTIMIZER = {"name": "AdamW", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}
C1 = {"hidden_dim": 64, "kernels": [3, 3, 3], "dilations": [1, 2, 4]}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def row_class(row):
    target = row["keyword_targets"]
    if target == [1, 0]:
        return "keyword-1-positive"
    if target == [0, 1]:
        return "keyword-2-positive"
    require(target == [0, 0], "this fixed D0 recipe requires single-keyword positives or known negatives")
    require(row.get("generation_category") == "near-negative" or
            row.get("intended_kind") in ("near-miss-intent", "hard-negative-intent"),
            "D0 negative source category must be explicitly nearphone; no inferred ordinary-negative pool")
    return "nearphone-negative"


def validate_row(row):
    require(row["consumer"] == "rnn-four-token" and row["purpose"] == "technical-research-only" and
            row["split"] == "train" and row["human_gold"] is False and row["independent_test"] is False,
            "only the explicit technical research train view is accepted")
    text, target = row["full_text"], row["target_ids"]
    require(type(text) is str and bool(text) and all(char in CHAR_IDS for char in text),
            "full text has OOV characters; OOV-to-blank is prohibited")
    require(type(target) is list and target == [CHAR_IDS[char] for char in text] and
            all(type(value) is int and 1 <= value <= 4 for value in target), "full-text CTC IDs differ")
    require(row["ctc_mask"] is True and type(row["ctc_weight"]) in (int, float) and
            math.isfinite(row["ctc_weight"]) and 0 < row["ctc_weight"] <= 1, "invalid weak CTC weight/mask")
    require(type(row["keyword_mask"]) is list and len(row["keyword_mask"]) == 2 and
            all(type(value) is bool for value in row["keyword_mask"]), "invalid keyword mask")
    require(type(row["keyword_weight"]) is list and len(row["keyword_weight"]) == 2, "invalid keyword weights")
    require(type(row["keyword_targets"]) is list and len(row["keyword_targets"]) == 2, "invalid keyword targets")
    for keyword, target, mask in zip(("你好小窝", "小窝小窝"), row["keyword_targets"], row["keyword_mask"]):
        require(not mask or (type(target) is int and target == int(keyword in text)),
                "known keyword target differs from the bound full text")
    for mask, weight in zip(row["keyword_mask"], row["keyword_weight"]):
        require(type(weight) in (int, float) and math.isfinite(weight) and 0 <= weight <= 1 and
                (mask or weight == 0), "masked keyword slots must have zero weight")
    require(all(type(row[key]) is str and row[key] for key in ("source_id", "speaker_id", "source_family_id", "item_id")),
            "source/speaker/item identity is required")
    return row_class(row)


def fraction_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator, "value": float(value)}


def make_recipe(rows, *, expected_items=40):
    require(len(rows) == expected_items and expected_items > 0, "unexpected snapshot item count")
    require(len({row["source_id"] for row in rows}) == len(rows) and
            len({row["item_id"] for row in rows}) == len(rows), "duplicate source/item identity")
    classes = [validate_row(row) for row in rows]
    names = ("keyword-1-positive", "keyword-2-positive", "nearphone-negative")
    masses = {name: sum((Fraction(str(row["ctc_weight"])) for row, kind in zip(rows, classes) if kind == name), Fraction()) for name in names}
    require(all(masses[name] > 0 for name in names), "both complete keywords and actual nearphone negatives are required")
    positive_each = min(masses[names[0]], masses[names[1]])
    positive_mass = 2 * positive_each
    near_mass = min(masses[names[2]], positive_mass / 9)
    scales = {names[0]: positive_each / masses[names[0]], names[1]: positive_each / masses[names[1]],
              names[2]: near_mass / masses[names[2]]}
    items = []
    for index, (row, kind) in enumerate(zip(rows, classes)):
        scale = scales[kind]
        items.append({"row_index": index, "item_id": row["item_id"], "source_id": row["source_id"],
                      "class": kind, "asr_ctc_weight": row["ctc_weight"], "class_multiplier": fraction_record(scale),
                      "effective_ctc_weight": float(Fraction(str(row["ctc_weight"])) * scale),
                      "keyword_mask": row["keyword_mask"], "asr_keyword_weight": row["keyword_weight"],
                      "effective_keyword_weight": [float(Fraction(str(w)) * scale) if mask else 0.0
                                                   for mask, w in zip(row["keyword_mask"], row["keyword_weight"])],
                      "label_binding_sha256": digest({key: value for key, value in row.items() if key != "audio_path"})})
    rng, order = random.Random(SEED), []
    for epoch in range(EPOCHS):
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        order.append(indices)
    ctc_normalizer = float((positive_mass + near_mass) / len(rows))
    keyword_normalizer = sum(sum(item["effective_keyword_weight"]) for item in items) / (2 * len(rows))
    require(keyword_normalizer > 0, "keyword probe needs known weighted targets")
    return {"schema": "d0-weak-consumption-recipe-v1", "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "head_probe_epochs": HEAD_EPOCHS, "head_seed": SEED + 1, "optimizer": OPTIMIZER,
            "gradient_clip_max_norm": 5.0, "ctc_blank_id": 0, "ctc_zero_infinity": False,
            "ctc_length_normalization": "divide each sequence NLL by its full target token count",
            "loss_denominator": "batch item count times fixed snapshot mean effective weight; never batch weight sum",
            "fixed_ctc_normalizer": ctc_normalizer, "fixed_keyword_normalizer": keyword_normalizer,
            "classes": {name: {"items": classes.count(name), "raw_asr_mass": fraction_record(masses[name]),
                                "multiplier": fraction_record(scales[name]), "effective_mass": fraction_record(masses[name] * scales[name])}
                        for name in names},
            "nearphone_fraction_of_effective_ctc_mass": fraction_record(near_mass / (positive_mass + near_mass)),
            "mass_definition": "fixed objective coefficient mass, not a guarantee about observed numeric loss contributions",
            "ordinary_negative_items": 0, "nearphone_exposure_replacement_experiment": False,
            "main_models": ["rnn64", "c1-ctc"], "c1": C1, "items": items, "epoch_row_order": order,
            "main_sample_exposures_per_model": len(rows) * EPOCHS,
            "main_exposure_count_per_source": dict(Counter(rows[index]["source_id"] for epoch in order for index in epoch)),
            "head_probe": "separate frozen C1 encoder and learned two-logit keyword head; excluded from main-pair comparison",
            "head_sample_exposures": len(rows), "source_roles": "train_only_no_independent_development",
            "human_gold": False, "quality_winner": None, "production_qualified": False,
            "training_set_readback_is_generalization": False}
