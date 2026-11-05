"""D1 v2 same-cohort recipes and actual fixed-exposure accounting.

The retained data sampler controls the draw counts. This module projects full
transcripts into the fixed extended vocabulary, preserves unknown masks, and
records which rows can enter each objective. It never turns generation intent
into an observed label or invents a missing negative pool.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import math

from research_d1_vocab import OUTPUTS, digest, fulltext_targets, require

SCHEMA = "kws-d1-fixed-exposure-recipe-v2"
SEED = 20261105
TOTAL_EXPOSURES = 600
POSITIVE_EXPOSURES = 300
BATCH_SIZE = 4
KEYWORDS = ("你好小窝", "小窝小窝")


def finite_weight(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def row_identity(row):
    return digest({key: value for key, value in row.items() if key not in ("audio", "audio_path")})


def project_row(row, vocabulary):
    require(row["consumer"] == "d1-verified-full-transcript-v2" and
            row["purpose"] == "technical-research-only" and row["human_gold"] is False and
            row["independent_test"] is False and row["split"] in ("train", "development") and
            row["source_role"] == ("train-candidate" if row["split"] == "train" else "development-only") and
            row["global_source_key"] == [row["source_plan_sha256"], row["source_id"]], "D1_VERIFIED_ROW_SCOPE")
    require(type(row["ctc_mask"]) is bool and finite_weight(row["ctc_weight"]), "D1_CTC_LABEL_MASK")
    require(type(row["keyword_targets"]) is list and type(row["keyword_mask"]) is list and
            type(row["keyword_weight"]) is list and
            len(row["keyword_targets"]) == len(row["keyword_mask"]) == len(row["keyword_weight"]) == 2,
            "D1_KEYWORD_LABEL_SHAPE")
    confirmed = row["label_status"] == "machine_candidate"
    # A retained native generation failure is never promoted by recognizer
    # agreement. This is a consumer boundary, not a new weak-label decision.
    outcome = row.get("generation_outcome")
    require(outcome is None or outcome in ("completed", "failed-native"), "D1_GENERATION_OUTCOME")
    if outcome == "failed-native":
        require(not confirmed and row["label_status"] == "uncertain" and row["full_text"] is None and
                row["ctc_mask"] is False and row["ctc_weight"] == 0 and row["target_ids"] is None and
                row["keyword_targets"] == [None, None] and all(mask is False for mask in row["keyword_mask"]) and
                row["keyword_weight"] == [0, 0], "D1_FAILED_NATIVE_MUST_HAVE_ZERO_LOSS")
    require(confirmed == row["ctc_mask"] and (row["ctc_weight"] > 0 if confirmed else row["ctc_weight"] == 0),
            "D1_CONFIRMATION_MASK_MISMATCH")
    text = row["full_text"]
    require(not confirmed or type(text) is str and bool(text), "D1_CONFIRMED_FULL_TEXT_REQUIRED")
    for index, (target, mask, weight) in enumerate(zip(row["keyword_targets"], row["keyword_mask"], row["keyword_weight"])):
        require(type(mask) is bool and finite_weight(weight), "D1_KEYWORD_WEIGHT_MASK")
        if mask:
            require(confirmed and type(target) is int and target in (0, 1) and weight > 0 and
                    target == int(KEYWORDS[index] in text), "D1_KEYWORD_TARGET_DIFFERS_FROM_FULL_TEXT")
        else:
            require(target is None and weight == 0, "D1_UNKNOWN_KEYWORD_IS_NOT_NEGATIVE")
    category = row["generation_category"]
    require(category in ("full-positive", "ordinary-negative", "partial-negative", "near-negative"),
            "D1_GENERATION_CATEGORY")
    observed_positive = any(target == 1 and mask for target, mask in zip(row["keyword_targets"], row["keyword_mask"]))
    intended_keyword = row.get("intended_keyword_id")
    require(intended_keyword is None or type(intended_keyword) is int and intended_keyword in (1, 2),
            "D1_INTENDED_KEYWORD_ID")
    conflict = confirmed and (
        observed_positive != (category == "full-positive") or
        category == "full-positive" and intended_keyword is not None and row["keyword_targets"][intended_keyword - 1] != 1)
    require(type(row["samples"]) is int and row["samples"] > 0, "D1_AUDIO_SAMPLE_COUNT")
    projection = fulltext_targets(text if confirmed else None, vocabulary)
    frames = {"rnn64": max(0, 1 + (row["samples"] - 400) // 320),
              "c1": max(0, 1 + (row["samples"] - 400) // 320),
              "fsmn": max(0, 1 + (row["samples"] - 400) // 160) // 3}
    minimum = None
    if projection["supported"]:
        ids = projection["target_ids"]
        minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if any(value < minimum for value in frames.values()):
            projection.update(supported=False, reason="common_unpadded_ctc_geometry_infeasible", target_ids=None)
    if conflict:
        projection.update(supported=False, reason="generation_intent_weak_label_conflict", target_ids=None)
    if outcome == "failed-native":
        projection.update(supported=False, reason="failed_native_generation_audit_only", target_ids=None)
    ctc = confirmed and projection["supported"]
    keyword = confirmed and not conflict and any(row["keyword_mask"]) and frames["c1"] > 0
    return {**row, "consumer": "d1-objective-projection-v2", "vocabulary_sha256": digest(vocabulary),
            "label_ctc_mask": row["ctc_mask"], "label_ctc_weight": row["ctc_weight"],
            "label_keyword_targets": list(row["keyword_targets"]), "label_keyword_mask": list(row["keyword_mask"]),
            "label_keyword_weight": list(row["keyword_weight"]),
            "keyword_targets": [None, None] if conflict else list(row["keyword_targets"]),
            "keyword_mask": [False, False] if conflict else list(row["keyword_mask"]),
            "keyword_weight": [0.0, 0.0] if conflict else list(row["keyword_weight"]),
            "ctc_mask": ctc, "ctc_weight": row["ctc_weight"] if ctc else 0.0,
            "target_ids": projection["target_ids"] if ctc else None,
            "ctc_reason": projection["reason"], "oov_characters": projection["oov_characters"],
            "ctc_minimum_frames": minimum, "expected_output_frames": frames,
            "keyword_eligible": keyword, "same_cohort_ctc_eligible": ctc,
            "quarantined": conflict, "quarantine_reason": "generation_intent_weak_label_conflict" if conflict else None}


def project_rows(rows, vocabulary):
    require(type(rows) is list and 1 <= len(rows) <= 256 and
            len({tuple(row["global_source_key"]) for row in rows}) == len(rows), "D1_DISTINCT_GLOBAL_ROWS")
    return [project_row(row, vocabulary) for row in rows]


def _interleave(groups):
    """Spread each finite group's rows without duplicating a source identity."""
    names = sorted(groups)
    total = sum(len(groups[name]) for name in names)
    used = dict.fromkeys(names, 0)
    for position in range(total):
        available = [name for name in names if used[name] < len(groups[name])]
        name = max(available, key=lambda key: len(groups[key]) * (position + 1) - used[key] * total)
        yield groups[name][used[name]]
        used[name] += 1


def cohort_rows(rows, *, objective="ctc"):
    require(objective in ("ctc", "keyword"), "D1_OBJECTIVE")
    eligible = [row for row in rows if row["split"] == "train" and
                (row["ctc_mask"] if objective == "ctc" else row["keyword_eligible"])]
    # Source-family round robin remains owned by the existing sampler. A stable
    # finite interleave keeps both positive words and ordinary/partial prompts
    # spread within each family; it does not invent or duplicate pool members.
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in eligible:
        category = row["generation_category"]
        if any(row["keyword_targets"]):
            bucket, subgroup = "positive", str(row["keyword_targets"])
        elif category == "near-negative":
            bucket, subgroup = "near", row["full_text"]
        else:
            bucket, subgroup = "ordinary-partial", category
        grouped[row["source_family_id"]][bucket][subgroup].append(row)
    result = []
    for family in sorted(grouped):
        for bucket in sorted(grouped[family]):
            groups = grouped[family][bucket]
            for values in groups.values():
                values.sort(key=lambda row: row["source_id"])
            result.extend(_interleave(groups))
    return result


def coefficient(value):
    fraction = Fraction(value)
    return {"numerator": fraction.numerator, "denominator": fraction.denominator, "value": float(fraction)}


def make_recipe(rows, vocabulary, sampler, *, near_percent):
    require(type(near_percent) is int and near_percent in (0, 5, 10), "D1_NEAR_DRAW_PERCENT")
    cohort = cohort_rows(rows)
    require(cohort and all(row["vocabulary_sha256"] == digest(vocabulary) for row in rows),
            "D1_COHORT_VOCABULARY")
    plan = sampler.exposure_plan(cohort, total_exposures=TOTAL_EXPOSURES,
                                 full_positive_exposures=POSITIVE_EXPOSURES, near_percent=near_percent)
    expected = {"full-positive": 300, "ordinary-or-partial": 300 - 6 * near_percent,
                "near-negative": 6 * near_percent}
    require(plan["counts"] == expected and len(plan["draws"]) == TOTAL_EXPOSURES,
            "D1_RETAINED_SAMPLER_DRAW_COUNTS")
    families, categories, positive_words = Counter(), Counter(), Counter()
    per_family_word, per_source = Counter(), Counter()
    masses = defaultdict(Fraction)
    for draw in plan["draws"]:
        row = cohort[draw["row_index"]]
        require(draw["source_id"] == row["source_id"] and draw["ctc_weight"] == row["ctc_weight"] and
                draw["keyword_mask"] == row["keyword_mask"] and draw["keyword_weight"] == row["keyword_weight"],
                "D1_DRAW_LOSS_FIELDS_CHANGED")
        families[row["source_family_id"]] += 1
        categories[row["generation_category"]] += 1
        per_source[row["source_id"]] += 1
        masses[row["generation_category"]] += Fraction(str(row["ctc_weight"]))
        for index, target in enumerate(row["keyword_targets"]):
            if target == 1 and row["keyword_mask"][index]:
                positive_words[KEYWORDS[index]] += 1
                per_family_word[row["source_family_id"] + ":" + KEYWORDS[index]] += 1
    items = [{"row_index": index, "source_id": row["source_id"],
              "global_source_key": row["global_source_key"], "local_item_id": row["item_id"],
              "label_binding_sha256": row_identity(row), "full_text": row["full_text"],
              "target_ids": row["target_ids"], "ctc_mask": row["ctc_mask"], "ctc_weight": row["ctc_weight"],
              "keyword_targets": row["keyword_targets"], "keyword_mask": row["keyword_mask"],
              "keyword_weight": row["keyword_weight"], "source_family_id": row["source_family_id"],
              "generation_category": row["generation_category"]} for index, row in enumerate(cohort)]
    result = {
        "schema": SCHEMA, "seed": SEED, "vocabulary_sha256": digest(vocabulary), "output_count": OUTPUTS,
        "ctc_blank_id": 0, "total_exposures": TOTAL_EXPOSURES, "full_positive_exposures": POSITIVE_EXPOSURES,
        "near_percent_of_all_draws": near_percent, "batch_size": BATCH_SIZE,
        "optimizer_steps": TOTAL_EXPOSURES // BATCH_SIZE,
        "exposure_budget_is_dataset_epochs": False, "every_pool_row_must_be_exposed": False,
        "cohort_items": len(cohort), "items": items, "cohort_sha256": digest(items),
        "exposure_plan": plan, "exposure_plan_sha256": digest(plan),
        "fixed_ctc_normalizer": 1.0, "fixed_keyword_normalizer": 1.0,
        "loss_reduction": "full-target-length-normalized CTC times saved confidence, then mean over batch items",
        "loss_denominator_changes_with_near_ratio": False,
        "draw_fraction_equals_effective_weight_fraction": False,
        "numerical_loss_contribution_requires_actual_measurement": True,
        "exposures_by_family": dict(sorted(families.items())),
        "exposures_by_generation_category": dict(sorted(categories.items())),
        "positive_exposures_by_word": dict(sorted(positive_words.items())),
        "positive_exposures_by_family_and_word": dict(sorted(per_family_word.items())),
        "exposures_by_source": dict(sorted(per_source.items())),
        "ctc_coefficient_mass_by_generation_category": {name: coefficient(value) for name, value in sorted(masses.items())},
        "initialization_policy": "same candidate initial checkpoint and fresh optimizer for every ratio; no arm continuation",
        "optimizers": {"fresh-rnn64-or-c1": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01},
                       "extended-fsmn-head": {"name": "AdamW", "lr": 0.0001, "weight_decay": 0.0}},
        "gradient_clip_max_norm": 5.0, "development_used_for_optimization": False,
        "synthetic_development_scope": "held-out source families; engineering diagnostic, not independent human evidence",
        "human_gold": False, "quality_winner": None, "actual_training_executed": False,
    }
    return result, cohort


def make_comparison_recipes(rows, vocabulary, sampler):
    recipes = {str(percent): make_recipe(rows, vocabulary, sampler, near_percent=percent)[0]
               for percent in (0, 5, 10)}
    def positives(recipe):
        return [(row["source_id"], row["ctc_weight"], row["keyword_mask"], row["keyword_weight"])
                for row in recipe["exposure_plan"]["draws"] if row["bucket"] == "full-positive"]
    require(positives(recipes["0"]) == positives(recipes["5"]) == positives(recipes["10"]),
            "D1_POSITIVE_EXPOSURES_OR_WEIGHTS_CHANGED")
    require(len({recipe["cohort_sha256"] for recipe in recipes.values()}) == 1,
            "D1_SAME_DATA_COHORT_REQUIRED")
    return recipes


def available_recipes(rows, vocabulary, sampler):
    """A missing near pool blocks its requested arms, not the zero arm."""
    available, blocked = {}, {}
    if not cohort_rows(rows):
        return {}, {"schema": "kws-d1-arm-availability-v2", "available": [],
            "blocked": {str(percent): {"status": "blocked", "reason": "NO_CONFIRMED_COMMON_CTC_ROWS",
                         "automatic_retry": False} for percent in (0, 5, 10)},
            "quarantined_items": sum(row["quarantined"] for row in rows),
            "unchanged_generation_intents": True, "automatic_pool_filling": False}
    for percent in (0, 5, 10):
        try:
            available[str(percent)] = make_recipe(rows, vocabulary, sampler, near_percent=percent)[0]
        except ValueError as error:
            if not str(error).startswith("MISSING_SAMPLING_POOL:"):
                raise
            blocked[str(percent)] = {"status": "blocked", "reason": str(error), "automatic_retry": False}
    positives = [[(draw["source_id"], draw["ctc_weight"], draw["keyword_mask"], draw["keyword_weight"])
                  for draw in recipe["exposure_plan"]["draws"] if draw["bucket"] == "full-positive"]
                 for recipe in available.values()]
    require(not positives or all(value == positives[0] for value in positives),
            "D1_AVAILABLE_ARMS_POSITIVE_EXPOSURES_CHANGED")
    return available, {"schema": "kws-d1-arm-availability-v2", "available": sorted(map(int, available)),
                       "blocked": blocked, "quarantined_items": sum(row["quarantined"] for row in rows),
                       "unchanged_generation_intents": True, "automatic_pool_filling": False}
