#!/usr/bin/env python3
"""Fixed-exposure research sampler; no generation, training or label changes."""
from __future__ import annotations

from collections import defaultdict


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def exposure_plan(rows, *, total_exposures, full_positive_exposures, near_percent):
    """Keep positive draws/weights fixed while replacing near-negative exposure.

    ``near_percent`` is a percentage of all training draws (0, 5 or 10), not
    the fraction of a generation plan. Missing ordinary/partial negatives are
    a real data shortage: never fill their slot with extra positives or OOV blanks.
    The returned row indices can be passed to a research trainer's sampler;
    weights remain explicit because legacy Manifest ignores weight metadata.
    """
    require(type(total_exposures) is int and 1 <= total_exposures <= 100000 and
            type(full_positive_exposures) is int and 0 <= full_positive_exposures <= total_exposures and
            type(near_percent) is int and near_percent in (0, 5, 10) and
            total_exposures * near_percent % 100 == 0, "EXPOSURE_BUDGET")
    near = total_exposures * near_percent // 100
    counts = {"full-positive": full_positive_exposures, "ordinary-or-partial": total_exposures - full_positive_exposures - near,
              "near-negative": near}
    require(counts["ordinary-or-partial"] >= 0, "EXPOSURE_BUDGET")
    buckets = {name: defaultdict(list) for name in counts}
    ids = set()
    for index, row in enumerate(rows):
        require(row["source_id"] not in ids and row["split"] == "train" and
                row["human_gold"] is False and row["independent_test"] is False and
                row["ctc_mask"] is True and 0 < row["ctc_weight"] <= 1 and
                row["keyword_mask"] == [True, True] and
                row["keyword_targets"] in ([1, 0], [0, 1], [1, 1], [0, 0]),
                "RESEARCH_TRAIN_ROWS_REQUIRED")
        ids.add(row["source_id"])
        category = row.get("generation_category")
        require("generation_category" not in row or
                category in ("full-positive", "ordinary-negative", "partial-negative", "near-negative"),
                "UNKNOWN_GENERATION_CATEGORY")
        if category is not None:
            require(bool(any(row["keyword_targets"])) == (category == "full-positive"),
                    "GENERATION_CATEGORY_TARGET_CONFLICT")
        if any(row["keyword_targets"]):
            bucket = "full-positive"
        elif (category == "near-negative" or
              (category is None and
               row["intended_kind"] in ("near-miss-intent", "hard-negative-intent"))):
            bucket = "near-negative"
        else:
            require(category in ("ordinary-negative", "partial-negative") or
                    row["intended_kind"] in ("speech-negative-intent", "partial-intent"),
                    "UNKNOWN_NEGATIVE_GENERATION_INTENT")
            bucket = "ordinary-or-partial"
        buckets[bucket][row["source_family_id"]].append(index)
    for name, count in counts.items():
        require(count == 0 or buckets[name], "MISSING_SAMPLING_POOL:" + name)
    used = dict.fromkeys(counts, 0)
    per_source = defaultdict(int)
    draws = []
    for position in range(total_exposures):
        available = [name for name in counts if used[name] < counts[name]]
        # Deficit scheduling interleaves classes instead of exhausting a class.
        name = max(available, key=lambda item: counts[item] * (position + 1) - used[item] * total_exposures)
        groups = sorted(buckets[name])
        family = groups[used[name] % len(groups)]
        choices = buckets[name][family]
        index = choices[per_source[name, family] % len(choices)]
        row = rows[index]
        draws.append({"row_index": index, "source_id": row["source_id"], "bucket": name,
                      "source_family_id": family, "ctc_weight": row["ctc_weight"],
                      "keyword_mask": list(row["keyword_mask"]), "keyword_weight": list(row["keyword_weight"])})
        per_source[name, family] += 1
        used[name] += 1
    return {"schema": "kws-research-fixed-exposure-plan-v1", "near_percent_of_all_draws": near_percent,
            "total_exposures": total_exposures, "full_positive_exposures": full_positive_exposures,
            "counts": counts, "draws": draws, "generation_mix_changed": False,
            "positive_weights_changed": False, "new_model_calls": 0, "training_runs": 0}
