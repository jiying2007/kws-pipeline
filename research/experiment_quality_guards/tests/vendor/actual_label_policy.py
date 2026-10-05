"""Actual human text policy for future dataset guards; never supplies CTC IDs."""
from audio_review import normalize, valid_text, require

def assess_actual_label(row, vocabulary="你好小窝屋"):
    require(type(row) is dict, "Expected actual-label row object")
    review = row.get("review", {})
    require(type(review) is dict, "Expected human review object")
    purposes = row.get("allowed_purposes", [])
    require(type(purposes) is list and all(type(p) is str for p in purposes), "Expected allowed-purpose string list")
    actual = row.get("actual_text")
    reasons = []
    if actual is None:
        reasons.append("missing_actual_text")
    else:
        valid_text(actual)
        if not normalize(actual):
            reasons.append("empty_actual_text_not_blank_truth")
    if review.get("status") != "clean":
        reasons.append("human_review_not_clean")
    if review.get("independent_human") is not True:
        reasons.append("independent_human_review_missing")
    if review.get("complete") is not True:
        reasons.append("acoustic_completeness_unconfirmed")
    text = normalize(actual or "")
    semantic_reasons = list(reasons)
    expected = None if semantic_reasons else [name for name, phrase in (("K1", "你好小窝"), ("K2", "小窝小窝")) if phrase in text]
    oov = sorted(set(text) - set(vocabulary))
    if oov:
        reasons.append("actual_text_out_of_vocabulary")
    if row.get("exposure") != "FRESH":
        reasons.append("exposed_or_unknown_exposure")
    if review.get("reviewed_before_predictions") is not True:
        reasons.append("review_before_predictions_not_established")
    if "asr_review_tool_validation" in purposes:
        reasons.append("review_tool_validation_only")
    # Eligibility is one guard, not full admission; coverage/lineage/rights gates remain.
    return {"actual_text": actual, "label_source": "human_actual_only", "expected_keywords": expected,
            "ctc_label_eligible": not reasons, "ctc_target": None, "oov_characters": oov,
            "reasons": reasons, "training_admitted": False, "machine_or_intended_fallback": False}

def expected_keywords(row):
    return assess_actual_label(row)["expected_keywords"]
