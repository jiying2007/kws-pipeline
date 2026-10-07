"""Frozen saved-output human review; stdlib only, no audio/model/network work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

COMPARISON_SHA256 = "7af81332103cf8174d8cbda9deda00e988027493e84f4643518993f20cdf280e"
BINDINGS_SHA256 = "a49fb0f9d0739098ea0ad5e6bb90f0729533e3cbf180e23b0a4103a2b44d9082"
LABELS_SHA256 = "9553d53d1cb0c95fcb6a552a614d6e8250234d103348cddc3895674c354c5027"
VOCABULARY_SHA256 = "42ef40ddc040ce394423eed78f17eb3c0a8ddb92e1c2a2e39a3ce0c4c5d339ed"
ALIASES = [f"M{i}" for i in range(1, 7)]
AUDIO_IDS = [f"clip-{i:06d}" for i in range(1, 7)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_json(raw, expected, name):
    require(hashlib.sha256(raw).hexdigest() == expected, name + " hash mismatch")
    return json.loads(raw)


def latest_revision(revisions):
    """Select the newest explicit revision, never the last non-null full text."""
    require(isinstance(revisions, list) and bool(revisions), "Missing human revisions")
    numbers = [r["revision"] for r in revisions]
    require(all(type(n) is int and n > 0 for n in numbers)
            and len(set(numbers)) == len(numbers), "Invalid/duplicate human revision")
    for row in revisions:
        if row["status"] == "full":
            require(isinstance(row["full_text"], str) and bool(row["full_text"])
                    and row["partial_text"] is None, "Invalid full transcript")
        else:
            require(row["status"] == "partial" and row["full_text"] is None
                    and isinstance(row["partial_text"], str) and bool(row["partial_text"])
                    and row["first_character"] == "UNKNOWN", "Invalid partial transcript")
    return max(revisions, key=lambda row: row["revision"])


def transcript_coverage(full_text, partial_text, vocabulary):
    """Report text coverage only. Unknown text and OOV are never blank targets."""
    known_text = full_text if full_text is not None else partial_text
    # This fixed review's bracketed uncertainty marker is metadata, not speech.
    known_text = (known_text or "").removeprefix("[首字听不清]")
    oov = sorted(set(known_text) - set(vocabulary))
    return {
        "full_transcript_available": full_text is not None,
        "full_transcript_representable": (not oov) if full_text is not None else None,
        "observed_out_of_vocabulary_characters": oov,
        "ctc_target": None,
        "training_admission": False,
    }


def adjudicate_saved(comparison_raw, labels_raw, bindings_raw, vocabulary_raw):
    """Pure join of four frozen byte strings; no input mutation or audio reading.

    Hashes establish this supplement's reviewed snapshots, not independent
    proof of listening or waveform truth. A future review needs a new snapshot.
    """
    comparison = frozen_json(comparison_raw, COMPARISON_SHA256, "Comparison")
    labels = frozen_json(labels_raw, LABELS_SHA256, "Human labels")
    bindings = frozen_json(bindings_raw, BINDINGS_SHA256, "Audio bindings")
    vocabulary = frozen_json(vocabulary_raw, VOCABULARY_SHA256, "Vocabulary")
    require(comparison["schema"] == "melo6-post-freeze-machine-comparison-v1"
            and labels["schema"] == "melo6-human-transcript-revisions-v1"
            and bindings["schema"] == "melo6-human-adjudication-audio-bindings-v1",
            "Wrong saved input schema")
    require(bindings["prior_machine_comparison_sha256"] == COMPARISON_SHA256,
            "Binding points to another comparison")
    require([r["audio_id"] for r in comparison["clips"]] == AUDIO_IDS
            and [r["alias"] for r in labels["clips"]] == ALIASES
            and [r["alias"] for r in bindings["bindings"]] == ALIASES,
            "Wrong alias/audio panel")
    require(labels["heard_representation"] == "native_44100_float32"
            and labels["prior_asr_hypotheses_exposed"] is True
            and labels["fully_blinded"] is False
            and labels["derived_audio_independently_heard"] is False,
            "Wrong listening scope")
    require(vocabulary["schema"] == "frozen-six-class-kws-vocabulary-binding-v1"
            and vocabulary["blank_id"] == 0
            and vocabulary["symbols"][0] == "<blank>", "Wrong vocabulary binding")
    characters = vocabulary["symbols"][1:]
    require(isinstance(characters, list) and len(set(characters)) == len(characters)
            and all(isinstance(c, str) and len(c) == 1 for c in characters),
            "Invalid nonblank character vocabulary")
    clips = []
    for index, (machine, human, binding) in enumerate(zip(
            comparison["clips"], labels["clips"], bindings["bindings"]), 1):
        require(human["asr_audio_id"] == binding["asr_audio_id"] == machine["audio_id"]
                and binding["source_id"] == f"melo6-{index:03d}"
                and human["heard_native_wav_sha256"] == binding["native_wav_sha256"]
                and binding["derived_wav_sha256"] == machine["wav_sha256"],
                "Human/native/derived alias or hash mismatch")
        current = latest_revision(human["revisions"])
        full_text, partial_text = current["full_text"], current["partial_text"]
        clips.append({
            "alias": human["alias"], "asr_audio_id": machine["audio_id"],
            "heard_native_wav_sha256": binding["native_wav_sha256"],
            "asr_derived_wav_sha256": machine["wav_sha256"],
            "intended_text": machine["intended_text"],
            "original_machine_assessment": machine["assessment"],
            "original_machine_text_agreement_only": machine["machine_text_agreement_only"],
            "saved_asr_texts": {r["model"]: r["normalized_text"]
                                for r in machine["recognizer_evidence"]},
            "current_human_revision": current["revision"],
            "human_transcript_status": current["status"],
            "human_full_text": full_text, "human_partial_text": partial_text,
            "first_character": current["first_character"],
            "human_prompt_match": ("MATCH" if full_text == machine["intended_text"]
                                   else "MISMATCH") if full_text is not None else "UNKNOWN",
            "superseded_human_revisions": [dict(r, superseded=True) for r in
                sorted(human["revisions"], key=lambda r: r["revision"])
                if r["revision"] < current["revision"]],
            "confidence": "UNKNOWN", "acoustic_tail_completeness": "UNKNOWN",
            "pronunciation_analysis": "NOT_ESTABLISHED_BY_TRANSCRIPT",
            **transcript_coverage(full_text, partial_text, characters),
        })
    return {
        "schema": "melo6-saved-human-adjudication-v1",
        "input_sha256": {"comparison": COMPARISON_SHA256, "human_labels": LABELS_SHA256,
                         "audio_bindings": BINDINGS_SHA256, "vocabulary": VOCABULARY_SHA256},
        "original_machine_gate": {key: comparison[key] for key in
            ("all_six_weak_lexical_gate_passed", "overall_result", "next_action",
             "automatic_source_advancement", "automatic_retry")},
        "original_both_asr_exact_intent_matches": comparison["counts"]["both_ASR_exact_intent_matches"],
        "human_review_scope": {
            "reviewer_count": labels["reviewer_count"],
            "prior_asr_hypotheses_exposed": True, "fully_blinded": False,
            "heard_representation": "native_44100_float32",
            "asr_representation": "derived_16000_pcm16",
            "derived_audio_independently_heard": False,
        },
        "clips": clips,
        "panel_readiness": {
            "full_human_transcripts": sum(c["full_transcript_available"] for c in clips),
            "partial_human_transcripts": sum(c["human_transcript_status"] == "partial" for c in clips),
            "human_prompt_matches": sum(c["human_prompt_match"] == "MATCH" for c in clips),
            "human_prompt_mismatches": sum(c["human_prompt_match"] == "MISMATCH" for c in clips),
            "human_prompt_unknown": sum(c["human_prompt_match"] == "UNKNOWN" for c in clips),
            "word_level_positive_coverage": {
                key: {"text": text, "human_transcript_aliases": [
                    c["alias"] for c in clips if c["human_full_text"] == text]}
                for key, text in (("K1", "你好小窝"), ("K2", "小窝小窝"))
            },
            "split_allocation": "UNASSIGNED",
            "split_integrity_qualified": False,
            "qualified_positive_development_set": False,
            "identity_held_out_evaluation_established": False,
            "voice_rights_cleared": "UNKNOWN",
            "training_admitted": 0,
            "reason": "Word-level K1/K2 examples are present. This increment has no frozen split allocation or split-integrity qualification; one identity group alone cannot fill isolated train/development/held-source partitions. Voice rights remain unestablished.",
        },
        "new_model_calls": 0, "kws_improvement_measured": False,
        "ctc_policy": "No targets admitted; partial/OOV speech is never encoded as blank CTC.",
    }


def serialized(result):
    return (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write_new(path, result):
    with Path(path).open("xb") as out:
        out.write(serialized(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "labels", "bindings", "vocabulary", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = adjudicate_saved(*(getattr(args, name).read_bytes() for name in
                               ("comparison", "labels", "bindings", "vocabulary")))
    write_new(args.out, result)


if __name__ == "__main__":
    main()
