"""Offline SOURCE-only listening handoff over retained screen32 evidence.

Hashes bind bytes and declared receipts, not a human's identity or honesty.
No model execution, target generation, admission, network, or training occurs.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import wave

import quality_gates as gates

BASE = Path("research/source-screen32-v1")
GEN = "evidence/run-38018029787"
ASR = "evidence/run-38021467257/asr"
PINS = {
    "execution-freeze.json": "242565d3dec5174c57572eafef68f5ea49a2ddd42d213b0debdd7f7b44d33079",
    "plan.json": "ecf8ff83e73953456e194741d77b6332bfc9c6fce51e71295e2b13bf452cb97d",
    GEN + "/generation/tts/tts-receipt.json": "e7898fabd0c1e75b18f0d29a105777a348ca592d7c78e2e5efe6931b4cf1958f",
    GEN + "/blind/job.json": "425a705515339ff7054e4f1148971a342e5f882a1e9b84ac2be144a3d7aeecc5",
    "evidence/run-38021467257/provenance.json": "f135d0131aee89a14630a15006225b3be3a8a4e5d6ffc6efba6938cb1da3d489",
    ASR + "/qwen06/qwen06/decoder-inputs.json": "3a61e9c35018c63d7117f791b3fd84ea7a32cbba5167891ef00f4271dc9dad1e",
    ASR + "/sensevoice/sensevoice/decoder-inputs.json": "3a61e9c35018c63d7117f791b3fd84ea7a32cbba5167891ef00f4271dc9dad1e",
    ASR + "/primary-raw.json": "79b1567e5739beab72b1da94490ea90c41ff8e0b32f604df85d12ac37ee054d5",
    ASR + "/primary-raw-freeze.json": "334c7e2df9649fd5431f97f7870fe57594d9c68094e2b034cf701242001defbc",
}
# Historical read-only evidence review only. Execution admission and staging
# must continue checking the live source paths against the unchanged freeze.
HISTORICAL_SOURCE_PATHS = {
    "research/experiment_quality_guards/quality_gates.py":
        "research/consolidation/maintained-sources/pr-479/research/experiment_quality_guards/quality_gates.py",
    "research/source-screen32-v1/test_contract.py":
        "research/source-screen32-v1/history/maintained-f47e81d/test_contract.py",
    "research/source-screen32-v1/test_hosted_run.py":
        "research/source-screen32-v1/history/maintained-f47e81d/test_hosted_run.py",
    "research/source-screen32-v1/test_sense_continuation.py":
        "research/source-screen32-v1/history/maintained-f47e81d/test_sense_continuation.py",
}
REPRESENTATION = "derived_16000_pcm16"
PRODUCTION_PUNCTUATION = frozenset("。，！？、；：…,.!?;: \t\n\r")
PACKET_SCHEMA = "screen32-label-free-listening-packet-v1"
RECEIPT_SCHEMA = "screen32-human-listening-receipts-v1"
require = gates.require


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(raw):
    def reject_constant(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(raw,
                      object_pairs_hook=unique_object, parse_constant=reject_constant)


def load(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def keyed(rows, field):
    require(type(rows) is list and all(type(r) is dict for r in rows), "object list required")
    result = {}
    for row in rows:
        key = row.get(field)
        require(type(key) is str and key and key not in result, "missing or duplicate " + field)
        result[key] = row
    return result


def verified_inputs(repo):
    """Read one call-local byte snapshot and parse exactly the verified bytes."""
    base = Path(repo) / BASE
    raw_inputs = {}

    def verified_bytes(path, expected, message):
        require(not path.is_symlink(), message)
        if path not in raw_inputs:
            raw_inputs[path] = path.read_bytes()
        raw = raw_inputs[path]
        require(digest(raw) == expected, message)
        return raw

    values = {}
    for relative, expected in PINS.items():
        raw = verified_bytes(base / relative, expected, "retained input hash mismatch: " + relative)
        values[relative] = parse_json(raw.decode("utf-8"))
    freeze = values["execution-freeze.json"]["files"]
    require(len(freeze) == 74, "frozen source denominator changed")
    for relative, expected in freeze.items():
        path = Path(repo) / HISTORICAL_SOURCE_PATHS.get(relative, relative)
        verified_bytes(path, expected, "frozen source changed: " + relative)
    raw_freeze = values[ASR + "/primary-raw-freeze.json"]
    require(raw_freeze["primary_raw_sha256"] == PINS[ASR + "/primary-raw.json"], "primary raw freeze mismatch")
    for name in ("qwen06", "sensevoice"):
        terminal = base / ASR / name / "terminal-outcomes.json"
        verified_bytes(terminal, raw_freeze["terminal_outcomes_sha256"][name],
                       "terminal outcomes freeze mismatch")
    return values


def build_packet(repo):
    """Verify bytes, then expose only audio bindings and empty listening fields."""
    return _build_packet(repo, verified_inputs(repo))


def _build_packet(repo, values):
    generated = keyed(values[GEN + "/generation/tts/tts-receipt.json"]["ledger"], "cell_id")
    require({key for key, row in generated.items() if row["status"] == "GENERATED"}
            == {f"screen32-{n:03d}" for n in range(1, 17)},
            "all 16 generated cells required")
    blind = values[GEN + "/blind/job.json"]
    clips = keyed(blind["clips"], "wav_sha256")
    decoders = keyed(values[ASR + "/qwen06/qwen06/decoder-inputs.json"]["clips"], "opaque_id")
    require(len(clips) == len(decoders) == 16, "all 16 input bindings required")
    rows = []
    for n in range(1, 17):
        cell = f"screen32-{n:03d}"
        entry = generated[cell]
        require(entry["status"] == "GENERATED", "generated audio required")
        path = Path(repo) / BASE / GEN / "generation/tts" / (cell + ".wav")
        require(not path.is_symlink(), "audio symlink rejected")
        wav_bytes = path.read_bytes()
        wav_hash = digest(wav_bytes)
        require(wav_hash == entry["audio"]["wav_sha256"], "WAV digest mismatch")
        with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
            require((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype())
                    == (1, 2, 16000, "NONE"), "derived PCM16 representation required")
            frames = audio.getnframes()
            pcm = audio.readframes(frames)
        require(len(pcm) == frames * 2, "truncated PCM")
        pcm_hash = digest(pcm)
        require(pcm_hash == entry["audio"]["pcm_sha256"], "PCM digest mismatch")
        require(wav_hash in clips, "WAV absent from original blind job")
        clip = clips[wav_hash]
        require(clip["audio_id"] in decoders, "decoder ID absent")
        decoder = decoders[clip["audio_id"]]
        descriptor = decoder["descriptor"]
        require(digest(canonical(descriptor)) == decoder["binding_sha256"], "decoder binding mismatch")
        require(descriptor["wav"]["sha256"] == wav_hash
                and descriptor["raw_pcm16"]["sha256"] == pcm_hash
                and descriptor["raw_pcm16"]["frame_count"] == frames, "decoder/audio mismatch")
        rows.append({"review_id": f"{n:02d}", "audio_file": f"{n:02d}.wav",
                     "wav_sha256": wav_hash, "pcm_sha256": pcm_hash,
                     "decoder_input_binding_sha256": decoder["binding_sha256"],
                     "representation": REPRESENTATION, "sample_rate_hz": 16000,
                     "frame_count": frames})
    provenance = values["evidence/run-38021467257/provenance.json"]
    packet = {"schema": PACKET_SCHEMA, "scope": "SOURCE_REVIEW_ONLY", "denominator": 16,
              "binding": {"generation_run_id": 38018029787, "continuation_run_id": 38021467257,
                          "generation_head_sha": provenance["original_executed_head_sha"],
                          "continuation_head_sha": provenance["executed_head_sha"],
                          "source_freeze_sha256": PINS["execution-freeze.json"],
                          "original_blind_job_raw_sha256": PINS[GEN + "/blind/job.json"],
                          "blind_job_canonical_sha256": digest(canonical(blind)),
                          "input_sha256": dict(PINS)}, "rows": rows}
    return dict(packet, packet_sha256=digest(canonical(packet)))


def blank_review():
    return {"status": "pending", "label_origin": None, "listened_to_audio": None,
            "listened_entire_clip": None, "independent_human": None, "complete": None,
            "hypothesis_exposure": "UNKNOWN"}


def receipt_template(packet):
    return {"schema": RECEIPT_SCHEMA, "packet_sha256": packet["packet_sha256"],
            "revision": 1, "supersedes_receipt_sha256": None,
            "reviewer_id": None, "source_reference": None,
            "rows": [dict(copy.deepcopy(row), actual_text=None, partial_text=None,
                          acoustic_completeness="UNKNOWN", review=blank_review())
                     for row in packet["rows"]]}


def validate_history(packet, documents):
    """Partial revisions retain history; latest submitted row wins, even pending."""
    require(type(documents) is list, "receipt history must be a list")
    bindings = keyed(packet["rows"], "review_id")
    histories = {key: [] for key in bindings}
    previous = None
    template = receipt_template(packet)
    for number, document in enumerate(documents, 1):
        require(type(document) is dict and set(document) == set(template), "receipt fields mismatch")
        require(document["schema"] == RECEIPT_SCHEMA, "receipt schema mismatch")
        require(document["packet_sha256"] == packet["packet_sha256"], "packet binding mismatch")
        require(type(document["revision"]) is int and document["revision"] == number
                and document["supersedes_receipt_sha256"] == previous, "stale or incomplete receipt history")
        for field in ("reviewer_id", "source_reference"):
            require(document[field] is None or (type(document[field]) is str and document[field].strip() and len(document[field]) <= 512),
                    field + " must be null or nonempty text")
        rows = keyed(document["rows"], "review_id")
        require(set(rows) <= set(bindings), "unknown review ID")
        for identity, row in rows.items():
            bound = bindings[identity]
            require(set(row) == set(template["rows"][0]), "review row fields mismatch")
            require(all(type(row[k]) is type(v) and row[k] == v for k, v in bound.items()),
                    "immutable audio binding mismatch: " + identity)
            review = row["review"]
            require(type(review) is dict and set(review) == set(blank_review()), "review fields mismatch")
            require(review["status"] in ("pending", "clean", "ambiguous", "incomplete", "inaudible"),
                    "unknown review status")
            require(review["label_origin"] in (None, "human_listening"), "human listening origin required")
            require(review["hypothesis_exposure"] in ("UNKNOWN", "EXPOSED", "NONE_DECLARED"),
                    "explicit hypothesis exposure required")
            for field in ("listened_to_audio", "listened_entire_clip", "independent_human", "complete"):
                require(review[field] is None or type(review[field]) is bool, "review boolean or null required")
            for field in ("actual_text", "partial_text"):
                require(row[field] is None or (type(row[field]) is str and len(row[field]) <= 8192),
                        field + " must be bounded text or null")
            require(row["acoustic_completeness"] in ("UNKNOWN", "COMPLETE", "INCOMPLETE"),
                    "unknown acoustic completeness")
            if review["status"] == "pending":
                require(row["actual_text"] is None and row["partial_text"] is None
                        and row["acoustic_completeness"] == "UNKNOWN" and review == blank_review(),
                        "pending row must remain unfilled")
            else:
                require(review["label_origin"] == "human_listening" and review["listened_to_audio"] is True,
                        "review requires declared direct human listening")
                if review["status"] == "clean":
                    require(row["partial_text"] is None, "clean review cannot retain unresolved partial text")
                    require(type(row["actual_text"]) is str and gates.normalized_actual(row["actual_text"]),
                            "clean review requires nonempty human actual words")
            histories[identity].append({"receipt_revision": number, "receipt_sha256": digest(canonical(document)),
                                       "reviewer_id": document["reviewer_id"],
                                       "source_reference": document["source_reference"], "row": copy.deepcopy(row)})
        previous = digest(canonical(document))
    return histories


def source_report(repo, documents):
    return _source_report(repo, documents, verified_inputs(repo))


def _source_report(repo, documents, values):
    packet = _build_packet(repo, values)
    histories = validate_history(packet, documents)
    # Hypotheses enter the derived report only after receipt validation.
    plan = keyed(values["plan.json"]["cells"], "cell_id")
    clips = keyed(values[GEN + "/blind/job.json"]["clips"], "wav_sha256")
    saved = {name: keyed(rows, "audio_id") for name, rows in values[ASR + "/primary-raw.json"].items()}
    rows = []
    for index, bound in enumerate(packet["rows"], 1):
        history = histories[bound["review_id"]]
        latest = history[-1]["row"] if history else receipt_template(packet)["rows"][index - 1]
        review = latest["review"]
        submitted = review["status"] != "pending"
        attributed = bool(history and history[-1]["reviewer_id"] and history[-1]["source_reference"])
        direct = (review["label_origin"] == "human_listening" and review["listened_to_audio"] is True
                  and review["listened_entire_clip"] is True)
        gate_row = {"actual_text": latest["actual_text"],
                    "review": {"status": review["status"],
                               "independent_human": attributed and direct and review["independent_human"] is True,
                               "complete": review["complete"] is True
                               and latest["acoustic_completeness"] == "COMPLETE"}}
        if submitted:
            opaque_id = clips[bound["wav_sha256"]]["audio_id"]
            gate_row.update(intended_text=plan[f"screen32-{index:03d}"]["intended_text"],
                            asr_results=[copy.deepcopy(saved[name][opaque_id]) for name in ("qwen06", "sensevoice")])
        truth = gates.human_truth(gate_row)
        actual = latest["actual_text"]
        rows.append(dict(copy.deepcopy(bound), actual_text=actual, partial_text=latest["partial_text"],
                         review=copy.deepcopy(review), review_history=history,
                         review_provenance_complete=attributed, reviewer_identity_authenticated=False,
                         reviewer_declared_acoustic_completeness=latest["acoustic_completeness"],
                         acoustic_completeness="UNKNOWN", human_actual_review_complete=truth is not None,
                         actual_expected_keywords=None if truth is None else sorted(truth),
                         label_diagnostics=gates.label_preparation(gate_row) if submitted else None,
                         production_transcript_policy="xiaowo-four-syllable-transcript-v1",
                         production_oov_characters=None if actual is None else sorted(set(actual) - set("你好小窝") - PRODUCTION_PUNCTUATION),
                         role="source_screen_only", split="UNASSIGNED", lineage_status="unverified",
                         ctc_target=None, automatic_relabel=False, silence_or_blank_target_inferred=False,
                         training_admitted=False, independent_accuracy_qualified=False))
    return {"schema": "screen32-human-source-review-report-v1", "scope": "SOURCE_REVIEW_ONLY",
            "packet_sha256": packet["packet_sha256"], "binding": packet["binding"], "denominator": 16,
            "submitted_rows": sum(row["review"]["status"] != "pending" for row in rows),
            "pending_rows": sum(row["review"]["status"] == "pending" for row in rows),
            "complete_human_actual_rows": sum(row["human_actual_review_complete"] for row in rows),
            "unqualified_submitted_rows": sum(row["review"]["status"] != "pending"
                                              and not row["human_actual_review_complete"] for row in rows),
            "receipt_history_sha256": [digest(canonical(doc)) for doc in documents], "rows": rows,
            "new_model_calls": 0, "training_admitted": False, "shipping_approved": False,
            "independent_accuracy_qualified": False, "acoustic_completeness": "UNKNOWN",
            "limits": "Declared listening evidence is not authenticated by hashes. No coverage, lineage, CTC or training admission is made."}


SCREENING_CATEGORIES = (
    "human_and_both_asr_agree",
    "dual_asr_agree_human_uncertain",
    "human_vs_dual_asr_disagreement",
    "dual_asr_disagree",
    "dual_asr_missing_incomplete_or_flagged",
)


def weak_screening(actual_text, human_complete, observations):
    """Comparison-only machine consensus; never replace a human declaration.

    A missing, failed, flagged, blank or punctuation-only observation is unresolved,
    even when both normalized strings would otherwise compare equal.
    """
    require(type(human_complete) is bool, "human completeness must be boolean")
    diagnostics = gates.label_preparation({"actual_text": actual_text,
                                          "asr_results": observations})
    require(not human_complete or bool(gates.normalized_actual(actual_text or "")),
            "complete human receipt requires nonempty actual words")
    flags = [flag for observation in observations if observation
             for flag in observation.get("quality_flags", [])]
    state = "unresolved" if flags else diagnostics["machine_state"]
    texts = diagnostics["machine_normalized_texts"]
    agreed = state == "machinesAgreeWeak"
    if state == "unresolved":
        category = "dual_asr_missing_incomplete_or_flagged"
    elif state == "disagree":
        category = "dual_asr_disagree"
    elif not human_complete:
        category = "dual_asr_agree_human_uncertain"
    elif gates.normalized_actual(actual_text) == texts[0]:
        category = "human_and_both_asr_agree"
    else:
        category = "human_vs_dual_asr_disagreement"
    return {"category": category, "machine_state": state,
            "asr_results": copy.deepcopy(observations), "asr_quality_flags": flags,
            "machine_normalized_texts": texts, "dual_asr_agreement": agreed,
            "screening_candidate_text": texts[0] if agreed else None,
            "screening_evidence_strength": "weak_machine_consensus" if agreed else "unresolved",
            "human_actual_review_complete": human_complete,
            "final_adjudicated_text": None, "human_gold": False,
            "acoustic_completeness": "UNKNOWN", "automatic_relabel": False,
            "silence_or_blank_target_inferred": False,
            "machine_consensus_overwrites_actual_text": False,
            "ctc_target": None, "training_admitted": False,
            "shipping_approved": False, "independent_accuracy_qualified": False}


def screening_report(repo, documents):
    """Explicit opt-in view of saved hypotheses, including pending human rows.

    The existing report and human/partial words remain intact. Receipt
    completeness records declarations, not adjudicated ground truth.
    """
    values = verified_inputs(repo)
    report = _source_report(repo, documents, values)
    original_digest = digest(canonical(report))
    clips = keyed(values[GEN + "/blind/job.json"]["clips"], "wav_sha256")
    names = ("qwen06", "sensevoice")
    saved = {name: keyed(values[ASR + "/primary-raw.json"].get(name, []), "audio_id")
             for name in names}
    partitions = {category: [] for category in SCREENING_CATEGORIES}
    candidate_ids = []
    rows = copy.deepcopy(report["rows"])
    for row in rows:
        opaque_id = clips[row["wav_sha256"]]["audio_id"]
        screening = weak_screening(row["actual_text"], row["human_actual_review_complete"],
                                   [saved[name].get(opaque_id) for name in names])
        row["screening"] = screening
        partitions[screening["category"]].append(row["review_id"])
        if screening["dual_asr_agreement"]:
            candidate_ids.append(row["review_id"])
    identities = [identity for group in partitions.values() for identity in group]
    require(len(identities) == len(set(identities)) == report["denominator"] == 16,
            "screening partition must retain all 16 unique rows")
    return dict(report, schema="screen32-dual-asr-screening-report-v1", scope="SOURCE_SCREENING_ONLY",
                source_review_schema=report["schema"], source_review_report_canonical_sha256=original_digest,
                screening_policy="dual-asr-consensus-weak-v1", asr_observation_order=list(names),
                partition_counts={key: len(ids) for key, ids in partitions.items()},
                partition_rows=partitions, dual_asr_agreement_rows=candidate_ids,
                dual_asr_agreement_weak_candidates=len(candidate_ids), rows=rows,
                human_gold=False, final_adjudicated_text=None, ctc_target=None, new_run_authorized=False,
                publication_authorized=False, machine_consensus_overwrites_actual_text=False,
                limits="Dual-ASR consensus is weak screening evidence, including when human declarations conflict. "
                       "Complete receipts record declarations, not adjudicated truth. Pending human words remain "
                       "unknown. No CTC, training, shipping, independent accuracy, new execution or publication is granted.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("packet", "template", "report", "screening"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--receipts", action="append", type=Path, default=[], help="Complete chronological receipt-document history")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        require(args.mode in ("report", "screening") or not args.receipts,
                "receipts are only valid in report or screening mode")
        if args.mode in ("report", "screening"):
            reporter = screening_report if args.mode == "screening" else source_report
            result = reporter(args.repo, [load(path) for path in args.receipts])
        else:
            packet = build_packet(args.repo)
            result = packet if args.mode == "packet" else receipt_template(packet)
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output:
            # Exclusive creation prevents accidentally replacing evidence or reviews.
            with args.output.open("x", encoding="utf-8") as output:
                output.write(encoded)
        else:
            print(encoded, end="")
        return 0
    except (ValueError, OSError, KeyError, TypeError, wave.Error) as error:
        print("screen32 review rejected: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
