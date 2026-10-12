"""Fixed new punctuation supplement. Stdlib metadata only; no runnable campaign.

Validators below check proposed metadata shape, never establish actual source,
rights, execution, storage durability or scientific admission by themselves.
"""
from copy import deepcopy
import hashlib
import json
import re

EXPERIMENT = "k1-punctuation6-v1"
PREFIX = "k1p6"
VOICES = ("Dylan", "Uncle_Fu", "Vivian")
TEXTS = ("你好小窝", "你好，小窝")
FAMILIES = ("qwen06", "sensevoice", "whisper")
PAIRED_SEEDS = (1012001, 1012002, 1012003)
MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"
OLD_COVERAGE = "83a0b0473049ca1b784274be5703ff38afd5e7d6922277350db3732265a2b128"
OLD_RECIPE = "bee9cc3d4a6c879705fe77e36195d2f2eeca2244e7ea01c5b2891e499eb4cd5f"
SHA = re.compile(r"[0-9a-f]{64}\Z")
RECIPE = {"language": "Chinese", "do_sample": True, "top_k": 50, "top_p": 1.0,
          "temperature": 0.9, "repetition_penalty": 1.05, "subtalker_dosample": True,
          "subtalker_top_k": 50, "subtalker_top_p": 1.0, "subtalker_temperature": 0.9,
          "max_new_tokens": 120, "non_streaming_mode": True}
OLD_CANDIDATES = (
    ("clip-000003", "3660c72e7db7392d4a430b71cc224b65527037129ef45d4387627dc6e262ef25", "old-k2", "小窝小窝"),
    ("clip-000012", "f96edce564f94ea7919c6de63613e88fe89ae07f01ad907097c4af095c3685db", "old-k2", "小窝小窝"),
    ("clip-000002", "06fd2d9f2186ad9c7d4386bbbfe13241a141bc792c7ea8e634b74cadea5e2980", "old-repeat-nihao", "你好你好"),
    ("clip-000006", "7c091117d6ef772c960bd818c4188a1ece8db9219f540d35ecb6850493697af4", "old-repeat-nihao", "你好你好"),
    ("clip-000007", "a9ddaa06fa6133cd9acdb37a761b0e3cc6b66e48a18c82a97b159738bdbc8229", "old-repeat-nihao", "你好你好"),
    ("clip-000008", "ae352e520daff27d394d726e3f35ea46b1745c6d8c0a335685e1f9b770227e20", "old-suffix-kw2", "窝小窝"),
)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def cells():
    return [{"cell_id": f"{PREFIX}-{2*i+j+1:03d}", "voice": voice,
             "condition": ("unpunctuated", "comma")[j], "text": text,
             "seed": PAIRED_SEEDS[i], "attempts_max": 1,
             "prospective_role": "train-only", "human_gold": False,
             "training_admitted": False}
            for i, voice in enumerate(VOICES) for j, text in enumerate(TEXTS)]


def slots():
    return [{"slot_id": f"{cell['cell_id']}-{family}", "cell_id": cell["cell_id"],
             "family": family, "attempts_max": 1}
            for family in FAMILIES for cell in cells()]


def expected_plan():
    return {"schema": "k1-punctuation6-preregistration-proposal-v1",
            "experiment": EXPERIMENT, "status": "disabled_pending_root_freeze",
            "execution_enabled": False, "derivation": {'schema': 'k1-punctuation6-derived-policy-v1', 'parent_plan_sha256': 'a799de29be9e41c283982942ed6eb2256e951d864c12104deb152bdcd4d1f229', 'parent_contract_sha256': 'aa860a9baa2b0aceb29d5a75f2693dec8c2bb150b68c60f49a68a08e8be017e0', 'parent_snapshot_manifest_sha256': '15bec2841e5cda32f556874a063afed42d498d173c0b1ac599559f3fd837934b', 'sole_behavior_change': 'Exact all-zero final PCM16 bytes add derived_pcm_silent; quarantine as not_run_quality before ASR. No amplitude/SNR threshold, retry, changed generation or changed denominator.'},
            "quality_policy": {'ordered_flags': ['silent', 'source_peak_ge_one', 'normalized_peak_ge_one', 'derived_pcm_silent'], 'derived_pcm_silent_definition': 'All final decoded PCM16 sample bytes are zero; compute after WAV write/readback and independently recompute from published WAV. Any nonzero int16 sample, including +1 or -1, is not silent under this rule.', 'generation_status': 'Natural generation may remain complete; quality exclusion is an independent pre-ASR disposition.', 'asr_disposition': 'not_run_quality unless an already-fixed duplicate quarantine also applies'}, "tts": {"model_id": MODEL, "revision": REVISION,
                "cells": cells(), "recipe": deepcopy(RECIPE),
                "cpu": {"dtype": "float32", "attention": "eager", "threads": 4, "interop_threads": 1},
                "seed_policy": "One predeclared seed per voice, paired across the two texts; no seed search.",
                "instruction": None, "phoneme_interface": None, "pronunciation_control_claim": False,
                "reference_audio": None, "voice_cloning": False},
            "asr": {"slots": slots(), "new_attempts_max": 18, "old_attempts_allowed": 0,
                "family_order": list(FAMILIES), "blind_inputs": True,
                "prepare_only_loads_have_zero_audio_forward": True,
                "all_six_inputs_frozen_before_first_asr": True,
                "reuse_historical_transcripts_as_new_outputs": False},
            "attempt_accounting": {
                "tts_denominator": 6, "asr_denominator": 18, "max_tts_calls": 6,
                "max_asr_calls": 18, "attempts_per_slot": 1, "retries": 0,
                "warmup_synthesis": 0, "best_of": 1, "hidden_audio_probes": 0,
                "tts_call_definition": "One top-level generate_custom_voice call for one cell. Its ordinary autoregressive subtalker work is internal, never another top-level request.",
                "asr_call_definition": "One native family audio-decode invocation for one newly generated, byte-frozen cell.",
                "consumed_boundary": "Exclusive started file plus file and containing-directory fsync before each call; failed/timeout/crashed started slots remain consumed.",
                "global_once_boundary": "Distinct fixed create-event refs for TTS and ASR, run_attempt=1, source-bound one-use release, and verified prior-claim history. Local exclusive files alone do not prevent new-workspace replay.",
                "failure_policy": "Stop on any failed call, timeout, resource violation or TTS output/token/duration limit; retain failed and unstarted slots. Never refill, rename or resume.",
                "denominator_policy": "Retain every six-cell and eighteen-slot row, including failures, not_run, disagreements and exclusions, stratified by voice and text condition."},
            "resources": {"provider": "public GitHub standard ubuntu-24.04 hosted runner or separately reviewed already-owned resource",
                "paid_resources": False, "actions_artifact_storage": False,
                "tts": {"controlled_input_bytes": 2934954359, "response_body_cap_bytes": 4*1024**3,
                    "setup_wall_seconds": 1200, "setup_cpu_seconds_per_process": 2400,
                    "setup_sampled_rss_bytes": 4*1024**3, "generation_wall_seconds": 1200,
                    "generation_cpu_seconds_per_process": 3600, "generation_sampled_rss_bytes": 7*1024**3,
                    "total_stage_wall_seconds": 2400, "workflow_minutes": 50,
                    "per_cell_cooperative_alarm_seconds": 300, "maximum_audio_seconds_per_cell": 12,
                    "maximum_audio_seconds_all_cells": 72, "maximum_evidence_bytes": 16*1024**2,
                    "maximum_blind_audio_bytes": 4*1024**2},
                "asr_proposal": {"setup_wall_seconds": 1800, "family_wall_seconds": 600,
                    "clip_wall_seconds": 120, "host_wall_seconds": 3600, "workflow_minutes": 70,
                    "cpu_threshold_seconds": {"setup": 3600, "qwen06": 600, "sensevoice": 600, "whisper": 1800},
                    "cpu_total_threshold_seconds": 6600, "maximum_cumulative_audio_seconds": 216,
                    "prior_audio_transport_reservation_reusable": False,
                    "new_complete_data_package_response_body_reservation_bytes": 16*1024**2,
                    "maximum_derived_wav_payload_bytes": 6*12*16000*2,
                    "proposed_mixed_scope_acquisition_reservation_bytes": 6056574976,
                    "reservation_slack_bytes": 385875968,
                    "total_acquisition_cap_bytes": 6*1024**3,
                    "resource_scope_requires_fresh_review": True},
                "caveats": ["RSS/disk thresholds are sampled, not hard continuous limits.",
                    "TTS RLIMIT_CPU is per process, not aggregate process-tree CPU.",
                    "Transport input body caps are not all-wire traffic caps.",
                    "Weight cache is not established. No download, installation or runtime check has run.",
                    "Model/native-I/O preflight may use fixed synthetic numeric samples but no model construction or audio generation."]},
            "stage_barrier": ["All six fixed TTS cells must finish once without failure/cap; no early ASR. Successful cells with signal-quality flags are quarantined before ASR.",
                "Verify genuine claims, complete runtime/source/model custody, 24k float source and 16k mono PCM16 derivative bytes, PCM/frames and EOS termination.",
                "Before ASR, quarantine every member of each same-batch equal-WAV or equal-PCM group symmetrically; quarantine any cell overlapping frozen historical consumed/selection WAV or PCM identities. Preserve all six TTS outcomes and all eighteen slots with not_run_duplicate or not_run_quality reasons; never reuse old transcripts or regenerate.",
                "Freeze all six actual WAV/PCM/frame identities, history-comparison inventory, deterministic quarantine decisions and fresh eligible-only blind job/source inventory before releasing any ASR. Each eligible unique new cell permits exactly three new slots; maximum eighteen. No eligible cells means no ASR release.",
                "Native policies, decode configurations, retained source closure, resource scope and worst-case evidence packaging must be frozen before audio-forward.",
                "TTS and ASR releases are independently reviewed; successful TTS does not automatically release ASR."],
            "publication": {"audio_repository": "jiying2007/kws-data",
                "audio_prefix": "datasets/k1-punctuation6-v1/", "evidence_repository": "jiying2007/kws-pipeline",
                "evidence_prefix": "research/k1-punctuation6-v1/evidence/",
                "transport": "TTS runs in kws-data and publishes WAVs only to same-repository create-only audio refs with its scoped GITHUB_TOKEN. Pipeline ASR reads the fixed verified data commit and publishes audio-free machine evidence to pipeline create-only refs; no Actions artifacts or WAV-bearing pipeline/log transport.",
                "tts_audio_derivation": "Each cell has one native 24k mono float32 source and one fixed 2:3 scipy-resample-poly 16k mono PCM16 derivative. Bind original float values, raw WAV, derived WAV, PCM and frame geometry in the same cell receipt; no additional synthesis, gain or trimming.",
                "data_read_path": "Read the one complete data-result ZIP as raw-content chunks and manifest at one verified immutable kws-data commit. Compressed archive <=12MiB plus manifest <=16KiB; <=16MiB reserved total data response bodies. No unaccounted base64/API download path. The derived WAV payload (<2.31MB) is consumed-audio size, not download size.",
                "data_reader_requests": {"maximum_gets": 49, "manifest_gets": 1, "maximum_chunk_gets": 48, "attempts_per_path": 1, "retries": 0, "redirects": 0, "manifest_body_counted": True, "other_data_body_requests_allowed": False},
                "transport_bounds": {"compressed_archive_bytes": 12*1024**2, "chunk_bytes": 256*1024, "maximum_chunks": 48, "outer_manifest_bytes": 16*1024, "per_member_wav_and_pcm_readback_required": True, "future_same_repository_write_tested": False},
                "tts_output_members": "Six derived PCM WAVs, six native FLOAT WAVs, and bounded allowlisted source/generation/EOS/resource JSON receipts, only in kws-data.",
                "no_cross_repository_credential": True, "no_persistent_token": True,
                "excluded": ["weights", "credentials", "environments", "unfiltered_logs", "private_human_history", "private_training_recipe"],
                "output_rights_scope": "research/development; commercial output rights not established",
                "model_license_is_not_output_rights_guarantee": True},
            "scientific_limits": {"punctuation_improvement_guaranteed": False, "human_gold": False,
                "training_admitted": False, "generalization_claim": False, "statistical_inference_claim": False, "shipping_approved": False,
                "normalization": "Existing fixed punctuation/whitespace removal only; no homophone, script, case or Unicode folding.",
                "forbidden_substitutions": ["沃→窝", "屋→窝", "我→窝"],
                "agreement": "All three successful, fully native/source/config-bound outputs must normalize identically and nonempty; no majority voting."}}


def expected_coverage():
    return {"schema": "k1-punctuation6-training-coverage-proposal-v1", "experiment": EXPERIMENT,
            "status": "disabled_pending_root_freeze_before_asr", "execution_enabled": False,
            "new_cells": [x["cell_id"] for x in cells()],
            "old_train_only_candidates": [{"audio_id": clip, "source_id": "qwen12-"+sha,
                "group": group, "required_actual_text": text, "formal_admission": False}
                for clip, sha, group, text in OLD_CANDIDATES],
            "minimum_support": [{"group": "new-k1", "eligible_members": [x["cell_id"] for x in cells()],
                "required_actual_text": "你好小窝", "minimum_legally_admitted": 1}] +
                [{"group": group, "eligible_members": ["qwen12-"+sha for _, sha, g, _ in OLD_CANDIDATES if g == group],
                  "required_actual_text": text, "minimum_legally_admitted": 1}
                 for group, text in (("old-k2", "小窝小窝"), ("old-repeat-nihao", "你好你好"), ("old-suffix-kw2", "窝小窝"))],
            "all_legally_admitted_rows_required": True, "maximum_training_rows": 12,
            "member_policy": "Six new cells are preregistered before ASR. Every legally admitted new/old candidate is included, with real label/kind preserved. No ASR-dependent cell selection or cross-group coverage substitution.",
            "admission_policy": "Actual source/rights/history/native verification and genuine registered train-only roots remain required for both old and new rows. This proposal does not register roots or admit rows.",
            "historical_contract": {"coverage_sha256": OLD_COVERAGE, "recipe_sha256": OLD_RECIPE,
                "old_twelve_status": "INSUFFICIENT_ADMITTED_COVERAGE", "old_group_candidate_counts": [0,2,3,1],
                "old_asr_slots_exhausted": 36, "mutate_old_contracts": False},
            "derived_recipe_required": {"update_only": ["new data identities", "new coverage binding", "source closure", "analytic workload bounds", "reviewed resource proposal"],
                "preserve": ["seed9101101", "TinyStreamingRNN32x64x5/logmel", "all objective weights", "AdamW and learning rate/weight decay/gradient clip", "1000 updates", "single fit", "last-only checkpoint", "no augmentation", "no warm start", "thresholds0.55", "no retry"],
                "maximum_extra_neural_readback_passes": 36, "readback_formula": "N + 24 with N <= 12",
                "recalculate_from_new_audio_geometry": True, "old_24_24_second_bound_reused": False,
                "training_execution_release": "separate future independent review; disabled"}}


def validate_plan(value):
    require(canonical(value) == canonical(expected_plan()), "fixed proposed plan drift")
    return value


def validate_coverage(value):
    require(canonical(value) == canonical(expected_coverage()), "fixed new coverage drift")
    return value


def initial_ledger(phase):
    require(phase in ("tts", "asr"), "fixed phase")
    rows = cells() if phase == "tts" else slots()
    key = "cell_id" if phase == "tts" else "slot_id"
    return [{"id": x[key], "status": "not_run", "attempts": 0} for x in rows]


def validate_ledger(phase, ledger, screening=None):
    expected = initial_ledger(phase)
    require(type(ledger) is list and len(ledger) == len(expected), "retain every denominator slot")
    skips = {}
    if phase == "asr" and screening is not None:
        require(type(screening) is list and [x.get("cell_id") for x in screening] == [x["cell_id"] for x in cells()], "fixed screening cells")
        require(all(x.get("disposition") in ("eligible", "not_run_duplicate", "not_run_quality") for x in screening), "screening disposition")
        skips = {x["cell_id"]: x["disposition"] for x in screening}
    terminal_stop = False
    for index, (actual, initial) in enumerate(zip(ledger, expected)):
        require(type(actual) is dict and set(actual) == {"id", "status", "attempts"}, "ledger fields")
        require(actual["id"] == initial["id"], "fixed fresh slot identity and order")
        require(type(actual["attempts"]) is int, "attempts must be integer, never boolean")
        status = actual["status"]
        disposition = skips.get(slots()[index]["cell_id"], "eligible") if phase == "asr" else "eligible"
        if disposition != "eligible":
            require(status == disposition and actual["attempts"] == 0, "quarantined slot cannot run")
            continue
        require(status in ("not_run", "started", "complete", "failed_no_retry"), "slot status")
        require(actual["attempts"] == (0 if status == "not_run" else 1), "one consumed call per started slot")
        require(not terminal_stop or status == "not_run", "no continuation after not_run/start/failure")
        terminal_stop = terminal_stop or status != "complete"
    return {"denominator": len(expected), "consumed": sum(r["attempts"] for r in ledger),
            "complete": sum(r["status"] == "complete" for r in ledger)}


def validate_tts_barrier_metadata(rows, historical_audio):
    """Metadata-only sanity check; real byte, capture and custody review is separate."""
    require(type(rows) is list and len(rows) == 6, "all six predeclared cells required")
    require(type(historical_audio) is list and bool(historical_audio), "frozen historical audio comparison inventory required")
    for item in historical_audio:
        require(type(item) is dict and set(item) == {"source_id", "wav_sha256", "pcm_sha256"}, "historical comparison fields")
        require(type(item["source_id"]) is str and item["source_id"], "historical source identity")
        require(all(type(item[k]) is str and SHA.fullmatch(item[k]) for k in ("wav_sha256", "pcm_sha256")), "historical audio hash")
    required = {"cell_id", "voice", "text", "seed", "attempts", "status", "wav_sha256", "pcm_sha256",
                "raw_wav_sha256", "raw_float_sha256", "source_frames", "frames", "sample_rate_hz",
                "channels", "sample_width_bytes", "eos_verified", "generation_iterations", "quality_flags"}
    for row, cell in zip(rows, cells()):
        require(type(row) is dict and set(row) == required, "complete bounded barrier metadata")
        for key in ("cell_id", "voice", "text", "seed"):
            require(canonical(row[key]) == canonical(cell[key]), "predeclared TTS cell drift")
        require(type(row["attempts"]) is int and row["attempts"] == 1 and row["status"] == "complete", "all six must complete once")
        for key in ("wav_sha256", "pcm_sha256", "raw_wav_sha256", "raw_float_sha256"):
            require(type(row[key]) is str and SHA.fullmatch(row[key]), "measured hash syntax")
        for key in ("source_frames", "frames", "sample_rate_hz", "channels", "sample_width_bytes", "generation_iterations"):
            require(type(row[key]) is int, "integer audio geometry")
        require(0 < row["source_frames"] < 12*24000 and 0 < row["frames"] < 12*16000, "duration limit reached or invalid")
        require(row["frames"] == (row["source_frames"]*2+2)//3, "fixed 2:3 resample frame count")
        require((row["sample_rate_hz"], row["channels"], row["sample_width_bytes"]) == (16000,1,2), "derived PCM format")
        require(row["eos_verified"] is True and 0 < row["generation_iterations"] < 120, "EOS unverified or token limit reached")
        require(type(row["quality_flags"]) is list and all(x in ("silent", "source_peak_ge_one", "normalized_peak_ge_one", "derived_pcm_silent") for x in row["quality_flags"]), "recognized signal-quality flags only")
    screening = []
    for row in rows:
        peers = [r["cell_id"] for r in rows if r["cell_id"] != row["cell_id"] and
                 (r["wav_sha256"] == row["wav_sha256"] or r["pcm_sha256"] == row["pcm_sha256"])]
        history = sorted({r["source_id"] for r in historical_audio if
                         r["wav_sha256"] == row["wav_sha256"] or r["pcm_sha256"] == row["pcm_sha256"]})
        disposition = "not_run_duplicate" if peers or history else ("not_run_quality" if row["quality_flags"] else "eligible")
        screening.append({"cell_id": row["cell_id"], "disposition": disposition,
                          "same_batch_matches": peers, "historical_matches": history,
                          "quality_flags": list(row["quality_flags"])})
    return {"metadata_status": "shape_only_not_execution_or_admission", "cells": 6,
            "audio_seconds": sum(r["frames"] for r in rows)/16000,
            "new_asr_slots_max": 3*sum(r["disposition"] == "eligible" for r in screening),
            "historical_inventory_sha256": digest(historical_audio), "screening": screening}
