#!/usr/bin/env python3
"""Real English63 fixed-keyword Zipformer FP32/INT8 event and CPU study.

A pretrained-model research comparator, NOT trained English FSMN or product KWS.
Uses unchanged source audio and official HEY/MERO lexicon tokens. All clip
labels remain synthetic requested intent, not acoustic truth or FAR/FRR.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import resource
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcm_frontend_parity as source

SCHEMA = "english63-zh-en-frozen-3m-quant-event-comparator-v1"
MODEL_SHA = "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
MODEL_BYTES = 32885699
MODEL_DIR = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
MODEL_STEM = "epoch-13-avg-2-chunk-"
MODES = {
    "fp32_c16": ("16", False),
    "int8_c16": ("16", True),
    "int8_c8": ("8", True),
}
PROMPTS = {
    "Hey Mero": "wake",
    "Hey, Mero!": "wake",
    "Hey Vosen": "wake",
    "Hey, Vosen!": "wake",
    "Hey Wosen": "near_miss",
    "Hey Miro": "near_miss",
    "Hey Rosen": "near_miss",
    "What time is it?": "ordinary_negative",
    "Hey, can you hear me?": "ordinary_negative",
}
TRAIN_VOICES = {"dylan", "eric", "ono_anna", "sohee", "vivian"}
DEV_VOICES = {"serena", "uncle_fu"}
SAMPLES_PER_FEED = 1600
END_TAIL_SAMPLES = 10560
KEYWORD_TEXT = "HH EY1 M EH1 R OW0 @HEY_MERO\n"
EXPECTED_PCM_SAMPLES = 2101760


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def correct_spelling_and_role(row):
    cell, voice, family = row.get("cell_id"), row.get("voice_id"), row.get("source_family_id")
    text = row.get("requested_text")
    role = row.get("role")
    require(type(cell) is str and source.CELL.fullmatch(cell) and
            voice in TRAIN_VOICES | DEV_VOICES and
            family == "qwen3-preset-" + voice and
            role == ("train-candidate" if voice in TRAIN_VOICES else "development-only") and
            type(text) is str and text in PROMPTS and
            row.get("requested_intent") == PROMPTS[text] and
            row.get("human_gold") is False and
            row.get("training_admitted") is False and
            row.get("trusted_negative") is False and
            row.get("phonetic_label_authenticated") is False and
            row.get("independent_evaluation") is False,
            "EN_ZHEN_REVIEW_ROW_UNSAFE")
    return True


def load_pinned_audio(tts_zip, review_json):
    """Source bytes and seven speaker roles must match the original evidence."""
    require(tts_zip.is_absolute() and review_json.is_absolute() and
            tts_zip.is_file() and review_json.is_file() and
            not tts_zip.is_symlink() and not review_json.is_symlink() and
            tts_zip.stat().st_size < 16 * 1024 * 1024 and
            review_json.stat().st_size < 1024 * 1024 and
            sha256_file(tts_zip) == source.TTS_SHA256 and
            sha256_file(review_json) == source.REVIEW_SHA256,
            "EN_ZHEN_SOURCE_NOT_EXACT")
    review = json.loads(review_json.read_bytes())
    rows = review.get("rows")
    require(type(review) is dict and
            review.get("schema") == "english63-independent-evidence-joined-review-priority-v1" and
            review.get("status") == "MACHINE_EVIDENCE_JOINED_REVIEW_NOT_GOLD" and
            review.get("source_tts_run_id") == 43242278962 and
            review.get("audio_rows_reused") == 63 and
            review.get("human_gold") is False and
            review.get("training_admitted") is False and
            type(rows) is list and len(rows) == 63,
            "EN_ZHEN_REVIEW_POLICY_DRIFT")
    found = {}
    prompts = set()
    with zipfile.ZipFile(tts_zip) as archive:
        all_audio = {n for n in archive.namelist() if n.endswith("/audio-16k.wav")}
        require(len(all_audio) == 63, "EN_ZHEN_WAV_MEMBER_COUNT")
        for row in rows:
            correct_spelling_and_role(row)
            cell = row["cell_id"]
            prompt = (row["voice_id"], row["requested_text"])
            require(cell not in found and prompt not in prompts,
                    "EN_ZHEN_DUPLICATE_CLIP_OR_PROMPT")
            prompts.add(prompt)
            member = "english-seven-voice-actual/capture/cells/" + cell + "/audio-16k.wav"
            require(member in all_audio and archive.getinfo(member).file_size < 400000,
                    "EN_ZHEN_WAV_MEMBER_MISSING")
            all_audio.remove(member)
            pcm = source.pcm_from_wav(archive.read(member), row)
            found[cell] = (row, pcm)
        require(not all_audio and
                prompts == {(v, p) for v in TRAIN_VOICES | DEV_VOICES for p in PROMPTS} and
                sum(len(pcm) for _, pcm in found.values()) == EXPECTED_PCM_SAMPLES,
                "EN_ZHEN_SOURCE_ROSTER_DRIFT")
    return found


def model_component_names(mode):
    require(mode in MODES, "EN_ZHEN_UNKNOWN_VARIANT")
    chunk, quant = MODES[mode]
    stem = MODEL_STEM + chunk + "-left-64"
    ending = ".int8.onnx" if quant else ".onnx"
    return {
        "encoder": "encoder-" + stem + ending,
        "decoder": "decoder-" + stem + ".onnx",
        "joiner": "joiner-" + stem + ending,
        "tokens": "tokens.txt",
        "lexicon": "en.phone",
    }


def extract_model(archive, target, mode):
    """Never trust tar entries, even after exact release SHA verification."""
    require(archive.is_absolute() and archive.is_file() and
            not archive.is_symlink() and archive.stat().st_size == MODEL_BYTES and
            sha256_file(archive) == MODEL_SHA and
            target.is_dir() and not target.is_symlink(),
            "EN_ZHEN_MODEL_ARCHIVE_IDENTITY")
    needed = model_component_names(mode)
    wanted = {MODEL_DIR + "/" + v: k for k, v in needed.items()}
    seen = set()
    parts = {}
    with tarfile.open(archive, "r:bz2") as tar:
        members = tar.getmembers()
        require(0 < len(members) <= 600 and
                sum(x.size for x in members if x.isfile()) < 180 * 1024 * 1024,
                "EN_ZHEN_TAR_TOO_LARGE")
        for member in members:
            p = PurePosixPath(member.name)
            require(not p.is_absolute() and p.parts and p.parts[0] == MODEL_DIR and
                    all(v not in ("", ".", "..") for v in p.parts) and
                    member.name not in seen and
                    (member.isdir() or member.isfile()) and
                    0 <= member.size <= 40 * 1024 * 1024,
                    "EN_ZHEN_TAR_MEMBER_UNSAFE")
            seen.add(member.name)
            if member.name not in wanted:
                continue
            require(member.isfile() and 0 < member.size <= 14 * 1024 * 1024,
                    "EN_ZHEN_MISSING_COMPONENT")
            destination = target / needed[wanted[member.name]]
            with tar.extractfile(member) as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            require(destination.stat().st_size == member.size,
                    "EN_ZHEN_COMPONENT_SHORT_WRITE")
            parts[wanted[member.name]] = destination
    require(set(parts) == set(needed), "EN_ZHEN_COMPONENT_ROSTER")
    return parts


def official_keyword(parts, directory):
    """The verified official en.phone transcription, no invented Vosen G2P."""
    with parts["lexicon"].open(encoding="utf-8") as stream:
        entry = {key: set() for key in ("HEY", "MERO", "VOSEN")}
        for line in stream:
            tokens = line.split()
            if tokens and tokens[0] in entry:
                entry[tokens[0]].add(tuple(tokens[1:]))
    require(("HH", "EY1") in entry["HEY"] and
            ("M", "EH1", "R", "OW0") in entry["MERO"] and
            not entry["VOSEN"],
            "EN_ZHEN_LEXICON_UNSUPPORTED")
    with parts["tokens"].open(encoding="utf-8") as f:
        available = {line.split()[0] for line in f if line.strip()}
    require(all(phone in available for phone in KEYWORD_TEXT.split()[:-1]),
            "EN_ZHEN_PHONE_TOKEN_NOT_FOUND")
    path = directory / "original_official_mero.txt"
    path.write_text(KEYWORD_TEXT, encoding="utf-8")
    return path


def stream_events(spotter, samples):
    """Per-record live decoder events, not continuous-background FAR."""
    import numpy as np
    stream = spotter.create_stream()
    events = []
    steps = 0
    fed = 0
    def drain():
        nonlocal steps
        while spotter.is_ready(stream):
            steps += 1
            require(steps <= 20000, "EN_ZHEN_UNBOUNDED_DECODE")
            spotter.decode_stream(stream)
            value = spotter.get_result(stream)
            if value:
                mapped = str(value).upper().strip().replace(" ", "_")
                require(mapped == "HEY_MERO" and len(events) < 16,
                        "EN_ZHEN_UNEXPECTED_EVENT")
                tokens = list(spotter.tokens(stream))
                stamps = [float(x) for x in spotter.timestamps(stream)]
                require(len(tokens) == len(stamps) and
                        all(math.isfinite(x) and x >= 0 for x in stamps),
                        "EN_ZHEN_BAD_EVENT_TIMESTAMPS")
                events.append({
                    "keyword": mapped,
                    "available_after_input_sec": round(fed / 16000, 4),
                    "token_count": len(tokens),
                    "timestamp_count": len(stamps),
                    "is_streaming_event": True,
                })
                spotter.reset_stream(stream)
    for start in range(0, len(samples), SAMPLES_PER_FEED):
        packet = samples[start:start + SAMPLES_PER_FEED]
        stream.accept_waveform(16000, packet)
        fed += len(packet)
        drain()
    stream.accept_waveform(16000, np.zeros(END_TAIL_SAMPLES, dtype=np.float32))
    fed += END_TAIL_SAMPLES
    drain()
    stream.input_finished()
    drain()
    return events, steps


def verify_receipt(result):
    """No model imports: hostile-receipt checks executed in standard CI."""
    require(type(result) is dict and result.get("schema") == SCHEMA and
            result.get("status") == "REAL_PRETRAINED_SYNTHETIC_EVENTS_NOT_QUALIFICATION" and
            result.get("model_archive_sha256") == MODEL_SHA and
            result.get("source_tts_zip_sha256") == source.TTS_SHA256 and
            result.get("source_review_json_sha256") == source.REVIEW_SHA256 and
            result.get("runner_sha256") == sha256_bytes(Path(__file__).read_bytes()) and
            result.get("mode") in MODES and
            result.get("source_clips") == 63 and
            result.get("source_pcm_samples") == EXPECTED_PCM_SAMPLES and
            result.get("keyword_line") == KEYWORD_TEXT.strip() and
            result.get("new_tts_asr_training_calls") == 0 and
            result.get("actual_human_speakers") == 0 and
            result.get("human_gold") is False and
            result.get("training_admitted") is False and
            result.get("independent_frr") == "NOT_RUN" and
            result.get("continuous_far_per_hour") == "NOT_RUN" and
            result.get("ssc305_qualified") is False and
            result.get("model_promotion_allowed") is False,
            "EN_ZHEN_RECEIPT_UNSAFE")
    rows = result.get("clips")
    require(type(rows) is list and len(rows) == 63 and
            len({x.get("cell_id") for x in rows}) == 63,
            "EN_ZHEN_RECEIPT_CLIPS")
    for row in rows:
        require(type(row) is dict and
                type(row.get("cell_id")) is str and source.CELL.fullmatch(row["cell_id"]) and
                row.get("requested_text") in PROMPTS and
                row.get("requested_intent") == PROMPTS[row["requested_text"]] and
                row.get("role") in ("train-candidate", "development-only") and
                type(row.get("wav_sha256")) is str and len(row["wav_sha256"]) == 64 and
                type(row.get("pcm_sha256")) is str and len(row["pcm_sha256"]) == 64 and
                type(row.get("events")) is list and len(row["events"]) < 16 and
                all(e.get("keyword") == "HEY_MERO" and
                    e.get("is_streaming_event") is True for e in row["events"]) and
                row.get("human_gold") is False and
                row.get("trusted_negative") is False and
                row.get("training_admitted") is False,
                "EN_ZHEN_RECEIPT_ROW_UNSAFE")
    return True


def execute(mode, audio_zip, review_json, model_archive, output):
    import numpy as np
    import sherpa_onnx
    require(mode in MODES and output.is_absolute() and
            output.parent.is_dir() and not output.parent.is_symlink() and
            not output.exists(), "EN_ZHEN_OUTPUT_CREATE_ONLY")
    clips = load_pinned_audio(audio_zip, review_json)
    with tempfile.TemporaryDirectory(prefix="en63-zh-en-quant-") as name:
        directory = Path(name)
        parts = extract_model(model_archive, directory, mode)
        keyword = official_keyword(parts, directory)
        spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(parts["tokens"]), encoder=str(parts["encoder"]),
            decoder=str(parts["decoder"]), joiner=str(parts["joiner"]),
            keywords_file=str(keyword), provider="cpu",
            num_threads=1, sample_rate=16000, feature_dim=80,
            max_active_paths=4, keywords_score=1.0,
            keywords_threshold=0.25, num_trailing_blanks=1)
        start = time.perf_counter()
        rows, counts = [], Counter()
        for cell in sorted(clips):
            row, pcm = clips[cell]
            events, steps = stream_events(spotter, pcm)
            counts[row["requested_text"]] += len(events)
            rows.append({
                "cell_id": cell,
                "source_family_id": row["source_family_id"],
                "role": row["role"], "requested_text": row["requested_text"],
                "requested_intent": row["requested_intent"],
                "wav_sha256": row["wav_sha256"],
                "pcm_sha256": row["pcm_sha256"], "decoder_steps": steps,
                "events": events, "human_gold": False,
                "training_admitted": False, "trusted_negative": False,
            })
        wall = time.perf_counter() - start
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {
            "schema": SCHEMA,
            "status": "REAL_PRETRAINED_SYNTHETIC_EVENTS_NOT_QUALIFICATION",
            "model_archive_sha256": MODEL_SHA,
            "source_tts_zip_sha256": source.TTS_SHA256,
            "source_review_json_sha256": source.REVIEW_SHA256,
            "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "mode": mode, "source_clips": len(rows),
            "source_pcm_samples": sum(len(pcm) for _, pcm in clips.values()),
            "keyword_line": keyword.read_text().strip(),
            "model_file_bytes": {part: parts[part].stat().st_size
                                 for part in ("encoder", "decoder", "joiner")},
            "host_inference_wall_sec_excludes_load": round(wall, 6),
            "host_inference_rtf_excludes_load": round(wall / 131.36, 7),
            "host_process_peak_rss_kib_includes_setup": peak_rss_kib,
            "events_by_requested_text_not_gold": dict(counts),
            "total_events": sum(len(r["events"]) for r in rows),
            "clips": rows,
            "new_tts_asr_training_calls": 0,
            "actual_human_speakers": 0,
            "human_gold": False, "training_admitted": False,
            "independent_frr": "NOT_RUN",
            "continuous_far_per_hour": "NOT_RUN",
            "ssc305_qualified": False,
            "model_promotion_allowed": False,
        }
        verify_receipt(result)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        output.chmod(0o600)
        return result


def compare_receipts(paths, output):
    require(len(paths) == 3 and len(set(paths)) == 3 and
            output.is_absolute() and output.parent.is_dir() and
            not output.exists(), "EN_ZHEN_COMPARE_OUTPUTS")
    reports = {}
    for path in paths:
        val = json.loads(path.read_text())
        verify_receipt(val)
        require(val["mode"] not in reports, "EN_ZHEN_COMPARE_DUPLICATE")
        reports[val["mode"]] = val
    require(set(reports) == set(MODES), "EN_ZHEN_COMPARE_MODE_ROSTER")
    base = {r["cell_id"]: r for r in reports["fp32_c16"]["clips"]}
    outputs = {}
    for mode, value in reports.items():
        changed = []
        for row in value["clips"]:
            old = base[row["cell_id"]]
            require((row["wav_sha256"],row["pcm_sha256"],
                     row["requested_text"],row["role"]) ==
                    (old["wav_sha256"],old["pcm_sha256"],
                     old["requested_text"],old["role"]),
                    "EN_ZHEN_COMPARE_SOURCE_CHANGED")
            if len(row["events"]) != len(old["events"]):
                changed.append(row["cell_id"])
        outputs[mode] = {
            "total_events": value["total_events"],
            "clips_with_event": sum(bool(r["events"]) for r in value["clips"]),
            "changed_event_count_clips_vs_fp32_c16": len(changed),
            "host_inference_wall_sec_excludes_load":
                value["host_inference_wall_sec_excludes_load"],
            "host_process_peak_rss_kib_includes_setup":
                value["host_process_peak_rss_kib_includes_setup"],
            "model_file_bytes": value["model_file_bytes"],
        }
    result = {
        "schema": "english63-zh-en-quant-comparison-summary-v1",
        "model_archive_sha256": MODEL_SHA,
        "source_tts_zip_sha256": source.TTS_SHA256,
        "source_review_json_sha256": source.REVIEW_SHA256,
        "source_clips": 63,
        "modes": outputs,
        "source_requested_text_is_gold": False,
        "model_selection_made": False,
        "real_far_frr_measured": False,
        "target_arm_performance_measured": False,
        "model_promotion_allowed": False,
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=sorted(MODES))
    p.add_argument("--tts-zip", type=Path)
    p.add_argument("--review-json", type=Path)
    p.add_argument("--model-archive", type=Path)
    p.add_argument("--compare", type=Path, nargs=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.compare:
            require(args.mode is None and args.tts_zip is None and
                    args.review_json is None and args.model_archive is None,
                    "EN_ZHEN_NO_MIXED_MODE")
            result = compare_receipts(args.compare, args.output)
            print(json.dumps(result, sort_keys=True))
        else:
            require(args.mode and args.tts_zip and args.review_json and
                    args.model_archive, "EN_ZHEN_EXPLICIT_INPUTS")
            result = execute(args.mode, args.tts_zip, args.review_json,
                             args.model_archive, args.output)
            print(json.dumps({
                "mode": result["mode"],
                "total_events": result["total_events"],
                "host_inference_wall_sec_excludes_load":
                    result["host_inference_wall_sec_excludes_load"],
                "model_promotion_allowed": False,
            }, sort_keys=True))
    except (ValueError, OSError, KeyError, RuntimeError,
            zipfile.BadZipFile, tarfile.TarError) as exc:
        print("EN_ZHEN_QUANT_STOP: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
