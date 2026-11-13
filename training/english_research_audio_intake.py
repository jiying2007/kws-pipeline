"""Reuse exactly the original English-63 WAV bytes as a non-admitted research corpus.

No audio/model generation, training, truth-label certification or cross-language
CTC conversion. This is a deliberately separate acoustic intake from formal
training admission; weak requested intents remain diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_research_label_bridge as labels

POLICY = "english63-original-audio-weak-intake-v1"
SOURCE_TTS_RUN_ID = 43242278962
SOURCE_TTS_ARTIFACT_ID = 13954744587
SOURCE_TTS_ZIP_SHA256 = "f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc"
MAX_ZIP_BYTES = 32 * 1024 * 1024
MAX_UNPACKED_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024
MAX_MEMBERS = 350
DIRS = {
    "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY": "weak_diagnostic_only",
    "EXPOSED_DEVELOPMENT_NO_TUNE": "development_no_tune",
    "QUARANTINE_AMBIGUOUS_NEARPHONE": "nearphone_quarantine",
}
WAV_MEMBER = re.compile(
    r"english-seven-voice-actual/capture/cells/"
    r"(en-[0-9a-f]{24})/(audio-16k|native-f32)\\.wav\\Z"
)


def need(condition: bool, why: str) -> None:
    if not condition:
        raise ValueError(why)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_wav(raw: bytes) -> str:
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            need(wav.getnchannels() == 1 and wav.getsampwidth() == 2 and
                 wav.getframerate() == 16000 and wav.getcomptype() == "NONE" and
                 400 <= wav.getnframes() <= 16000 * 120,
                 "EN_AUDIO_WAV_GEOMETRY")
            frames = wav.getnframes()
            pcm = wav.readframes(frames)
            need(len(pcm) == frames * 2 and not wav.readframes(1),
                 "EN_AUDIO_TRUNCATED_PCM")
    except (wave.Error, EOFError) as exc:
        raise ValueError("EN_AUDIO_WAV_INVALID") from exc
    return sha(pcm)


def verify_zip(source: Path) -> dict:
    need(source.is_absolute() and source.is_file() and
         not source.is_symlink() and 0 < source.stat().st_size <= MAX_ZIP_BYTES,
         "EN_AUDIO_SOURCE_PATH_INVALID")
    need(sha(source.read_bytes()) == SOURCE_TTS_ZIP_SHA256,
         "EN_AUDIO_ORIGINAL_TTS_DIGEST_DRIFT")
    sounds = {}
    pcm_cells, native_cells = set(), set()
    with zipfile.ZipFile(source) as z:
        infos = z.infolist()
        need(0 < len(infos) <= MAX_MEMBERS, "EN_AUDIO_ZIP_MEMBER_COUNT")
        seen_paths = set()
        total = 0
        for item in infos:
            name = item.filename
            path = PurePosixPath(name)
            mode = item.external_attr >> 16
            need(type(name) is str and bool(name) and
                 not path.is_absolute() and "\\" not in name and
                 "\x00" not in name and all(p not in (".", "..", "") for p in path.parts)
                 and name not in seen_paths and not stat.S_ISLNK(mode),
                 "EN_AUDIO_ZIP_TRAVERSAL_OR_ALIAS")
            seen_paths.add(name)
            need(0 <= item.file_size <= MAX_MEMBER_BYTES,
                 "EN_AUDIO_ZIP_MEMBER_OVERSIZE")
            total += item.file_size
            need(total <= MAX_UNPACKED_BYTES, "EN_AUDIO_ZIP_UNPACK_BUDGET")
            if item.is_dir() or not name.lower().endswith(".wav"):
                continue
            match = WAV_MEMBER.fullmatch(name)
            need(match is not None, "EN_AUDIO_UNEXPECTED_WAV_MEMBER")
            cell_id, kind = match.groups()
            if kind == "native-f32":
                # The original archive also retains 63 FLOAT/native WAVs.
                # They are provenance companions, NOT PCM16 trainer inputs.
                need(cell_id not in native_cells, "EN_AUDIO_DUPLICATE_NATIVE_CELL")
                native_cells.add(cell_id)
                continue
            need(cell_id not in pcm_cells, "EN_AUDIO_DUPLICATE_PCM16_CELL")
            pcm_cells.add(cell_id)
            raw = z.read(item)
            need(len(raw) == item.file_size, "EN_AUDIO_ZIP_MEMBER_SHORT_READ")
            identity = sha(raw)
            need(identity not in sounds, "EN_AUDIO_DUPLICATE_WAV")
            sounds[identity] = {"wav": raw, "pcm_sha256": verify_wav(raw),
                                "cell_id": cell_id}
    need(len(sounds) == len(pcm_cells) == len(native_cells) == 63 and
         pcm_cells == native_cells,
         "EN_AUDIO_EXPECTED_63_PCM16_PLUS_63_NATIVE_FLOAT_WAVS")
    return sounds


def prepare(source_zip: Path, review_path: Path) -> tuple[dict, dict]:
    """Read all verified bytes, then fail if any independent source row differs."""
    diagnostic = labels.diagnostic_plan(labels.read_verified_report(review_path))
    sounds = verify_zip(source_zip)
    roster = {row["wav_sha256"]: row for row in diagnostic["rows"]}
    need(len(roster) == 63 and set(roster) == set(sounds),
         "EN_AUDIO_SOURCE_WAV_ROSTER_DRIFT")
    for identity, row in roster.items():
        need(row["pcm_sha256"] == sounds[identity]["pcm_sha256"] and
             row["cell_id"] == sounds[identity]["cell_id"] and
             row["action"] in DIRS and row["training_admitted"] is False and
             row["human_gold"] is False and
             row["ctc_target_ids"] is None,
             "EN_AUDIO_PCM_OR_ADMISSION_DRIFT")
    need(diagnostic["admitted_training_rows"] == 0 and
         diagnostic["optimizer_steps"] == 0,
         "EN_AUDIO_FORMAL_TRAINING_NOT_AUTHORIZED")
    return sounds, diagnostic


def materialize(source_zip: Path, review_path: Path, output: Path) -> dict:
    need(output.is_absolute() and not output.exists() and
         output.parent.is_dir() and not output.parent.is_symlink(),
         "EN_AUDIO_DESTINATION_CREATE_ONLY")
    sounds, diagnostic = prepare(source_zip, review_path)
    # No output is opened until the original archive and all 63 PCM identities pass.
    output.mkdir(mode=0o700)
    out_rows = []
    for row in diagnostic["rows"]:
        bucket = DIRS[row["action"]]
        folder = output / bucket
        folder.mkdir(mode=0o700, exist_ok=True)
        file = folder / (row["cell_id"] + ".wav")
        audio = sounds[row["wav_sha256"]]["wav"]
        with file.open("xb") as stream:
            stream.write(audio)
        file.chmod(0o600)
        need(sha(file.read_bytes()) == row["wav_sha256"],
             "EN_AUDIO_MATERIALIZED_BYTES_CHANGED")
        out_rows.append({
            "cell_id": row["cell_id"],
            "source_family_id": row["source_family_id"],
            "role": row["role"],
            "action": row["action"],
            "local_relative_wav": bucket + "/" + file.name,
            "wav_sha256": row["wav_sha256"],
            "pcm_sha256": row["pcm_sha256"],
            "weak_wholeclip_class": row["weak_wholeclip_class"],
            "human_gold": False, "training_admitted": False,
            "trusted_negative": False, "formal_training_eligible": False,
            "ctc_target_ids": None,
        })
    result = {
        "schema": POLICY,
        "status": "EXACT_AUDIO_REUSED_FOR_WEAK_DIAGNOSTIC_ONLY",
        "source_tts_run_id": SOURCE_TTS_RUN_ID,
        "source_tts_artifact_id": SOURCE_TTS_ARTIFACT_ID,
        "source_tts_zip_sha256": SOURCE_TTS_ZIP_SHA256,
        "source_review_sha256": labels.SOURCE_BYTES_SHA256,
        "english_class_roster": labels.CLASSES,
        "head_kind": "whole-recording-3-class-not-ctc",
        "counts": diagnostic["counts"],
        "rows": out_rows,
        "no_audio_generated": True,
        "new_asr_calls": 0, "new_ipa_calls": 0,
        "optimizer_steps": 0, "training_admitted": False,
        "human_gold": False, "independent_human_validation": "NOT_RUN",
        "formal_training_eligible": False, "model_promotion_allowed": False,
    }
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    (output / "MANIFEST.json").chmod(0o600)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-zip", type=Path)
    parser.add_argument("--source-review", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if all(v is None for v in (args.tts_zip, args.source_review, args.output_dir)):
            print(json.dumps({"schema": POLICY, "source_tts_run_id":
                SOURCE_TTS_RUN_ID, "training_admitted": False, "optimizer_steps": 0}))
            return 0
        need(all(v is not None for v in
                 (args.tts_zip, args.source_review, args.output_dir)),
             "EN_AUDIO_ALL_EXPLICIT_INPUTS_REQUIRED")
        result = materialize(args.tts_zip, args.source_review, args.output_dir)
        print(json.dumps({"status": result["status"], "counts": result["counts"],
                          "training_admitted": False}, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile,
            zipfile.LargeZipFile) as exc:
        print("EN_AUDIO_INTAKE_STOP: " + type(exc).__name__ + ": " + str(exc),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
