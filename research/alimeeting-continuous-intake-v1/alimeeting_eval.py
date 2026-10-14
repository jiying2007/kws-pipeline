#!/usr/bin/env python3
"""Bounded AliMeeting Eval intake preparation, stdlib only; no model execution.

Default command is an offline plan. `acquire --allow-download` is deliberately
separate and must not be run as part of this preparation task. `audit` consumes
an already local archive and emits review candidates, never approved negatives.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tarfile
import tempfile
import time
import unicodedata
import urllib.request
import wave

SOURCE_URL = "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz"
SOURCE_PAGE = "https://www.openslr.org/119/"
OBSERVED_BYTES = 3673718355  # Previous HEAD observation; not a cryptographic lock.
MAX_ARCHIVE_BYTES = 4 * 1024**3
MAX_EXPANDED_BYTES = 12 * 1024**3
MAX_MEMBER_BYTES = 1024**3
MAX_MEMBERS = 4096
MAX_TEXTGRID_BYTES = 8 * 1024**2
EXPECTED_SESSIONS = 8
SAMPLE_RATE = 16000
TARGETS = ("你好小窝", "小窝小窝")
NEAR_FOR_REVIEW = ("小屋", "小我", "小沃", "小卧", "小窝")
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}\Z")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def plan():
    return {
        "schema": "alimeeting-eval-intake-plan-v1", "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE, "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "archive": "Eval_Ali.tar.gz", "previous_head_bytes": OBSERVED_BYTES,
        "publisher_sha256": None, "sha256_lock_status": "not-yet-acquired-or-locked",
        "selection": "all Eval_Ali_far/audio_dir/*.wav; fixed channel index 0; whole sessions",
        "transcripts": "Eval_Ali_far/textgrid_dir/*.TextGrid; all speaker tiers",
        "expected_sessions": EXPECTED_SESSIONS, "advertised_hours_not_denominator": 4,
        "network_default": False, "training_allowed": False, "models_run": False,
        "evidence_class": "proposed-development-evaluation-not-acquired",
        "sealed_qualification": False, "shipping_approved": False,
        "recalibration_required": True,
    }


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect rejected; re-review the official URL")


def acquire(out, allow_download=False, expected_sha256=None, *, opener=None,
            expected_bytes=OBSERVED_BYTES):
    check(allow_download, "network disabled; acquire requires --allow-download")
    if expected_sha256 is not None:
        check(bool(HEX256.fullmatch(expected_sha256)), "invalid expected SHA256")
    check(0 < expected_bytes <= MAX_ARCHIVE_BYTES, "invalid expected archive length")
    out = Path(out)
    out.mkdir(parents=False, exist_ok=False)
    partial = out / "Eval_Ali.tar.gz.partial"
    opener = opener or urllib.request.build_opener(NoRedirect()).open
    h, count, started = hashlib.sha256(), 0, time.monotonic()
    try:
        request = urllib.request.Request(SOURCE_URL, headers={"Accept-Encoding": "identity"})
        with opener(request, timeout=30) as response, partial.open("xb") as stream:
            check(response.geturl() == SOURCE_URL, "unreviewed response URL")
            check(response.status == 200, "expected HTTP 200 (no partial/resume)")
            check(response.headers.get("Content-Encoding", "identity") == "identity", "encoded response rejected")
            check(response.headers.get("Content-Length") == str(expected_bytes), "source length changed or missing")
            while True:
                data = response.read(1024 * 1024)
                if not data:
                    break
                count += len(data)
                check(count <= expected_bytes, "archive exceeds expected length")
                check(time.monotonic() - started <= 3600, "download exceeded one-hour budget")
                stream.write(data)
                h.update(data)
        check(count == expected_bytes, "truncated archive")
        check(expected_sha256 is None or h.hexdigest() == expected_sha256, "archive SHA256 mismatch")
        partial.rename(out / "Eval_Ali.tar.gz")
        receipt = plan() | {
            "schema": "alimeeting-eval-acquisition-v1", "archive_bytes": count,
            "archive_sha256": h.hexdigest(), "expected_sha256": expected_sha256,
            "hash_status": "verified-against-supplied-lock" if expected_sha256 else "first-acquisition-observation-only",
            "sha256_lock_status": "supplied-lock-verified" if expected_sha256 else "first-acquisition-observed-not-reviewed",
            "evidence_class": "acquired-development-candidates-pending-review",
            "approval_required_before_evaluation": True,
        }
        write_json(out / "acquisition.json", receipt)
        return receipt
    except BaseException as exc:
        # Authorized acquisition keeps partial bytes and a bounded failure receipt;
        # there is no resume or implicit retry and no successful-intake claim.
        retained = partial if partial.exists() else out / "Eval_Ali.tar.gz"
        failure = {"schema": "alimeeting-eval-acquisition-failed-v1",
                   "source_url": SOURCE_URL, "attempt_count": 1,
                   "exception_type": type(exc).__name__, "error": str(exc),
                   "elapsed_s": time.monotonic() - started,
                   "retained_file": retained.name if retained.exists() else None,
                   "retained_bytes": retained.stat().st_size if retained.exists() else 0,
                   "retained_sha256": digest_file(retained) if retained.exists() else None,
                   "accepted_negative_duration_s": 0, "acquisition_complete": False}
        try:
            write_json(out / "acquisition-failed.json", failure)
        except OSError:
            pass  # Preserve the original failure if the filesystem is full.
        raise


class PlainTarInfo(tarfile.TarInfo):
    def _proc_member(self, bundle):
        # Check BEFORE tarfile can consume an unbounded PAX/GNU metadata body.
        # Archive compatibility is deliberately narrow until actual intake.
        check(self.type not in (tarfile.XHDTYPE, tarfile.XGLTYPE,
              tarfile.SOLARIS_XHDTYPE, tarfile.GNUTYPE_LONGNAME,
              tarfile.GNUTYPE_LONGLINK, tarfile.GNUTYPE_SPARSE),
              "extended/PAX/GNU metadata or sparse header unsupported")
        return super()._proc_member(bundle)


def safe_member(member):
    raw = member.name
    check("\\" not in raw and "\x00" not in raw, "unsafe archive path")
    p = PurePosixPath(raw)
    check(not p.is_absolute() and ".." not in p.parts and p.parts, "unsafe archive path")
    check(member.isfile() or member.isdir(), "archive links/devices/special members rejected")
    check(not member.sparse, "sparse archive members rejected")
    check(0 <= member.size <= MAX_MEMBER_BYTES, "archive member size exceeds bound")
    return p


def selected_kind(path):
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "Eval_Ali_far":
            tail = parts[i + 1:]
            if len(tail) == 2 and tail[0] == "audio_dir" and tail[1].endswith(".wav"):
                return "audio"
            if len(tail) == 2 and tail[0] == "textgrid_dir" and tail[1].endswith(".TextGrid"):
                return "textgrid"
    return None


NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
Q = r'"(?:[^"\n]|""|\n)*"'


def quoted(token):
    return token[1:-1].replace('""', '"')


def textgrid_audit(data):
    """Strict standard long-text IntervalTier TextGrid; reject unsupported syntax.

    No dependency on Praat/pypinyin/ASR. Full manual listening remains required:
    textual screening is neither pronunciation-complete nor a negative label.
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        text = data.decode("utf-8-sig")
    header = re.fullmatch(
        rf'\s*File type\s*=\s*"ooTextFile"\s+Object class\s*=\s*"TextGrid"\s+'
        rf'xmin\s*=\s*({NUM})\s+xmax\s*=\s*({NUM})\s+tiers\?\s*<exists>\s+'
        rf'size\s*=\s*(\d+)\s+item\s*\[\]:\s*(.*)', text, re.S)
    check(header is not None, "unsupported/malformed TextGrid (only standard long text supported)")
    lo, hi, count = float(header[1]), float(header[2]), int(header[3])
    check(math.isfinite(lo) and math.isfinite(hi) and lo == 0 and hi > 0 and count > 0, "invalid TextGrid extent")
    tiers = re.split(r'(?m)^\s*item\s*\[(\d+)\]:\s*', header[4])
    check(not tiers[0].strip() and len(tiers) == 1 + count * 2, "TextGrid tier count mismatch")
    flags, intervals_total, annotation_gap_count = [], 0, 0
    for index in range(count):
        check(int(tiers[1 + index * 2]) == index + 1, "TextGrid tier index mismatch")
        body = tiers[2 + index * 2]
        tier = re.fullmatch(rf'class\s*=\s*"IntervalTier"\s+name\s*=\s*({Q})\s+'
                            rf'xmin\s*=\s*({NUM})\s+xmax\s*=\s*({NUM})\s+'
                            rf'intervals:\s*size\s*=\s*(\d+)\s*(.*)', body, re.S)
        check(tier is not None, "unsupported/malformed TextGrid tier")
        check(float(tier[2]) == lo and float(tier[3]) == hi, "tier extent mismatch")
        n = int(tier[4])
        chunks = re.split(r'(?m)^\s*intervals\s*\[(\d+)\]:\s*', tier[5])
        check(n > 0 and not chunks[0].strip() and len(chunks) == 1 + n * 2, "interval count mismatch")
        marks, spans, previous = [], [], lo
        for j in range(n):
            check(int(chunks[1 + j * 2]) == j + 1, "interval index mismatch")
            iv = re.fullmatch(rf'xmin\s*=\s*({NUM})\s+xmax\s*=\s*({NUM})\s+text\s*=\s*({Q})\s*', chunks[2 + j * 2], re.S)
            check(iv is not None, "malformed interval")
            start, end = float(iv[1]), float(iv[2])
            check(math.isfinite(start) and math.isfinite(end) and
                  start + 1e-6 >= previous and lo <= start <= end <= hi,
                  "interval timing overlap/out of range")
            annotation_gap_count += int(start - previous > 1e-6)
            previous = end
            marks.append(quoted(iv[3]))
            spans.append((start, end))
        annotation_gap_count += int(hi - previous > 1e-6)
        # Chinese-only concatenation also screens across adjacent annotations and
        # (SIL)/(OVERLAP) markers. False positives are intentionally review flags.
        pieces = ["".join(c for c in unicodedata.normalize("NFKC", mark)
                          if "\u3400" <= c <= "\u9fff") for mark in marks]
        chinese = "".join(pieces)
        owners = [j for j, piece in enumerate(pieces) for _ in piece]
        for phrase in TARGETS + NEAR_FOR_REVIEW:
            for match in re.finditer("(?=" + re.escape(phrase) + ")", chinese):
                first, last = owners[match.start()], owners[match.start()+len(phrase)-1]
                flags.append({"tier_index": index, "phrase": phrase,
                              "kind": "possible-full-target" if phrase in TARGETS else "near-or-partial-review",
                              "first_interval_index": first, "last_interval_index": last,
                              "start_s": spans[first][0], "end_s": spans[last][1],
                              "cross_interval": first != last,
                              "timing_precision": "annotation-interval-span-not-word-alignment"})
        intervals_total += n
    return {"duration_s": hi, "tier_count": count, "interval_count": intervals_total,
            "annotation_gap_count": annotation_gap_count,
            "annotation_gap_semantics": "unannotated-speaker-intervals-not-missing-PCM",
            "flags": flags, "text_scan_complete": True,
            "acoustic_target_absence_verified": False, "screening_complete": False,
            "targeted_audio_review_pending": True}


def validate_pcm16_riff(source):
    """Validate exact PCM16 RIFF (classic or explicit PCM extensible subtype)."""
    size = Path(source).stat().st_size
    with Path(source).open("rb") as stream:
        header = stream.read(12)
        check(len(header) == 12 and header[:4] == b"RIFF" and header[8:] == b"WAVE",
              "expected RIFF WAVE")
        check(struct.unpack("<I", header[4:8])[0] + 8 == size, "RIFF length mismatch")
        fmt, data_size = None, None
        while stream.tell() < size:
            header = stream.read(8)
            check(len(header) == 8, "truncated RIFF chunk header")
            kind, count = struct.unpack("<4sI", header)
            end = stream.tell() + count + (count & 1)
            check(end <= size, "truncated RIFF chunk")
            if kind == b"fmt ":
                check(fmt is None and 16 <= count <= 4096, "invalid/duplicate fmt chunk")
                fmt_bytes = stream.read(count)
                fmt = struct.unpack("<HHIIHH", fmt_bytes[:16])
            elif kind == b"data":
                check(data_size is None and fmt is not None, "duplicate or pre-fmt data chunk")
                data_size = count
            stream.seek(end)
        check(fmt is not None and fmt[1:] == (8, SAMPLE_RATE, SAMPLE_RATE * 16, 16, 16),
              "expected 8-channel 16-kHz PCM16 RIFF format")
        channel_mask = None
        if fmt[0] == 0xFFFE:
            check(len(fmt_bytes) == 40, "invalid extensible fmt size")
            cb_size, valid_bits, channel_mask = struct.unpack("<HHI", fmt_bytes[16:24])
            check(cb_size == 22 and valid_bits == 16 and
                  fmt_bytes[24:] == bytes.fromhex("0100000000001000800000aa00389b71"),
                  "extensible WAV must have explicit 16-bit PCM subtype")
            check(channel_mask == 0 or channel_mask.bit_count() == 8,
                  "extensible WAV channel mask/count mismatch")
        else:
            check(fmt[0] == 1, "unsupported WAV format tag")
        check(data_size is not None and data_size > 0 and data_size % 16 == 0,
              "empty or partial PCM frame")
        return {"frames": data_size // 16, "source_format_tag": fmt[0],
                "source_channel_mask": channel_mask}


def extract_mono(source, dest):
    """Fixed first channel, no gain, mixing, resampling, segmentation, or trimming."""
    riff = validate_pcm16_riff(source)
    pcm_hash = hashlib.sha256()
    with wave.open(str(source), "rb") as src:
        check(src.getcomptype() == "NONE" and src.getnchannels() == 8 and
              src.getsampwidth() == 2 and src.getframerate() == SAMPLE_RATE,
              "expected 8-channel 16-kHz PCM16; no implicit conversion permitted")
        frames = src.getnframes()
        check(frames == riff["frames"], "WAV frame count mismatch")
        check(0 < frames <= SAMPLE_RATE * 3600, "empty or over-one-hour session")
        with wave.open(str(dest), "wb") as dst:
            dst.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            remaining = frames
            while remaining:
                count = min(remaining, 16384)
                raw = src.readframes(count)
                check(len(raw) == count * 16, "truncated multichannel WAV")
                values = array.array("h")
                values.frombytes(raw)
                # Endianness is unchanged: frombytes/tobytes both use native order;
                # stride selects the original two bytes without numeric operations.
                mono = values[0::8].tobytes()
                pcm_hash.update(mono)
                dst.writeframesraw(mono)
                remaining -= count
    return {"source_format_tag": riff["source_format_tag"],
            "source_channel_mask": riff["source_channel_mask"],
            "sample_rate_hz": SAMPLE_RATE, "sample_count": frames,
            "duration_s": frames / SAMPLE_RATE, "channel_index": 0,
            "pcm_sha256": pcm_hash.hexdigest(), "wav_sha256": digest_file(dest)}


def audit(archive, archive_sha256, out, *, expected_sessions=EXPECTED_SESSIONS):
    check(bool(HEX256.fullmatch(archive_sha256)), "archive SHA256 required")
    archive, out = Path(archive), Path(out)
    check(stat.S_ISREG(archive.lstat().st_mode), "archive must be a regular, non-symlink file")
    initial_stat = archive.stat()
    check(initial_stat.st_size <= MAX_ARCHIVE_BYTES, "archive exceeds compressed bound")
    check(digest_file(archive) == archive_sha256, "archive SHA256 mismatch")
    out.mkdir(parents=False, exist_ok=False)
    (out / "mono").mkdir()
    (out / "textgrid").mkdir()
    seen, audio, texts, inventory = set(), {}, {}, []
    expanded = 0
    # Stream archive, never extractall. Only one original multichannel WAV is
    # materialized at a time, then removed; near-field audio is not retained.
    with tarfile.open(archive, "r|gz", tarinfo=PlainTarInfo) as bundle:
        for member in bundle:
            path = safe_member(member)
            key = path.as_posix()
            check(key not in seen, "duplicate canonical archive path")
            seen.add(key)
            check(len(seen) <= MAX_MEMBERS, "too many archive members")
            expanded += member.size
            check(expanded <= MAX_EXPANDED_BYTES, "expanded archive exceeds bound")
            kind = selected_kind(path) if member.isfile() else None
            inventory.append({"path": key, "bytes": member.size, "selected_kind": kind})
            if kind is None:
                continue
            stem = path.stem
            check(bool(SAFE_ID.fullmatch(stem)), "unsafe source identifier")
            check(stem not in (audio if kind == "audio" else texts), "duplicate selected source identifier")
            stream = bundle.extractfile(member)
            check(stream is not None, "unreadable archive member")
            with stream:
                if kind == "textgrid":
                    check(member.size <= MAX_TEXTGRID_BYTES, "TextGrid exceeds bound")
                    data = stream.read(MAX_TEXTGRID_BYTES + 1)
                    check(len(data) == member.size, "truncated TextGrid")
                    dest = out / "textgrid" / (stem + ".TextGrid")
                    with dest.open("xb") as dst:
                        dst.write(data)
                    info = textgrid_audit(data)
                    texts[stem] = info | {"member": key, "sha256": hashlib.sha256(data).hexdigest()}
                else:
                    h = hashlib.sha256()
                    with tempfile.NamedTemporaryFile(dir=out, suffix=".wav") as tmp:
                        copied = 0
                        for data in iter(lambda: stream.read(1024 * 1024), b""):
                            copied += len(data)
                            check(copied <= member.size, "member exceeded declared size")
                            h.update(data)
                            tmp.write(data)
                        check(copied == member.size, "truncated WAV member")
                        tmp.flush()
                        info = extract_mono(tmp.name, out / "mono" / (stem + ".wav"))
                    audio[stem] = info | {"member": key, "source_wav_sha256": h.hexdigest()}
    check(len(audio) == expected_sessions, "unexpected Eval far-WAV count; review archive layout, do not silently subsample")
    rows, used_texts, pcm_seen, sessions = [], set(), set(), set()
    for stem, wav_info in sorted(audio.items()):
        # Official baseline --mars mapping uses the first two underscore fields.
        short = "_".join(stem.split("_")[:2])
        candidates = {key for key in (stem, short) if key in texts}
        check(len(candidates) == 1, "missing or ambiguous WAV/TextGrid mapping")
        session = candidates.pop()
        check(session not in sessions, "multiple recordings of one session cannot multiply exposure")
        sessions.add(session)
        used_texts.add(session)
        text_info = texts[session]
        extent_delta = wav_info["duration_s"] - text_info["duration_s"]
        check(extent_delta >= -1 / SAMPLE_RATE,
              "WAV/TextGrid extent mismatch: transcript extends beyond audio")
        check(wav_info["pcm_sha256"] not in pcm_seen, "duplicate decoded mono PCM across sessions")
        pcm_seen.add(wav_info["pcm_sha256"])
        rows.append(wav_info | {"recording": "alimeeting-eval-" + session,
                    "session_id": session, "wav": "mono/" + stem + ".wav",
                    "textgrid": text_info,
                    "audio_after_textgrid_end_s": max(0.0, extent_delta),
                    "annotation_extent_matches_audio": abs(extent_delta) <= 1 / SAMPLE_RATE,
                    "timing_alignment_status": "official-timestamps-preserved-no-offset-inferred",
                    "review_status": "pending",
                    "independence_status": "pending-training-and-source-family-audit"})
    check(used_texts == set(texts), "orphan far-field TextGrid; review archive layout")
    final_stat = archive.stat()
    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
        check(getattr(initial_stat, field) == getattr(final_stat, field), "archive changed during audit")
    check(digest_file(archive) == archive_sha256, "archive changed during audit")
    report = plan() | {"schema": "alimeeting-eval-audit-v1", "archive_sha256": archive_sha256,
             "sha256_lock_status": "local-archive-verified-against-supplied-sha256",
             "evidence_class": "audited-development-candidates-pending-review",
             "archive_bytes": archive.stat().st_size, "expanded_member_bytes": expanded,
             "rows": rows, "unique_session_count": len(rows),
             "candidate_duration_s": sum(r["sample_count"] for r in rows) / SAMPLE_RATE,
             "accepted_negative_duration_s": 0, "independence_verified": False,
             "archive_inventory": inventory}
    write_json(out / "audit.json", report)
    write_json(out / "review-template.json", {
        "audit_sha256": digest_file(out / "audit.json"),
        "purpose": "development only; complete BEFORE inspecting any KWS outputs",
        "training_source_audit": "pending", "sealed_qualification": False,
        "rows": [{"recording": r["recording"], "pcm_sha256": r["pcm_sha256"],
                  "textgrid_sha256": r["textgrid"]["sha256"], "decision": "pending",
                  "all_speakers_and_boundaries_listened": False, "reason": ""} for r in rows]})
    with (out / "reference-candidates.jsonl").open("x", encoding="utf-8") as stream:
        for r in rows:
            # expected=null is deliberate: existing scorer must reject pending data.
            stream.write(json.dumps({"recording": r["recording"], "path": r["wav"],
                                    "duration_s": r["duration_s"], "expected": None,
                                    "evidence_class": "development-pending-review"}) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("plan")
    fetch = commands.add_parser("acquire")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--allow-download", action="store_true")
    fetch.add_argument("--expected-sha256")
    inspect = commands.add_parser("audit")
    inspect.add_argument("--archive", type=Path, required=True)
    inspect.add_argument("--archive-sha256", required=True)
    inspect.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "plan"):
            result = plan()
        elif args.command == "acquire":
            result = acquire(args.out, args.allow_download, args.expected_sha256)
        else:
            result = audit(args.archive, args.archive_sha256, args.out)
            result = {key: result[key] for key in ("schema", "unique_session_count", "candidate_duration_s", "accepted_negative_duration_s")}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError, tarfile.TarError, wave.Error, UnicodeError, EOFError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
