#!/usr/bin/env python3
"""Independent, *non-admitting* KWS-data bulk clone effect readback.

Consumes the fixed private kws-data bulk plan + saved machine screen and
re-checks generated 16k PCM from bytes. No TTS/ASR/training or external calls.
A report from here is ONLY a development feedback card, not label proof,
source-rights verification, blind acceptance, or production qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import inspect_pcm16_wav, sha256_file  # noqa: E402

POLICY = "kws-data-bulk-effect-consumer-v1"
PLAN_SCHEMA = "kws-bulk-clone-corpus-plan-v1"
SCREEN_SCHEMA = "kws-bulk-clone-machine-screen-v1"
TTS_RECEIPT = "kws-bulk-clone-complete-v1"
MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
HEX = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[a-z][a-z0-9_-]{1,63}\Z")
MAX_ROWS = 16384
MAX_JSON_BYTES = 16 * 1024 * 1024
KINDS = ("positive-intent", "hard-negative-intent", "speech-negative-intent")
SCREEN_STATES = ("NO_GENERATION", "ASR_INCOMPLETE_HOLD",
                 "MACHINE_POSITIVE_CANDIDATE", "MACHINE_NEGATIVE_CANDIDATE",
                 "NEGATIVE_SPOKE_WAKE_OR_ASR_CONFLICT",
                 "ASR_DISAGREEMENT_OR_TTS_MISPRONUNCIATION")
# Exact approved bulk_pool.py v1 text requests, not a claim of acoustic truth.
PROMPTS = {
    "你好小窝": ("positive-intent", 1, "wake:kw1_exact_requested"),
    "你好，小窝": ("positive-intent", 1, "wake:kw1_pause_requested"),
    "小窝小窝": ("positive-intent", 2, "wake:kw2_exact_requested"),
    "小窝，小窝": ("positive-intent", 2, "wake:kw2_pause_requested"),
    "你好小屋": ("hard-negative-intent", None, "near_miss:homophone_wu_requested"),
    "你好小吴": ("hard-negative-intent", None, "near_miss:homophone_wu_requested"),
    "你好小我": ("hard-negative-intent", None, "near_miss:homophone_wo_requested"),
    "你好小王": ("hard-negative-intent", None, "near_miss:homophone_wang_requested"),
    "小屋小屋": ("hard-negative-intent", None, "near_miss:double_homophone_requested"),
    "小沃小沃": ("hard-negative-intent", None, "near_miss:double_homophone_requested"),
    "你好你好": ("hard-negative-intent", None, "near_miss:incomplete_or_repeated_requested"),
    "小窝": ("hard-negative-intent", None, "near_miss:partial_keyword_requested"),
    "今天晚上吃什么": ("speech-negative-intent", None, "ordinary:conversation_requested"),
    "请把灯打开": ("speech-negative-intent", None, "ordinary:command_requested"),
    "我想听一首音乐": ("speech-negative-intent", None, "ordinary:conversation_requested"),
    "现在外面下雨了": ("speech-negative-intent", None, "ordinary:conversation_requested"),
}


def require(predicate: bool, code: str) -> None:
    if not predicate:
        raise ValueError(code)


def pairs(items: list[tuple[str, object]]) -> dict:
    result = {}
    for k, v in items:
        require(k not in result, "DUPLICATE_JSON_KEY")
        result[k] = v
    return result


def read_json(path: Path, limit: int = MAX_JSON_BYTES) -> dict:
    require(not path.is_symlink() and path.is_file(), "SOURCE_JSON_NOT_REGULAR")
    require(path.stat().st_size <= limit, "SOURCE_JSON_OVERSIZE")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("SOURCE_JSON_INVALID") from exc
    require(type(obj) is dict, "SOURCE_JSON_NOT_OBJECT")
    return obj


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def source_file(root: Path, relative: Path) -> Path:
    require(not relative.is_absolute() and all(x not in (".", "..", "") for x in relative.parts),
            "SOURCE_PATH_INVALID")
    path = root
    for segment in relative.parts:
        path = path / segment
        require(not path.is_symlink(), "SOURCE_PATH_ALIAS")
    source = path.resolve(strict=True)
    require(source.is_file() and root in source.parents and
            stat.S_ISREG(source.stat().st_mode) and source.stat().st_nlink == 1,
            "SOURCE_FILE_NOT_PRIVATE_REGULAR")
    return source


def validate_plan(plan: dict) -> tuple[dict, dict]:
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("model_id") == MODEL and
            plan.get("training_admitted") is False and
            plan.get("qualification_allowed") is False and
            plan.get("shipping_approved") is False and
            plan.get("source_rights_authenticated") is False and
            plan.get("actual_spoken_labels_verified") is False,
            "UNTRUSTED_BULK_PLAN")
    refs, cells, budgets = plan.get("references"), plan.get("cells"), plan.get("budgets")
    require(type(refs) is list and 1 <= len(refs) <= 512 and
            type(cells) is list and 1 <= len(cells) <= MAX_ROWS, "BULK_PLAN_COUNT")
    require(type(budgets) is dict and budgets.get("max_retries") == 0 and
            budgets.get("max_paid_usd") == 0 and
            type(budgets.get("max_total_calls")) is int and
            1 <= budgets["max_total_calls"] <= len(cells), "BULK_PLAN_BUDGET")
    refmap, roles = {}, {}
    for ref in refs:
        require(type(ref) is dict and
                all(type(ref.get(k)) is str and ID.fullmatch(ref[k])
                    for k in ("reference_id", "speaker_group_id", "source_family_id")) and
                ref.get("role") in ("train-candidate", "development-only") and
                ref["reference_id"] not in refmap,
                "BULK_REFERENCE_IDENTITY")
        for group in ("speaker:" + ref["speaker_group_id"], "family:" + ref["source_family_id"]):
            require(group not in roles or roles[group] == ref["role"],
                    "SPEAKER_FAMILY_ROLE_LEAKAGE")
            roles[group] = ref["role"]
        refmap[ref["reference_id"]] = ref
    cmap = {}
    for cell in cells:
        require(type(cell) is dict and type(cell.get("cell_id")) is str and
                ID.fullmatch(cell["cell_id"]) and cell["cell_id"] not in cmap and
                cell.get("reference_id") in refmap, "BULK_CELL_IDENTITY")
        ref = refmap[cell["reference_id"]]
        expected = PROMPTS.get(cell.get("prompt"))
        require(expected is not None and
                (cell.get("intended_kind"), cell.get("intended_keyword_id")) == expected[:2] and
                cell.get("role") == ref["role"] and
                cell.get("source_family_id") == ref["source_family_id"] and
                cell.get("speaker_group_id") == ref["speaker_group_id"] and
                cell.get("label_authority") == "intended_only" and
                cell.get("training_admitted") is False and
                cell.get("qualification_allowed") is False and
                cell.get("synthetic") is True, "BULK_CELL_INTENT_OR_LINEAGE")
        cmap[cell["cell_id"]] = cell
    return refmap, cmap


def validate_screen(screen: dict, plan: dict, cmap: dict) -> dict:
    require(screen.get("schema") == SCREEN_SCHEMA and
            screen.get("plan_sha256") == canonical_sha(plan) and
            type(screen.get("adapter_lock_sha256")) is str and
            HEX.fullmatch(screen["adapter_lock_sha256"]) and
            all(screen.get(key) is False for key in
                ("training_admitted", "qualification_allowed", "shipping_approved",
                 "human_gold", "rights_authenticated")), "UNTRUSTED_BULK_SCREEN")
    rows = screen.get("rows")
    require(type(rows) is list and len(rows) == len(cmap), "SCREEN_COUNT_MISMATCH")
    indexed, counts = {}, Counter()
    for row in rows:
        require(type(row) is dict and row.get("cell_id") in cmap and
                row["cell_id"] not in indexed and row.get("status") in SCREEN_STATES and
                row.get("training_admitted") is False, "SCREEN_ROW_IDENTITY")
        cell = cmap[row["cell_id"]]
        if row["status"] == "NO_GENERATION":
            require(set(row) == {"cell_id", "status", "training_admitted"},
                    "NO_GENERATION_ROW_FIELDS")
        else:
            require(set(row) == {"cell_id", "reference_id", "source_family_id",
                    "role", "intended_kind", "asr_completed", "status", "training_admitted"} and
                    all(row[k] == cell[k] for k in
                        ("reference_id", "source_family_id", "role", "intended_kind")) and
                    type(row["asr_completed"]) is int and 0 <= row["asr_completed"] <= 3,
                    "SCREEN_ROW_LINEAGE_OR_COVERAGE")
            expected_count = 3 if row["status"] not in ("ASR_INCOMPLETE_HOLD",) else None
            require((expected_count is None and row["asr_completed"] < 3) or
                    row["asr_completed"] == expected_count,
                    "SCREEN_COMPLETION_INCONSISTENT")
            if row["status"] == "MACHINE_POSITIVE_CANDIDATE":
                require(cell["intended_kind"] == "positive-intent", "MACHINE_POSITIVE_KIND")
            if row["status"] in ("MACHINE_NEGATIVE_CANDIDATE",
                                 "NEGATIVE_SPOKE_WAKE_OR_ASR_CONFLICT"):
                require(cell["intended_kind"] != "positive-intent", "MACHINE_NEGATIVE_KIND")
        indexed[row["cell_id"]] = row
        counts[row["status"]] += 1
    require(set(indexed) == set(cmap) and screen.get("counts") == dict(counts),
            "SCREEN_COUNTS_OR_MEMBERSHIP_DRIFT")
    return indexed


def measure(plan: dict, screen: dict, generation_root: Path) -> dict:
    refmap, cmap = validate_plan(plan)
    srows = validate_screen(screen, plan, cmap)
    root = generation_root.resolve(strict=True)
    require(root.is_dir(), "GENERATED_ROOT_MISSING")
    lock = read_json(source_file(root, Path("plan-lock.json")), 65536)
    require(lock.get("schema") == "kws-bulk-clone-execution-lock-v1" and
            lock.get("plan_sha256") == canonical_sha(plan) and
            lock.get("model_id") == MODEL and lock.get("training_admitted") is False,
            "GENERATION_PLAN_DRIFT")
    counts, by_challenge, by_keyword, by_source, pcm_owner = (
        Counter(), defaultdict(Counter), defaultdict(Counter),
        defaultdict(Counter), {})
    measured_rows = []
    for cell_id, cell in sorted(cmap.items()):
        row = srows[cell_id]
        family = cell["source_family_id"]
        challenge = PROMPTS[cell["prompt"]][2]
        counts[row["status"]] += 1
        by_challenge[challenge][row["status"]] += 1
        if cell["intended_keyword_id"] in (1, 2):
            by_keyword[str(cell["intended_keyword_id"])][row["status"]] += 1
        by_source[family][row["status"]] += 1
        verified = None
        folder = Path("cells") / cell_id
        completed_path = root / folder / "completed.json"
        if row["status"] == "NO_GENERATION":
            require(not completed_path.exists(), "UNREPORTED_GENERATED_WAV")
        else:
            completion = read_json(source_file(root, folder / "completed.json"), 65536)
            require(completion.get("schema") == TTS_RECEIPT and
                    completion.get("status") == "GENERATED_PENDING_ASR" and
                    completion.get("cell_id") == cell_id and
                    completion.get("source_family_id") == family and
                    completion.get("intended_kind") == cell["intended_kind"] and
                    completion.get("training_admitted") is False and
                    completion.get("qualification_allowed") is False and
                    completion.get("shipping_approved") is False, "GENERATION_RECEIPT_DRIFT")
            wav = source_file(root, folder / "audio-16k.wav")
            actual = inspect_pcm16_wav(wav)
            reported = completion.get("audio")
            require(type(reported) is dict and
                    all(reported.get(key) == actual[expected] for key, expected in (
                        ("wav_sha256", "file_sha256"), ("pcm_sha256", "pcm_sha256"),
                        ("audio_frames", "frames"))) and
                    reported.get("wav_bytes") == wav.stat().st_size,
                    "GENERATION_WAV_OR_PCM_CHANGED")
            pcm = actual["pcm_sha256"]
            if pcm in pcm_owner:
                require(pcm_owner[pcm] == (family, cell["speaker_group_id"]),
                        "CROSS_SPEAKER_DUPLICATE_PCM")
                counts["duplicate_pcm_within_original_group"] += 1
            else:
                pcm_owner[pcm] = (family, cell["speaker_group_id"])
            verified = {"file_sha256": actual["file_sha256"],
                        "pcm_sha256": actual["pcm_sha256"], "frames": actual["frames"]}
        measured_rows.append({"cell_id": cell_id, "source_family_id": family,
                              "speaker_group_id": cell["speaker_group_id"],
                              "role": cell["role"], "requested_text": cell["prompt"],
                              "requested_challenge": challenge,
                              "requested_keyword_id": cell["intended_keyword_id"],
                              "screen_status": row["status"], "asr_completed": row.get("asr_completed", 0),
                              "generated_audio": verified,
                              "synthetic": True, "training_admitted": False,
                              "qualification_allowed": False})
    gaps = []
    if counts["NO_GENERATION"] or counts["ASR_INCOMPLETE_HOLD"]:
        gaps.append("SOURCE_OR_ASR_EXECUTION_INCOMPLETE")
    if counts["NEGATIVE_SPOKE_WAKE_OR_ASR_CONFLICT"]:
        gaps.append("HARD_NEGATIVE_WAKE_CONFUSION_NEEDS_DEVELOPMENT_REVIEW")
    for word in ("1", "2"):
        if by_keyword[word]["MACHINE_POSITIVE_CANDIDATE"] == 0:
            gaps.append("NO_MACHINE_SCREENED_POSITIVE_FOR_KW" + word)
    gaps += ["ACTUAL_LABEL_AUTHORITY_NOT_VERIFIED",
             "INDEPENDENT_REAL_HUMAN_FINAL_AFE_NOT_MEASURED",
             "CONTINUOUS_FAR_FRR_NOT_MEASURED",
             "PHYSICAL_SSC305_NOT_MEASURED"]
    return {"schema_version": 1, "policy": POLICY,
            "kws_data_plan_sha256": canonical_sha(plan),
            "kws_data_screen_sha256": canonical_sha(screen),
            "adapter_lock_sha256": screen["adapter_lock_sha256"],
            "total_planned": len(cmap), "total_generated_audio_verified":
                sum(x["generated_audio"] is not None for x in measured_rows),
            "unique_source_families_declared": len({c["source_family_id"] for c in cmap.values()}),
            "unique_pcm_verified": len(pcm_owner),
            "screen_counts": dict(sorted(counts.items())),
            "screen_by_challenge": {k: dict(sorted(v.items())) for k, v in sorted(by_challenge.items())},
            "screen_by_keyword": {k: dict(sorted(v.items())) for k, v in sorted(by_keyword.items())},
            "screen_by_source_family": {k: dict(sorted(v.items())) for k, v in sorted(by_source.items())},
            "rows": measured_rows, "gaps": sorted(gaps),
            "source_rights_authenticated": False, "asr_source_authenticated": False,
            "human_gold": False, "training_admitted": False,
            "qualification_allowed": False, "shipping_approved": False,
            "conclusion": "INDEPENDENT_DEVELOPMENT_DIAGNOSTIC_ONLY"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--generated-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="new private JSON; never overwrite an existing report")
    args = parser.parse_args(argv)
    plan, screen = read_json(args.plan), read_json(args.screen)
    report = measure(plan, screen, args.generated_root)
    output = args.output
    require(output.resolve() != args.plan.resolve() and
            output.resolve() != args.screen.resolve() and
            output.resolve() != args.generated_root.resolve() and
            args.generated_root.resolve() not in output.resolve().parents,
            "DIAGNOSTIC_OUTPUT_ALIASES_SOURCE")
    require(output.parent.exists() and output.parent.is_dir(), "PRIVATE_OUTPUT_DIRECTORY_REQUIRED")
    raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"policy": POLICY, "total_planned": report["total_planned"],
                      "total_generated_audio_verified": report["total_generated_audio_verified"],
                      "screen_counts": report["screen_counts"], "conclusion": report["conclusion"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
