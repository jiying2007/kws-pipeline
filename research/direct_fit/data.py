"""Project verified *actual* whole-source labels, never requested TTS text.

No synthetic-admission bypass exists here. Windows retain every original PCM
sample and append declared zeros; zeros are context, not background examples.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import wave

TOKEN_MAP = {"<blk>": 0, "ni3": 1, "hao3": 2, "xiao3": 3, "wo1": 4}
WAKE = {1: ["ni3", "hao3", "xiao3", "wo1"],
        2: ["xiao3", "wo1", "xiao3", "wo1"]}
REUSE_PURPOSE = "verified-actual-label-facts-for-exposed-direct-fit-v1"
WINDOW_POLICY = "whole-original-pcm-right-zero-context-v1"


def project_actual_label(row):
    """Call only after live admission validation; retains its restricted purpose."""
    tokens = row.get("tokens")
    text = row.get("actual_text")
    if (type(tokens) is not list or any(type(t) is not str or t not in TOKEN_MAP or
                                      t == "<blk>" for t in tokens)
            or type(text) is not str):
        raise ValueError("complete actual transcript and canonical tokens are required")
    if row.get("target_ids") != [TOKEN_MAP[t] for t in tokens]:
        raise ValueError("actual token/target identity mismatch")
    kind, keyword = row.get("kind"), row.get("keyword_id")
    if kind == "positive":
        if type(keyword) is not int or keyword not in WAKE or tokens != WAKE[keyword]:
            raise ValueError("positive actual tokens must be exactly one canonical keyword")
        return keyword
    if kind not in ("negative", "confusable", "background") or keyword is not None:
        raise ValueError("missing/unknown/non-single-keyword labels are not background")
    if kind == "background":
        if tokens or text:
            raise ValueError("background requires a verified complete no-speech source")
    elif not tokens or not text.strip():
        raise ValueError("negative speech requires its actual transcript")
    # Match the admission rule: even a wake-token subsequence is not negative.
    for word in WAKE.values():
        iterator = iter(tokens)
        if all(any(token == expected for token in iterator) for expected in word):
            raise ValueError("negative transcript contains a wake-token subsequence")
    return 0


def validate_window(spec, frames):
    if type(frames) is not int or not 1 <= frames <= 1500:
        raise ValueError("window_frames must be an integer in 1..1500")
    # Receptive span alone can hide holes in a sparse dilation schedule.
    reach = 0
    for dilation, kernel in sorted(zip(spec.dilations, spec.kernels)):
        if kernel > 1:
            if dilation > reach + 1:
                raise ValueError("backbone receptive field has temporal holes")
            reach += (kernel - 1) * dilation
    if frames > reach + 1:
        raise ValueError("terminal class head cannot observe the whole labeled window")
    return 400 + (frames - 1) * 320


def whole_source_features(raw_pcm, window_frames, *, torch, features, frontend):
    """No crop and no waveform resampling. Include tail samples in a full frame."""
    if type(raw_pcm) is not bytes or len(raw_pcm) % 2 or len(raw_pcm) < 800:
        raise ValueError("whole PCM16 source must have at least 400 complete samples")
    samples = 400 + (window_frames - 1) * 320
    if len(raw_pcm) > samples * 2:
        raise ValueError("source exceeds declared window; cropping is forbidden")
    padded = raw_pcm + bytes(samples * 2 - len(raw_pcm))
    pcm = torch.frombuffer(bytearray(padded), dtype=torch.int16).float() / 32768.0
    result = features(pcm, 32, frame_len=400, hop=320, frontend=frontend)
    if tuple(result.shape) != (window_frames, 32) or not torch.isfinite(result).all():
        raise ValueError("frontend output violates fixed-window contract")
    return result, {"source_samples": len(raw_pcm) // 2, "left_zero_samples": 0,
                    "right_zero_samples": samples - len(raw_pcm) // 2,
                    "window_pcm_sha256": hashlib.sha256(padded).hexdigest()}


def read_source(path, expected):
    """Recheck consumed bytes against existing PCM identity using one snapshot."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected["file_sha256"]:
        raise ValueError("source WAV changed since admission/identity check")
    with wave.open(io.BytesIO(raw), "rb") as reader:
        if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth(),
                reader.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("expected mono 16-kHz PCM16")
        pcm = reader.readframes(reader.getnframes())
    if (len(pcm) != expected["frames"] * 2 or
            hashlib.sha256(pcm).hexdigest() != expected["pcm_sha256"]):
        raise ValueError("source PCM changed since admission/identity check")
    return pcm


def load_development(config, rt):
    """Real admission only; original sources, no hidden augmentation or holdout."""
    from audit_dataset import verify_manifest_lineage
    from corpus_identity import inspect_pcm16_wav, canonical_hash
    from kws_vocab import load_tokens
    from training_admission import verify_training_manifests

    torch, _, Manifest, features, frontend, _, _ = rt
    if config["label_reuse_purpose"] != REUSE_PURPOSE:
        raise ValueError("explicit direct diagnostic label-fact reuse purpose required")
    tokens = Path(config["tokens"]).resolve(strict=True)
    if load_tokens(tokens) != TOKEN_MAP:
        raise ValueError("this controlled pair requires the exact five-token vocabulary")
    paths = [Path(p).resolve(strict=True) for p in config["manifests"]]
    for path in paths:
        if path.suffix != ".jsonl" or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("bounded JSONL manifests required")
    before_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    tokens_hash = hashlib.sha256(tokens.read_bytes()).hexdigest()
    admission = verify_training_manifests(paths, TOKEN_MAP, diagnostic=False,
                                         evidence_dir=config["evidence_dir"])
    dataset = Manifest(paths, 32, 5, frontend)
    raw_rows = [(path, row) for path in paths for row in verify_manifest_lineage(path)]
    if ([entry["sha256"] for entry in admission["manifests"]] != before_hashes
            or [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] != before_hashes
            or hashlib.sha256(tokens.read_bytes()).hexdigest() != tokens_hash):
        raise ValueError("manifest/token bytes changed during admission projection")
    if len(raw_rows) != len(dataset) or len(raw_rows) > 10000:
        raise ValueError("manifest projection count mismatch or row cap exceeded")
    if len(raw_rows) * config["window_frames"] > 100000:
        raise ValueError("fixed feature allocation exceeds 100000 frames")
    items, identities, by_pcm = [], [], {}
    for index, ((manifest, row), rendered) in enumerate(zip(raw_rows, dataset.identity_rows)):
        if row.get("split") != "train" or row.get("protected_evidence", False) is not False:
            raise ValueError("only exposed train rows are accepted; no protected/held-out reuse")
        target = project_actual_label(row)
        source = Path(row["source_path"])
        source = (source if source.is_absolute() else manifest.parent / source).resolve(strict=True)
        measured = inspect_pcm16_wav(source)
        if measured["frames"] > 480000:
            raise ValueError("source exceeds 30-second diagnostic cap")
        if (measured["file_sha256"] != row["source_wav_sha256"] or
                measured["pcm_sha256"] != row["source_pcm_sha256"]):
            raise ValueError("original source binding mismatch")
        raw = read_source(source, measured)
        acoustic, window = whole_source_features(raw, config["window_frames"],
            torch=torch, features=features, frontend=frontend)
        target_ids = row["target_ids"]
        if len(target_ids) + sum(a == b for a, b in zip(target_ids, target_ids[1:])) > len(acoustic):
            raise ValueError("fixed window cannot align the actual CTC target")
        binding = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   "row_index": index, "reviewed_row": row, "rendered_identity": rendered}
        key = measured["pcm_sha256"]
        if key in by_pcm:
            previous = by_pcm[key]
            if previous["class_target"] != target or previous["target_ids"] != target_ids:
                raise ValueError("duplicate original PCM has conflicting actual labels")
            previous["manifest_bindings"].append(binding)
            continue
        recording = "source-" + key
        identity = dict(recording=recording, source_path=str(source), **measured,
            **window, class_target=target, target_ids=target_ids,
            manifest_bindings=[binding])
        by_pcm[key] = identity
        identities.append(identity)
        items.append((acoustic, torch.tensor(target_ids, dtype=torch.long), target))
    coverage = {str(k): sum(item[2] == k for item in items) for k in range(3)}
    if config["coverage_mode"] == "all-three-classes" and not all(coverage.values()):
        raise ValueError("all-three-classes requires admitted actual examples of every class")
    if not items:
        raise ValueError("no usable original sources")
    if ([hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] != before_hashes
            or hashlib.sha256(tokens.read_bytes()).hexdigest() != tokens_hash):
        raise ValueError("manifest/token bytes changed during source consumption")
    receipt = {"schema": "exposed-whole-source-fit-data-v1", "promotion_allowed": False,
        "independent_evaluation": False, "label_reuse_purpose": REUSE_PURPOSE,
        "original_ctc_admission": admission, "window_policy": WINDOW_POLICY,
        "source_policy": "unique-original-pcm-first-occurrence-v1",
        "rendered_augmentations_consumed": False, "coverage": coverage,
        "token_sha256": tokens_hash,
        "recordings": identities, "window_frames": config["window_frames"]}
    receipt["data_sha256"] = canonical_hash(receipt)
    return items, receipt
