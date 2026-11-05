#!/usr/bin/env python3
"""Run one fixed native Zipformer system over a hash-bound PCM view.

Outputs are private execution observations, not an independent quality score.
No transcript, label, extra silence, or threshold search enters inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import sys
import time
import wave

from research_job_lock import heavy_model_slot


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_path(root, relative):
    rel = PurePosixPath(relative)
    require(type(relative) is str and relative and not rel.is_absolute()
            and ".." not in rel.parts and str(rel) == relative, "invalid relative evidence path")
    path = root.joinpath(*rel.parts)
    require(not any(part.is_symlink() for part in (path, *path.parents)), "evidence alias")
    require(path.is_file(), "missing evidence file: " + relative)
    return path


def checked_bytes(path, expected, size=None):
    raw = path.read_bytes()
    require(sha256(raw) == expected, "hash mismatch: " + str(path))
    require(size is None or len(raw) == size, "size mismatch: " + str(path))
    return raw


def pcm_wave(raw):
    require(raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
            and int.from_bytes(raw[4:8], "little") + 8 == len(raw), "invalid complete RIFF")
    with wave.open(io.BytesIO(raw), "rb") as wav:
        require(wav.getnchannels() == 1 and wav.getframerate() == 16000
                and wav.getsampwidth() == 2 and wav.getcomptype() == "NONE", "expected mono PCM16 16kHz")
        count = wav.getnframes()
        pcm = wav.readframes(count)
    require(count > 0 and len(pcm) == count * 2, "empty or incomplete PCM")
    return pcm, count


def verified_audio(view, expected_manifest):
    manifest_raw = checked_bytes(view / "MANIFEST.json", expected_manifest)
    inventory = json.loads(manifest_raw)
    require(type(inventory) is dict and inventory, "empty view inventory")
    audio = []
    for relative, digest in sorted(inventory.items()):
        require(type(digest) is str and len(digest) == 64, "invalid file digest")
        raw = checked_bytes(bound_path(view, relative), digest)
        if relative.startswith("audio/") and relative.endswith(".wav"):
            pcm, samples = pcm_wave(raw)
            audio.append({"path": relative, "wav_sha256": digest,
                          "pcm_sha256": sha256(pcm), "samples": samples})
    require(audio, "view contains no verified PCM audio")
    return audio, len(inventory)


def write_new_json(path, value):
    """Create a complete immutable receipt without replacing an old one."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False).encode() + b"\n"
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def check_profile(profile, models):
    require(profile.get("schema") == 1, "unsupported native profile")
    fixed = {"provider": "cpu", "num_threads": 1, "sample_rate": 16000,
             "feature_dim": 80, "feed_samples": 160, "automatic_padding_samples": 0,
             "input_finished_calls_per_clip": 1, "reset_after_detection": True}
    require(all(profile["inference"].get(k) == v for k, v in fixed.items()),
            "native streaming contract changed")
    for asset in profile["assets"].values():
        checked_bytes(bound_path(models, asset["filename"]), asset["sha256"], asset["bytes"])
    runtime = profile["runtime"]
    require(sys.version.split()[0] == runtime["python"], "Python patch mismatch")
    require(importlib.metadata.version("numpy") == runtime["numpy"], "NumPy version mismatch")
    for package, entry in runtime["packages"].items():
        require(importlib.metadata.version(package) == entry["version"], "package version mismatch: " + package)
    for entry in runtime["bound_files"]:
        distribution = importlib.metadata.distribution(entry["distribution"])
        checked_bytes(Path(distribution.locate_file(entry["path"])), entry["sha256"], entry["bytes"])


def decode_samples(engine, samples, block=160, deadline=math.inf):
    """Preserve the supplied sample coordinate and drain once at actual EOF."""
    require(type(block) is int and block > 0 and len(samples) > 0, "invalid feed")
    def check_deadline():
        require(time.monotonic() < deadline, "native diagnostic time budget exhausted")

    check_deadline()
    stream = engine.create_stream()
    events = []
    supplied = 0
    feeds = 0

    def drain(eof):
        while engine.is_ready(stream):
            check_deadline()
            engine.decode_stream(stream)
            result = engine.keyword_spotter.get_result(stream)
            keyword = result.keyword.strip()
            if keyword:
                stamps = [float(value) for value in result.timestamps]
                require(all(math.isfinite(value) and value >= 0 for value in stamps),
                        "invalid native result timing")
                require(keyword in ("kw1", "kw2"), "unexpected native keyword")
                events.append({"keyword_id": int(keyword[-1]), "native_keyword": keyword,
                               "tokens": list(result.tokens), "native_timestamps_seconds": stamps,
                               "native_start_time_seconds": None, "available_samples": supplied,
                               "native_timestamp_note": "Pinned Python KeywordResult exposes timestamps, not start_time; no inferred offset.",
                               "after_input_finished": eof})
                engine.reset_stream(stream)

    for offset in range(0, len(samples), block):
        check_deadline()
        chunk = samples[offset:offset + block]
        stream.accept_waveform(16000, chunk)
        supplied += len(chunk)
        feeds += 1
        drain(False)
    check_deadline()
    stream.input_finished()
    drain(True)
    check_deadline()
    return {"events": events, "audio_samples": supplied, "feed_calls": feeds,
            "input_finished_calls": 1, "added_padding_samples": 0}


def cached_result(path, identity, audio):
    if not path.exists():
        return None
    value = json.loads(path.read_bytes())
    claim_path = path.parent.parent / "claims" / path.name
    claim_raw = claim_path.read_bytes()
    claim = json.loads(claim_raw)
    require(claim.get("identity_sha256") == identity and claim.get("input") == audio
            and claim.get("attempt") == 1 and value.get("claim_sha256") == sha256(claim_raw),
            "cached claim identity drift")
    require(value.get("schema") == 1 and value.get("identity_sha256") == identity
            and value.get("input") == audio and value.get("status") == "completed", "cached result identity drift")
    require(value.get("audio_samples") == audio["samples"]
            and value.get("added_padding_samples") == 0
            and value.get("input_finished_calls") == 1, "cached stream accounting drift")
    digest = value.get("result_sha256")
    require(digest == sha256(canonical({k: v for k, v in value.items() if k != "result_sha256"})),
            "cached native result bytes drift")
    return value


def make_engine(profile, models, keyword_path):
    import sherpa_onnx
    def asset(role):
        return str(models / profile["assets"][role]["filename"])
    infer = profile["inference"]
    return sherpa_onnx.KeywordSpotter(
        tokens=asset("tokens"), encoder=asset("encoder"), decoder=asset("decoder"),
        joiner=asset("joiner"), keywords_file=str(keyword_path), num_threads=1,
        sample_rate=16000, feature_dim=80, max_active_paths=infer["max_active_paths"],
        keywords_score=infer["keywords_score"], keywords_threshold=infer["keywords_threshold"],
        num_trailing_blanks=infer["num_trailing_blanks"], provider="cpu")


def run(args):
    root = Path(__file__).resolve().parents[1]
    output, view, models = (Path(value).absolute() for value in (args.output, args.view, args.models))
    require(not output.is_relative_to(root) and not output.is_relative_to(view)
            and not output.is_relative_to(models), "use a separate private output directory")
    profile_path = Path(args.profile)
    profile_raw = profile_path.read_bytes()
    profile = json.loads(profile_raw)
    require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 999 * 60,
            "time budget must be positive and at most 999 minutes")
    start = time.monotonic()
    deadline = start + args.max_seconds
    with heavy_model_slot(args.heavy_lock):
        audio, verified = verified_audio(view, args.view_manifest_sha256)
        check_profile(profile, models)
        identity = {"schema": 1, "profile_sha256": sha256(profile_raw),
                    "view_manifest_sha256": args.view_manifest_sha256,
                    "runner_sha256": sha256(Path(__file__).read_bytes()),
                    "lock_helper_sha256": sha256((root / "tools/research_job_lock.py").read_bytes()),
                    "audio": audio, "python": sys.version.split()[0]}
        identity_hash = sha256(canonical(identity))
        require(not output.exists() or not any(output.iterdir())
                or (output / "identity.json").is_file(), "unbound existing output")
        output.mkdir(parents=True, exist_ok=True)
        (output / "claims").mkdir(exist_ok=True)
        (output / "results").mkdir(exist_ok=True)
        receipt = output / "identity.json"
        if receipt.exists():
            require(json.loads(receipt.read_bytes()) == identity, "output belongs to another fixed experiment")
        else:
            write_new_json(receipt, identity)
        results = []
        missing = []
        for index, item in enumerate(audio):
            name = f"clip-{index:03d}.json"
            cached = cached_result(output / "results" / name, identity_hash, item)
            if cached is not None:
                results.append(cached)
            else:
                require(not (output / "claims" / name).exists(), "incomplete model claim: explicit review required")
                missing.append((name, item))
        keyword_lines = [" ".join(k["tokens"]) + " @kw" + str(k["id"]) for k in profile["keywords"]]
        keyword_text = "\n".join(keyword_lines) + "\n"
        keyword_path = output / "keywords.txt"
        if keyword_path.exists():
            require(keyword_path.read_text() == keyword_text, "keyword recipe drift")
        else:
            require(not results, "missing native keyword receipt")
            keyword_path.write_text(keyword_text)
        require(time.monotonic() < deadline, "native diagnostic time budget exhausted")
        if not missing and (output / "summary.json").exists():
            saved = json.loads((output / "summary.json").read_bytes())
            require(saved.get("identity_sha256") == identity_hash and saved.get("clips") == len(results)
                    and saved.get("status") == "completed"
                    and saved.get("summary_sha256") == sha256(canonical(
                        {k: v for k, v in saved.items() if k != "summary_sha256"})), "cached summary drift")
            return {"status": "reused", "clips": len(results), "new_model_calls": 0,
                    "identity_sha256": identity_hash}
        load_seconds = 0.0
        if missing:
            import numpy as np
            require(time.monotonic() < deadline, "native diagnostic time budget exhausted")
            load_start = time.monotonic()
            engine = make_engine(profile, models, keyword_path)
            load_seconds = time.monotonic() - load_start
            require(time.monotonic() < deadline, "native diagnostic time budget exhausted")
        for name, item in missing:
            claim = {"schema": 1, "identity_sha256": identity_hash, "input": item,
                     "attempt": 1, "started_unix_seconds": time.time()}
            write_new_json(output / "claims" / name, claim)
            pcm, _ = pcm_wave(checked_bytes(bound_path(view, item["path"]), item["wav_sha256"]))
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768.0)
            call_start = time.monotonic()
            result = {"schema": 1, "identity_sha256": identity_hash, "input": item,
                      "claim_sha256": sha256((output / "claims" / name).read_bytes()),
                      "status": "completed", **decode_samples(engine, samples, 160, deadline)}
            result["inference_seconds"] = time.monotonic() - call_start
            result["result_sha256"] = sha256(canonical(result))
            write_new_json(output / "results" / name, result)
            results.append(result)
        counts = Counter(str(event["keyword_id"]) for row in results for event in row["events"])
        summary = {"schema": 1, "status": "completed", "identity_sha256": identity_hash,
                   "clips": len(results), "new_model_calls": len(missing), "cached_clips": len(audio)-len(missing),
                   "verified_view_files": verified, "samples": sum(r["audio_samples"] for r in results),
                   "inference_seconds": sum(r["inference_seconds"] for r in results),
                   "model_load_seconds_this_attempt": load_seconds,
                   "elapsed_seconds_this_attempt": time.monotonic()-start,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "detections_by_keyword": dict(counts), "clips_with_detection": sum(bool(r["events"]) for r in results),
                   "added_padding_samples": 0, "independent_human_evidence": False,
                   "scope": "Synthetic train-only system observations; no FRR, population FAR, device latency or model winner.",
                   "timing_note": "available_samples is supplied PCM, not acoustic completion or hardware callback latency"}
        summary["summary_sha256"] = sha256(canonical(summary))
        write_new_json(output / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(Path(__file__).resolve().parents[1] / "configs/research/zipformer-system-20261105.lock.json"))
    for field in ("models", "view", "view-manifest-sha256", "output", "heavy-lock"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
