#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import corpus_digest, inspect_pcm16_wav  # noqa: E402

IDENTITY_FIELDS = ("speaker_id", "session_id", "source_id", "room_id", "device_id")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def trace_cache_paths(
    cache_root: pathlib.Path,
    *,
    model_sha256: str,
    audio_sha256: str,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    root = cache_root / model_sha256 / audio_sha256[:2]
    trace = root / f"{audio_sha256}.kwtr"
    sidecar = root / f"{audio_sha256}.json"
    lock = root / f"{audio_sha256}.lock"
    return trace, sidecar, lock


def cached_trace_valid(
    trace: pathlib.Path,
    sidecar: pathlib.Path,
    *,
    model_sha256: str,
    audio_sha256: str,
) -> dict | None:
    if not trace.is_file() or not sidecar.is_file():
        return None
    try:
        value = load_json_object(sidecar)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if (
        value.get("schema_version") != 1
        or value.get("evidence_class") != "kws-posterior-trace-cache-v1"
        or value.get("model_sha256") != model_sha256
        or value.get("audio_sha256") != audio_sha256
        or value.get("trace_sha256") != sha256_file(trace)
        or int(value.get("frames", 0)) <= 0
    ):
        return None
    return value


def ensure_cached_trace(
    *,
    posterior_dump: pathlib.Path,
    cache_root: pathlib.Path,
    model: pathlib.Path,
    audio: pathlib.Path,
    model_sha256: str,
    audio_sha256: str,
) -> tuple[pathlib.Path, dict, bool]:
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("posterior replay cache requires POSIX fcntl locking") from exc

    trace, sidecar, lock = trace_cache_paths(
        cache_root,
        model_sha256=model_sha256,
        audio_sha256=audio_sha256,
    )
    trace.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        cached = cached_trace_valid(
            trace,
            sidecar,
            model_sha256=model_sha256,
            audio_sha256=audio_sha256,
        )
        if cached is not None:
            return trace, cached, True

        trace.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        temp_trace = trace.with_name(f"{trace.name}.{os.getpid()}.tmp")
        temp_trace.unlink(missing_ok=True)
        try:
            completed = subprocess.run(
                [str(posterior_dump), str(model), str(audio), str(temp_trace)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if len(lines) != 1:
                raise ValueError("posterior dump must emit exactly one provenance JSON line")
            summary = json.loads(lines[0])
            if (
                not isinstance(summary, dict)
                or summary.get("schema_version") != 1
                or summary.get("evidence_class") != "kws-posterior-trace-v1"
                or summary.get("model_sha256") != model_sha256
                or int(summary.get("frames", 0)) <= 0
                or summary.get("trace_sha256") != sha256_file(temp_trace)
            ):
                raise ValueError("posterior dump provenance mismatch")
            os.replace(temp_trace, trace)
            cached_summary = {
                "schema_version": 1,
                "evidence_class": "kws-posterior-trace-cache-v1",
                "model_sha256": model_sha256,
                "audio_sha256": audio_sha256,
                "trace_sha256": sha256_file(trace),
                "frames": int(summary["frames"]),
                "vocab_size": int(summary["vocab_size"]),
            }
            sidecar.write_text(
                json.dumps(
                    cached_summary,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            return trace, cached_summary, False
        finally:
            temp_trace.unlink(missing_ok=True)


def load_references(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        recording = str(row.get("recording", ""))
        audio_path = row.get("audio_path") or row.get("path")
        if not recording or recording in seen:
            raise ValueError(f"{path}:{line_no}: recording must be non-empty and unique")
        if not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError(f"{path}:{line_no}: path is required for corpus execution")
        row["_execution_path"] = audio_path.strip()
        seen.add(recording)
        rows.append(row)
    if not rows:
        raise ValueError("reference corpus is empty")
    return rows


def audio_identity(row: dict, audio: pathlib.Path) -> dict:
    measured = inspect_pcm16_wav(audio.resolve(strict=True))
    item = {
        "recording": str(row["recording"]),
        "path": str(row["_execution_path"]),
        **measured,
    }
    for field in IDENTITY_FIELDS:
        value = row.get(field)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{row['recording']}: {field} must be non-empty text")
            item[field] = value.strip()
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--detections", required=True, type=pathlib.Path)
    parser.add_argument("--provenance", type=pathlib.Path)
    parser.add_argument("--corpus-identity", type=pathlib.Path)
    parser.add_argument("--posterior-dump", type=pathlib.Path)
    parser.add_argument("--decoder-replay", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    args = parser.parse_args()

    cache_values = (
        args.posterior_dump,
        args.decoder_replay,
        args.posterior_cache,
    )
    cache_enabled = all(value is not None for value in cache_values)
    if any(value is not None for value in cache_values) and not cache_enabled:
        raise ValueError(
            "--posterior-dump, --decoder-replay, and --posterior-cache "
            "must be supplied together"
        )

    rows = load_references(args.references)
    output_lines: list[str] = []
    identities: list[dict] = []
    posterior_traces: list[dict] = []
    posterior_cache_hits = 0
    posterior_cache_misses = 0
    model_sha256 = sha256_file(args.model)
    for row in rows:
        recording = str(row["recording"])
        audio = pathlib.Path(str(row["_execution_path"]))
        if not audio.is_absolute():
            audio = args.audio_root / audio
        identity = audio_identity(row, audio)
        identities.append(identity)
        if cache_enabled:
            audio_sha256 = str(identity["sha256"])
            trace, trace_summary, cache_hit = ensure_cached_trace(
                posterior_dump=args.posterior_dump,
                cache_root=args.posterior_cache,
                model=args.model,
                audio=audio,
                model_sha256=model_sha256,
                audio_sha256=audio_sha256,
            )
            if cache_hit:
                posterior_cache_hits += 1
            else:
                posterior_cache_misses += 1
            posterior_traces.append(
                {
                    "recording": recording,
                    "audio_sha256": audio_sha256,
                    "trace_sha256": trace_summary["trace_sha256"],
                    "frames": trace_summary["frames"],
                    "cache_hit": cache_hit,
                }
            )
            command = [
                str(args.decoder_replay),
                str(args.model),
                str(args.keywords),
                str(trace),
                recording,
            ]
        else:
            command = [
                str(args.runner),
                str(args.model),
                str(args.keywords),
                str(audio),
                recording,
            ]
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        for line_no, raw in enumerate(completed.stdout.splitlines(), 1):
            if not raw.strip():
                continue
            try:
                detection = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"runner emitted invalid JSON for {recording}:{line_no}: {exc}"
                ) from exc
            if not isinstance(detection, dict):
                raise ValueError(
                    f"runner emitted non-object JSON for {recording}:{line_no}"
                )
            if detection.get("recording") != recording:
                raise ValueError(
                    f"runner recording mismatch: expected {recording}, got {detection.get('recording')}"
                )
            output_lines.append(json.dumps(detection, ensure_ascii=False, allow_nan=False))

    corpus_identity = {
        "schema_version": 1,
        "corpus_sha256": corpus_digest(identities),
        "recordings": identities,
    }
    args.detections.parent.mkdir(parents=True, exist_ok=True)
    args.detections.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8"
    )

    if args.corpus_identity:
        args.corpus_identity.parent.mkdir(parents=True, exist_ok=True)
        args.corpus_identity.write_text(
            json.dumps(corpus_identity, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    if args.provenance:
        provenance = {
            "schema_version": 2,
            "runner_sha256": sha256_file(
                args.decoder_replay if cache_enabled else args.runner
            ),
            "evaluation_mode": (
                "posterior-replay-cache-v1"
                if cache_enabled
                else "direct-runner-v1"
            ),
            "model_sha256": model_sha256,
            "keyword_pack_sha256": sha256_file(args.keywords),
            "references_sha256": sha256_file(args.references),
            "detections_sha256": sha256_file(args.detections),
            "audio_corpus_sha256": corpus_identity["corpus_sha256"],
            "audio_files": identities,
            "recordings": len(rows),
            "detections": len(output_lines),
        }
        if cache_enabled:
            provenance.update(
                {
                    "posterior_dump_sha256": sha256_file(args.posterior_dump),
                    "decoder_replay_sha256": sha256_file(args.decoder_replay),
                    "posterior_cache_hits": posterior_cache_hits,
                    "posterior_cache_misses": posterior_cache_misses,
                    "posterior_traces": posterior_traces,
                }
            )
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print(
        f"processed {len(rows)} recording(s), emitted {len(output_lines)} detection(s), "
        f"corpus={corpus_identity['corpus_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
