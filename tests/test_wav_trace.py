#!/usr/bin/env python3
"""Bounded artificial-only CLI contract: 1 model, 3 WAV runs, 2 replays.

No training, Torch, downloaded weights, speech corpus, or quality claims.
An existing output directory is refused. With --output-root, failures preserve
a partial receipt; default workspaces are temporary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import wave

FINGERPRINT = 0x1122334455667788
LOGITS = (-4.0, 4.0, 3.0, -4.0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_model() -> bytes:
    # Handcrafted ABI fixture; not a trained/exported research checkpoint.
    data = bytearray(264)
    struct.pack_into("<4sHHHHHHIIIfffQIIIIII", data, 0,
                     b"KWSP", 2, 72, 32, 4, 4, 0, 16000, 400, 320,
                     .01, .01, .01, FINGERPRINT, 72, 200, 216, 232, 248, 264)
    struct.pack_into("<4f", data, 248, *LOGITS)
    return bytes(data)


def fixture_pack() -> bytes:
    data = bytearray(72)
    struct.pack_into("<4sHHHHIQ", data, 0,
                     b"KWKP", 3, 24, 1, 4, 72, FINGERPRINT)
    struct.pack_into("<IfHBBBBH", data, 24, 42, .30, 1, 0, 0, 0, 0, 0)
    struct.pack_into("<H", data, 40, 1)
    return bytes(data)


def write_wav(path: Path, count: int, signal: bool) -> None:
    values = [(-12000 if (i // 20) % 2 == 0 else 12000) if signal else 0
              for i in range(count)]
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, count, "NONE", "not compressed"))
        stream.writeframes(struct.pack("<" + "h" * count, *values))


def read_trace(path: Path, model_sha: str) -> list[dict]:
    data = path.read_bytes()
    require(len(data) >= 120, "trace header truncated")
    header = struct.unpack_from("<8sIHHIIIIQQ64s8s", data)
    require(header[:9] == (b"KWTRACE1", 1, 4, 0, 16000, 400, 320, 0,
                           FINGERPRINT), "trace identity/geometry mismatch")
    require(header[10] == model_sha.encode(), "trace loaded-model hash mismatch")
    require(header[11] == bytes(8), "trace reserved header bytes")
    require(len(data) == 120 + header[9] * 32, "trace size/frame count mismatch")
    rows = []
    for index in range(header[9]):
        end, flags, *logits = struct.unpack_from("<Q8s4f", data, 120 + index * 32)
        require(flags[0] in (0, 1) and flags[1:] == bytes(7), "trace flags")
        require(tuple(logits) == LOGITS, "raw float32 logits differ from fixture")
        rows.append({"end_sample": end, "speech_active": flags[0], "logits": logits})
    return rows


def events(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line]


def canonical(rows: list[dict]) -> list[dict]:
    return [{key: row[key] for key in ("recording", "keyword_id", "time_s", "confidence")}
            for row in rows]


def run(args: argparse.Namespace) -> None:
    out = args.output_root.resolve()
    out.mkdir(parents=False, exist_ok=False)
    report = {"schema": "wav-single-pass-trace-fixture-v1", "status": "running",
              "evidence_class": "artificial-model-and-pcm-contract-only",
              "training_updates": 0, "real_audio_calls": 0,
              "wav_runs_reserved": 0, "replay_runs_reserved": 0,
              "acoustic_frames_reserved": 0, "decoder_frames_reserved": 0,
              "invocations": [], "fixtures": {}, "assertions": []}

    def save() -> None:
        (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")

    def invoke(label: str, command: list[str], kind: str, frames: int = 0,
               expected: int = 0) -> subprocess.CompletedProcess:
        if kind == "wav":
            report["wav_runs_reserved"] += 1
            report["acoustic_frames_reserved"] += frames
        elif kind == "replay":
            report["replay_runs_reserved"] += 1
            report["decoder_frames_reserved"] += frames
        require(report["wav_runs_reserved"] <= 3 and report["replay_runs_reserved"] <= 2,
                "fixed process budget exceeded")
        entry = {"label": label, "kind": kind, "command": command,
                 "frames_reserved": frames, "expected_returncode": expected,
                 "status": "reserved"}
        report["invocations"].append(entry)
        save()
        result = subprocess.run(command, text=True, capture_output=True, timeout=15)
        (out / (label + ".stdout")).write_text(result.stdout)
        (out / (label + ".stderr")).write_text(result.stderr)
        entry.update(status="completed", returncode=result.returncode,
                     stdout_sha256=digest(out / (label + ".stdout")),
                     stderr_sha256=digest(out / (label + ".stderr")))
        save()
        require(result.returncode == expected, label + ": unexpected exit status")
        if expected != 0:
            require(result.stdout == "", label + ": rejected call produced events")
        return result

    try:
        wav_runner = str(args.wav_runner.resolve(strict=True))
        replay_runner = str(args.replay_runner.resolve(strict=True))
        report["runner_sha256"] = {"wav": digest(Path(wav_runner)),
                                   "replay": digest(Path(replay_runner))}
        model, pack = out / "fixture.kwm", out / "fixture.kwk"
        model.write_bytes(fixture_model())
        pack.write_bytes(fixture_pack())
        model_sha = digest(model)
        report["model_sha256"], report["keyword_pack_sha256"] = model_sha, digest(pack)
        specs = [("short", 399, False, 160, [], []),
                 ("full", 400, True, 160, [400], [1]),
                 ("tail", 1201, True, 320, [400, 720, 1040], [0, 1, 1])]
        for name, count, signal, block, ends, speech in specs:
            write_wav(out / (name + ".wav"), count, signal)
        config_sha = "a" * 64
        timeline = out / "tail.tsv"
        timeline.write_text("kws-afe-timeline-v1\n"
                            f"0\t400\t1\t0\t16\t0\t0\t0\t{config_sha}\n"
                            f"400\t801\t2\t25000000\t16\t0\t1\t0\t{config_sha}\n")
        bad_timeline = out / "reset.tsv"
        bad_timeline.write_text("kws-afe-timeline-v1\n"
                                f"0\t399\t1\t0\t1\t0\t0\t0\t{config_sha}\n")
        malformed = out / "malformed.tsv"
        malformed.write_text("not-a-timeline\n")
        common = [wav_runner, str(model), str(pack), str(out / "short.wav"), "preflight"]
        sentinel = out / "existing.kwtr"
        sentinel.write_bytes(b"DO NOT TRUNCATE\n")
        sentinel_sha = digest(sentinel)
        invoke("reject-existing", common + ["--trace-out", str(sentinel)], "preflight", expected=1)
        require(digest(sentinel) == sentinel_sha, "existing trace overwritten")
        invoke("reject-duplicate", common + ["--trace-out", str(out / "duplicate.kwtr"),
                 "--trace-out", str(out / "duplicate2.kwtr")], "preflight", expected=2)
        collision = out / "collision"
        invoke("reject-collision", common + ["--trace-out", str(collision),
                 "--stats-json", str(collision)], "preflight", expected=2)
        require(not collision.exists(), "collision output created")
        for name, metadata in (("reset", bad_timeline), ("malformed", malformed)):
            dest = out / ("reject-" + name + ".kwtr")
            invoke("reject-" + name, common + ["--metadata-tsv", str(metadata),
                     "--trace-out", str(dest)], "preflight", expected=1)
            require(not dest.exists(), "unsupported timeline left a trace")
        alias = out / "aliased-stats"
        alias.symlink_to("alias-trace.kwtr")
        invoke("reject-stats-alias", common + ["--trace-out", str(out / "alias-trace.kwtr"),
                 "--stats-json", str(alias)], "preflight", expected=1)
        require(alias.is_symlink() and not (out / "alias-trace.kwtr").exists(),
                "aliased stats changed output")
        report["assertions"].append("six pre-inference rejections and no-clobber checks passed")

        for name, count, signal, block, ends, speech in specs:
            wav, trace, stats = (out / (name + suffix) for suffix in (".wav", ".kwtr", ".stats.json"))
            command = [wav_runner, str(model), str(pack), str(wav), name,
                       "--block-samples", str(block), "--trace-out", str(trace),
                       "--stats-json", str(stats)]
            if name == "tail":
                command += ["--metadata-tsv", str(timeline)]
            live = invoke("live-" + name, command, "wav", len(ends))
            rows = read_trace(trace, model_sha)
            require([row["end_sample"] for row in rows] == ends, "frame coverage/EOF changed")
            require([row["speech_active"] for row in rows] == speech, "actual VAD trace mismatch")
            metrics = json.loads(stats.read_text())
            require(metrics["processed_samples"] == count and metrics["processed_frames"] == len(ends),
                    "runtime sample/frame accounting mismatch")
            require(metrics["speech_frames"] == sum(speech), "runtime speech frame mismatch")
            require(metrics["external_vad_frames"] == (3 if name == "tail" else 0),
                    "VAD source accounting mismatch")
            require(metrics["decoder_hits"] == metrics["detections"] + metrics["refractory_suppressed"],
                    "event/suppression accounting mismatch")
            live_events = events(live.stdout)
            require(metrics["detections"] == len(live_events), "event count differs from stats")
            report["fixtures"][name] = {"wav_sha256": digest(wav), "trace_sha256": digest(trace),
                                         "samples": count, "frames": rows,
                                         "events": live_events, "stats": metrics}
            if name == "short":
                require(not live_events, "incomplete window produced a detection")
            else:
                require(len(live_events) == 1 and live_events[0]["keyword_id"] == 42,
                        "fixed fixture must produce one accepted event")
                replay = invoke("replay-" + name,
                                [replay_runner, str(model), str(pack), str(trace), name],
                                "replay", len(ends))
                require(canonical(live_events) == events(replay.stdout), "live/replay events differ")
            save()

        # Corrupt only trace identity; official replay must reject before frames.
        original = (out / "full.kwtr").read_bytes()
        for label, offset, payload in (("model", 48, b"b" * 64),
                                       ("frontend", 14, struct.pack("<H", 1)),
                                       ("vocabulary", 32, struct.pack("<Q", FINGERPRINT + 1))):
            data = bytearray(original)
            data[offset:offset + len(payload)] = payload
            path = out / ("wrong-" + label + ".kwtr")
            path.write_bytes(data)
            invoke("reject-" + label, [replay_runner, str(model), str(pack), str(path), label],
                   "preflight", expected=1)
        report["assertions"] += ["exact trace geometry/logits/VAD/frame coverage passed",
                                   "live events equal official decoder replay for both nonempty traces",
                                   "official model/frontend/vocabulary identity rejection unchanged"]
        require(report["wav_runs_reserved"] == 3 and report["replay_runs_reserved"] == 2 and
                report["acoustic_frames_reserved"] == report["decoder_frames_reserved"] == 4,
                "fixed test budget receipt mismatch")
        report["status"] = "passed"
        save()
        print(json.dumps({"status": "passed", "receipt": str(out / "receipt.json"),
                          "acoustic_frames": 4, "decoder_only_frames": 4}))
    except BaseException as exc:
        report.update(status="failed_no_automatic_retry", error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-runner", type=Path, required=True)
    parser.add_argument("--replay-runner", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        help="new persistent evidence directory; default is temporary")
    args = parser.parse_args()
    if args.output_root is not None:
        run(args)
    else:
        with tempfile.TemporaryDirectory(prefix="kws-wav-trace-") as directory:
            args.output_root = Path(directory) / "evidence"
            run(args)
