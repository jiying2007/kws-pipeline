#!/usr/bin/env python3
"""Bounded artificial CLI contracts: base WAV fixture plus decoder completion.

The base fixture keeps 1 model, 3 WAV runs and 2 replays. Completion checks
use separate handcrafted decoder-only traces; they do not run acoustic inference.

No training, Torch, downloaded weights, speech corpus, or quality claims.
An existing output directory is refused. With --output-root, failures preserve
a partial receipt; default workspaces are temporary.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import select
from pathlib import Path
import struct
import subprocess
import sys
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



def verify_event_readback_contract() -> None:
    # Only temporary artificial files and public source snapshots. No runner,
    # model inference, audio corpus or event scoring occurs in these checks.
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "tools"))
    from event_readback_contract import (CONTEXT, SCHEMA, SOURCE_FILES, verify)
    from gen_parameter_limits import load_contract, render_header

    with tempfile.TemporaryDirectory(prefix="event-readback-contract-") as tmp:
        root = Path(tmp)

        def binding(path: Path) -> dict:
            return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                    "sha256": digest(path)}

        def put(name: str, data: bytes) -> dict:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return binding(path)

        sources = {}
        for name in SOURCE_FILES:
            sources[name] = put("source/" + name, (repo / name).read_bytes())
        # Bindings in the source manifest are relative to that manifest.
        artifacts = {"source_manifest": put("sources.json", json.dumps(sources).encode())}
        artifacts["model"] = put("fixture.kwm", fixture_model())
        artifacts["keyword_pack"] = put("fixture.kwk", fixture_pack())
        write_wav(root / "fixture.wav", 720, True)
        artifacts["source_wav"] = binding(root / "fixture.wav")
        model_sha = artifacts["model"]["sha256"].encode()
        trace_bytes = (struct.pack("<8sIHHIIIIQQ64s8x", b"KWTRACE1", 1, 4, 0,
                                  16000, 400, 320, 0, FINGERPRINT, 2, model_sha)
                       + b"".join(struct.pack("<QB7x4f", end, 1, *LOGITS)
                                  for end in (400, 720)))
        artifacts["trace"] = put("fixture.kwtr", trace_bytes)
        artifacts["capture_binary"] = put("capture-fixture", b"not executed capture fixture")
        artifacts["replay_binary"] = put("replay-fixture", b"not executed replay fixture")
        contract_path = root / "source/configs/parameter-contract.json"
        artifacts["parameter_contract"] = binding(contract_path)
        contract, contract_sha = load_contract(contract_path)
        artifacts["generated_header"] = put("kws_parameter_limits.h", render_header(contract, contract_sha).encode())
        completion = {"exit_code": 0, "frames_consumed": 2, "decoder_hits": 1,
                      "detections": 1, "refractory_suppressed": 0}
        artifacts["completion"] = put("completion.json", json.dumps(completion).encode())
        event_row = {"recording": "fixture", "keyword_id": 42, "time_s": .025, "confidence": .9}
        artifacts["events"] = put("events.jsonl", (json.dumps(event_row) + "\n").encode())
        recipe = {"schema": SCHEMA, "scope": "observed-per-clip-diagnostic",
                  "source_id": "fixture", "data_role": "synthetic-control",
                  "label_grade": "constructed", "context": copy.deepcopy(CONTEXT),
                  "runtime": {key: row["default"] for key, row in contract["runtime"].items()},
                  "keyword_policy": [{"id": 42, "threshold": struct.unpack("<f", struct.pack("<f", .30))[0],
                                      "tokens": [1], "min_trailing_blanks": 0, "priority": 0,
                                      "prefix_policy": 0, "grace_frames": 0}], "artifacts": artifacts}
        receipt = root / "receipt.json"

        def checked(value: dict) -> dict:
            receipt.write_text(json.dumps(value))
            return verify(receipt)

        result = checked(recipe)
        require(result["status"] == "VERIFIED_BINDINGS_ONLY" and result["execution_observed"] is False,
                "identity verification must not authenticate execution")
        require(result["promotion_allowed"] is False and result["far_per_hour"] is None and
                result["frr"] is None and result["word_end_latency_ms"] is None,
                "diagnostic receipt must not invent quality metrics")

        def rejected(change, message: str) -> None:
            value = copy.deepcopy(recipe)
            change(value)
            try:
                checked(value)
            except (ValueError, KeyError) as exc:
                require(message in str(exc), f"wrong rejection: {exc}")
            else:
                raise AssertionError("accepted invalid event receipt: " + message)

        rejected(lambda v: v.update(frr=0), "receipt fields")
        rejected(lambda v: v.update(scope="product-qualified-events"), "scope")
        rejected(lambda v: v.update(data_role="sealed-test"), "data role")
        rejected(lambda v: v["context"].update(mode="continuous"), "context")
        rejected(lambda v: v["context"].update(vad="from-target"), "context")
        rejected(lambda v: v["context"].update(sample_origin=False), "context")
        rejected(lambda v: v["runtime"].update(refractory_ms=0), "runtime")
        rejected(lambda v: v["runtime"].update(refractory_ms=True), "runtime")
        rejected(lambda v: v["keyword_policy"][0].update(threshold=.5), "keyword policy")
        rejected(lambda v: v["artifacts"]["model"].update(sha256="0" * 64), "hash")
        rejected(lambda v: v["artifacts"]["trace"].update(bytes=True), "integer")
        rejected(lambda v: v["artifacts"]["trace"].update(path="../outside"), "outside")
        # Rehashing corrupted artifacts does not bypass semantic validation.
        for offset, payload, reason in ((128, bytes([2]), "VAD"),
                                         (120, struct.pack("<Q", 401), "clock"),
                                         (48, b"b" * 64, "trace model"),
                                         (136, struct.pack("<f", float("nan")), "non-finite")):
            damaged = bytearray(trace_bytes)
            damaged[offset:offset + len(payload)] = payload
            new_binding = put("damaged.kwtr", bytes(damaged))
            rejected(lambda v: v["artifacts"].update(trace=new_binding), reason)
        raw_binding = put("raw-logits.bin", b"transcript-only")
        rejected(lambda v: v["artifacts"].update(trace=raw_binding), "trace header")
        missing = dict(sources)
        missing.pop("src/decoder.c")
        new_binding = put("missing-sources.json", json.dumps(missing).encode())
        rejected(lambda v: v["artifacts"].update(source_manifest=new_binding), "complete supported")
        bad_completion = dict(completion, exit_code=True)
        new_binding = put("bad-completion.json", json.dumps(bad_completion).encode())
        rejected(lambda v: v["artifacts"].update(completion=new_binding), "integer")
        bad_completion = dict(completion, frames_consumed=1)
        new_binding = put("short-completion.json", json.dumps(bad_completion).encode())
        rejected(lambda v: v["artifacts"].update(completion=new_binding), "incomplete")
        for update, reason in (({"time_s": 0}, "frame clock"),
                               ({"time_s": .026}, "frame clock"),
                               ({"confidence": .29}, "below keyword threshold"),
                               ({"confidence": True}, "numeric field")):
            changed_event = dict(event_row, **update)
            new_binding = put("bad-event.jsonl", json.dumps(changed_event).encode())
            rejected(lambda v: v["artifacts"].update(events=new_binding), reason)
        # Six-decimal CLI rounding of a float32 threshold is allowed.
        rounded = put("rounded.jsonl", json.dumps(dict(event_row, confidence=.3)).encode())
        value = copy.deepcopy(recipe)
        value["artifacts"]["events"] = rounded
        checked(value)
        for second_time in (.025, .045):
            double = put("double.jsonl", (json.dumps(event_row) + "\n" +
                         json.dumps(dict(event_row, time_s=second_time))).encode())
            double_completion = put("double-completion.json", json.dumps(dict(
                completion, decoder_hits=2, detections=2)).encode())
            rejected(lambda v: v["artifacts"].update(events=double, completion=double_completion),
                     "frame uniqueness/refractory")
        no_events = put("no-events.jsonl", b"")
        suppressed_only = put("suppressed-only.json", json.dumps(dict(
            completion, detections=0, refractory_suppressed=1)).encode())
        rejected(lambda v: v["artifacts"].update(events=no_events, completion=suppressed_only),
                 "suppress its first hit")
        empty_trace = bytearray(trace_bytes[:120])
        struct.pack_into("<Q", empty_trace, 40, 0)
        empty_binding = put("empty.kwtr", bytes(empty_trace))
        rejected(lambda v: v["artifacts"].update(trace=empty_binding), "non-empty trace")
        receipt.write_text('{"schema":"a","schema":"b"}')
        try:
            verify(receipt)
        except ValueError as exc:
            require("duplicate" in str(exc), "duplicate receipt key rejection")
        else:
            raise AssertionError("duplicate receipt key accepted")
    print("event_readback_contract: ok (artificial files only)")


def verify_replay_completion(runner: Path) -> None:
    """Bounded decoder-only execution on handcrafted traces; no acoustic model run."""
    runner = runner.resolve(strict=True)
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="replay-completion-") as directory:
        root = Path(directory)
        model, pack = root / "fixture.kwm", root / "fixture.kwk"
        model.write_bytes(fixture_model())
        pack_bytes = bytearray(fixture_pack())
        struct.pack_into("<fH", pack_bytes, 28, .55, 2)
        struct.pack_into("<2H", pack_bytes, 40, 1, 2)
        pack.write_bytes(pack_bytes)
        blank = (8., -8., -8., -8.)
        first = (-8., 8., -8., -8.)
        second = (-8., -8., 8., -8.)
        calls = 0

        def trace(name: str, frames: list[tuple], flags: list[int] | None = None) -> Path:
            active = flags if flags is not None else [1] * len(frames)
            path = root / (name + ".kwtr")
            path.write_bytes(struct.pack("<8sIHHIIIIQQ64s8x", b"KWTRACE1", 1, 4, 0,
                                        16000, 400, 320, 0, FINGERPRINT, len(frames),
                                        digest(model).encode()) + b"".join(
                struct.pack("<QB7x4f", 400 + i * 320, active[i], *frame)
                for i, frame in enumerate(frames)))
            return path

        def command(source: Path, dest: Path | None, extra: tuple = ()) -> list[str]:
            return [str(runner), str(model), str(pack), str(source), "completion-fixture",
                    *extra, *(["--completion", str(dest)] if dest is not None else [])]

        def invoke(source: Path, dest: Path | None, extra: tuple = (), success: bool = True):
            nonlocal calls
            calls += 1
            require(calls <= 24, "completion fixture invocation budget exceeded")
            result = subprocess.run(command(source, dest, extra), capture_output=True,
                                    text=True, timeout=15)
            require((result.returncode == 0) == success, "completion invocation exit: " + result.stderr)
            if not success and dest is not None:
                require(not dest.exists(), "failed replay published completion")
            return result

        def checked(source: Path, expected: tuple[int, int, int], extra: tuple = ()) -> dict:
            dest = root / (source.stem + ".json")
            result = invoke(source, dest, extra)
            receipt = json.loads(dest.read_text())
            (root / (source.stem + ".events.jsonl")).write_text(result.stdout)
            require(receipt["schema"] == "kws-decoder-replay-completion-v1", "completion schema")
            require(receipt["production_qualified"] is False, "completion is not qualification")
            require(receipt["execution_kind"] == "native-linux-decoder-replay", "execution kind")
            c = receipt["completion"]
            frame_count = struct.unpack_from("<Q", source.read_bytes(), 40)[0]
            require(c == dict(exit_code=0, frames_consumed=frame_count,
                              decoder_hits=expected[0], detections=expected[1],
                              refractory_suppressed=expected[2]), "actual engine completion counts")
            identity = receipt["identity"]
            for field, path in (("model_sha256", model), ("keyword_pack_sha256", pack),
                                ("trace_sha256", source), ("tool_sha256", runner),
                                ("parameter_contract_sha256", repo / "configs/parameter-contract.json")):
                require(identity[field] == digest(path), "execution identity: " + field)
            require(receipt["context"]["recording_id"] == "completion-fixture" and
                    receipt["context"]["mode"] == "per-clip-fresh-engine", "execution context")
            require(len(events(result.stdout)) == expected[1], "visible events differ from engine detections")
            plain = invoke(source, None, extra)
            require(result.stdout == plain.stdout, "completion option changed event bytes")
            return receipt

        word = [first, second]
        cooldown = trace("cooldown", word + [blank] * 2 + word + [blank] * 60 + word)
        baseline = checked(cooldown, (3, 2, 1))
        from event_readback_contract import verify_completion
        # Run the whole saved-file verifier with a real native completion.
        # WAV/capture bytes here remain artificial identity fixtures: this does
        # not claim the trace was acoustically produced from that WAV.
        from event_readback_contract import CONTEXT, SCHEMA, SOURCE_FILES, verify
        from gen_parameter_limits import load_contract, render_header

        def binding(path: Path) -> dict:
            return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                    "sha256": digest(path)}

        def put(name: str, payload: bytes) -> dict:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return binding(path)

        sources = {name: put("source/" + name, (repo / name).read_bytes()) for name in SOURCE_FILES}
        contract_path = root / "source/configs/parameter-contract.json"
        contract, contract_sha = load_contract(contract_path)
        frame_count = baseline["completion"]["frames_consumed"]
        wav = root / "identity-fixture.wav"
        write_wav(wav, 400 + 320 * (frame_count - 1), False)
        # Copy the actual runner so every binding remains under the receipt root.
        artifacts = {"source_manifest": put("sources.json", json.dumps(sources).encode()),
                     "parameter_contract": binding(contract_path),
                     "generated_header": put("generated.h", render_header(contract, contract_sha).encode()),
                     "model": binding(model), "keyword_pack": binding(pack),
                     "trace": binding(cooldown), "source_wav": binding(wav),
                     "capture_binary": put("capture-fixture", b"artificial identity only"),
                     "replay_binary": put("replay-binary", runner.read_bytes()),
                     "events": binding(root / "cooldown.events.jsonl"),
                     "completion": binding(root / "cooldown.json")}
        recipe = {"schema": SCHEMA, "scope": "observed-per-clip-diagnostic",
                  "source_id": "completion-fixture", "data_role": "synthetic-control",
                  "label_grade": "constructed", "context": copy.deepcopy(CONTEXT),
                  "runtime": {key: value["default"] for key, value in contract["runtime"].items()},
                  "keyword_policy": [{"id": 42, "threshold": struct.unpack("<f", struct.pack("<f", .55))[0],
                                      "tokens": [1, 2], "min_trailing_blanks": 0, "priority": 0,
                                      "prefix_policy": 0, "grace_frames": 0}], "artifacts": artifacts}
        complete_receipt = root / "full-receipt.json"
        complete_receipt.write_text(json.dumps(recipe))
        verified = verify(complete_receipt)
        require(verified["status"] == "VERIFIED_BINDINGS_ONLY" and
                verified["execution_observed"] is False, "native envelope invented authentication")
        tampered = copy.deepcopy(baseline)
        tampered["identity"]["trace_sha256"] = "0" * 64
        changed_recipe = copy.deepcopy(recipe)
        changed_recipe["artifacts"]["completion"] = put("rehashed-completion.json", json.dumps(tampered).encode())
        complete_receipt.write_text(json.dumps(changed_recipe))
        try:
            verify(complete_receipt)
        except ValueError as exc:
            require("identity" in str(exc), "wrong full-envelope tamper rejection")
        else:
            raise AssertionError("accepted rehashed but misbound native envelope")
        identities, runtime = baseline["identity"], baseline["runtime"]
        require(verify_completion(baseline, identities, runtime, "completion-fixture") ==
                baseline["completion"], "native envelope verifier rejected execution")
        for field in identities:
            damaged = copy.deepcopy(baseline)
            damaged["identity"][field] = "0" * 64
            try:
                verify_completion(damaged, identities, runtime, "completion-fixture")
            except ValueError:
                pass
            else:
                raise AssertionError("accepted altered completion identity: " + field)
        for kind in ("runtime", "recording", "production"):
            damaged = copy.deepcopy(baseline)
            if kind == "runtime":
                damaged["runtime"]["refractory_ms"] = 0
            elif kind == "recording":
                damaged["context"]["recording_id"] = "another-recording"
            else:
                damaged["production_qualified"] = True
            try:
                verify_completion(damaged, identities, runtime, "completion-fixture")
            except ValueError:
                pass
            else:
                raise AssertionError("accepted altered completion " + kind)
        checked(trace("no-hit", [blank] * 3), (0, 0, 0))
        checked(trace("empty", []), (0, 0, 0))
        checked(trace("inactive-reset", [first] + [blank] * 12 + [second],
                      [1] + [0] * 12 + [1]), (0, 0, 0))
        checked(trace("fresh-engine", word), (1, 1, 0))
        override = checked(trace("no-refractory", word + [blank] + word), (2, 2, 0),
                           ("--refractory-ms", "0"))
        require(override["runtime"]["refractory_ms"] == 0, "actual runtime override not recorded")
        truncated = root / "truncated.kwtr"
        truncated.write_bytes(cooldown.read_bytes()[:-1])
        invoke(truncated, root / "truncated.json", success=False)
        corrupt = root / "corrupt.kwtr"
        corrupt.write_bytes(b"not a trace")
        invoke(corrupt, root / "corrupt.json", success=False)
        invoke(root / "missing.kwtr", root / "missing.json", success=False)
        sentinel = root / "existing.json"
        sentinel.write_bytes(b"do not replace\n")
        sentinel_digest = digest(sentinel)
        result = subprocess.run(command(cooldown, sentinel), capture_output=True, timeout=15)
        require(result.returncode != 0 and digest(sentinel) == sentinel_digest,
                "existing completion overwritten")
        dangling = root / "dangling.json"
        dangling.symlink_to("absent.json")
        result = subprocess.run(command(cooldown, dangling), capture_output=True, timeout=15)
        require(result.returncode != 0 and dangling.is_symlink() and not (root / "absent.json").exists(),
                "dangling completion symlink followed")
        # A real output failure must prevent a success sidecar after decoding.
        if sys.platform.startswith("linux"):
            failed = root / "stdout-failed.json"
            with open("/dev/full", "wb") as full:
                result = subprocess.run(command(cooldown, failed), stdout=full,
                                        stderr=subprocess.PIPE, timeout=15)
            require(result.returncode != 0 and not failed.exists(), "stdout failure published success")
            # Backpressure, not a scheduling sleep, holds a live execution before
            # its completion boundary. Termination cannot publish success.
            long_trace = trace("interrupted", (word + [blank]) * 30000)
            interrupted = root / "interrupted.json"
            proc = subprocess.Popen(command(long_trace, interrupted, ("--refractory-ms", "0")),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                require(bool(select.select([proc.stdout], [], [], 15)[0]), "interruption fixture produced no output")
                require(proc.stdout.readline().startswith(b'{"recording":'), "no actual event before interruption")
                require(proc.poll() is None and not interrupted.exists(), "execution completed before interruption")
                proc.terminate()
                proc.communicate(timeout=15)
                require(proc.returncode != 0 and not interrupted.exists(), "interruption published success")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate(timeout=15)
            # Install a destination only after actual execution started. The
            # no-replace commit must preserve this concurrent writer's bytes.
            raced = root / "raced.json"
            proc = subprocess.Popen(command(long_trace, raced, ("--refractory-ms", "0")),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                require(bool(select.select([proc.stdout], [], [], 15)[0]), "race fixture produced no output")
                require(proc.stdout.readline().startswith(b'{"recording":'), "race fixture did not execute")
                require(proc.poll() is None and not raced.exists(), "race fixture already completed")
                raced.write_bytes(b"concurrent owner\n")
                proc.communicate(timeout=15)
                require(proc.returncode != 0 and raced.read_bytes() == b"concurrent owner\n",
                        "atomic completion replaced concurrent destination")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate(timeout=15)
        print("replay_completion: ok (handcrafted decoder-only traces; no acoustic inference)")


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
        for suffix, second_value in (("same", "160"), ("different", "320")):
            trace_dest = out / ("duplicate-block-" + suffix + ".kwtr")
            stats_dest = out / ("duplicate-block-" + suffix + ".stats.json")
            invoke("reject-duplicate-block-" + suffix,
                   common + ["--block-samples", "160", "--block-samples", second_value,
                             "--trace-out", str(trace_dest), "--stats-json", str(stats_dest)],
                   "preflight", expected=2)
            require(not trace_dest.exists() and not stats_dest.exists(),
                    "duplicate --block-samples created an output file")
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
        report["assertions"].append("eight pre-inference rejections and no-clobber checks passed")

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
    parser.add_argument("--wav-runner", type=Path)
    parser.add_argument("--replay-runner", type=Path)
    parser.add_argument("--contract-only", action="store_true",
                        help="verify artificial receipt fixtures without executing runners")
    parser.add_argument("--output-root", type=Path,
                        help="new persistent evidence directory; default is temporary")
    args = parser.parse_args()
    verify_event_readback_contract()
    if args.contract_only:
        raise SystemExit(0)
    if args.wav_runner is None or args.replay_runner is None:
        parser.error("--wav-runner and --replay-runner are required unless --contract-only")
    verify_replay_completion(args.replay_runner)
    if args.output_root is not None:
        run(args)
    else:
        with tempfile.TemporaryDirectory(prefix="kws-wav-trace-") as directory:
            args.output_root = Path(directory) / "evidence"
            run(args)
