#!/usr/bin/env python3
from __future__ import annotations

import json
import copy
import hashlib
import argparse
import pathlib
import subprocess
import sys
import tempfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCORER = ROOT / "eval" / "score_events.py"


def fixture_preparation_cases(root, fixtures):
    """Preparation failure cases use local bytes only, never a live network."""
    script = ROOT / "tools/prepare_eval_context_fixtures.py"
    destination = root / "prepared-fixtures"
    argv = [sys.executable, str(script), "--source-dir", str(fixtures),
            "--output-dir", str(destination)]
    for _ in range(2):
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["files"] == 3
    item = destination / "original.raw.jsonl"
    raw = item.read_bytes()
    item.write_bytes(b"!" + raw[1:])
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert result.returncode == 1 and "size/SHA-256 mismatch" in result.stderr
    missing = root / "missing-local-fixtures"
    failed_output = root / "must-not-be-created"
    result = subprocess.run([sys.executable, str(script), "--source-dir", str(missing),
                             "--output-dir", str(failed_output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1 and not failed_output.exists()
    sys.path.insert(0, str(ROOT / "tools"))
    import prepare_eval_context_fixtures as preparation
    source = json.loads(preparation.MANIFEST.read_text(encoding="utf-8"))
    row = source["files"][0]
    good = (fixtures / row["name"]).read_bytes()
    for status, body, accepted in ((200, good, True), (302, good, False),
                                    (200, good[:-1], False), (200, b"!" + good[1:], False)):
        connection = mock.Mock()
        response = connection.getresponse.return_value
        response.status = status
        response.getheader.return_value = None
        response.read.return_value = body
        with mock.patch.object(preparation.http.client, "HTTPSConnection", return_value=connection):
            try:
                result = preparation.fetch(source, row)
            except ValueError:
                assert not accepted
            else:
                assert accepted and result == good
        connection.close.assert_called_once()
        assert connection.request.call_args.args == (
            "GET", "/jiying2007/kws-data/" + source["commit"] + "/" + row["path"])
    print("test_eval explicit fixture preparation: 8 cases ok (local bytes/mocked HTTP)")


def context_cli_cases(root, fixtures):
    """Exercise the real CLI with saved public data only: never invoke a runner."""
    hashes = {
        "original": "b93d9068eba66a8a744d9e706b0d16a763240df566942714029fa3f053b6f4ff",
        "positive300": "90196e0730db17fbae25d604d7e57d51dec4bb1b6ef1192b3c36d7f2fae29cf2",
        "negative300": "743e97c86540a1917c938170fc21b511fc6a4b75e61f5877a13bce8cd226b98d",
    }
    labels = {"M1": [1], "M2": [2], "M3": [], "M4": [], "M5": []}
    saved = {}
    for key, digest in hashes.items():
        path = fixtures / (key + ".raw.jsonl")
        assert path.is_file(), (
            "Missing pinned fixture; first run python3 tools/prepare_eval_context_fixtures.py "
            "(or prepare --source-dir and pass --context-fixtures). Offline tests never download or skip.")
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, "Pinned fixture SHA-256 mismatch: " + str(path)
        records = [json.loads(line) for line in raw.splitlines()]
        assert records[-1]["kind"] == "run_end" and records[-1]["complete"] is True
        refs, dets = [], []
        for row in records:
            if row["kind"] == "clip_start":
                refs.append(dict(recording=row["recording"], duration_s=row["frames"] / 16000,
                                 audio_sha256=row["wav_sha256"],
                                 annotation_status="complete",
                                 expected_keywords=labels[row["recording"]]))
            if row["kind"] == "callback" and row["state"] == 1:
                dets.append(dict(recording=row["recording"], keyword_id=row["keyword"],
                                 time_s=row["available_samples"] / 16000,
                                 confidence=row["score"]))
        assert len(dets) == records[-1]["event_count"]
        saved[key] = refs, dets

    def policy(tail):
        return dict(sample_rate_hz=16000, feed_chunk_samples=4800,
                    appended_context_samples=tail,
                    appended_context_kind="digital-zero" if tail else "none",
                    reset_frontend="per-recording", reset_model="per-recording",
                    reset_decoder="per-recording", reset_clocks="per-recording",
                    eof_partial_chunk="process-retained", eof_flush=False, eof_padding="none")

    original = saved["original"]
    matched = tuple(saved["positive300"][i] + saved["negative300"][i] for i in (0, 1))
    mixed = (saved["positive300"][0] + original[0][2:], saved["positive300"][1])
    seq = 0

    def invoke(data=matched, *, tails=None, edit=None, strict=True, manifest=True,
               clip=True, extra=(), expected_code=0, error=None):
        nonlocal seq
        seq += 1
        case = root / ("context-" + str(seq))
        case.mkdir()
        refs, dets = (case / name for name in ("references.jsonl", "detections.jsonl"))
        summary = case / "summary.json"
        rows, events = copy.deepcopy(data)
        for path, values in ((refs, rows), (dets, events)):
            path.write_text("".join(json.dumps(r) + "\n" for r in values), encoding="utf-8")
        declaration = dict(schema="eval-context-v1",
                           references_sha256=hashlib.sha256(refs.read_bytes()).hexdigest(),
                           detections_sha256=hashlib.sha256(dets.read_bytes()).hexdigest(),
                           recordings={r["recording"]: dict(
                               role="positive" if r["expected_keywords" if clip else "expected"] else "negative",
                               audio_sha256=r["audio_sha256"],
                               policy=policy((tails or {}).get(r["recording"], 4800))) for r in rows})
        if edit:
            edit(declaration)
        sidecar = case / "context.json"
        sidecar.write_text(json.dumps(declaration), encoding="utf-8")
        argv = [sys.executable, str(SCORER), "--references", str(refs),
                "--detections", str(dets), "--summary", str(summary)]
        if clip:
            argv += ["--clip-presence"]
        if strict:
            argv += ["--require-matched-context"]
        if manifest:
            argv += ["--context-manifest", str(sidecar)]
        completed = subprocess.run(argv + list(extra), capture_output=True, text=True, check=False)
        assert completed.returncode == expected_code, (completed.stdout, completed.stderr)
        if error:
            assert error in completed.stderr, completed.stderr
        if expected_code:
            assert not summary.exists(), "rejected claim must not write a fresh score"
            return None
        return json.loads(summary.read_text(encoding="utf-8"))

    old = invoke(original, tails={name: 0 for name in labels})
    new = invoke()
    assert (old["positive_clips_with_target"], old["positive_clips"],
            old["negative_clips_with_events"], old["negative_clips"]) == (0, 2, 0, 3)
    assert (new["positive_clips_with_target"], new["positive_clips"],
            new["negative_clips_with_events"], new["negative_clips"]) == (1, 2, 1, 3)
    for report in (old, new):
        assert report["context"]["status"] == "MATCHED_DECLARED_CONTEXT"
        assert report["qualification_allowed"] is False
        assert report["event_annotations_available"] is False
        assert not any(key in report for key in ("frr", "far_per_hour", "p95_post_end_latency_ms"))
    mixed_tails = {"M3": 0, "M4": 0, "M5": 0}
    invoke(mixed, tails=mixed_tails, expected_code=2, error="CONTEXT_MISMATCH")
    report = invoke(mixed, tails=mixed_tails, strict=False)
    assert report["context"]["matched_context"] is False
    assert report["context"]["status"] == "CONTEXT_MISMATCH"
    assert report["positive_clips_with_target"] == 1 and report["negative_clips_with_events"] == 0
    report = invoke(mixed, tails=mixed_tails, strict=False,
                    edit=lambda d: [e["policy"].pop("eof_flush")
                                    for e in d["recordings"].values()])
    assert report["context"]["status"] == "CONTEXT_MISMATCH"
    assert report["context"]["unknown"]
    assert "appended_context_samples" in report["context"]["differences"]

    for field, value in (("reset_frontend", "continuous"), ("reset_model", "continuous"),
                         ("reset_decoder", "continuous"), ("reset_clocks", "continuous"),
                         ("eof_flush", True), ("eof_padding", "zero"),
                         ("eof_partial_chunk", "drop"), ("feed_chunk_samples", 1600)):
        invoke(edit=lambda d, f=field, v=value: d["recordings"]["M3"]["policy"].update({f: v}),
               expected_code=2, error=field)
    for remove in (False, True):
        def unknown(d):
            for entry in d["recordings"].values():
                if remove:
                    entry["policy"].pop("eof_flush")
                else:
                    entry["policy"]["eof_flush"] = "unknown"
        invoke(edit=unknown, expected_code=2, error="CONTEXT_UNVERIFIED")
        assert invoke(edit=unknown, strict=False)["context"]["matched_context"] is False
    invoke(manifest=False, expected_code=2, error="CONTEXT_UNVERIFIED")
    assert invoke(manifest=False, strict=False)["context"]["status"] == "CONTEXT_UNVERIFIED"
    invoke(edit=lambda d: d.update(references_sha256="f" * 64), expected_code=2,
           error="references_sha256 binding mismatch")
    invoke(edit=lambda d: d.update(detections_sha256="f" * 64), expected_code=2,
           error="detections_sha256 binding mismatch")
    invoke(edit=lambda d: d["recordings"]["M3"].update(audio_sha256="f" * 64),
           expected_code=2, error="audio identity binding mismatch")
    invoke(edit=lambda d: d["recordings"]["M3"].update(role="positive"),
           expected_code=2, error="role mismatch")
    invoke(edit=lambda d: d["recordings"].pop("M5"), expected_code=2, error="recording set")
    invoke(edit=lambda d: d.update(schema="unknown"), expected_code=2, error="schema")
    invoke(edit=lambda d: d["recordings"]["M3"].pop("policy"),
           expected_code=2, error="CONTEXT_UNVERIFIED")
    invoke(edit=lambda d: d["recordings"]["M3"]["policy"].update(eof_flush=0),
           expected_code=2, error="must be boolean")
    invoke(mixed, tails=mixed_tails, edit=lambda d: d.update(matched_context=True),
           expected_code=2, error="CONTEXT_MISMATCH")
    invoke(saved["positive300"], expected_code=2, error="both positive and negative")
    for status in (None, "unknown", "partial"):
        incomplete = copy.deepcopy(matched)
        incomplete[0][2]["annotation_status"] = status
        invoke(incomplete, strict=False, expected_code=2, error="complete annotations")
    incomplete = copy.deepcopy(matched)
    incomplete[0][2]["expected_keywords"] = None
    invoke(incomplete, strict=False, expected_code=2, error="explicit expected_keywords")
    zero_identity = copy.deepcopy(matched)
    zero_identity[0][2]["audio_sha256"] = "0" * 64
    invoke(zero_identity, expected_code=2, error="audio identity")

    # A wrong keyword cannot rescue a positive; duplicate saved events stay visible.
    altered = copy.deepcopy(matched)
    altered[1].append(copy.deepcopy(altered[1][1]))
    altered[1].append(dict(recording="M2", keyword_id=1, time_s=.5, confidence=.9))
    report = invoke(altered)
    assert report["positive_clips_with_target"] == 1
    assert report["missing_keywords"] == 1 and report["wrong_keyword_events"] == 1
    assert report["negative_events"] == 2 and report["repeat_events"] == 1
    for flag in ("--max-frr", "--max-far-per-hour", "--max-p95-latency-ms",
                 "--min-negative-hours", "--min-continuous-negative-seconds",
                 "--max-negative-far-upper-95-per-hour"):
        invoke(extra=(flag, "0"), expected_code=2, error="clip presence cannot satisfy")

    # Aligned synthetic scorer contract, distinct from unaligned Melo data.
    aligned = ([dict(recording="P", duration_s=2, audio_sha256="1" * 64,
                     expected=[dict(keyword_id=1, start_s=.5, end_s=1)]),
                dict(recording="N", duration_s=2, audio_sha256="2" * 64, expected=[])],
               [dict(recording="P", keyword_id=1, time_s=1.1, confidence=.9)])
    report = invoke(aligned, clip=False, extra=("--max-frr", "0"))
    assert report["frr"] == 0 and report["matched"] == 1
    assert report["context"]["matched_context"] is True
    legacy = invoke(aligned, clip=False, manifest=False, strict=False)
    assert legacy["frr"] == 0 and legacy["context"]["status"] == "CONTEXT_UNVERIFIED"
    invoke(aligned, clip=False, manifest=False, expected_code=2, error="CONTEXT_UNVERIFIED")
    invoke(aligned, clip=False, tails={"N": 0}, expected_code=2, error="CONTEXT_MISMATCH")
    module = subprocess.run([sys.executable, "-m", "eval.score_events", "--help"],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    assert module.returncode == 0, module.stderr
    print("test_eval context CLI cases:", seq, "ok (saved data only)")


def run_score(references: pathlib.Path, detections: pathlib.Path, summary: pathlib.Path):
    return subprocess.run(
        [
            sys.executable,
            str(SCORER),
            "--references",
            str(references),
            "--detections",
            str(detections),
            "--summary",
            str(summary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-fixtures", type=pathlib.Path,
                        default=ROOT / "build/eval-context-fixtures")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        references = root / "references.jsonl"
        detections = root / "detections.jsonl"
        summary = root / "summary.json"
        false_positives = root / "false_positives.jsonl"
        false_rejects = root / "false_rejects.jsonl"

        references.write_text(
            json.dumps(
                {
                    "recording": "room-1",
                    "path": "room-1.wav",
                    "duration_s": 3600.0,
                    "expected": [
                        {"keyword_id": 1, "start_s": 9.0, "end_s": 10.0},
                        {"keyword_id": 2, "start_s": 19.0, "end_s": 20.0},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        detections.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "recording": "room-1",
                            "keyword_id": 1,
                            "time_s": 10.2,
                            "confidence": 0.8,
                        }
                    ),
                    json.dumps(
                        {
                            "recording": "room-1",
                            "keyword_id": 1,
                            "time_s": 100.0,
                            "confidence": 0.7,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        subprocess.check_call(
            [
                sys.executable,
                str(SCORER),
                "--references",
                str(references),
                "--detections",
                str(detections),
                "--summary",
                str(summary),
                "--false-positives",
                str(false_positives),
                "--false-rejects",
                str(false_rejects),
                "--max-far-per-hour",
                "1.0",
                "--max-frr",
                "0.5",
                "--max-p95-latency-ms",
                "250",
            ]
        )

        result = json.loads(summary.read_text(encoding="utf-8"))
        assert result["expected"] == 2
        assert result["matched"] == 1
        assert result["false_rejects"] == 1
        assert result["false_accepts"] == 1
        assert abs(result["frr"] - 0.5) < 1.0e-9
        assert abs(result["far_per_hour"] - 1.0) < 1.0e-9
        assert result["event_match_pre_tolerance_ms"] == 150.0
        assert result["event_match_post_tolerance_ms"] == 500.0
        assert 199.0 <= result["p95_post_end_latency_ms"] <= 201.0

        early_refs = root / "early-references.jsonl"
        early_dets = root / "early-detections.jsonl"
        early_summary = root / "early-summary.json"
        early_refs.write_text(
            json.dumps({"recording": "early", "duration_s": 3.0,
                        "expected": [{"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}]}) + "\n",
            encoding="utf-8",
        )
        early_dets.write_text(
            json.dumps({"recording": "early", "keyword_id": 1,
                        "time_s": 0.9, "confidence": 0.9}) + "\n",
            encoding="utf-8",
        )
        early_default = run_score(early_refs, early_dets, early_summary)
        assert early_default.returncode == 0, early_default.stderr
        assert json.loads(early_summary.read_text(encoding="utf-8"))["matched"] == 1
        strict = subprocess.run(
            [sys.executable, str(SCORER), "--references", str(early_refs),
             "--detections", str(early_dets), "--pre-tolerance-ms", "0",
             "--summary", str(early_summary)],
            check=False, capture_output=True, text=True,
        )
        assert strict.returncode == 0, strict.stderr
        strict_summary = json.loads(early_summary.read_text(encoding="utf-8"))
        assert strict_summary["matched"] == 0
        assert strict_summary["false_rejects"] == 1
        assert strict_summary["false_accepts"] == 1

        fp = [
            json.loads(line)
            for line in false_positives.read_text(encoding="utf-8").splitlines()
        ]
        assert len(fp) == 1
        assert fp[0]["time_s"] == 100.0
        assert fp[0]["path"] == "room-1.wav"

        fr = [
            json.loads(line)
            for line in false_rejects.read_text(encoding="utf-8").splitlines()
        ]
        assert len(fr) == 1
        assert fr[0]["keyword_id"] == 2
        assert fr[0]["start_s"] == 19.0
        assert fr[0]["end_s"] == 20.0
        assert fr[0]["path"] == "room-1.wav"

        negative_refs = root / "negative-references.jsonl"
        negative_dets = root / "negative-detections.jsonl"
        negative_summary = root / "negative-summary.json"
        negative_refs.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "recording": "positive",
                            "duration_s": 1800.0,
                            "expected": [
                                {"keyword_id": 1, "start_s": 10.0, "end_s": 11.0}
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "recording": "negative",
                            "duration_s": 3600.0,
                            "expected": [],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        negative_dets.write_text(
            json.dumps(
                {
                    "recording": "positive",
                    "keyword_id": 2,
                    "time_s": 100.0,
                    "confidence": 0.7,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        completed = run_score(negative_refs, negative_dets, negative_summary)
        assert completed.returncode == 0, completed.stderr
        negative_result = json.loads(
            negative_summary.read_text(encoding="utf-8")
        )
        assert abs(negative_result["audio_hours"] - 1.5) < 1.0e-12
        assert negative_result["false_accepts"] == 1
        assert abs(negative_result["far_per_hour"] - (2.0 / 3.0)) < 1.0e-12
        assert negative_result["negative_recording_false_accepts"] == 0
        assert abs(negative_result["negative_recording_audio_hours"] - 1.0) < 1.0e-12
        assert negative_result["longest_negative_recording_seconds"] == 3600.0
        assert negative_result["negative_recording_far_per_hour"] == 0.0
        assert (
            2.9957
            < negative_result["negative_recording_far_upper_95_per_hour"]
            < 2.9958
        )
        assert negative_result["negative_recording_far_confidence"] == 0.95
        assert (
            negative_result["negative_recording_far_policy"]
            == "negative-only-recordings-poisson-upper-v1"
        )

        negative_dets.write_text(
            json.dumps(
                {
                    "recording": "negative",
                    "keyword_id": 1,
                    "time_s": 120.0,
                    "confidence": 0.8,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        completed = run_score(negative_refs, negative_dets, negative_summary)
        assert completed.returncode == 0, completed.stderr
        negative_result = json.loads(
            negative_summary.read_text(encoding="utf-8")
        )
        assert negative_result["negative_recording_false_accepts"] == 1
        assert negative_result["negative_recording_far_per_hour"] == 1.0
        assert (
            4.7438
            < negative_result["negative_recording_far_upper_95_per_hour"]
            < 4.7440
        )

        positive_only_refs = root / "positive-only-references.jsonl"
        positive_only_dets = root / "positive-only-detections.jsonl"
        positive_only_summary = root / "positive-only-summary.json"
        positive_only_refs.write_text(
            json.dumps(
                {
                    "recording": "only-positive",
                    "duration_s": 10.0,
                    "expected": [
                        {"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        positive_only_dets.write_text("", encoding="utf-8")
        completed = run_score(
            positive_only_refs,
            positive_only_dets,
            positive_only_summary,
        )
        assert completed.returncode == 0, completed.stderr
        positive_only_result = json.loads(
            positive_only_summary.read_text(encoding="utf-8")
        )
        assert positive_only_result["negative_recording_audio_hours"] == 0.0
        assert positive_only_result["negative_recording_far_per_hour"] is None
        assert (
            positive_only_result["negative_recording_far_upper_95_per_hour"]
            is None
        )
        evidence_gate = subprocess.run(
            [sys.executable, str(SCORER), "--references", str(positive_only_refs),
             "--detections", str(positive_only_dets), "--max-frr", "1",
             "--max-p95-latency-ms", "500",
             "--min-expected-per-keyword", "1", "1",
             "--min-expected-per-keyword", "2", "1",
             "--min-negative-hours", "1",
             "--min-continuous-negative-seconds", "60",
             "--max-negative-far-upper-95-per-hour", "3"],
            check=False, capture_output=True, text=True,
        )
        assert evidence_gate.returncode == 1
        assert "latency has no matched events" in evidence_gate.stderr
        assert "keyword 2 expected coverage" in evidence_gate.stderr
        assert "negative recording hours" in evidence_gate.stderr
        assert "continuous negative recording duration" in evidence_gate.stderr
        assert "negative FAR upper 95/hour" in evidence_gate.stderr

        no_positive_refs = root / "no-positive-references.jsonl"
        no_positive_refs.write_text(
            json.dumps({"recording": "negative", "duration_s": 3600.0,
                        "expected": []}) + "\n", encoding="utf-8",
        )
        no_positive_gate = subprocess.run(
            [sys.executable, str(SCORER), "--references", str(no_positive_refs),
             "--detections", str(positive_only_dets), "--max-frr", "1"],
            check=False, capture_output=True, text=True,
        )
        assert no_positive_gate.returncode == 1
        assert "FRR has no positive events" in no_positive_gate.stderr

        sufficient_negative_gate = subprocess.run(
            [sys.executable, str(SCORER), "--references", str(negative_refs),
             "--detections", str(positive_only_dets),
             "--min-expected-per-keyword", "1", "1",
             "--min-negative-hours", "1",
             "--max-negative-far-upper-95-per-hour", "3"],
            check=False, capture_output=True, text=True,
        )
        assert sufficient_negative_gate.returncode == 0, sufficient_negative_gate.stderr

        unlabeled_refs = root / "unlabeled.jsonl"
        unlabeled_refs.write_text(json.dumps({"recording": "unlabeled", "duration_s": 3600.0}) + "\n")
        unlabeled = run_score(unlabeled_refs, positive_only_dets, summary)
        assert unlabeled.returncode == 2
        assert "expected annotation is required" in unlabeled.stderr
        for invalid_args in (
            ["--min-negative-hours", "nan"],
            ["--max-negative-far-upper-95-per-hour", "inf"],
            ["--min-continuous-negative-seconds", "-1"],
            ["--min-expected-per-keyword", "1", "0"],
            ["--min-expected-per-keyword", "1", "1", "--min-expected-per-keyword", "1", "1"],
        ):
            invalid = subprocess.run(
                [sys.executable, str(SCORER), "--references", str(negative_refs),
                 "--detections", str(positive_only_dets), *invalid_args],
                check=False, capture_output=True, text=True,
            )
            assert invalid.returncode == 2, (invalid_args, invalid.stderr)
        insufficient = subprocess.run(
            [sys.executable, str(SCORER), "--references", str(negative_refs),
             "--detections", str(positive_only_dets), "--max-far-per-hour", "0",
             "--max-negative-far-upper-95-per-hour", "2"],
            check=False, capture_output=True, text=True,
        )
        assert insufficient.returncode == 1
        assert "negative FAR upper 95/hour" in insufficient.stderr

        overlap_refs = root / "overlap-references.jsonl"
        overlap_dets = root / "overlap-detections.jsonl"
        overlap_summary = root / "overlap-summary.json"
        overlap_refs.write_text(
            json.dumps(
                {
                    "recording": "overlap",
                    "duration_s": 100.0,
                    "expected": [
                        {"keyword_id": 1, "start_s": 4.0, "end_s": 5.45},
                        {"keyword_id": 1, "start_s": 5.4, "end_s": 5.8},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        overlap_dets.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "recording": "overlap",
                            "keyword_id": 1,
                            "time_s": 5.2,
                            "confidence": 0.8,
                        }
                    ),
                    json.dumps(
                        {
                            "recording": "overlap",
                            "keyword_id": 1,
                            "time_s": 5.5,
                            "confidence": 0.8,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        completed = run_score(overlap_refs, overlap_dets, overlap_summary)
        assert completed.returncode == 0, completed.stderr
        overlap_result = json.loads(overlap_summary.read_text(encoding="utf-8"))
        assert overlap_result["matched"] == 2
        assert overlap_result["false_rejects"] == 0
        assert overlap_result["false_accepts"] == 0

        invalid_refs = root / "invalid-references.jsonl"
        empty_dets = root / "empty-detections.jsonl"
        invalid_summary = root / "invalid-summary.json"
        invalid_refs.write_text(
            '{"recording":"bad","duration_s":NaN,"expected":[]}\n',
            encoding="utf-8",
        )
        empty_dets.write_text("", encoding="utf-8")
        completed = run_score(invalid_refs, empty_dets, invalid_summary)
        assert completed.returncode == 2
        assert "finite" in completed.stderr

        context_cli_cases(root, args.context_fixtures)
        fixture_preparation_cases(root, args.context_fixtures)

    print("test_eval: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
