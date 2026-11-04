#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import subprocess
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_wav(path: pathlib.Path, marker: int, seconds: float = 1.0) -> None:
    frames = int(16000 * seconds)
    samples = [((index + marker * 17) % 257) - 128 for index in range(frames)]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(
            b"".join(struct.pack("<h", value * 32) for value in samples)
        )


def run(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != expect:
        raise AssertionError(
            f"expected exit {expect}, got {completed.returncode}:\n{completed.stdout}"
        )
    return completed



def test_phase_a_field_plan_preflight() -> None:
    """No actual recording, source right, AFE, qualification or model involved."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import phase_a_field_plan_preflight as preflight
    import plan_phase_a_capture_supplement as supplement

    policy = json.loads((ROOT / "configs/research/real-human-coverage-v1.json")
                        .read_text(encoding="utf-8"))

    def draft(session_roles: list[tuple[str, str, str, str]]) -> dict:
        slots = []
        for speaker, session, family, role in session_roles:
            for kind, prompt, distance, azimuth, repetition in sorted(preflight.expected_tasks()):
                row = {
                    "speaker_id": speaker, "session_id": session,
                    "source_family_id": family, "role": role,
                    "room_id": "room-one", "device_id": "microphone-one",
                    "afe_profile_sha256": None,
                    "distance_m": distance, "azimuth_deg": azimuth,
                    "repetition": repetition, "prompt": prompt,
                    "intent_kind": kind,
                    "intended_keyword_id": preflight.WORDS.get(prompt),
                    "audio_recorded": False, "human_verbatim_reviewed": False,
                    "rights_approved": False, "training_admitted": False,
                }
                row["slot_id"] = preflight.slot_identity(row)
                slots.append(row)
        slots.sort(key=lambda row: row["slot_id"])
        return {
            "schema": preflight.FIELD_SCHEMA,
            "source_type": "human-field-recording-candidate",
            "scope": "private-local-pilot-only",
            "roles_are_declared_not_authenticated": True,
            "participants": len(session_roles),
            "slots": slots,
            "human_rights_authenticated": False,
            "audio_labels_verified": False, "training_admitted": False,
            "qualification_allowed": False, "shipping_approved": False,
            "model_calls": 0,
        }

    roles = [
        ("speaker-a", "speaker-a-1", "family-a", "train-candidate"),
        ("speaker-b", "speaker-b-1", "family-b", "development"),
        ("speaker-c", "speaker-c-1", "family-c", "sealed-evaluation"),
    ]
    plan = draft(roles)
    result = preflight.assess(plan, policy, preflight.canonical_sha(plan))
    assert result["declared_session_participants"] == 3
    assert result["declared_sealed_speakers"] == 1
    assert result["sealed_requested_wakes_by_keyword"] == {"1": 12, "2": 12}
    assert result["sealed_rear_azimuth_candidate_wakes"] == {"1": 6, "2": 6}
    assert result["verified_negative_audio_hours"] == 0
    assert result["verified_real_speakers"] == 0
    assert not result["qualification_allowed"] and not result["training_admitted"]
    assert result["research_policy_id"] == "xiaowo-research-real-human-coverage-v1"
    assert "deployment_tag" not in policy and "shipping_approved" not in policy
    assert result["planned_quota_gaps"]["declared_sealed_speakers_missing"] == 39
    assert result["planned_quota_gaps"]["declared_sealed_speakers_short_of_session_min"] == 1
    tagged = {r["name"]: r for r in result["critical_positive_slice_plans"]}
    for name in ("rear", "playback", "double_talk", "robot_motion"):
        assert tagged[name]["requested_support"]["1"]["planned_candidate_slots"] == 0
        assert not tagged[name]["actual_qualification_support_verified"]

    # A full *declared* 40-speaker, two-session sealed plan can meet the raw
    # slot-count minimum while STILL having zero verified acoustic evidence.
    full = draft([
        (f"speaker-{i:03d}", f"session-{i:03d}-{j}",
         f"family-{i:03d}", "sealed-evaluation")
        for i in range(40) for j in (1, 2)
    ])
    full_report = preflight.assess(full, policy, preflight.canonical_sha(full))
    assert full_report["declared_sealed_speakers"] == 40
    assert full_report["declared_sealed_sessions"] == 80
    assert full_report["sealed_requested_wakes_by_keyword"] == {"1": 960, "2": 960}
    assert full_report["planned_quota_gaps"]["planned_wakes_total_missing"] == 0
    assert full_report["planned_quota_gaps"]["declared_sealed_speakers_missing"] == 0
    assert full_report["verified_actual_keyword_events"] == 0
    assert full_report["verified_negative_audio_hours"] == 0
    assert full_report["critical_negative_exposure_hours_missing"] == \
        policy["critical_negative_exposure_hours"]
    assert not full_report["qualification_allowed"] and "shipping_approved" not in full_report
    assert len(json.dumps(full, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode()) > 1024 * 1024
    # Upstream kws-data v1 verifies individual manifests only up to 1 MiB.
    # Two <=1 MiB shards are allowed, but MUST be evaluated together to
    # count each speaker once and preserve both of that speaker's sessions.
    shard_one = draft([
        (f"speaker-{i:03d}", f"session-{i:03d}-1",
         f"family-{i:03d}", "sealed-evaluation") for i in range(40)
    ])
    shard_two = draft([
        (f"speaker-{i:03d}", f"session-{i:03d}-2",
         f"family-{i:03d}", "sealed-evaluation") for i in range(40)
    ])
    for shard in (shard_one, shard_two):
        # Mirror the upstream _publish() serialization, not just compact JSON.
        assert len(json.dumps(shard, ensure_ascii=False, sort_keys=True,
                              indent=2).encode()) < 1024 * 1024
    hashes = [preflight.canonical_sha(shard_one), preflight.canonical_sha(shard_two)]
    joint = preflight.assess_many([shard_one, shard_two], policy, hashes)
    assert joint["source_shard_count"] == 2
    assert joint["source_shard_plan_sha256s"] == sorted(hashes)
    reversed_joint = preflight.assess_many([shard_two, shard_one],
                                          policy, [hashes[1], hashes[0]])
    assert reversed_joint["source_plan_sha256"] == joint["source_plan_sha256"]
    assert reversed_joint["source_shard_plan_sha256s"] == joint["source_shard_plan_sha256s"]
    assert joint["declared_sealed_speakers"] == 40
    assert joint["declared_sealed_sessions"] == 80

    # The 37-slot v1 pilot lacks explicit playback/double-talk/robot-motion
    # positive requests and does not allocate actual negative audio hours.
    # A separate product-policy-derived supplement now prepares physical
    # collection tasks without claiming that ANY audio has been captured.
    capture = supplement.build([shard_one, shard_two], policy, hashes)
    assert capture["declared_sealed_speakers"] == 40
    assert capture["declared_sealed_sessions"] == 80
    assert capture["new_capture_requests"] == 1600
    assert capture["requested_negative_hours_total"] == 26.4
    assert capture["requested_negative_duration_s_per_session"] == {
        "household": 594, "speech_confusion": 198,
        "playback": 198, "robot_motion": 198
    }
    for tag in ("rear", "playback", "double_talk", "robot_motion"):
        assert capture["requested_positive_by_scene_keyword"][tag] == {"1": 160, "2": 160}
    for tag, value in (("household", 13.2), ("speech_confusion", 4.4),
                       ("playback", 4.4), ("robot_motion", 4.4)):
        assert capture["requested_negative_hours_by_scene"][tag] == value
    assert len({x["request_id"] for x in capture["requests"]}) == 1600
    assert capture["capture_requests_sha256"] == preflight.canonical_sha(capture["requests"])
    assert all(x["qualification_allowed"] is False and
               x["actual_acoustic_tags_verified"] is False and
               x["training_admitted"] is False for x in capture["requests"])
    assert capture["observed_wake_events"] == 0
    assert capture["observed_post_afe_negative_hours"] == 0
    assert capture["research_policy_id"] == "xiaowo-research-real-human-coverage-v1"
    assert "shipping_approved" not in capture and "deployment_tag" not in capture
    opposite = supplement.build([shard_two, shard_one], policy, hashes[::-1])
    assert opposite == capture
    try:
        supplement.build([shard_one, shard_two], policy, [hashes[0], "0" * 64])
    except ValueError as exc:
        assert "INDEPENDENT_FROZEN_FIELD_PLAN_DIGEST_MISMATCH" in str(exc)
    else:
        raise AssertionError("supplement accepted forged source freeze")
    try:
        supplement.build([shard_one], policy, [hashes[0]])
    except ValueError as exc:
        assert "FRESH_SEALED_COHORT_MINIMUM_NOT_DECLARED" in str(exc)
    else:
        raise AssertionError("single-session capture was treated as ready")
    with tempfile.TemporaryDirectory() as supplemental_tmp:
        folder = pathlib.Path(supplemental_tmp)
        for name, value in (("a.json", shard_one), ("b.json", shard_two),
                            ("policy.json", policy)):
            (folder / name).write_text(json.dumps(value, ensure_ascii=False,
                                                 sort_keys=True, indent=2), encoding="utf-8")
        output = folder / "supplemental-capture-requests.json"
        args = ["--field-plan", str(folder / "a.json"),
                "--field-plan", str(folder / "b.json"),
                "--expected-plan-sha256", hashes[0],
                "--expected-plan-sha256", hashes[1],
                "--research-coverage-policy", str(folder / "policy.json"),
                "--output", str(output)]
        assert supplement.main(args) == 0
        assert (output.stat().st_mode & 0o777) == 0o600
        assert json.loads(output.read_text())["qualification_allowed"] is False
        try:
            supplement.main(args)
        except FileExistsError:
            pass
        else:
            raise AssertionError("supplemental private plan overwritten")
        args[-1] = str(ROOT / "private-capture-requests-not-for-public-repo.json")
        try:
            supplement.main(args)
        except ValueError as exc:
            assert "PRIVATE_CREATE_ONLY_OUTPUT_REQUIRED" in str(exc)
        else:
            raise AssertionError("supplemental private plan published in public repo")
    assert joint["declared_sealed_source_family_groups"] == 40
    assert joint["sealed_requested_wakes_by_keyword"] == {"1": 960, "2": 960}
    assert not joint["aggregate_source_identity_authenticated"]
    assert joint["verified_negative_audio_hours"] == 0
    assert not joint["qualification_allowed"]
    # Individual plans may each be structurally valid and still cause cross-
    # shard speaker/voice-family split leakage. The aggregate must reject it.
    leaking = json.loads(json.dumps(shard_two))
    for row in leaking["slots"]:
        if row["speaker_id"] == "speaker-000":
            row["role"] = "development"
    try:
        preflight.assess_many(
            [shard_one, leaking], policy,
            [hashes[0], preflight.canonical_sha(leaking)])
    except ValueError as exc:
        assert "FIELD_SPEAKER_OR_FAMILY_CROSSES_SPLITS" in str(exc)
    else:
        raise AssertionError("cross-shard source split leakage accepted")
    # A pseudonymous speaker cannot silently switch source family between
    # sessions, and sharing one family across different "speakers" must not
    # inflate independent planned source-family breadth.
    drift = json.loads(json.dumps(shard_two))
    for row in drift["slots"]:
        if row["speaker_id"] == "speaker-000":
            row["source_family_id"] = "family-renamed"
    try:
        preflight.assess_many(
            [shard_one, drift], policy,
            [hashes[0], preflight.canonical_sha(drift)])
    except ValueError as exc:
        assert "SPEAKER_SOURCE_FAMILY_DRIFT_BETWEEN_SESSIONS" in str(exc)
    else:
        raise AssertionError("same speaker's family changed between sessions")
    shared = json.loads(json.dumps(full))
    for row in shared["slots"]:
        if row["speaker_id"] == "speaker-001":
            row["source_family_id"] = "family-000"
    shared_report = preflight.assess(shared, policy, preflight.canonical_sha(shared))
    assert shared_report["declared_sealed_speakers"] == 40
    assert shared_report["declared_sealed_source_family_groups"] == 39
    assert shared_report["planned_quota_gaps"]["declared_source_family_groups_missing"] == 1
    assert shared_report["verified_real_speakers"] == 0

    try:
        preflight.assess_many([shard_one, shard_one], policy, [hashes[0], hashes[0]])
    except ValueError as exc:
        assert "FIELD_INTENT_OR_SLOT_ID_CHANGED" in str(exc)
    else:
        raise AssertionError("duplicated shard increased observed coverage")
    with tempfile.TemporaryDirectory() as shard_tmp:
        shard_root = pathlib.Path(shard_tmp)
        inputs = [shard_root / "plan-a.json", shard_root / "plan-b.json"]
        policy_file = shard_root / "policy.json"
        output = shard_root / "aggregated-private-coverage.json"
        for target, shard in zip(inputs, (shard_one, shard_two)):
            target.write_text(json.dumps(shard, ensure_ascii=False, sort_keys=True,
                                         indent=2), encoding="utf-8")
        policy_file.write_text(json.dumps(policy), encoding="utf-8")
        assert preflight.main([
            "--field-plan", str(inputs[0]), "--field-plan", str(inputs[1]),
            "--research-coverage-policy", str(policy_file),
            "--expected-plan-sha256", hashes[0],
            "--expected-plan-sha256", hashes[1], "--output", str(output),
        ]) == 0
        reported = json.loads(output.read_text(encoding="utf-8"))
        assert reported["declared_sealed_speakers"] == 40
        assert reported["source_shard_count"] == 2
        assert not reported["qualification_allowed"]

    bad = json.loads(json.dumps(plan))
    bad["slots"][0]["audio_recorded"] = True
    try:
        preflight.assess(bad, policy, preflight.canonical_sha(bad))
    except ValueError as exc:
        assert "FIELD_SLOT_CANNOT_CLAIM_OBSERVED_AUDIO_OR_AUTHORITY" in str(exc)
    else:
        raise AssertionError("forged recorded audio was accepted")
    bad = json.loads(json.dumps(plan))
    bad["slots"][0]["slot_id"] = "s-forged"
    try:
        preflight.assess(bad, policy, preflight.canonical_sha(bad))
    except ValueError as exc:
        assert "FIELD_INTENT_OR_SLOT_ID_CHANGED" in str(exc)
    else:
        raise AssertionError("forged slot identity was accepted")
    try:
        preflight.assess(plan, policy, "0" * 64)
    except ValueError as exc:
        assert "INDEPENDENT_FROZEN_FIELD_PLAN_DIGEST_MISMATCH" in str(exc)
    else:
        raise AssertionError("edited frozen plan digest accepted")
    assert policy["minimums"]["negative_audio_hours"] == 24.0
    # The source policy stores measured hours as a positive JSON float,
    # while speakers/sessions/wake-event counts are genuine integers.
    for value in (True, False, -1.0, 0.0, float("nan"), float("inf")):
        changed = json.loads(json.dumps(policy))
        changed["minimums"]["negative_audio_hours"] = value
        try:
            preflight.assess(plan, changed, preflight.canonical_sha(plan))
        except ValueError as exc:
            assert "PHASE_A_NEGATIVE_HOURS_INVALID" in str(exc)
        else:
            raise AssertionError("invalid negative-audio exposure quota accepted")
    bad_policy = json.loads(json.dumps(policy))
    bad_policy["minimums"]["speakers"] = True
    try:
        preflight.assess(plan, bad_policy, preflight.canonical_sha(plan))
    except ValueError as exc:
        assert "PHASE_A_QUOTA_NOT_POSITIVE_INTEGER" in str(exc)
    else:
        raise AssertionError("boolean speaker policy accepted")
    forged_approval = json.loads(json.dumps(policy))
    forged_approval["shipping_approved"] = True
    try:
        preflight.assess(plan, forged_approval, preflight.canonical_sha(plan))
    except ValueError as exc:
        assert "UNEXPECTED_PHASE_A_POLICY" in str(exc)
    else:
        raise AssertionError("commercial approval flag unexpectedly accepted in research policy")
    weakened = json.loads(json.dumps(policy))
    weakened["minimums"]["expected_wakes_per_keyword"] = 1
    try:
        preflight.assess(plan, weakened, preflight.canonical_sha(plan))
    except ValueError as exc:
        assert "PHASE_A_POLICY_NOT_CURRENT_SOURCE" in str(exc)
    else:
        raise AssertionError("caller-supplied weakened Phase A policy accepted")
    # A private 0600, create-once record only; never public Git worktree.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        plan_file, policy_file, output = (
            tmp / "field-plan.json", tmp / "phase-a.json", tmp / "coverage.json")
        plan_file.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        policy_file.write_text(json.dumps(policy), encoding="utf-8")
        args = ["--field-plan", str(plan_file), "--research-coverage-policy", str(policy_file),
                "--expected-plan-sha256", preflight.canonical_sha(plan),
                "--output", str(output)]
        assert preflight.main(args) == 0
        assert (output.stat().st_mode & 0o777) == 0o600
        assert json.loads(output.read_text())["qualification_allowed"] is False
        try:
            preflight.main(args)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing report overwritten")
        args[-1] = str(ROOT / "never-publish-private-field-plan-report.json")
        try:
            preflight.main(args)
        except ValueError as exc:
            assert "PRIVATE_CREATE_ONLY_OUTPUT_REQUIRED" in str(exc)
        else:
            raise AssertionError("private field-plan output accepted inside repository")


def main() -> int:
    test_phase_a_field_plan_preflight()
    production_policy = json.loads(
        (ROOT / "commercial/real-human-qualification.policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert production_policy["shipping_approved"] is False
    assert production_policy["deployment_tag"] == "deployment-c20f3eb88e43"
    assert production_policy["minimums"]["speakers"] == 40
    assert production_policy["minimums"]["sessions_per_speaker"] == 2
    assert production_policy["minimums"]["expected_wakes_per_keyword"] == 600
    assert production_policy["minimums"]["negative_audio_hours"] == 24.0
    assert production_policy["gates"]["max_frr"] == 0.05
    assert production_policy["gates"]["max_frr_per_keyword"] == 0.05
    assert production_policy["gates"]["max_frr_upper_bound_per_keyword"] == 0.08
    assert production_policy["gates"]["max_far_per_hour"] == 0.10
    assert production_policy["event_matching"] == {
        "pre_tolerance_ms": 0.0,
        "post_tolerance_ms": 500.0,
    }
    assert {
        row["name"] for row in production_policy["critical_positive_slices"]
    } >= {"distance_3_5m", "rear", "playback", "double_talk", "robot_motion"}
    assert production_policy["privacy"]["raw_audio_public_upload_allowed"] is False
    assert production_policy["privacy"]["post_afe_audio_public_upload_allowed"] is False

    summary_schema = json.loads(
        (ROOT / "commercial/real-human-qualification-summary.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary_schema["properties"]["shipping_approved"]["const"] is False
    assert "critical_negative_slices" in summary_schema["required"]

    # The historical manual Release publisher was retired for research-only
    # execution. All corpus/privacy/AFE/scoring negatives below still run.
    assert not (ROOT / ".github/workflows/real-human-qualification.yml").exists()
    for tool in ("tools/validate_real_human_corpus.py",
                 "tools/score_real_human_qualification.py",
                 "tools/run_final_afe_corpus.py"):
        assert (ROOT / tool).is_file(), tool

    with tempfile.TemporaryDirectory(prefix="kws-real-human-contract-") as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        audio_root = tmp / "audio"
        audio_root.mkdir()
        for index, name in enumerate(("k1.wav", "k2.wav", "neg.wav"), 1):
            write_wav(audio_root / name, index)

        draft = {
            "schema_version": 1,
            "qualification_id": "fixture-q-0001",
            "deployment_tag": "deployment-fixture0000",
            "corpus_role": "fresh-held-out-qualification",
            "recordings": [],
        }
        rows = [
            (
                "k1",
                "k1.wav",
                "spk-a",
                "s1",
                [{"keyword_id": 1, "start_s": 0.10, "end_s": 0.30}],
                ["household"],
            ),
            (
                "k2",
                "k2.wav",
                "spk-a",
                "s1",
                [{"keyword_id": 2, "start_s": 0.10, "end_s": 0.30}],
                ["playback"],
            ),
            (
                "neg",
                "neg.wav",
                "ambient-a",
                "n1",
                [],
                ["household", "speech_confusion", "playback", "robot_motion"],
            ),
        ]
        for recording, filename, speaker, session, expected, tags in rows:
            draft["recordings"].append(
                {
                    "recording": recording,
                    "input_path": filename,
                    "speaker_id": speaker,
                    "session_id": session,
                    "source_id": f"src-{recording}",
                    "room_id": "room-a",
                    "device_id": "device-a",
                    "distance_m": 0.5,
                    "azimuth_deg": 0.0,
                    "snr_db": 20.0,
                    "tags": tags,
                    "expected": expected,
                    "consent_scope": "product-kws-qualification",
                    "retention_class": "restricted-raw-audio",
                }
            )
        draft_path = tmp / "draft.json"
        draft_path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
        manifest_path = tmp / "manifest.json"
        run(
            "python3",
            "tools/seal_real_human_corpus.py",
            "--draft",
            str(draft_path),
            "--audio-root",
            str(audio_root),
            "--output",
            str(manifest_path),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["recordings"]) == 3
        for row in manifest["recordings"]:
            path = audio_root / row["input_path"]
            assert row["input_sha256"] == sha(path)
            assert row["input_bytes"] == path.stat().st_size
            assert row["capture"] == {
                "sample_rate_hz": 16000,
                "channels": 1,
                "sample_format": "pcm_s16le",
            }
            assert abs(row["duration_s"] - 1.0) < 1e-12

        policy = {
            "schema_version": 1,
            "deployment_tag": "deployment-fixture0000",
            "confidence_level": 0.95,
            "event_matching": {
                "pre_tolerance_ms": 0.0,
                "post_tolerance_ms": 500.0,
            },
            "minimums": {
                "speakers": 1,
                "sessions_per_speaker": 1,
                "expected_wakes_total": 2,
                "expected_wakes_per_keyword": 1,
                "negative_audio_hours": 1.0 / 3600.0,
            },
            "gates": {
                "max_frr": 0.01,
                "max_frr_upper_bound": 0.90,
                "max_frr_per_keyword": 0.01,
                "max_frr_upper_bound_per_keyword": 0.90,
                "max_far_per_hour": 0.01,
                "max_far_upper_bound_per_hour": 20000.0,
                "max_p95_post_end_latency_ms": 500.0,
            },
            "critical_positive_slices": [],
            "critical_negative_exposure_hours": {
                "household": 1.0 / 3600.0
            },
            "identity_policy": {"forbidden_fields": ["name", "email", "phone"]},
        }
        policy_path = tmp / "policy.json"
        policy_path.write_text(
            json.dumps(policy, indent=2) + "\n", encoding="utf-8"
        )
        intake = tmp / "intake.json"
        run(
            "python3",
            "tools/validate_real_human_corpus.py",
            "--manifest",
            str(manifest_path),
            "--audio-root",
            str(audio_root),
            "--policy",
            str(policy_path),
            "--public-summary",
            str(intake),
        )
        intake_json = json.loads(intake.read_text(encoding="utf-8"))
        assert intake_json["raw_audio_hashes_verified"] is True
        assert intake_json["expected_by_keyword"] == {"1": 1, "2": 1}

        zero_length = json.loads(manifest_path.read_text(encoding="utf-8"))
        zero_length["recordings"][0]["expected"][0]["end_s"] = 0.10
        zero_length_path = tmp / "zero-length-manifest.json"
        zero_length_path.write_text(json.dumps(zero_length), encoding="utf-8")
        run(
            "python3", "tools/validate_real_human_corpus.py",
            "--manifest", str(zero_length_path),
            "--audio-root", str(audio_root),
            "--policy", str(policy_path),
            "--public-summary", str(tmp / "zero-length-intake.json"),
            expect=2,
        )

        bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bad_manifest["recordings"][0]["email"] = "forbidden@example.invalid"
        bad_path = tmp / "bad-manifest.json"
        bad_path.write_text(json.dumps(bad_manifest) + "\n", encoding="utf-8")
        run(
            "python3",
            "tools/validate_real_human_corpus.py",
            "--manifest",
            str(bad_path),
            "--audio-root",
            str(audio_root),
            "--policy",
            str(policy_path),
            "--public-summary",
            str(tmp / "bad.json"),
            expect=2,
        )

        fake_afe = tmp / "fake_afe.py"
        fake_afe.write_text(
            "#!/usr/bin/env python3\n"
            "import json, shutil, sys\n"
            "_, src, dst, result, config = sys.argv\n"
            "shutil.copyfile(src, dst)\n"
            "open(result, 'w', encoding='utf-8').write(json.dumps({'latency_samples': 0}) + '\\n')\n",
            encoding="utf-8",
        )
        fake_afe.chmod(0o755)
        config = tmp / "afe.cfg"
        config.write_text("fixture=true\n", encoding="utf-8")
        adapter = {
            "schema_version": 1,
            "executable_path": str(fake_afe),
            "command_argv": [
                "{executable}",
                "{input}",
                "{output}",
                "{result}",
                str(config),
            ],
            "config_files": [str(config)],
            "pipeline_source_sha": "1" * 40,
            "sku": "fixture",
            "microphone_revision": "mic-a",
            "enclosure_revision": "enc-a",
            "audio_route": "fixture-route",
            "toolchain": "fixture-python",
        }
        adapter_path = tmp / "adapter.json"
        adapter_path.write_text(
            json.dumps(adapter, indent=2) + "\n", encoding="utf-8"
        )
        identity_path = tmp / "afe-identity.json"
        run(
            "python3",
            "tools/final_afe_identity.py",
            "--adapter",
            str(adapter_path),
            "--output",
            str(identity_path),
        )
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        assert identity["research_measurement_only"] is True
        assert "shipping_authority" not in identity
        assert "command_argv" not in identity
        assert len(identity["command_argv_sha256"]) == 64
        assert len(identity["identity_sha256"]) == 64

        afe_root = tmp / "afe-out"
        run(
            "python3",
            "tools/run_final_afe_corpus.py",
            "--manifest",
            str(manifest_path),
            "--audio-root",
            str(audio_root),
            "--adapter",
            str(adapter_path),
            "--expected-identity",
            str(identity_path),
            "--output-dir",
            str(afe_root),
        )
        afe_summary = json.loads(
            (afe_root / "afe-corpus-summary.json").read_text(encoding="utf-8")
        )
        assert afe_summary["afe"]["research_measurement_only"] is True
        assert "shipping_authority" not in afe_summary["afe"]
        assert afe_summary["afe"]["identity_sha256"] == identity["identity_sha256"]
        assert afe_summary["latency_samples"] == {"min": 0, "max": 0}

        detections = tmp / "detections.jsonl"
        detections.write_text(
            json.dumps(
                {
                    "recording": "k1",
                    "keyword_id": 1,
                    "time_s": 0.30,
                    "confidence": 0.9,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "recording": "k2",
                    "keyword_id": 2,
                    "time_s": 0.30,
                    "confidence": 0.9,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        score_summary = tmp / "score.json"
        false_accepts = tmp / "fa.jsonl"
        false_rejects = tmp / "fr.jsonl"
        references = afe_root / "references.post-afe.jsonl"
        run(
            "python3",
            "eval/score_events.py",
            "--references",
            str(references),
            "--detections",
            str(detections),
            "--pre-tolerance-ms",
            "0",
            "--post-tolerance-ms",
            "500",
            "--summary",
            str(score_summary),
            "--false-positives",
            str(false_accepts),
            "--false-rejects",
            str(false_rejects),
        )
        final = tmp / "qualification.json"
        gate_args = (
            "--manifest",
            str(manifest_path),
            "--policy",
            str(policy_path),
            "--intake-summary",
            str(intake),
            "--afe-summary",
            str(afe_root / "afe-corpus-summary.json"),
            "--references",
            str(references),
            "--score-summary",
            str(score_summary),
            "--false-accepts",
            str(false_accepts),
            "--false-rejects",
            str(false_rejects),
            "--detections",
            str(detections),
            "--output",
            str(final),
        )
        run("python3", "tools/score_real_human_qualification.py", *gate_args)
        result = json.loads(final.read_text(encoding="utf-8"))
        assert result["qualified"] is True
        assert result["shipping_approved"] is False
        assert result["false_rejects"] == 0
        assert result["false_accepts_negative"] == 0
        assert abs(result["negative_audio_hours"] - 1.0 / 3600.0) < 1e-12
        assert result["per_keyword"]["1"]["frr"] == 0.0
        assert result["per_keyword"]["2"]["frr"] == 0.0
        assert abs(
            result["critical_negative_slices"]["household"]["audio_hours"]
            - 1.0 / 3600.0
        ) < 1e-12
        assert result["next_gate"] == "physical-target-board-performance-and-soak"

        afe_summary_path = afe_root / "afe-corpus-summary.json"
        valid_afe_summary = afe_summary_path.read_text(encoding="utf-8")
        drifted_afe = json.loads(valid_afe_summary)
        drifted_afe["afe"]["shipping_authority"] = True
        afe_summary_path.write_text(json.dumps(drifted_afe), encoding="utf-8")
        invalid = run(
            "python3", "tools/score_real_human_qualification.py", *gate_args, expect=2
        )
        assert "not research-scoped command backend" in invalid.stdout
        afe_summary_path.write_text(valid_afe_summary, encoding="utf-8")
        drifted_afe = json.loads(valid_afe_summary)
        drifted_afe["references_sha256"] = "0" * 64
        afe_summary_path.write_text(json.dumps(drifted_afe), encoding="utf-8")
        invalid = run(
            "python3", "tools/score_real_human_qualification.py", *gate_args, expect=2
        )
        assert "final AFE references hash differs" in invalid.stdout
        afe_summary_path.write_text(valid_afe_summary, encoding="utf-8")

        drifted_afe = json.loads(valid_afe_summary)
        drifted_afe["qualification_id"] = "different-qualification"
        afe_summary_path.write_text(json.dumps(drifted_afe), encoding="utf-8")
        invalid = run(
            "python3", "tools/score_real_human_qualification.py", *gate_args, expect=2
        )
        assert "qualification ID differs" in invalid.stdout
        afe_summary_path.write_text(valid_afe_summary, encoding="utf-8")

        valid_intake = intake.read_text(encoding="utf-8")
        drifted_intake = json.loads(valid_intake)
        drifted_intake["manifest_sha256"] = "0" * 64
        intake.write_text(json.dumps(drifted_intake), encoding="utf-8")
        invalid = run(
            "python3", "tools/score_real_human_qualification.py", *gate_args, expect=2
        )
        assert "corpus intake manifest hash differs" in invalid.stdout
        intake.write_text(valid_intake, encoding="utf-8")

        valid_detections = detections.read_text(encoding="utf-8")
        valid_summary = score_summary.read_text(encoding="utf-8")
        valid_false_accepts = false_accepts.read_text(encoding="utf-8")
        valid_false_rejects = false_rejects.read_text(encoding="utf-8")
        detections.write_text(
            valid_detections.replace('"time_s": 0.3', '"time_s": 0.05', 1),
            encoding="utf-8",
        )
        run(
            "python3", "eval/score_events.py",
            "--references", str(references),
            "--detections", str(detections),
            "--pre-tolerance-ms", "0",
            "--post-tolerance-ms", "500",
            "--summary", str(score_summary),
            "--false-positives", str(false_accepts),
            "--false-rejects", str(false_rejects),
        )
        run("python3", "tools/score_real_human_qualification.py", *gate_args, expect=1)
        early_result = json.loads(final.read_text(encoding="utf-8"))
        assert early_result["qualified"] is False
        assert "unexpected-detections-in-positive-recordings" in early_result["failures"]
        detections.write_text(valid_detections, encoding="utf-8")
        score_summary.write_text(valid_summary, encoding="utf-8")
        false_accepts.write_text(valid_false_accepts, encoding="utf-8")
        false_rejects.write_text(valid_false_rejects, encoding="utf-8")

        wrong_matching = json.loads(valid_summary)
        wrong_matching["event_match_pre_tolerance_ms"] = 150.0
        score_summary.write_text(json.dumps(wrong_matching), encoding="utf-8")
        invalid = run(
            "python3", "tools/score_real_human_qualification.py", *gate_args, expect=2
        )
        assert "event matching differs from policy" in invalid.stdout

        forged_score = json.loads(valid_summary)
        forged_score["p95_post_end_latency_ms"] = 1000.0
        score_summary.write_text(json.dumps(forged_score), encoding="utf-8")
        invalid = run(
            "python3", "tools/score_real_human_qualification.py", *gate_args, expect=2
        )
        assert "score summary disagrees with detections" in invalid.stdout
        score_summary.write_text(valid_summary, encoding="utf-8")

        false_accepts.write_text(
            json.dumps({"recording": "k1", "keyword_id": 2, "time_s": 0.5}) + "\n",
            encoding="utf-8",
        )
        invalid = run(
            "python3", "tools/score_real_human_qualification.py", *gate_args, expect=2
        )
        assert "false-accept rows disagree" in invalid.stdout

        # Once the identity has been frozen, changing any AFE config must fail
        # before the consumed holdout is evaluated with a different tuple.
        config.write_text("fixture=false\n", encoding="utf-8")
        run(
            "python3",
            "tools/run_final_afe_corpus.py",
            "--manifest",
            str(manifest_path),
            "--audio-root",
            str(audio_root),
            "--adapter",
            str(adapter_path),
            "--expected-identity",
            str(identity_path),
            "--output-dir",
            str(tmp / "drift-out"),
            expect=2,
        )

    print("test_real_human_qualification: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
