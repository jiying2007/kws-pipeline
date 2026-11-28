"""One-shot independent V7 FSMN-CTC training for the two candidate-relative manifests.

This is synthetic research training only. It reads TRAIN PCM and DEVELOPMENT
PCM after the final optimizer update; it never opens CALIBRATION or LOCKED-TEST
audio, never retries, and never promotes a product model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
import sys
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(ROOT / "automation"))
import english_kws_dual_track_v7_silver as admission

PLAN_PATH = ROOT / "configs/research/english-kws-dual-track-v7-training-plan.json"
REPO = "jiying2007/kws-data"
MAGIC = b"KWSV7C01"
FLOAT_COUNT = 54236
PARAMETERS = 43876
GAIN_LEVELS = (0.75, 1.0, 1.25)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "REGULAR_JSON_INPUT")
    return json.loads(path.read_bytes(), object_pairs_hook=unique_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def write_once(path, value):
    raw = canonical(value) + b"\n"
    with Path(path).open("xb") as handle:
        handle.write(raw)
        handle.flush()
    return {"bytes": len(raw), "sha256": sha(raw)}


def load_plan():
    plan = read_json(PLAN_PATH)
    need(plan.get("schema") == "english-kws-dual-track-v7-independent-training-plan-v1"
         and plan.get("enabled") is True
         and plan.get("status") == "READY_FOR_MULTISOURCE_G3"
         and plan.get("source_repository") == REPO
         and plan.get("next_main_before") == "5902f1f8d4fdc8096a864eb5d814c253d6f58ff2",
         "EXACT_PREDECLARED_MULTISOURCE_G3_PLAN")
    bundle = plan.get("source_bundle")
    expected_pins = [
        ("fresh24", "source", 14422271973, "26396073f5be2f91e35575bedb586737405162388e32bfe85633ed3d796a9583", 4505217),
        ("fresh24", "screen", 14429385634, "5a9b226b8fe475331d9f03d3ca4f91d5bb94c5270113f6d0cf3e8b4c36fe8316", 19501),
        ("fresh18", "source", 14429578824, "34cf7879f974f62c9f6046a72a964002f30bf59d2186a423ac382a76ff69d0e9", 3359173),
        ("fresh18", "screen", 14429884666, "4a685afa1d8e8c35767900290a7fd590f2153fb69f3b2e5ae68d5dada58b11e1", 15068),
        ("piper24", "source", 14434542792, "d1f75716fefd954d59c3289c37e7e6670965ee98917436676896312e4b781aa1", 3362081),
        ("piper24", "screen", 14436377961, "c6b8573bae44ffa110b776482da74919d16e6b1437c6c3e1f462fd1b7a28a519", 19503),
    ]
    need(isinstance(bundle, dict)
         and bundle.get("schema") == "english-canonical-multisource-reviewed-pcm-v1"
         and bundle.get("source_policy_git_blob_sha1") == "0a688325d0d36e83ef6623ac678511df24a92374"
         and bundle.get("source_plan_git_blobs") == {
             "fresh24": "dbe911fbcf8a697492ad29c7ca11858e707a72ff",
             "fresh18": "c2ee2f7bf2fbdeb8821920d6cb5e6890d0df1f76",
             "piper24": "d477ecec2a07910fc1a91063752aff1acc9006ef",
         }
         and isinstance(bundle.get("artifacts"), list)
         and [(x.get("key"), x.get("kind"), x.get("artifact_id"), x.get("sha256"), x.get("bytes"))
              for x in bundle["artifacts"]] == expected_pins
         and bundle.get("qualified_groups_by_role") == {
             "train": 8, "development": 4, "calibration": 3, "locked-test": 3}
         and bundle.get("minimum_groups_by_role") == {
             "train": 6, "development": 2, "calibration": 2, "locked-test": 2}
         and bundle.get("selected_machine_reviewed_rows") == 63
         and bundle.get("original_rows_authenticated") == 462
         and bundle.get("new_tts_calls") == bundle.get("new_asr_calls") == bundle.get("new_phone_calls") == 0
         and bundle.get("audio_regenerated") is False
         and bundle.get("human_gold") is False
         and bundle.get("independent_human_speaker_identity_claimed") is False
         and bundle.get("original_screen_training_snapshot_allowed") is False
         and bundle.get("fresh_candidate_snapshot_required") is True,
         "EXACT_SIX_ARTIFACT_MULTISOURCE_BUNDLE")
    need(plan.get("source_plan_sha256") == "COMPUTED_FROM_EXACT_SOURCE_BUNDLE_AT_RUNTIME",
         "SOURCE_BUNDLE_HASH_DERIVATION_MODE")
    plan["source_plan_sha256"] = sha(canonical(bundle))
    model = plan["model"]
    need(model == {
        "family": "causal-fsmn-ctc-v1", "parameters": PARAMETERS,
        "frontend": "PCM16 mono 16 kHz -> FFT512 Hann400 hop160 center=false -> 40 HTK log1p Mel",
        "normalization": "per-bin mean/std computed only from eligible TRAIN PCM",
        "hidden": 96, "blocks": 4, "left_memory": 4, "outputs": 4,
        "ctc_target_token_ids": [1, 2, 3], "dropout_train_only": 0.1
    }, "FROZEN_43K_CAUSAL_FSMN_CTC_ARCHITECTURE")
    optimizer = plan["optimizer"]
    need(optimizer["name"] == "AdamW" and optimizer["learning_rate"] == 0.00012
         and optimizer["weight_decay"] == 0.001 and optimizer["max_grad_norm"] == 5.0
         and optimizer["updates_per_candidate"] == 192
         and optimizer["training_forwards_per_candidate"] == 384
         and optimizer["gain_levels_train_only"] == list(GAIN_LEVELS)
         and optimizer["weak_row_weight"] == 0.5
         and optimizer["group_schedule"] ==
             "round-robin eight TRAIN source groups; cycle each candidate's admitted rows deterministically",
         "FIXED_SINGLE_RUN_TRAINING_BUDGET")
    need(plan["source_roles"] == {
        "train": 8, "development": 4, "calibration": 3, "locked-test": 3},
        "EXACT_8_4_3_3_MULTISOURCE_ROLE_SPLIT")
    need(plan["admission"] == {
        "minimum_train_groups": 8, "minimum_dev_groups": 4,
        "minimum_positive_rows_per_train_group": 1,
        "minimum_negative_rows_per_train_group": 2,
        "minimum_positive_rows_per_dev_group": 1,
        "minimum_negative_rows_per_dev_group": 2,
        "train_only_silver_weak": True, "unknown_is_background": False,
        "no_old_audio_or_legacy_labels": True,
        "candidate_mapping": {
            "VOSEN": "observed class 1 positive; classes 0 and 2 are negatives",
            "MERO": "observed class 2 positive; classes 0 and 1 are negatives",
        },
    }, "CANDIDATE_RELATIVE_CROSS_TRIGGER_ADMISSION")
    need(plan["training_input"] == {
        "schema": "english-kws-dual-track-v7-training-snapshot-v1",
        "allowed_roles": ["train", "development"],
        "role_counts": {"train": 8, "development": 4},
        "calibration_audio_available": False,
        "calibration_labels_available": False,
        "locked_test_audio_available": False,
        "locked_test_labels_available": False,
        "full_source_packet_downloaded_by_training_job": False,
        "full_admission_recomputed_by_training_job": False,
        "admission_stage_builds_candidate_specific_snapshots": True,
    }, "TRAIN_DEV_SNAPSHOT_ONLY_INPUT_BOUNDARY")
    need(plan["independence"]["model_parameters_shared"] is False
         and plan["independence"]["optimizer_state_shared"] is False
         and plan["independence"]["checkpoints_shared"] is False
         and plan["independence"]["training_runs_per_candidate"] == 1
         and plan["execution"]["retries"] == 0
         and plan["execution"]["run_attempt"] == 1
         and plan["promotion_allowed"] is False,
         "INDEPENDENT_NO_RETRY_NO_PROMOTION")
    need(set(plan["candidates"]) == {"VOSEN", "MERO"}
         and plan["candidates"]["VOSEN"]["seed"] != plan["candidates"]["MERO"]["seed"]
         and plan["candidates"]["VOSEN"]["token_ids"] == [1, 2, 3]
         and plan["candidates"]["MERO"]["token_ids"] == [1, 2, 3],
         "TWO_SEPARATE_CANDIDATE_IDENTITIES_AND_SEEDS")
    pack = plan["pack"]
    need(pack == {"magic": "KWSV7C01", "version": 1, "float_count": FLOAT_COUNT,
                  "header_bytes": 16, "total_bytes": 216960,
                  "little_endian_float32": True, "candidate_id_in_header": False},
         "EXACT_KWSV7C01_PACK_GEOMETRY")
    return plan

def guard(plan, env=None, event=None):
    need(plan.get("enabled") is True
         and plan.get("status") == "READY_FOR_MULTISOURCE_G3"
         and re.fullmatch(r"[0-9a-f]{64}", plan.get("source_plan_sha256", "")) is not None,
         "G3_MULTISOURCE_BUNDLE_NOT_READY")
    env = os.environ if env is None else env
    required = {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    need(all(env.get(key) == value for key, value in required.items()),
         "PUBLIC_MAIN_FIRST_ATTEMPT_TRAINING_ONLY")
    head = env.get("GITHUB_SHA", "")
    run_id = env.get("GITHUB_RUN_ID", "")
    need(re.fullmatch(r"[0-9a-f]{40}", head) is not None
         and re.fullmatch(r"[1-9][0-9]*", run_id) is not None,
         "VALID_GITHUB_MAIN_RUN_ID")
    if event is None:
        event = read_json(env["GITHUB_EVENT_PATH"])
    need(event.get("before") == plan["next_main_before"]
         and event.get("after") == head
         and event.get("ref") == "refs/heads/main"
         and event.get("forced") is False and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO
         and event.get("repository", {}).get("private") is False,
         "EXACT_FIRST_MAIN_SUCCESSOR_AND_PUBLIC_REPOSITORY")
    return {
        "repository": REPO, "run_id": int(run_id), "commit": head,
        "parent": plan["next_main_before"], "attempt": 1,
        "source_bundle_sha256": plan["source_plan_sha256"],
    }

def read_pcm(row, snapshot):
    raw = admission.read_member(snapshot, row["pcm_path"], row["pcm_sha256"], limit=1024 * 1024)
    wav = admission.read_member(snapshot, row["wav_path"], row["wav_sha256"], limit=4 * 1024 * 1024)
    with wave.open(io.BytesIO(wav), "rb") as stream:
        need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
              stream.getcomptype(), stream.getnframes())
             == (1, 2, 16000, "NONE", row["pcm_samples"]), "ORIGINAL_PCM16_WAV_GEOMETRY")
        decoded = stream.readframes(stream.getnframes())
    need(decoded == raw and sha(decoded) == row["pcm_sha256"], "ORIGINAL_PCM_WAV_IDENTITY")
    need(row["pcm_samples"] >= 6400 and row["pcm_samples"] <= 28672,
         "35_FRAME_CTC_MODEL_INPUT_GEOMETRY")
    return raw


def verify_training_snapshot(snapshot, candidate, plan):
    """Verify only candidate-specific TRAIN+DEV bytes; no full G2 packet is available here."""
    snapshot = Path(snapshot)
    need(snapshot.is_dir() and not snapshot.is_symlink(), "REGULAR_TRAIN_DEV_SNAPSHOT_DIRECTORY")
    snapshot_meta = read_json(snapshot / "SNAPSHOT.json")
    snapshot_raw = (snapshot / "SNAPSHOT.json").read_bytes()
    manifest = read_json(snapshot / "MANIFEST.json")
    manifest_raw = (snapshot / "MANIFEST.json").read_bytes()
    files = read_json(snapshot / "FILES.json")
    files_raw = (snapshot / "FILES.json").read_bytes()
    need(sha(manifest_raw) == snapshot_meta.get("manifest_sha256"),
         "TRAIN_DEV_MANIFEST_SHA256")
    need(sha(files_raw) == snapshot_meta.get("files_manifest_sha256"),
         "TRAIN_DEV_FILES_MANIFEST_SHA256")
    need(snapshot_meta.get("schema") == "english-kws-dual-track-v7-training-snapshot-v1"
         and snapshot_meta.get("candidate") == candidate
         and manifest.get("schema") == "english-kws-dual-track-v7-training-snapshot-manifest-v1"
         and manifest.get("candidate") == candidate
         and manifest.get("training_snapshot_allowed") is True,
         "EXACT_CANDIDATE_TRAIN_DEV_SNAPSHOT")
    need(files.get("schema") == "english-kws-dual-track-v7-training-snapshot-files-v1"
         and files.get("candidate") == candidate and isinstance(files.get("files"), list),
         "TRAIN_DEV_FILES_SCHEMA")
    rows = manifest.get("rows")
    groups = manifest.get("source_groups")
    need(isinstance(rows, list) and isinstance(groups, list)
         and all(row.get("role") in ("train", "development") for row in rows),
         "NO_CALIBRATION_OR_LOCKED_ROWS_IN_TRAINING_SNAPSHOT")
    group_roles = {group.get("source_group_id"): group.get("role") for group in groups}
    role_counts = Counter(group_roles.values())
    need(role_counts == Counter({"train": 8, "development": 4})
         and len(group_roles) == 12
         and len(set(group.get("source_speaker_id") for group in groups)) == 12
         and all(role in ("train", "development") for role in group_roles.values()),
         "TRAIN_DEV_ONLY_GROUP_PARTITION")
    row_ids = set()
    expected_files = {"MANIFEST.json"}
    train_rows = []
    dev_rows = []
    for row in rows:
        row_id = admission.safe_relative(row.get("id"))
        need("/" not in row_id and "\\" not in row_id and row_id not in row_ids,
             "UNIQUE_SAFE_TRAIN_DEV_ROW_ID")
        row_ids.add(row_id)
        role = row["role"]
        need(group_roles.get(row.get("source_group_id")) == role
             and row.get("optimizer_allowed") is False
             and row.get("locked_test_sealed") is False
             and row.get("tier") in ("silver", "weak")
             and row.get("research_class") in (0, 1),
             "TRAIN_DEV_ROW_PROVENANCE_AND_LABEL")
        pcm_name = admission.safe_relative(row.get("pcm_path"))
        wav_name = admission.safe_relative(row.get("wav_path"))
        need(pcm_name not in expected_files and wav_name not in expected_files,
             "UNIQUE_TRAIN_DEV_AUDIO_PATHS")
        expected_files.update((pcm_name, wav_name))
        if role == "train":
            need(row.get("training_eligible") is True
                 and row.get("training_class") == row.get("research_class")
                 and row.get("training_class") in (0, 1)
                 and isinstance(row.get("training_weight"), (int, float))
                 and row["training_weight"] > 0.0,
                 "TRAIN_ROW_OPTIMIZER_ELIGIBILITY")
            train_rows.append(row)
        else:
            need(row.get("training_eligible") is False
                 and row.get("training_class") is None
                 and row.get("training_weight") == 0.0,
                 "DEVELOPMENT_ROW_NEVER_OPTIMIZER_ELIGIBLE")
            dev_rows.append(row)
    eligible_dev_rows = [row for row in dev_rows
                         if row["tier"] in ("silver", "weak") and row["research_class"] in (0, 1)]
    for group_id in (group_id for group_id, role in group_roles.items() if role == "development"):
        group_rows = [row for row in eligible_dev_rows if row["source_group_id"] == group_id]
        need(sum(row["research_class"] == 1 for row in group_rows) >= 1
             and sum(row["research_class"] == 0 for row in group_rows) >= 2,
             "BALANCED_DEVELOPMENT_GROUP_ROWS")
    # Reject extra calibration/locked paths before opening any listed file bytes.
    seen = set()
    for item in files["files"]:
        need(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"},
             "TRAIN_DEV_FILE_ENTRY")
        name = admission.safe_relative(item["path"])
        need(name not in seen and type(item["bytes"]) is int and item["bytes"] >= 0
             and re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])) is not None,
             "UNIQUE_SIZED_TRAIN_DEV_FILE")
        seen.add(name)
    need(seen == expected_files, "NO_EXTRA_OR_MISSING_TRAIN_DEV_FILES")
    total_bytes = 0
    for item in files["files"]:
        raw = admission.read_member(snapshot, item["path"], item["sha256"], limit=4 * 1024 * 1024)
        need(len(raw) == item["bytes"], "TRAIN_DEV_FILE_BYTE_COUNT")
        total_bytes += len(raw)
    need(snapshot_meta.get("role_counts") == {"train": 8, "development": 4}
         and snapshot_meta.get("training_rows") == len(train_rows)
         and snapshot_meta.get("development_rows") == len(eligible_dev_rows)
         and snapshot_meta.get("file_count") == len(files["files"])
         and snapshot_meta.get("verified_bytes") == total_bytes
         and re.fullmatch(r"[0-9a-f]{64}", str(snapshot_meta.get("source_candidate_manifest_sha256"))) is not None
         and re.fullmatch(r"[0-9a-f]{64}", str(snapshot_meta.get("source_packet_files_manifest_sha256"))) is not None,
         "TRAIN_DEV_SNAPSHOT_COUNTS_AND_BYTES")
    for key in ("calibration_rows_included", "locked_test_rows_included",
                "calibration_labels_included", "locked_test_labels_included",
                "calibration_audio_included", "locked_test_audio_included",
                "locked_test_access_in_training_job"):
        need(snapshot_meta.get(key) is False, "SEALED_ROLE_EXCLUDED:" + key)
    source_run = manifest.get("source_run")
    need(isinstance(source_run, dict) and source_run == snapshot_meta.get("source_run")
         and source_run.get("repository") == REPO
         and str(source_run.get("run_id", "")).isdigit()
         and re.fullmatch(r"[0-9a-f]{40}", str(source_run.get("commit", ""))) is not None
         and source_run.get("parent") == plan["next_main_before"]
         and source_run.get("attempt") == 1
         and source_run.get("source_bundle_sha256") == plan["source_plan_sha256"]
         and manifest.get("source_plan_sha256") == plan["source_plan_sha256"]
         and snapshot_meta.get("source_plan_sha256") == plan["source_plan_sha256"],
         "TRAIN_DEV_SNAPSHOT_SOURCE_BUNDLE_IDENTITY")
    candidate_plan = plan["candidates"][candidate]
    need(manifest.get("candidate_id") == candidate_plan["candidate_id"]
         and snapshot_meta.get("candidate_id") == candidate_plan["candidate_id"]
         and manifest.get("phrase") == candidate_plan["phrase"]
         and manifest.get("ipa") == candidate_plan["ipa"]
         and manifest.get("ctc_tokens") == candidate_plan["token_labels"]
         and manifest.get("ctc_token_ids") == candidate_plan["token_ids"],
         "TRAIN_DEV_CANDIDATE_PRONUNCIATION_BINDING")
    return manifest, {
        **snapshot_meta,
        "snapshot_sha256": sha(snapshot_raw),
        "manifest_sha256": sha(manifest_raw),
        "files_manifest_sha256": sha(files_raw),
    }


def make_mel_bank(np):
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(8000.0), 42)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor(513 * hz_points / 16000.0).astype(int)
    bank = np.zeros((257, 40), dtype=np.float32)
    for band in range(40):
        left, center, right = bins[band:band + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for index in range(left, min(center, 257)):
            bank[index, band] = (index - left) / float(center - left)
        for index in range(center, min(right, 257)):
            bank[index, band] = (right - index) / float(right - center)
    return bank


def build_model(torch):
    nn = torch.nn
    functional = torch.nn.functional

    class MemoryBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(96, 96)
            self.memory = nn.Conv1d(96, 96, kernel_size=5, groups=96)
            self.dropout = nn.Dropout(p=0.1)

        def forward(self, value):
            projected = self.linear(value)
            delayed = self.memory(functional.pad(projected.transpose(1, 2), (4, 0))).transpose(1, 2)
            return self.dropout(functional.relu(value + delayed))

    class CausalFsmnCtc(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(40, 96)
            self.layers = nn.ModuleList([MemoryBlock() for _ in range(4)])
            self.output = nn.Linear(96, 4)

        def forward(self, value):
            value = self.input(value)
            for layer in self.layers:
                value = layer(value)
            return self.output(value)

    model = CausalFsmnCtc()
    need(sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS,
         "EXACT_43876_PARAMETER_COUNT")
    return model


def extract_features(pcm, bank, torch, np):
    waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    need(waveform.size >= 6400, "35_FRAME_CTC_MODEL_INPUT_GEOMETRY")
    signal = torch.from_numpy(waveform)
    window = torch.hann_window(400, periodic=True)
    spectrum = torch.stft(signal, n_fft=512, hop_length=160, win_length=400,
                          window=window, center=False, return_complex=True)
    power = spectrum.abs().square().transpose(0, 1)
    mel = power @ torch.from_numpy(bank)
    features = torch.log1p(mel).cpu().numpy().astype(np.float32)
    need(features.ndim == 2 and features.shape[1] == 40 and features.shape[0] >= 35
         and np.isfinite(features).all(), "FINITE_40_MEL_FEATURES")
    return features


def gain_features(features, gain, np):
    power = np.maximum(np.expm1(features.astype(np.float64)), 0.0)
    return np.log1p(power * (gain * gain)).astype(np.float32)


def positive_sequence_loss(logits, torch, functional):
    log_probs = functional.log_softmax(logits, dim=-1).transpose(0, 1)
    frames = logits.shape[1]
    target = torch.tensor([1, 2, 3], dtype=torch.long)
    positive = functional.ctc_loss(
        log_probs, target, torch.tensor([frames], dtype=torch.long),
        torch.tensor([3], dtype=torch.long), blank=0, reduction="none",
        zero_infinity=False
    )[0] / float(frames)
    blank = -log_probs[:, :, 0].mean()
    return positive, blank, blank - positive


def greedy_sequence(logits, torch):
    indices = logits.argmax(dim=-1)[0].detach().cpu().tolist()
    collapsed = []
    previous = None
    for frame_index, token in enumerate(indices):
        if token != previous and token != 0:
            collapsed.append((int(token), frame_index))
        previous = token
    tokens = [item[0] for item in collapsed]
    events = []
    for index in range(max(0, len(collapsed) - 2)):
        if tokens[index:index + 3] == [1, 2, 3]:
            events.append(collapsed[index + 2][1])
    return tokens, events


def pack_model(state, bank, mean, scale, candidate_id, np):
    ordered = [
        state["input.weight"], state["input.bias"],
    ]
    for index in range(4):
        ordered.extend([
            state[f"layers.{index}.linear.weight"],
            state[f"layers.{index}.linear.bias"],
            state[f"layers.{index}.memory.weight"],
            state[f"layers.{index}.memory.bias"],
        ])
    ordered.extend([state["output.weight"], state["output.bias"]])
    arrays = []
    for tensor in ordered:
        value = tensor.detach().cpu().numpy().astype("<f4", copy=False)
        need(np.isfinite(value).all(), "FINITE_MODEL_TENSOR")
        arrays.append(value.reshape(-1))
    arrays.extend([
        np.asarray(bank, dtype="<f4").reshape(-1),
        np.asarray(mean, dtype="<f4").reshape(-1),
        np.asarray(scale, dtype="<f4").reshape(-1),
    ])
    floats = np.concatenate(arrays).astype("<f4", copy=False)
    need(floats.size == FLOAT_COUNT and np.isfinite(floats).all(),
         "EXACT_54236_FLOAT_PACK")
    need(candidate_id in (1, 2), "CANDIDATE_ID")
    return MAGIC + struct.pack("<II", 1, FLOAT_COUNT) + floats.tobytes()


def run(snapshot, candidate, output, *, env=None, event=None):
    plan = load_plan()
    need(candidate in ("VOSEN", "MERO"), "EXACT_CANDIDATE")
    identity = guard(plan, env=env, event=event)
    snapshot = Path(snapshot)
    output = Path(output)
    need(not output.exists() and output.parent.is_dir(), "NEW_ONE_SHOT_MODEL_OUTPUT")
    manifest, snapshot_info = verify_training_snapshot(snapshot, candidate, plan)
    manifest_sha = snapshot_info["source_candidate_manifest_sha256"]
    snapshot_sha = snapshot_info["snapshot_sha256"]
    candidate_plan = plan["candidates"][candidate]
    output.mkdir(mode=0o700)
    started = {
        "schema": "english-kws-dual-track-v7-training-start-v1",
        "run": identity, "candidate": candidate,
        "candidate_id": candidate_plan["candidate_id"],
        "candidate_manifest_sha256": manifest_sha,
        "training_snapshot_sha256": snapshot_sha,
        "training_plan_sha256": sha(canonical(plan)),
        "seed": candidate_plan["seed"],
        "optimizer_updates_budget": plan["optimizer"]["updates_per_candidate"],
        "training_forwards_budget": plan["optimizer"]["training_forwards_per_candidate"],
        "development_forwards_max": plan["evaluation"]["development_forwards_max_per_candidate"],
        "calibration_pcm_opened": False, "locked_test_pcm_opened": False,
        "retry": False,
    }
    write_once(output / "STARTED.json", started)
    try:
        train_rows = [row for row in manifest["rows"]
                      if row["role"] == "train" and row["training_eligible"] is True]
        dev_rows = [row for row in manifest["rows"]
                    if row["role"] == "development" and row["tier"] in ("silver", "weak")
                    and row["research_class"] in (0, 1)]
        need(len({row["source_group_id"] for row in train_rows}) == 6
             and len({row["source_group_id"] for row in dev_rows}) == 2
             and all(row["role"] == "train" and row["training_class"] in (0, 1)
                     and row["optimizer_allowed"] is False for row in train_rows)
             and all(row["role"] == "development" and row["training_class"] is None
                     and row["optimizer_allowed"] is False for row in dev_rows),
             "TRAIN_AND_DEV_ROLE_ISOLATION")
        train_groups = sorted({row["source_group_id"] for row in train_rows})
        positives = {group: sorted([row for row in train_rows
                                    if row["source_group_id"] == group and row["training_class"] == 1],
                                   key=lambda row: row["id"]) for group in train_groups}
        negatives = {group: sorted([row for row in train_rows
                                    if row["source_group_id"] == group and row["training_class"] == 0],
                                   key=lambda row: row["id"]) for group in train_groups}
        need(all(len(positives[group]) >= 2 and len(negatives[group]) >= 4 for group in train_groups),
             "BALANCED_TRAIN_GROUPS")
        import numpy as np
        import torch
        import torch.nn.functional as functional
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(candidate_plan["seed"])
        np.random.seed(candidate_plan["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        bank = make_mel_bank(np)
        train_features = {}
        for row in train_rows:
            pcm = read_pcm(row, snapshot)
            train_features[row["id"]] = extract_features(pcm, bank, torch, np)
        concatenated = np.concatenate([train_features[row["id"]] for row in train_rows], axis=0)
        mean = concatenated.mean(axis=0).astype(np.float32)
        scale = np.maximum(concatenated.std(axis=0), 1e-5).astype(np.float32)
        need(np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(),
             "TRAIN_ONLY_NORMALIZATION")
        model = build_model(torch)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=plan["optimizer"]["learning_rate"],
            weight_decay=plan["optimizer"]["weight_decay"]
        )
        trace = []
        began = time.perf_counter()
        updates = plan["optimizer"]["updates_per_candidate"]
        for step in range(updates):
            group = train_groups[step % len(train_groups)]
            position = step // len(train_groups)
            positive_row = positives[group][position % len(positives[group])]
            negative_row = negatives[group][position % len(negatives[group])]
            positive_gain = GAIN_LEVELS[(step + int(positive_row["pcm_sha256"][:2], 16)) % len(GAIN_LEVELS)]
            negative_gain = GAIN_LEVELS[(step + int(negative_row["pcm_sha256"][:2], 16) + 1) % len(GAIN_LEVELS)]
            pos_values = gain_features(train_features[positive_row["id"]], positive_gain, np)
            neg_values = gain_features(train_features[negative_row["id"]], negative_gain, np)
            pos_input = torch.from_numpy(((pos_values - mean) / scale).astype(np.float32)).unsqueeze(0)
            neg_input = torch.from_numpy(((neg_values - mean) / scale).astype(np.float32)).unsqueeze(0)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pos_logits = model(pos_input)
            neg_logits = model(neg_input)
            pos_loss, pos_blank, pos_margin = positive_sequence_loss(pos_logits, torch, functional)
            neg_positive, neg_blank, neg_margin = positive_sequence_loss(neg_logits, torch, functional)
            pos_total = (pos_loss + functional.softplus(0.5 - 4.0 * pos_margin)
                         ) * float(positive_row["training_weight"])
            neg_total = (0.35 * neg_blank + functional.softplus(0.5 + 4.0 * neg_margin)
                         ) * float(negative_row["training_weight"])
            total = (pos_total + neg_total) / 2.0
            need(bool(torch.isfinite(total)), "FINITE_TRAINING_LOSS")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 16 == 0 or step == updates - 1:
                trace.append({
                    "step": step + 1, "loss": float(total.detach()),
                    "positive_loss": float(pos_total.detach()),
                    "negative_loss": float(neg_total.detach()),
                    "positive_blank_loss": float(pos_blank.detach()),
                    "negative_positive_ctc_loss": float(neg_positive.detach()),
                    "negative_blank_loss": float(neg_blank.detach()),
                    "positive_group": group,
                })
        model.eval()
        model_dir = output / "models"
        model_dir.mkdir()
        pt_path = model_dir / "candidate.pt"
        torch.save(model.state_dict(), pt_path)
        reloaded = build_model(torch)
        reloaded.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True), strict=True)
        reloaded.eval()
        probe_row = positives[train_groups[0]][0]
        probe = torch.from_numpy(((train_features[probe_row["id"]] - mean) / scale).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            need(torch.equal(model(probe), reloaded(probe)), "FROZEN_WEIGHT_RELOAD_EXACT")
        npz_path = model_dir / "candidate.npz"
        np.savez(npz_path, mel_filters=bank, train_mean=mean, train_scale=scale,
                 **{key: value.detach().cpu().numpy() for key, value in reloaded.state_dict().items()})
        onnx_path = model_dir / "candidate.onnx"
        torch.onnx.export(
            reloaded, probe, onnx_path.as_posix(),
            input_names=["fbank40"], output_names=["frame_ctc_logits"],
            opset_version=17, dynamo=False,
            dynamic_axes={"fbank40": {1: "frames"}, "frame_ctc_logits": {1: "frames"}}
        )
        need(onnx_path.is_file() and onnx_path.stat().st_size > 10000, "ONNX_EXPORT")
        candidate_id = candidate_plan["keyword_id"]
        pack = pack_model(reloaded.state_dict(), bank, mean, scale, candidate_id, np)
        pack_path = model_dir / "candidate.kwv7"
        with pack_path.open("xb") as handle:
            handle.write(pack)
        need(len(pack) == 216960 and pack[:8] == MAGIC
             and struct.unpack("<II", pack[8:16]) == (1, FLOAT_COUNT),
             "KWSV7C01_PACK_READBACK")
        # The DEVELOPMENT set is opened only after final weights are frozen.
        dev_results = []
        with torch.no_grad():
            for row in dev_rows:
                pcm = read_pcm(row, snapshot)
                features = extract_features(pcm, bank, torch, np)
                model_input = torch.from_numpy(((features - mean) / scale).astype(np.float32)).unsqueeze(0)
                logits = reloaded(model_input)
                tokens, event_frames = greedy_sequence(logits, torch)
                dev_results.append({
                    "id": row["id"], "source_group_id": row["source_group_id"],
                    "pcm_sha256": row["pcm_sha256"], "research_class": row["research_class"],
                    "tier": row["tier"], "tokens": tokens, "event_frames": event_frames,
                    "full_sequence_event": bool(event_frames),
                })
        dev_positive = [row for row in dev_results if row["research_class"] == 1]
        dev_negative = [row for row in dev_results if row["research_class"] == 0]
        need(len(dev_positive) >= 2 and len(dev_negative) >= 2, "MINIMUM_ADMITTED_DEV_ROWS")
        dev_summary = {
            "positive_rows": len(dev_positive),
            "positive_hits": sum(row["full_sequence_event"] for row in dev_positive),
            "negative_rows": len(dev_negative),
            "negative_false_or_extra_events": sum(len(row["event_frames"]) for row in dev_negative),
            "source_groups": sorted({row["source_group_id"] for row in dev_results}),
            "single_post_freeze_pass": True,
            "threshold_tuning": False,
            "model_selection": False,
        }
        model_files = [pt_path, npz_path, onnx_path, pack_path]
        model_hashes = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
                        for path in model_files}
        result = {
            "schema": "english-kws-dual-track-v7-training-result-v1",
            "status": "SYNTHETIC_RESEARCH_TRAINED_NOT_PRODUCT_QUALIFIED",
            "candidate": candidate, "candidate_id": candidate_plan["candidate_id"],
            "training_snapshot_sha256": snapshot_sha,
            "source_candidate_manifest_sha256": manifest_sha,
            "phrase": candidate_plan["phrase"], "ipa": candidate_plan["ipa"],
            "token_labels": candidate_plan["token_labels"], "token_ids": [1, 2, 3],
            "seed": candidate_plan["seed"], "model_parameters": PARAMETERS,
            "optimizer": plan["optimizer"]["name"],
            "actual_optimizer_updates": updates,
            "actual_training_forwards": updates * 2,
            "actual_development_forwards": len(dev_rows),
            "train_rows": len(train_rows),
            "train_source_groups": len(train_groups),
            "dev_rows": len(dev_rows),
            "dev": dev_summary,
            "development_rows": dev_results,
            "train_only_normalization": True,
            "train_only_gain_augmentation": list(GAIN_LEVELS),
            "calibration_pcm_opened": False,
            "locked_test_pcm_opened": False,
            "locked_test_used_for_training_or_mining": False,
            "real_human_far_frr": "NOT_RUN",
            "physical_ssc305": "NOT_RUN",
            "product_promotion_allowed": False,
            "model_files": model_hashes,
            "training_cpu_seconds": round(time.perf_counter() - began, 3),
            "loss_trace": trace,
        }
        training = {
            "schema": "english-kws-dual-track-v7-training-ledger-v1",
            "candidate": candidate, "candidate_manifest_sha256": manifest_sha,
            "training_snapshot_sha256": snapshot_sha,
            "training_plan_sha256": sha(canonical(plan)),
            "seed": candidate_plan["seed"], "actual_optimizer_updates": updates,
            "actual_training_forwards": updates * 2, "training_rows": len(train_rows),
            "training_source_groups": train_groups,
            "development_gradients": False, "calibration_gradients": False,
            "locked_test_gradients": False, "model_initialization_shared": False,
            "optimizer_state_shared": False, "retry": False,
        }
        write_once(output / "TRAINING.json", training)
        write_once(output / "RESULT.json", result)
        manifest_out = {
            "schema": "english-kws-dual-track-v7-model-manifest-v1",
            "candidate": candidate, "candidate_id": candidate_plan["candidate_id"],
            "phrase": candidate_plan["phrase"], "ipa": candidate_plan["ipa"],
            "token_labels": candidate_plan["token_labels"], "token_ids": [1, 2, 3],
            "model_parameters": PARAMETERS, "pack_magic": MAGIC.decode("ascii"),
            "pack_bytes": len(pack), "pack_sha256": sha(pack),
            "candidate_manifest_sha256": manifest_sha,
            "training_plan_sha256": sha(canonical(plan)),
            "source_run_id": plan["source_run_id"], "source_commit": plan["source_commit"],
            "training_run_id": identity["run_id"], "training_commit": identity["commit"],
            "product_promotion_allowed": False,
        }
        write_once(output / "MODEL_MANIFEST.json", manifest_out)
        print(json.dumps({"status": result["status"], "candidate": candidate,
                          "optimizer_updates": updates, "dev": dev_summary,
                          "pack_sha256": sha(pack), "product_promotion_allowed": False}, sort_keys=True))
        return result
    except BaseException as exc:
        write_once(output / "STOP.json", {
            "schema": "english-kws-dual-track-v7-training-stop-v1",
            "status": "STOP_NO_RETRY",
            "candidate": candidate, "exception_type": type(exc).__name__,
            "reason": str(exc)[:520], "retry_allowed": False,
            "calibration_pcm_opened": False, "locked_test_pcm_opened": False,
            "product_promotion_allowed": False,
        })
        raise


def source_check():
    plan = load_plan()
    return {
        "status": "BLOCKED_AWAITING_NEW_VOICE_CLONE_G2_SOURCE_NO_MODEL_EXECUTION",
        "training_enabled": plan["enabled"],
        "source_compatible": plan["source_compatibility"]["compatible"],
        "candidates": sorted(plan["candidates"]),
        "optimizer_updates_per_candidate": plan["optimizer"]["updates_per_candidate"],
        "training_forwards_per_candidate": plan["optimizer"]["training_forwards_per_candidate"],
        "model_parameters": plan["model"]["parameters"],
        "pack_bytes": plan["pack"]["total_bytes"],
        "training_input_schema": plan["training_input"]["schema"],
        "calibration_and_locked_test_available_to_training": False,
        "shared_parameters": plan["independence"]["model_parameters_shared"],
        "shared_optimizer": plan["independence"]["optimizer_state_shared"],
        "training_runs_per_candidate": plan["independence"]["training_runs_per_candidate"],
        "retries": plan["execution"]["retries"],
        "source_run_id": plan["source_run_id"],
        "source_commit": plan["source_commit"],
        "promotion_allowed": plan["promotion_allowed"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--candidate", choices=("VOSEN", "MERO"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    need(sum((args.source_check, args.execute)) == 1, "EXACT_ONE_STAGE")
    if args.source_check:
        print(json.dumps(source_check(), sort_keys=True))
    else:
        need(args.snapshot and args.candidate and args.output, "TRAIN_SNAPSHOT_CANDIDATE_OUTPUT_REQUIRED")
        run(args.snapshot, args.candidate, args.output)


if __name__ == "__main__":
    main()
