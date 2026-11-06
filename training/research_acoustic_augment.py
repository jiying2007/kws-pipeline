"""Traceable acoustic and onset variants of already observed full-text rows.

No TTS/ASR is called. Label inheritance is explicit; additive untranscribed
noise disables full-text CTC and all absent-keyword labels. Digital gain,
recorded RIR convolution and onset offsets never become real distance or
soft-spoken-human measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import wave

SCHEMA = "kws-observed-acoustic-variants-v1"
VOCAB_SHA = "104575dab3a9f7ed622d8fa29aa8a897e0590cca8aed9214b26ef4cd1c8b4e4b"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode() + b"\n"


def once(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(body)


def pinned(path, sha):
    path = Path(path)
    require(path.is_file() and digest(path.read_bytes()) == sha, "PIN_CHANGED:" + str(path))
    return path


def safe_path(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and ".." not in name.parts, "UNSAFE_RELATIVE_PATH")
    result = Path(root) / str(name)
    require(result.resolve().is_relative_to(Path(root).resolve()), "PATH_ESCAPE")
    return result


def audio(body):
    import numpy as np
    with wave.open(io.BytesIO(body), "rb") as r:
        require(r.getframerate() == 16000 and r.getsampwidth() == 2 and
                r.getcomptype() == "NONE", "PCM16_16K_REQUIRED")
        channels, frames = r.getnchannels(), r.getnframes()
        raw = r.readframes(frames)
        require(len(raw) == frames * channels * 2 and frames > 0, "TRUNCATED_AUDIO")
    return np.frombuffer(raw, dtype="<i2").reshape(frames, channels).astype(np.float64) / 32768.0


def wav_bytes(samples):
    import numpy as np
    require(samples.ndim == 1 and samples.size and np.isfinite(samples).all(), "INVALID_SAMPLES")
    require(float(np.max(np.abs(samples))) <= 0.98 + 1e-10, "CLIPPING_POLICY")
    pcm = np.rint(samples * 32767.0).astype("<i2").tobytes()
    stream = io.BytesIO()
    with wave.open(stream, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
    return stream.getvalue(), digest(pcm)


def active_mask(samples):
    """20-ms frame RMS >= 5% of maximum frame RMS; an energy mask, not a VAD."""
    import numpy as np
    require(samples.ndim == 1 and samples.size > 0 and np.isfinite(samples).all(), "SIGNAL")
    padding = (-len(samples)) % 320
    frames = np.pad(samples, (0, padding)).reshape(-1, 320)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    require(float(rms.max()) > 0, "ZERO_ENERGY_PARENT")
    mask = np.repeat(rms >= 0.05 * rms.max(), 320)[:len(samples)]
    require(mask.any(), "NO_ACTIVE_SAMPLES")
    return mask


def scale_peak(components):
    import numpy as np
    mixed = sum(components)
    peak = float(np.max(np.abs(mixed)))
    gain = min(1.0, 0.98 / peak) if peak else 1.0
    return [component * gain for component in components], gain


def noise_mix(clean, noise, snr_db, offset):
    import numpy as np
    require(noise.ndim == 1 and len(noise) > 0 and np.isfinite(noise).all(), "NOISE")
    require(type(offset) is int and 0 <= offset < len(noise), "NOISE_OFFSET")
    require(snr_db in (0, 5, 10, 20), "SNR_RECIPE")
    mask = active_mask(clean)
    selected = noise[(np.arange(len(clean), dtype=np.int64) + offset) % len(noise)]
    clean_rms = float(np.sqrt(np.mean(clean[mask] ** 2)))
    noise_rms = float(np.sqrt(np.mean(selected[mask] ** 2)))
    require(noise_rms > 0, "ZERO_ENERGY_NOISE")
    factor = clean_rms / (noise_rms * 10 ** (snr_db / 20.0))
    (speech, background), gain = scale_peak([clean, selected * factor])
    measured = 10 * float(np.log10(np.mean(speech[mask] ** 2) / np.mean(background[mask] ** 2)))
    require(abs(measured - snr_db) < 1e-8, "SNR_NUMERICAL_MISMATCH")
    return speech + background, {"requested_snr_db": snr_db, "measured_component_snr_db": measured,
        "snr_before_pcm_quantization": True, "noise_offset_samples": offset,
        "noise_gain": factor, "common_peak_gain": gain,
        "active_samples": int(mask.sum()),
        "energy_mask": "20ms-frame-rms-ge-0.05-of-max-frame-rms",
        "noise_tiled": len(clean) > len(noise), "target_audibility_human_verified": False}


def convolve(clean, rir):
    import numpy as np
    from scipy.signal import fftconvolve
    require(rir.ndim == 1 and len(rir) > 0 and np.isfinite(rir).all(), "RIR")
    norm = float(np.sqrt(np.sum(rir * rir)))
    require(norm > 0, "ZERO_RIR")
    impulse = rir / norm
    rendered = fftconvolve(clean, impulse, mode="full")
    require(len(rendered) == len(clean) + len(rir) - 1, "RIR_TAIL_TRUNCATED")
    (rendered,), gain = scale_peak([rendered])
    onset = int(np.flatnonzero(np.abs(impulse) >= np.max(np.abs(impulse)) * 1e-3)[0])
    return rendered, {"rir_normalization": "unit-l2-energy", "rir_original_l2": norm,
        "rir_samples": len(rir), "input_samples": len(clean), "output_samples": len(rendered),
        "common_peak_gain": gain, "rir_onset_samples": onset,
        "onset_definition": "first-sample-ge-0.001-of-rir-absolute-peak",
        "full_tail_preserved": True, "physical_distance_m": None, "rt60_measured_s": None}


def inherited_labels(parent, additive_noise):
    ctc = bool(parent["ctc_mask"]) and not additive_noise
    masks = list(parent["keyword_mask"])
    targets = list(parent["keyword_targets"])
    weights = list(parent["keyword_weight"])
    if additive_noise:
        for i, target in enumerate(targets):
            if target != 1:
                masks[i], targets[i], weights[i] = False, None, 0.0
    return {"ctc_mask": ctc, "ctc_weight": parent["ctc_weight"] if ctc else 0.0,
        "target_ids": parent["target_ids"] if ctc else None,
        "full_text": parent["full_text"] if ctc else None,
        "keyword_mask": masks, "keyword_targets": targets, "keyword_weight": weights,
        "label_source": "inherited-observed-machine-weak-parent",
        "label_inheritance_is_new_asr_verification": False,
        "ctc_reason": "untranscribed_additive_background" if additive_noise else "parent_fulltext_preserved",
        "human_gold": False, "independent_test": False}


def checkpoint_rows(spec):
    root = Path(spec["root"])
    manifest_path = pinned(root / "MANIFEST.json", spec["manifest_sha256"])
    manifest = json.loads(manifest_path.read_bytes())["files"]
    rows_path = safe_path(root, spec["rows"])
    require(spec["rows"] in manifest, "ROWS_NOT_ARCHIVED")
    pinned(rows_path, manifest[spec["rows"]]["sha256"])
    by_original = {}
    for name, record in manifest.items():
        original = record.get("source_path")
        if original:
            by_original.setdefault(original, []).append((name, record))
    result = []
    seen, family_splits = set(), {}
    for index, line in enumerate(rows_path.read_text().splitlines()):
        row = json.loads(line)
        key = tuple(row["global_source_key"])
        require(key not in seen, "DUPLICATE_PARENT")
        seen.add(key)
        require(row["split"] in ("train", "development") and
                family_splits.setdefault(row["source_family_id"], row["split"]) == row["split"],
                "PARENT_FAMILY_SPLIT")
        require(row["human_gold"] is False and row["vocabulary_sha256"] == VOCAB_SHA,
                "PARENT_LABEL_SCOPE")
        candidates = by_original.get(row["audio_path"], [])
        require(bool(candidates), "PARENT_AUDIO_NOT_ARCHIVED")
        matches = [(name, rec) for name, rec in candidates if rec["sha256"] == row["file_sha256"]]
        require(len(matches) == 1, "PARENT_AUDIO_IDENTITY")
        path = safe_path(root, matches[0][0])
        pinned(path, row["file_sha256"])
        require(path.stat().st_size == matches[0][1]["bytes"], "PARENT_AUDIO_BYTES")
        if row["ctc_mask"]:
            require(row["ctc_weight"] > 0 and row["target_ids"] and row["full_text"] and
                    not row["quarantined"] and row["generation_outcome"] == "completed",
                    "PARENT_CTC_ADMISSION")
        result.append((row, path, {"rows_manifest_sha256": spec["manifest_sha256"],
            "rows_sha256": manifest[spec["rows"]]["sha256"], "row_index": index,
            "original_row_sha256": digest(line.encode()), "audio_member": matches[0][0]}))
    return result


def acoustic_sources(spec):
    root = Path(spec["root"])
    manifest = json.loads(pinned(root / "MANIFEST.json", spec["manifest_sha256"]).read_bytes())
    files = manifest["files"]
    for name, info in files.items():
        path = safe_path(root, name)
        require(path.stat().st_size == info["bytes"], "ASSET_BYTES")
        pinned(path, info["sha256"])
    source = json.loads((root / "source-subset.json").read_bytes())
    require(source["schema"] == "kws-acoustic-source-subset-v1" and
            source["whole_archive_crc_verified"] is False and source["model_calls"] == 0, "ASSET_SCOPE")
    groups = {}
    for item in source["assets"]:
        if item["kind"] == "metadata":
            continue
        require(groups.setdefault(item["source_group"], item["split"]) == item["split"], "ACOUSTIC_GROUP_SPLIT")
        require(item["split"] in ("train", "development") and item["zip_crc_verified"] is True, "ASSET_KIND")
        require(files[item["relative_path"]]["sha256"] == item["sha256"], "ASSET_MANIFEST_BINDING")
    return root, source


def render(request_path, output):
    import numpy as np
    import scipy
    request_path, output = Path(request_path), Path(output)
    request = json.loads(request_path.read_bytes())
    require(request["schema"] == SCHEMA and request["model_calls"] == 0 and
            request["purpose"] == "technical-research-only", "REQUEST")
    require(request["onset_offsets_ms"] == [160, 480] and request["digital_gains"] == [0.1, 0.3] and
            request["noise_snr_db"] == [20, 10, 5, 0], "FROZEN_RECIPE")
    require(not output.exists(), "CREATE_ONLY_OUTPUT")
    parents = checkpoint_rows(request["parents"])
    source_root, source = acoustic_sources(request["acoustics"])
    eligible = [(row, path, proof) for row, path, proof in parents if row["ctc_mask"]]
    require(bool(eligible), "NO_ELIGIBLE_PARENTS")
    output.mkdir(parents=True)
    once(output / "request.json", request_path.read_bytes())
    once(output / "acoustic-source-subset.json", canonical(source))
    once(output / "execution-source.py", Path(__file__).read_bytes())
    outputs, pcm_owners = [], {}
    retained_assets = {}
    for asset in source["assets"]:
        if asset["kind"] == "metadata":
            continue
        body = (source_root / asset["relative_path"]).read_bytes()
        name = "assets/" + asset["sha256"] + ".wav"
        once(output / name, body)
        retained_assets[asset["sha256"]] = name
    for parent, path, proof in eligible:
        original = path.read_bytes()
        clean_array = audio(original)
        require(clean_array.shape[1] == 1, "MONO_PARENT_REQUIRED")
        clean = clean_array[:, 0]
        require(len(clean) == parent["samples"], "PARENT_SAMPLE_COUNT")
        parent_key = digest(canonical(parent["global_source_key"]))
        once(output / "parents" / (parent_key + ".wav"), original)
        once(output / "parents" / (parent_key + ".json"), canonical({"original_row": parent, "proof": proof}))
        variants = []
        for milliseconds in request["onset_offsets_ms"]:
            samples = milliseconds * 16
            variants.append(("onset-offset", {"leading_silence_samples": samples,
                "leading_silence_ms": milliseconds}, np.pad(clean, (samples, 0)), False))
        for gain in request["digital_gains"]:
            variants.append(("digital-level", {"digital_gain": gain,
                "human_soft_speech": False}, clean * gain, False))
        for asset in sorted(source["assets"], key=lambda x: x["member"]):
            if asset["kind"] != "measured-rir" or asset["split"] != parent["split"]:
                continue
            rir = audio((source_root / asset["relative_path"]).read_bytes())[:, 0]
            wet, metrics = convolve(clean, rir)
            variants.append(("measured-rir-convolution", {**metrics, "asset_sha256": asset["sha256"],
                "asset_group": asset["source_group"], "asset_split": asset["split"],
                "selected_channel": 0}, wet, False))
        noises = sorted([a for a in source["assets"] if a["kind"] == "recorded-noise" and
                         a["split"] == parent["split"]], key=lambda a: a["member"])
        for index, snr in enumerate(request["noise_snr_db"]):
            if not noises:
                break
            asset = noises[index % len(noises)]
            background = audio((source_root / asset["relative_path"]).read_bytes())[:, 0]
            offset = int(digest(canonical([parent["global_source_key"], asset["sha256"], snr]))[:16], 16) % len(background)
            mixed, metrics = noise_mix(clean, background, snr, offset)
            variants.append(("recorded-noise-mix", {**metrics, "asset_sha256": asset["sha256"],
                "asset_group": asset["source_group"], "asset_split": asset["split"],
                "selected_channel": 0, "background_transcript_verified": False}, mixed, True))
        for kind, recipe, samples, noisy in variants:
            identity = {"parent_key": parent["global_source_key"], "kind": kind, "recipe": recipe,
                        "producer_sha256": digest(Path(__file__).read_bytes())}
            child = "aug-" + digest(canonical(identity))[:24]
            body, pcm_sha = wav_bytes(samples)
            require(pcm_sha not in pcm_owners, "DUPLICATE_DERIVED_PCM")
            pcm_owners[pcm_sha] = child
            rel = "audio/" + child + ".wav"
            once(output / rel, body)
            labels = inherited_labels(parent, noisy)
            record = {**identity, **labels, "sample_id": child, "audio": rel,
                "file_sha256": digest(body), "pcm_sha256": pcm_sha, "samples": len(samples),
                "sample_rate_hz": 16000, "parent_file_sha256": parent["file_sha256"],
                "parent_proof": proof, "speaker_id": parent["speaker_id"],
                "source_family_id": parent["source_family_id"], "source_role": parent["source_role"],
                "split": parent["split"], "vocabulary_sha256": VOCAB_SHA,
                "new_independent_speaker": False, "new_tts_call": False, "new_asr_call": False,
                "actual_distance_m": None, "event_times": None, "keyword_loss_used": False,
                "purpose": "technical-research-only", "training_admitted": False}
            outputs.append(record)
    once(output / "rows.jsonl", b"".join(canonical(row) for row in outputs))
    summary = {"schema": SCHEMA, "parents_observed": len(parents), "parents_used": len(eligible),
        "parents_excluded_zero_ctc_mask": len(parents)-len(eligible), "derived_rows": len(outputs),
        "by_split": dict(Counter(row["split"] for row in outputs)),
        "by_kind": dict(Counter(row["kind"] for row in outputs)),
        "fulltext_ctc_inheritance_rows": sum(row["ctc_mask"] for row in outputs),
        "additive_noise_ctc_masked_rows": sum(not row["ctc_mask"] for row in outputs),
        "unique_parent_families": len({row["source_family_id"] for row, _, _ in eligible}),
        "model_calls": 0, "training_steps": 0, "new_independent_speakers": 0,
        "accepted_natural_negative_hours": 0, "actual_device_farfield_examples": 0,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "human_gold": False, "scope": "derived_research_views_not_new_independent_evidence"}
    once(output / "summary.json", canonical(summary))
    files = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}
             for p in sorted(output.rglob("*")) if p.is_file()}
    once(output / "MANIFEST.json", canonical({"schema": SCHEMA, "files": files}))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render(args.request, args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
