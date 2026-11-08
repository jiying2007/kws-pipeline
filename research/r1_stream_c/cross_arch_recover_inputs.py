#!/usr/bin/env python3
"""Recover fixed R1 PCM locally from owner-held, hash-pinned inputs."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import wave
import zipfile

CAPSULE_SHA256 = "84048654b73de0d6da955472eacaff608b7a54534d0f48c035b56c9b46b9a6bc"
MODEL_CARD_SHA256 = "2079fce70bcd27d2991ce558807b3ef9707309051e6915946fc01821f8350936"
CHECKPOINT_SHA256 = "99ac2239807ce5a93c8723a01df2e53c446f868245caccb2e7809deb6e943571"

def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read_wav(raw):
    with wave.open(io.BytesIO(raw), "rb") as w:
        p = w.getparams()
        require((p.nchannels, p.sampwidth, p.framerate) == (1, 2, 16000),
                "source WAV must be mono PCM16 16kHz")
        pcm = w.readframes(p.nframes)
        samples = w.getnframes()
    require(len(pcm) == samples * 2, "WAV PCM length mismatch")
    return pcm, samples

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capsule", type=Path, required=True)
    ap.add_argument("--model-package", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        require(args.capsule.is_file() and not args.capsule.is_symlink(), "capsule missing")
        require(sha_file(args.capsule) == CAPSULE_SHA256, "capsule SHA256 mismatch")
        card_path = args.model_package / "MODEL.json"
        bind_path = args.model_package / "evidence/training-feature-bindings.json"
        require(card_path.is_file() and bind_path.is_file(), "R1 model package incomplete")
        require(sha_file(card_path) == MODEL_CARD_SHA256, "model-card SHA256 mismatch")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        require(card.get("checkpoint_sha256") == CHECKPOINT_SHA256, "checkpoint mismatch")
        pin = card.get("files", {}).get("evidence/training-feature-bindings.json")
        require(isinstance(pin, dict) and bind_path.stat().st_size == pin.get("bytes")
                and sha_file(bind_path) == pin.get("sha256"), "feature-binding file mismatch")
        bindings = json.loads(bind_path.read_text(encoding="utf-8"))
        require(isinstance(bindings, list) and len(bindings) == 134, "expected 134 retained inputs")
        require(args.output.is_absolute() and not args.output.exists() and
                not args.output.is_symlink(), "output must be a fresh absolute directory")
        ids = set()
        with zipfile.ZipFile(str(args.capsule), "r") as capsule:
            capsule_manifest = json.loads(capsule.read("MANIFEST.json"))
            files = capsule_manifest["files"]
            args.output.mkdir(parents=True)
            wav_dir = args.output / "wav"
            pcm_dir = args.output / "pcm"
            wav_dir.mkdir()
            pcm_dir.mkdir()
            records = []
            for index, binding in enumerate(bindings):
                source_id = binding["source_id"]
                require(source_id not in ids, "duplicate source ID")
                ids.add(source_id)
                require(binding["split"] in ("train", "development"), "unexpected source role")
                candidates = []
                for member, info in files.items():
                    if source_id in info.get("path", "") and info.get("path", "").lower().endswith(".wav"):
                        raw = capsule.read(member)
                        require(len(raw) == info["bytes"] and sha_bytes(raw) == info["sha256"],
                                "capsule WAV member hash mismatch")
                        pcm, samples = read_wav(raw)
                        if sha_bytes(pcm) == binding["pcm_sha256"]:
                            candidates.append((member, info, raw, pcm, samples))
                require(bool(candidates), "no WAV matches retained PCM identity: " + source_id)
                if len(candidates) > 1:
                    preferred = [c for c in candidates if "/outputs/audio/" in c[1]["path"]]
                    if len(preferred) == 1:
                        candidates = preferred
                require(len(candidates) == 1, "ambiguous WAV source: " + source_id)
                member, info, raw, pcm, samples = candidates[0]
                (wav_dir / ("%03d.wav" % index)).write_bytes(raw)
                (pcm_dir / ("%03d.pcm" % index)).write_bytes(pcm)
                records.append({
                    "index": index, "source_id": source_id, "split": binding["split"],
                    "wav_bytes": len(raw), "wav_sha256": sha_bytes(raw),
                    "pcm_bytes": len(pcm), "pcm_sha256": sha_bytes(pcm),
                    "samples": samples, "feature_shape": binding["shape"],
                    "capsule_member": member
                })
        counts = {}
        for row in records:
            counts[row["split"]] = counts.get(row["split"], 0) + 1
        manifest = {
            "schema": "kws-r1-arm-input-recovery-v1",
            "input_capsule_sha256": CAPSULE_SHA256,
            "model_checkpoint_sha256": CHECKPOINT_SHA256,
            "counts": counts,
            "total_samples": sum(r["samples"] for r in records),
            "records": records,
            "raw_audio_recovered_locally": True,
            "raw_audio_published": False,
            "new_tts": 0, "new_asr": 0, "training_updates": 0
        }
        out = args.output / "INPUT_MANIFEST.json"
        out.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"records": len(records), "counts": counts,
                          "audio_seconds": manifest["total_samples"] / 16000.0,
                          "input_manifest_sha256": sha_file(out),
                          "raw_audio_published": False}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "FAILED", "reason": str(exc),
                          "training_updates": 0, "tts": 0, "asr": 0}), file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
