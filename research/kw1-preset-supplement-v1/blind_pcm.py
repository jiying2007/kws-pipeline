"""Strict audio-only transport and immutable evidence; standard library only."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import tempfile

MAX_FILES = 16
MAX_FRAMES = 192000
MAX_INPUT_BYTES = 8 * 1024**2
MAX_JOB_BYTES = 65536
SHA = re.compile(r"[0-9a-f]{64}\Z")
JOB_SCHEMA = "kw1-blind-pcm-job-v1"
RECORD_FIELDS = {"opaque_id", "file", "sha256", "bytes", "frames", "sample_rate_hz", "channels", "sample_format"}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw, maximum=MAX_JOB_BYTES):
    require(type(raw) is bytes and len(raw) <= maximum, "bounded JSON bytes required")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    def invalid(_):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def unaliased(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "aliased path")
    require(path.resolve() == path, "noncanonical path")
    return path


def read_regular(path, maximum):
    path = unaliased(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum,
                "unaliased bounded regular file required")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(descriptor)
        entry = path.lstat()
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(stamp(before) == stamp(after) == stamp(entry) and len(raw) == before.st_size,
                "file changed during read")
        return raw
    finally:
        os.close(descriptor)


def fsync_directory(path):
    descriptor = os.open(unaliased(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_once(path, value):
    """An existing filename is a consumed claim even if its write was interrupted."""
    path = Path(path)
    raw = canonical(value)
    require(len(raw) <= MAX_INPUT_BYTES, "evidence size cap")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        # Make the exclusive claim durable before any model work can start.
        fsync_directory(path.parent)
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)
    return sha(raw)


def atomic(path, value):
    path = Path(path)
    temporary = path.parent / ("." + path.name + ".tmp")
    write_once(temporary, value)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def validate_job(raw):
    job = decode(raw)
    require(type(job) is dict and set(job) == {"schema", "clips"} and job["schema"] == JOB_SCHEMA, "blind PCM job schema")
    rows = job["clips"]
    require(type(rows) is list and 1 <= len(rows) <= MAX_FILES, "one to sixteen clips required")
    ids = []
    for row in rows:
        require(type(row) is dict and set(row) == RECORD_FIELDS, "only PCM identity fields allowed")
        digest = row["sha256"]
        require(type(digest) is str and SHA.fullmatch(digest) is not None, "PCM SHA256")
        require(row["opaque_id"] == digest and row["file"] == digest + ".pcm", "content-addressed PCM identity")
        require(type(row["frames"]) is int and 1 <= row["frames"] <= MAX_FRAMES, "PCM frame cap")
        require(type(row["bytes"]) is int and row["bytes"] == 2 * row["frames"], "PCM byte count")
        require(type(row["sample_rate_hz"]) is int and row["sample_rate_hz"] == 16000 and
                type(row["channels"]) is int and row["channels"] == 1 and row["sample_format"] == "s16le", "fixed PCM format")
        ids.append(digest)
    require(ids == sorted(set(ids)), "unique PCM content-hash order required")
    require(len(raw) + sum(row["bytes"] for row in rows) <= MAX_INPUT_BYTES, "aggregate input cap")
    return job


def bind_input(root):
    root = unaliased(root)
    require(root.is_dir(), "input directory")
    raw = read_regular(root / "job.json", MAX_JOB_BYTES)
    job = validate_job(raw)
    require({p.name for p in root.iterdir()} == {"job.json", *(row["file"] for row in job["clips"])}, "only job.json and raw PCM files allowed")
    pcm = {}
    for row in job["clips"]:
        value = read_regular(root / row["file"], 2 * MAX_FRAMES)
        require(len(value) == row["bytes"] and sha(value) == row["sha256"], "PCM byte identity mismatch")
        pcm[row["opaque_id"]] = value
    return job, raw, pcm


def canonical_wav(pcm):
    require(type(pcm) is bytes and len(pcm) % 2 == 0 and 2 <= len(pcm) <= 2 * MAX_FRAMES, "bounded complete PCM16 bytes")
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16,
                       1, 1, 16000, 32000, 2, 16, b"data", len(pcm)) + pcm


def identity_map(job, job_raw, pcm):
    rows = []
    for row in job["clips"]:
        require(sha(pcm[row["opaque_id"]]) == row["sha256"], "identity-map PCM binding")
        wav = canonical_wav(pcm[row["opaque_id"]])
        rows.append({"opaque_id": row["opaque_id"], "pcm_sha256": row["sha256"], "wav_sha256": sha(wav),
                     "wav_bytes": len(wav), "frames": row["frames"]})
    wav_order = sorted(row["wav_sha256"] for row in rows)
    require(len(set(wav_order)) == len(rows), "unique canonical WAV identities")
    for row in rows:
        row["retained_opaque_id"] = f"clip-{wav_order.index(row['wav_sha256']) + 1:06d}"
    return {"schema": "kw1-blind-pcm-wav-map-v1", "job_sha256": sha(job_raw), "clips": rows}


def reconstruct(job, job_raw, pcm, scratch):
    """New private canonical WAVs only; the model never receives input metadata."""
    scratch = unaliased(scratch)
    require(scratch.is_dir(), "private scratch mount")
    root = Path(tempfile.mkdtemp(prefix="blind-asr-", dir=scratch))
    os.chmod(root, 0o700)
    audio = root / "audio"; audio.mkdir(mode=0o700)
    mapping = identity_map(job, job_raw, pcm)
    rows = []
    for row in sorted(mapping["clips"], key=lambda r: r["wav_sha256"]):
        basename = row["wav_sha256"] + ".wav"
        with (audio / basename).open("xb") as target:
            target.write(canonical_wav(pcm[row["opaque_id"]]))
            target.flush(); os.fsync(target.fileno())
        rows.append({"audio_id": row["retained_opaque_id"], "audio_path": "audio/" + basename, "wav_sha256": row["wav_sha256"]})
    write_once(root / "job.json", {"schema": "blind-source-screen32-job-v1", "clips": rows})
    return root, mapping
RETAINED_PINS = {'research/qwen6_asr/FUNASR_MODEL_LICENSE': '7dba975a2069691db4992b0592d70828b330d2f8a30a71450f4e152a554e84f8', 'research/qwen6_asr/NOTICE.md': 'a01fd4891fd878f2bbb4de93e13dbef67dd10b127974240046304803c96685a7', 'research/qwen6_asr/core/asr6_contract.py': 'd1c3368133e06a3de9758b23f136c3a37a74a63057f317ece863470ad0304b10', 'research/qwen6_asr/core/asr_stage/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'research/qwen6_asr/core/asr_stage/adapters.py': '85512f20280cfa0aeb06b50699664e360db39bbf5b58034d3a2ede5cde8ff6ca', 'research/qwen6_asr/core/asr_stage/architecture.py': 'd3f0d67bcf724c34763ac4666a1570b02dfd979182e6183403ef91e0897d58ab', 'research/qwen6_asr/core/asr_stage/assets.py': '91a8c7e75f37872fc1b07cabc2afd1790901bf44a3fda1e48fb5abc2b129ff6e', 'research/qwen6_asr/core/asr_stage/decoding.py': 'fd401bd85e4ca3d871e5a9354753d3599facbb32b9df22e34f05e5476e74b503', 'research/qwen6_asr/core/asr_stage/execution.py': '149befff56ef9da603d8b27174ca1a1d4eae638cc86f356cc5a1b8412ac06c9f', 'research/qwen6_asr/core/asr_stage/runtime_support.py': 'f3a6ee85e30dc5a1bc2351a76843c44bc988aed83745773e543e1b69cf4a6d90', 'research/qwen6_asr/core/pcm/pcm_binding.py': '1f8cb1e8fba2c99dfd969b7f33a12011243aa68f0a6e1eb0dbf45d90fc9b1684', 'research/qwen6_asr/core/runtime_version_gate.py': 'c6d2e35c69f3ba34e3a7f92bd9b486eb00476f76ed5c8687319f4382b23bd512', 'research/qwen6_asr/core/sense_binding.py': '1a2fd8474d347591939b3ce9ab7411aa9a8f9ce0ffa732d93a95402b677503d2', 'research/qwen6_asr/core/vendor/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'research/qwen6_asr/core/vendor/checkpoint_guard/__init__.py': 'ca1bfd88d26591efeb70bb58155294be4ebc2af7a6e45cc5f09f3e90dc5c541e', 'research/qwen6_asr/core/vendor/checkpoint_guard/checkpoint_contract.py': '80cd48b0ce8beb2f7473372a3d904ba42af316c7df4a3c44f8a8b54dd50896c4', 'research/qwen6_asr/core/vendor/checkpoint_guard/exact_apply.py': '10bc8634053a602b1669a007865c9d18955577b8aab4b7e7f14dfdcc6ce9e22b', 'research/qwen6_asr/core/vendor/qwen_loading_gate_v2.py': 'b21139d3d63467b910f13b4bc2045d08a737799f9d4bc4dae96972f2a9cd1dad', 'research/qwen6_asr/import_preflight.py': '30cfe60347ccc6074d98a21406fccee3e3be5c54822aa8d874aae3655693ae1c', 'research/qwen6_asr/model-locks.json': '8b0813aba3c2447c7dc30b47e856e4ce9a619a98d32ed6dfa9f44dc6646714e4', 'research/qwen6_asr/runtime-lock.json': '2d496a76bd0e044a6812bfc9afdd8965474bf135bd21f3a24f71695638d1183f', 'research/qwen6_asr/sense_parser.py': '878278e46ae9af349a93e37dddffa83e916f447ef82fb6dd61fa8948bb8261d7', 'research/qwen6_asr/setup_diagnostics.py': '1b98e253d7054d0bfcd5ec128cb3a331bb05de48776330e84ea348694b3a1bf2', 'research/qwen6_asr/setup_locked.py': 'd7c580b59b0bd249bc62c9093b9de5f8842170510cf11e7e61c29057e6287af4', 'research/qwen6_asr/wheel_identity.py': '623ee07a0d8f85ba1d7218a56fcd0e270df8c6e2c3e353dd52b3184ca915d8d5', 'research/source-screen32-v1/asr_worker.py': 'b5cef29fe4e2c67f9121d93f83eb351f3e7a8ca14136ad684669832a77164f17', 'research/source-screen32-v1/runtime_scope.py': 'c08bd46d34bea011a63d32867386b26637d5a02753769654e08e4c32c31cb927', 'research/source-screen32-v1/sense_adapter.py': '307a3a5985ae616e777f0b71ffc37ac31cd4b78bd12b4d7cbd69f816813057c1', 'research/source-screen32-v1/setup_adapter.py': 'b7f6c3493ad0da909c7e6856d27f3247f5e1716ce6c981cb396a03687aa7e547', 'research/source-screen32-whisper-v1/adapter.py': 'd142605a62ff990633194b43992b698673780b065e5f3599d7e27eccc74d693b', 'research/source-screen32-whisper-v1/common.py': '21cc72171d071cc9ac0113307af751315b69ce70fbf522a1b2586a7617900861', 'research/source-screen32-whisper-v1/runtime-lock.json': '881443c6e66acb99223137bd8342788468f814d0a1c3e73ba33a27da3ad1fe6d', 'research/source-screen32-whisper-v1/source-lock.json': '0635fb8759fb56439f59db6956612f78cc8903c239f81f3eb7f9b0c47eb99649', 'research/source-screen32-whisper-v1/setup_runtime.py': '526d4c5b282c61644ee8357fb045e371288b5143b6a4ec9981ce867a758704c4', 'tools/asr_native_contracts.py': '752f192fb125fb31291a8246845c3015dc97348fc27822c389e14b52da3ef674'}
