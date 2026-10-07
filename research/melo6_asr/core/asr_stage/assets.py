"""Local exact-asset verification; never downloads, installs, or imports models."""
import hashlib
import os
import re
import stat
from pathlib import Path
from .architecture import canonical_sha
from pcm.pcm_binding import _open_root

MODELS = {
    "Qwen/Qwen3-ASR-0.6B": ("5eb144179a02acc5e5ba31e748d22b0cf3e303b0", frozenset((
        "chat_template.json", "config.json", "generation_config.json", "merges.txt", "model.safetensors",
        "preprocessor_config.json", "tokenizer_config.json", "vocab.json"))),
    "FunAudioLLM/SenseVoiceSmall": ("3847d57b6bdf2dd8875cb1508d2af43d80a16bf7", frozenset((
        "am.mvn", "chn_jpn_yue_eng_ko_spectok.bpe.model", "config.yaml", "configuration.json", "model.pt"))),
}
SHA = re.compile(r"[0-9a-f]{64}\Z")


def validate_asset_lock(lock, approved_sha256, model_id):
    if type(lock) is not dict or set(lock) != {"schema", "model_id", "revision", "files"}:
        raise ValueError("Malformed asset lock")
    if type(approved_sha256) is not str or not SHA.fullmatch(approved_sha256) or canonical_sha(lock) != approved_sha256:
        raise ValueError("Asset lock differs from predeclaration")
    revision, names = MODELS[model_id]
    if lock["schema"] != "asr-exact-model-assets-v1" or lock["model_id"] != model_id or lock["revision"] != revision:
        raise ValueError("Model identity/revision drift")
    files = lock["files"]
    if type(files) is not list or len(files) != len(names): raise ValueError("Asset inventory count mismatch")
    seen, total = set(), 0
    for row in files:
        if type(row) is not dict or set(row) != {"filename", "bytes", "sha256", "source_url"}: raise ValueError("Malformed asset row")
        name = row["filename"]
        if type(name) is not str or name not in names or name in seen: raise ValueError("Unknown or duplicate asset")
        seen.add(name)
        if type(row["bytes"]) is not int or not 0 < row["bytes"] <= 2_000_000_000: raise ValueError("Invalid asset byte budget")
        if type(row["sha256"]) is not str or not SHA.fullmatch(row["sha256"]): raise ValueError("Invalid asset SHA256")
        if row["source_url"] != f"https://huggingface.co/{model_id}/resolve/{revision}/{name}": raise ValueError("Asset is not exact official revision URL")
        total += row["bytes"]
    if seen != names or total > 2_000_000_000: raise ValueError("Incomplete or excessive asset lock")
    return {row["filename"]: dict(row) for row in files}


def verify_assets(root, lock, approved_sha256, model_id):
    files = validate_asset_lock(lock, approved_sha256, model_id)
    directory = _open_root(str(root))
    observed = []
    try:
        if set(os.listdir(directory)) != set(files): raise ValueError("Model directory contains missing/extra files")
        for name in sorted(files):
            expected = files[name]
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != expected["bytes"]:
                    raise ValueError("Model asset type/alias/length mismatch")
                h, size = hashlib.sha256(), 0
                while True:
                    block = os.read(fd, min(1024 * 1024, expected["bytes"] + 1 - size))
                    if not block: break
                    size += len(block)
                    if size > expected["bytes"]: raise ValueError("Model asset exceeded declared length")
                    h.update(block)
                after = os.fstat(fd); named = os.stat(name, dir_fd=directory, follow_symlinks=False)
                stamp = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_mode)
                if stamp(before) != stamp(after) or stamp(after) != stamp(named): raise ValueError("Model asset changed during verification")
                if size != expected["bytes"] or h.hexdigest() != expected["sha256"]: raise ValueError("Model body hash/length mismatch")
                observed.append({"filename":name,"bytes":size,"sha256":h.hexdigest()})
            finally:
                os.close(fd)
    finally:
        os.close(directory)
    return {"model_id":model_id,"revision":lock["revision"],"asset_lock_sha256":approved_sha256,
            "verified_files":observed,"verified_bytes":sum(r["bytes"] for r in observed)}


def read_small_locked(root, filename, files):
    # Caller already mounted the directory read-only and verified all file bodies.
    row = files[filename]
    if row["bytes"] > 1024 * 1024: raise ValueError("Not a bounded config file")
    directory = _open_root(str(root))
    fd = None
    try:
        fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != row["bytes"]: raise ValueError("Config file identity mismatch")
        parts, size = [], 0
        while size <= row["bytes"]:
            part = os.read(fd, min(65536, row["bytes"] + 1 - size))
            if not part: break
            parts.append(part); size += len(part)
        raw = b"".join(parts)
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]: raise ValueError("Config body mismatch")
        return raw.decode("utf-8")
    finally:
        if fd is not None: os.close(fd)
        os.close(directory)
