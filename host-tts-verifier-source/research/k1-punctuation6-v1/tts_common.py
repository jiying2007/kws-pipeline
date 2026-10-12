"""Fixed K1 TTS filesystem and process metadata helpers. No model/network import."""
from pathlib import Path
import hashlib
import json
import os
import stat
import sys

from contract import canonical, require

SAFE_EXCEPTIONS={"ValueError","RuntimeError","TimeoutError","TimeoutExpired","OSError","FileNotFoundError",
                 "PermissionError","MetadataMismatch","ModuleNotFoundError","ImportError","MemoryError","CalledProcessError",
                 "AttributeError","TypeError","KeyboardInterrupt","SystemExit","OtherException"}

def safe_exception(error):
    name=type(error).__name__
    return name if name in SAFE_EXCEPTIONS else "OtherException"



def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def unaliased(path, *, file=False):
    p = Path(path)
    require(not any(x.is_symlink() for x in (p, *p.parents)), "no symlink paths")
    if file:
        s = p.stat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, "unaliased regular file")
    return p


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_once(path, value):
    """Exclusive durable claim; call only after fresh parent directories exist."""
    path = unaliased(path)
    raw = canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.write(raw); target.flush(); os.fsync(target.fileno())
    fsync_directory(path.parent)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def atomic_json(path, value):
    path = unaliased(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.write(canonical(value)); target.flush(); os.fsync(target.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)


def child_environment(runtime, interpreter):
    """Construct from nothing, not os.environ; no host token or credentials."""
    runtime, interpreter = Path(runtime).absolute(), Path(interpreter).absolute()
    return {"PATH": str(interpreter.parent) + ":/usr/bin:/bin", "HOME": str(runtime/"home"),
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1", "ORT_DISABLE_TELEMETRY": "1", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "NUMBA_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_XET": "1"}


def assert_no_host_secrets(environment):
    """Defense in depth before imports; never logs key values."""
    banned = {"GITHUB_TOKEN", "GH_TOKEN", "KWS_EVIDENCE_TOKEN", "KWS_DATA_TOKEN", "SSH_AUTH_SOCK",
              "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "HF_TOKEN",
              "HUGGING_FACE_HUB_TOKEN", "NETRC", "GIT_ASKPASS", "PYTHONPATH", "LD_PRELOAD"}
    require(not banned.intersection(environment), "host credential key reached child")
    require(not any("TOKEN" in k or "SECRET" in k or "PASSWORD" in k or k.startswith("ACTIONS_")
                    for k in environment if k != "TOKENIZERS_PARALLELISM"), "unexpected secret-bearing child key")


def isolated_command(interpreter, script, arguments=()):
    return [str(interpreter), "-I", "-B", "-c",
            "import pathlib,runpy,sys; p=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0,str(p.parent)); sys.argv=sys.argv[1:]; runpy.run_path(str(p),run_name='__main__')",
            str(script), *map(str, arguments)]
