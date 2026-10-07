"""Stdlib-only, fail-closed contracts for the bounded public CosyVoice pilot."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
CONFIG_SHA256 = "f3076727adaf2c5521ec78ce9deccb79fc949385e203677eb955c84fd451ab6d"
OUTPUT_LIMIT = 128 * 1024**2
RSS_LIMIT = 12 * 1024**3
HOST_AVAILABLE_MIN = 2 * 1024**3
RESERVE = 1024**3
MODEL_FREE_MIN = 6635020288
LOG_LIMIT = 8 * 1024**2
JOB_LIMIT = 14000000000


class GateError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise GateError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_config():
    path = HERE / "pilot-config.json"
    require(sha256(path) == CONFIG_SHA256, "frozen pilot config changed")
    cfg = read_json(path)
    rows = cfg["design"]["phrase_rows"]
    require(len(rows) == 6 and cfg["design"]["maximum_generated_clips"] == 12, "scope changed")
    for row in rows:
        require(set(row["condition_order"]) == {"control", "inpaint"}, "condition order invalid")
        for key in ("control_text", "inpaint_text"):
            require(unicodedata.normalize("NFC", row[key]) == row[key], "non-NFC input")
        require(row["inpaint_text"].replace("[w][ō]", "窝").replace("[w][ū]", "屋") == row["control_text"], "phone contract changed")
    return cfg


def require_runner():
    require(os.environ.get("GITHUB_ACTIONS") == "true", "acquisition/execution is runner-only")
    require(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-pipeline", "wrong repository")
    require(os.environ.get("RUNNER_OS") == "Linux" and os.environ.get("RUNNER_ARCH") == "X64", "wrong runner platform")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "no reruns allowed")


def safe_relative(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and bool(p.parts) and all(x not in (".", "..", "") for x in p.parts), "unsafe relative path")
    require("\\" not in name, "unsafe path separator")
    return p


def regular_file(path):
    s = Path(path).lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, f"not a single regular file: {Path(path).name}")
    return s


def verify_file(path, entry):
    s = regular_file(path)
    size = entry.get("bytes", entry.get("size"))
    require(s.st_size == size, f"size mismatch: {Path(path).name}")
    h = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            blob.update(chunk)
    expected_sha = entry.get("sha256") or entry.get("lfs", {}).get("sha256")
    if expected_sha:
        require(h.hexdigest() == expected_sha, f"SHA256 mismatch: {Path(path).name}")
    expected_blob = entry.get("git_blob_sha1") or (entry.get("blobId") if "lfs" not in entry else None)
    if expected_blob:
        require(blob.hexdigest() == expected_blob, f"Git blob mismatch: {Path(path).name}")
    require(bool(expected_sha or expected_blob), "unpinned acquisition")
    return {"bytes": size, "sha256": h.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def tree_bytes(path):
    total = 0
    for p in Path(path).rglob("*"):
        require(not p.is_symlink(), "symlink in bounded tree")
        if p.is_file():
            total += regular_file(p).st_size
    return total


def disk_gate(path, minimum=RESERVE):
    free = shutil.disk_usage(path).free
    require(free >= minimum, f"free disk gate failed: {free} < {minimum}")
    return free


def allocated_bytes(root):
    """Count newly allocated job files, not symlink targets such as venv Python."""
    return sum(p.lstat().st_blocks * 512 for p in Path(root).rglob("*") if not p.is_symlink())


def job_gate(root, needed=RESERVE, scan=True):
    used = allocated_bytes(root) if scan else 0
    free = disk_gate(root, needed)
    require(JOB_LIMIT - used >= needed, "14,000,000,000-byte job allocation envelope exhausted")
    baseline = read_json(Path(root) / "controller-state.json")["baseline_free_bytes"]
    require(type(baseline) is int and baseline > 0, "invalid disk baseline")
    delta = max(0, baseline - free)
    require(JOB_LIMIT - delta >= needed, "14GB whole-filesystem delta envelope exhausted")
    return {"allocated_job_bytes": used, "actual_free_bytes": free, "new_filesystem_bytes": delta, "required_remaining_bytes": needed}


def available_memory():
    data = Path("/proc/meminfo").read_text()
    return int(next(x.split()[1] for x in data.splitlines() if x.startswith("MemAvailable:"))) * 1024


def memory_gate():
    available = available_memory()
    require(available >= HOST_AVAILABLE_MIN, "host available memory below 2 GiB")
    return available


class AllowlistedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_url(url):
    p = urllib.parse.urlparse(url)
    host = p.hostname or ""
    require(p.scheme == "https" and not p.username and not p.password and p.port in (None, 443), "non-HTTPS acquisition")
    require(host in ("raw.githubusercontent.com", "huggingface.co") or host.endswith(".hf.co") or host.endswith(".xethub.hf.co"), "unexpected download/redirect host")


def download_one(root, name, entry, url):
    """Exactly one streaming transfer, no retry, no snapshot or duplicate copy."""
    path = Path(root).joinpath(*safe_relative(name).parts)
    require(not path.exists() and not path.with_name(path.name + ".part").exists(), "acquisition would overwrite/retry")
    path.parent.mkdir(parents=True, exist_ok=True)
    size = entry.get("bytes", entry.get("size"))
    disk_gate(root, size + RESERVE)
    validate_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "bounded-cosyvoice-pilot/1"})
    part = path.with_name(path.name + ".part")
    started = time.monotonic()
    with urllib.request.build_opener(AllowlistedRedirect()).open(request, timeout=60) as response, open(part, "xb") as f:
        validate_url(response.url)
        count = 0
        while True:
            require(time.monotonic() - started <= 900, "single-download timeout")
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            require(count <= size, "download exceeded pinned length")
            disk_gate(root)
            memory_gate()
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    receipt = verify_file(part, entry)
    os.replace(part, path)
    return dict(path=name, **receipt)


def verify_source(source):
    manifest = read_json(HERE / "source-allowlist.json")
    expected = {row["path"] for row in manifest["files"]}
    actual = {str(p.relative_to(source)) for p in Path(source).rglob("*") if p.is_file()}
    require(actual == expected, "source tree contains missing/unapproved files")
    rows = [dict(path=r["path"], **verify_file(Path(source) / r["path"], r)) for r in manifest["files"]]
    return {"allowlist_sha256": sha256(HERE / "source-allowlist.json"), "files": rows}


def verify_models(model, cfg):
    expected = {r["rfilename"] for r in cfg["weights"]["files"]}
    actual = {str(p.relative_to(model)) for p in Path(model).rglob("*") if p.is_file()}
    require(actual == expected, "model tree contains missing/unapproved files")
    files = [dict(path=r["rfilename"], **verify_file(Path(model) / r["rfilename"], r)) for r in cfg["weights"]["files"]]
    validate_pretrained_configs(model)
    return files


def validate_pretrained_configs(model):
    forbidden = {"auto_map", "trust_remote_code", "custom_pipelines", "custom_generate", "_attn_implementation_internal", "attn_implementation", "_attn_implementation"}
    def walk(value):
        if isinstance(value, dict):
            require(not (set(value) & forbidden), "custom-code/attention configuration forbidden")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        walk(read_json(Path(model) / "CosyVoice-BlankEN" / name))
    config = read_json(Path(model) / "CosyVoice-BlankEN/config.json")
    require(config.get("model_type") == "qwen2" and config.get("architectures") == ["Qwen2ForCausalLM"], "unexpected architecture")
    tokenizer = read_json(Path(model) / "CosyVoice-BlankEN/tokenizer_config.json")
    require(tokenizer.get("tokenizer_class") in (None, "Qwen2Tokenizer", "Qwen2TokenizerFast"), "custom tokenizer class forbidden")


def verify_reference(path, cfg):
    r = cfg["reference"]
    return verify_file(path, {"bytes": r["bytes"], "sha256": r["sha256"], "git_blob_sha1": r["blob_sha1"]})


def runtime_gate(receipt_path, lock_path):
    r = read_json(receipt_path)
    require(r.get("status") == "qualified", "runtime not qualified")
    require(r.get("lock_sha256") == sha256(lock_path), "runtime lock receipt mismatch")
    require(Path(r.get("python_executable", "")).resolve() == Path(sys.executable).resolve(), "runtime interpreter changed")
    require(r.get("network_attempts") == [], "runtime qualification attempted network")
    require(r.get("pip_check", {}).get("returncode") == 0, "runtime pip check receipt failed")
    require(sys.prefix != sys.base_prefix, "runtime must remain isolated")
    import importlib.metadata
    import re
    canonical = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    actual = {canonical(d.metadata["Name"]): d.version for d in importlib.metadata.distributions()}
    require(actual == r.get("versions"), "installed distribution set changed after qualification")
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120)
    require(check.returncode == 0, "dependency check failed")
    return {"receipt_sha256": sha256(receipt_path), "lock_sha256": sha256(lock_path), "pip_check": "passed"}


def execution_env():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", WANDB_DISABLED="true", TORCH_FORCE_WEIGHTS_ONLY_LOAD="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    env.pop("PYTHONPATH", None)
    return env
