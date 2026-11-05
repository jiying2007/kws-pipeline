#!/usr/bin/env python3
"""Create the active ASR role from retained artifacts, without changing old runs.

Downloads are hash checked and reusable. Installation is offline from those
artifacts, with no dependency resolution. No model is executed by this helper.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import venv
import zipfile

import research_runtime

ROOT = Path(__file__).resolve().parents[1]
WHISPER_SOURCE = ROOT / "research/source-screen32-whisper-v1/source-lock.json"
MODEL_LOCK = ROOT / "research/qwen6_asr/model-locks.json"
WHISPER_SHA = "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
WHISPER_URL = "https://openaipublic.azureedge.net/main/whisper/models/" + WHISPER_SHA + "/small.pt"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or not name or "\\" in name or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("unsafe artifact path")
    return path


def acquire(row, path, deadline):
    path = Path(path)
    expected_size = row.get("size_bytes", row.get("bytes"))
    digest = row["sha256"]
    if path.is_file() and (expected_size is None or path.stat().st_size == expected_size) and sha_file(path) == digest:
        return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size, "reused": True}
    if path.exists():
        raise ValueError(f"existing artifact differs: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    url = row.get("url", row.get("source_url"))
    if not url.startswith("https://"):
        raise ValueError("artifact URL must use HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": "kws-pipeline-active-asr/1"})
    count, sha = 0, hashlib.sha256()
    cap = expected_size or 1024**3
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while block := response.read(1024**2):
            if time.monotonic() >= deadline:
                raise TimeoutError("setup deadline")
            count += len(block)
            if count > cap:
                raise ValueError(f"artifact byte cap: {path.name}")
            sha.update(block)
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
    if (expected_size is not None and count != expected_size) or sha.hexdigest() != digest:
        raise ValueError(f"artifact identity mismatch: {path.name}")
    temporary.replace(path)
    print(f"acquired {path.name} {count} bytes", flush=True)
    return {"path": str(path), "sha256": digest, "bytes": count, "reused": False}


def merged_packages(manifest):
    research_runtime.verify_reference_locks(manifest, "asr")
    rows = {}
    for ref in manifest["roles"]["asr"]["reference_locks"]:
        for row in json.loads((ROOT / ref["path"]).read_text())["files"]:
            key = row["name"].lower().replace("_", "-")
            previous = rows.get(key)
            if previous and any(previous[k] != row[k] for k in ("version", "filename", "sha256")):
                raise ValueError(f"artifact conflict: {key}")
            if previous is None:
                rows[key] = row
    return rows


def locked_assets(root):
    jobs = []
    for name, spec in json.loads(MODEL_LOCK.read_text()).items():
        for row in spec["asset_lock"]["files"]:
            jobs.append((row, root / "models" / name / safe_relative(row["filename"])))
    jobs.append(({"url": WHISPER_URL, "sha256": WHISPER_SHA}, root / "models/whisper/small.pt"))
    for row in json.loads(WHISPER_SOURCE.read_text())["files"]:
        jobs.append((row, root / "whisper-source" / safe_relative(row["path"])))
    return jobs


def wheel_identity(path, row):
    with zipfile.ZipFile(path) as archive:
        # Setuptools contains vendored distributions; the wheel's own identity
        # lives in the one top-level dist-info directory.
        names = [x for x in archive.namelist() if x.endswith(".dist-info/METADATA") and x.count("/") == 1]
        if len(names) != 1:
            raise ValueError(f"wheel must contain one top-level METADATA: {path.name}")
        from email.parser import BytesParser
        metadata = BytesParser().parsebytes(archive.read(names[0]))
    normalize = lambda x: x.lower().replace("_", "-").replace(".", "-")
    if normalize(metadata["Name"]) != normalize(row["name"]) or metadata["Version"] != row["version"]:
        raise ValueError(f"wheel identity differs: {path.name}")
    return {"path": str(path), "sha256": sha_file(path), "name": row["name"], "version": row["version"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--setup-minutes", type=int, default=30)
    parser.add_argument("--download-workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.setup_minutes <= 90:
        parser.error("setup budget must be 1..90 minutes")
    manifest = research_runtime.load_manifest()
    observed = research_runtime.check_interpreter(manifest)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("retained wheels require Linux x86_64")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record = {"schema": "active-asr-setup-v1", "attempt": attempt, "python": observed,
              "historical_lock_python": "3.12.14", "active_python": manifest["python"]["version"],
              "models_executed": 0, "status": "running", "stage": "download", "commands": [],
              "lock_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in
                              (research_runtime.MANIFEST, MODEL_LOCK, WHISPER_SOURCE)},
              "installer_sha256": sha_file(__file__)}
    receipt = root / "setup-attempts" / (attempt + ".json")
    save(receipt, record)
    deadline = time.monotonic() + args.setup_minutes * 60
    env = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PIP_NO_CACHE_DIR="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               NUMEXPR_NUM_THREADS="4", MAX_JOBS="4", CMAKE_BUILD_PARALLEL_LEVEL="4")

    def command(argv, stage):
        record["stage"] = stage
        record["commands"].append({"argv": [str(x) for x in argv], "stage": stage})
        save(receipt, record)
        subprocess.run([str(x) for x in argv], env=env, check=True,
                       timeout=max(1, deadline - time.monotonic()))

    try:
        packages = merged_packages(manifest)
        for ref in manifest["roles"]["asr"]["reference_locks"]:
            record["lock_sha256"][ref["path"]] = ref["sha256"]
        jobs = [(row, root / "downloads" / safe_relative(row["filename"])) for row in packages.values()]
        jobs.extend(locked_assets(root))
        with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
            record["acquired"] = list(pool.map(lambda pair: acquire(*pair, deadline), jobs))
        save(receipt, record)
        python = root / "venv/bin/python"
        if not python.exists():
            record["stage"] = "create_venv"
            save(receipt, record)
            venv.EnvBuilder(with_pip=True).create(root / "venv")
        command([python, "-B", ROOT / "tools/research_runtime.py", "check"], "check_python")
        prebuilt, sources, build_tools = [], [], []
        for row in packages.values():
            path = root / "downloads" / row["filename"]
            if path.suffix == ".whl":
                wheel_identity(path, row)
                (build_tools if row.get("build_tool") else prebuilt).append(path)
            else:
                sources.append((row, path))
        pip = [python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", "--disable-pip-version-check"]
        command([*pip, *build_tools], "install_build_tools")
        command([*pip, *prebuilt], "install_runtime_wheels")
        built = []
        for row, path in sources:
            out = root / "built-wheels" / row["name"]
            out.mkdir(parents=True, exist_ok=True)
            candidates = list(out.glob("*.whl"))
            if not candidates:
                command([python, "-I", "-m", "pip", "wheel", "--no-index", "--no-deps",
                         "--no-build-isolation", "--wheel-dir", out, path], "build_" + row["name"])
                candidates = list(out.glob("*.whl"))
            if len(candidates) != 1:
                raise ValueError("one built wheel required")
            built.append(wheel_identity(candidates[0], row))
        record["built_wheels"] = built
        command([*pip, *(x["path"] for x in built)], "install_built_wheels")
        command([python, "-I", "-m", "pip", "check"], "pip_check")
        command([python, "-B", ROOT / "tools/research_runtime.py", "check", "--role", "asr", "--packages"], "check_role")
        code = "import importlib.metadata,json;print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()},sort_keys=True))"
        inventory = json.loads(subprocess.check_output([str(python), "-I", "-c", code], env=env, text=True))
        for name, row in packages.items():
            if inventory.get(name) != row["version"]:
                raise ValueError("installed version mismatch: " + name)
        record["installed_packages"] = inventory
        record.update(status="complete", stage="complete", completed_at=datetime.now(timezone.utc).isoformat())
        save(receipt, record)
        save(root / "setup.json", record)
        print(json.dumps({"status": "complete", "python": str(python), "packages": len(packages),
                          "receipt": str(receipt), "models_executed": 0}), flush=True)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        save(receipt, record)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
