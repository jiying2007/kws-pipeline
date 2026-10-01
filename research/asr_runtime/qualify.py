#!/usr/bin/env python3
"""Fail-closed, dependency-free controller for a no-model hosted qualification.

Default mode validates candidate inputs only. Real execution is deliberately not
exposed until reviewed artifact identities and the execution admission are complete.
This module never imports third-party packages on the host.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile

SCHEMA = "kws.asr-runtime-receipt.v1"
HEX = re.compile(r"[0-9a-f]{64}\Z")
MAX_RECEIPT = 65536
MAX_LOG = 1048576
OFFICIAL_HOST = "files.pythonhosted.org"
ROOT = Path(__file__).resolve().parent


class GateError(RuntimeError):
    """A failed qualification requirement; never a successful skipped check."""


def require(condition, message):
    if not condition:
        raise GateError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path, maximum=2 * 1024 * 1024):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing or linked JSON input")
    require(path.stat().st_size <= maximum, "JSON input exceeds byte limit")
    return json.loads(path.read_bytes())


def canonical(record):
    return (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + "\n").encode()


def write_receipt(path, record):
    data = canonical(record)
    require(len(data) <= MAX_RECEIPT, "receipt exceeds 64 KiB limit")
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "receipt destination is not fresh")
    with path.open("xb") as out:
        out.write(data)
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
            "unsafe archive path")
    require(not any(ord(c) < 32 for c in name), "control character in archive path")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and ":" not in name,
            "unsafe archive path")
    require(str(p) == name.rstrip("/"), "noncanonical archive path")
    return str(p)


def validate_inventory(record, expected_count, byte_limit):
    files = record.get("files")
    require(isinstance(files, list) and len(files) == expected_count, "inventory count mismatch")
    names = set()
    total = 0
    for item in files:
        name = item.get("filename", "")
        require(safe_name(name) == name and "/" not in name, "artifact must have a flat filename")
        require(name not in names, "duplicate input filename")
        names.add(name)
        require(HEX.fullmatch(item.get("sha256", "")), "invalid artifact SHA256")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", item.get("name", "")), "unsafe package name")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]*", item.get("version", "")), "unsafe package version")
        size = item.get("size_bytes")
        require(type(size) is int and 0 < size <= byte_limit, "invalid artifact size")
        u = urllib.parse.urlsplit(item.get("url", ""))
        require(u.scheme == "https" and u.hostname == OFFICIAL_HOST and not u.query and
                not u.fragment and not u.username and u.port is None and
                u.path.startswith("/packages/") and urllib.parse.unquote(u.path.rsplit("/", 1)[-1]) == name,
                "artifact URL is not an exact official PyPI file")
        require(name.endswith((".whl", ".tar.gz", ".zip")), "unsupported input artifact")
        total += size
    require(total == byte_limit, "inventory total differs from approved exact byte budget")
    return [dict(item, bytes=item["size_bytes"]) for item in files]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise GateError("artifact URL redirect rejected")


def download_one(item, destination, opener=None, deadline=None):
    """One exact GET. No retries, mirrors, proxy inheritance, extraction or execution."""
    destination = Path(destination)
    require(not destination.exists() and not destination.is_symlink(), "download destination is not fresh")
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(item["url"], headers={"Accept-Encoding": "identity"})
    digest, consumed = hashlib.sha256(), 0
    deadline = time.monotonic() + 300 if deadline is None else deadline
    partial = destination.with_name(destination.name + ".partial")
    require(not partial.exists() and not partial.is_symlink(), "partial destination is not fresh")
    try:
        with opener.open(request, timeout=30) as source, partial.open("xb") as out:
            require(source.status == 200, "artifact HTTP status is not 200")
            require(source.geturl() == item["url"], "artifact URL changed")
            require(source.headers.get("Content-Encoding", "identity") == "identity", "encoded response rejected")
            require(source.headers.get("Content-Length") == str(item["bytes"]), "artifact HTTP length mismatch")
            while True:
                require(time.monotonic() < deadline, "artifact download wall deadline exceeded")
                chunk = source.read(min(1024 * 1024, item["bytes"] - consumed + 1))
                require(time.monotonic() < deadline, "artifact download wall deadline exceeded")
                if not chunk:
                    break
                consumed += len(chunk)
                require(consumed <= item["bytes"], "artifact exceeds exact download byte cap")
                out.write(chunk)
                digest.update(chunk)
        require(consumed == item["bytes"] and digest.hexdigest() == item["sha256"], "artifact size/hash mismatch")
        partial.rename(destination)
    except BaseException:
        # Partial bytes remain named as such for local failure diagnosis; never admitted.
        raise
    return consumed


def inspect_archive(path, expanded_limit, member_limit, allowed_pth=None):
    """Inspect and fully stream all members without extraction or source execution.

    Counts actual decompressed bytes, verifies ZIP CRC, and rejects links, duplicate
    paths, path traversal and executable .pth unless exact bytes were reviewed.
    A directory listing alone is not represented as a payload review.
    """
    path = Path(path)
    names, total, payloads = set(), 0, {}
    allowed_pth = allowed_pth or {}

    def member(name, size, source, mode=stat.S_IFREG):
        nonlocal total
        name = safe_name(name)
        require(name not in names, "duplicate archive path")
        names.add(name)
        require(len(names) <= member_limit, "archive member limit exceeded")
        require(size >= 0 and total + size <= expanded_limit, "archive expanded limit exceeded")
        digest, actual, pth = hashlib.sha256(), 0, bytearray()
        if source is not None:
            for chunk in iter(lambda: source.read(min(1024 * 1024, size - actual + 1)), b""):
                actual += len(chunk)
                require(actual <= size and total + actual <= expanded_limit, "actual expanded byte limit exceeded")
                digest.update(chunk)
                if name.endswith(".pth"):
                    require(actual <= 65536, "oversized .pth")
                    pth.extend(chunk)
            require(actual == size, "archive member length mismatch")
            if name.endswith(".pth"):
                require(allowed_pth.get(name) == digest.hexdigest(), "unreviewed executable/path .pth rejected")
            total += actual
            payloads[name] = {"bytes": actual, "sha256": digest.hexdigest()}

    is_wheel = path.name.endswith(".whl")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                require(not info.flag_bits & 1, "encrypted ZIP member rejected")
                mode = info.external_attr >> 16
                require(not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR),
                        "nonregular ZIP member rejected")
                if info.is_dir():
                    require(info.file_size == 0, "directory has payload")
                    member(info.filename, 0, None, mode)
                else:
                    with archive.open(info) as src:
                        member(info.filename, info.file_size, src, mode)
            if is_wheel:
                records = [n for n in payloads if n.count('/') == 1 and n.endswith('.dist-info/RECORD')]
                require(len(records) == 1, 'wheel RECORD count mismatch')
                record_name = records[0]
                require(payloads[record_name]['bytes'] <= 16 * 1024 * 1024, 'wheel RECORD too large')
                rows = list(csv.reader(io.StringIO(archive.read(record_name).decode('utf-8'))))
                seen = set()
                for row in rows:
                    require(len(row) == 3 and row[0] not in seen and row[0] in payloads, 'invalid/duplicate RECORD path')
                    seen.add(row[0])
                    if row[0] == record_name:
                        require(row[1:] == ['', ''], 'RECORD self hash must be empty')
                    else:
                        require(row[1].startswith('sha256='), 'RECORD must use SHA256')
                        expected = base64.urlsafe_b64encode(bytes.fromhex(payloads[row[0]]['sha256'])).rstrip(b'=').decode()
                        require(row[1] == 'sha256=' + expected and row[2] == str(payloads[row[0]]['bytes']), 'RECORD payload identity mismatch')
                require(seen == set(payloads), 'RECORD must cover every wheel payload')
    else:
        with tarfile.open(path, "r:gz") as archive:
            for info in archive:
                require(info.isfile() or info.isdir(), "nonregular TAR member rejected")
                src = archive.extractfile(info) if info.isfile() else None
                try:
                    member(info.name, info.size, src)
                finally:
                    if src:
                        src.close()
    return {"filename": path.name, "members": len(names), "expanded_bytes": total,
            "payload_sha256": hashlib.sha256(canonical(payloads)).hexdigest(), "payloads": payloads}


def validate_image_metadata(image, manifest_bytes, config_bytes):
    require(image["repository"] == "docker.io/library/python" and image["platform"] == "linux/amd64",
            "unapproved base image repository/platform")
    require("sha256:" + hashlib.sha256(manifest_bytes).hexdigest() == image["manifest_digest"],
            "OCI manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    require(manifest.get("schemaVersion") == 2 and isinstance(manifest.get("layers"), list), "invalid OCI manifest")
    config = manifest["config"]
    require(config["digest"] == "sha256:" + hashlib.sha256(config_bytes).hexdigest() and
            config["size"] == len(config_bytes), "OCI config identity mismatch")
    runtime = json.loads(config_bytes)
    require(runtime.get("os") == "linux" and runtime.get("architecture") == "amd64", "OCI platform mismatch")
    require("PYTHON_VERSION=" + image["python_version"] in runtime["config"]["Env"], "OCI Python version mismatch")
    require(len(runtime["rootfs"]["diff_ids"]) == len(manifest["layers"]), "OCI layers/diff IDs mismatch")
    total = config["size"]
    for layer in manifest["layers"]:
        require(HEX.fullmatch(layer.get("digest", "").removeprefix("sha256:")), "invalid layer digest")
        require(type(layer.get("size")) is int and layer["size"] > 0, "invalid layer size")
        total += layer["size"]
    require(total <= image["maximum_compressed_bytes"], "image compressed budget exceeded")
    require(image["compressed_bytes"] == total and image["registry_manifest_verified"] is True,
            "image metadata not independently admitted")
    return {"compressed_bytes": total, "config_digest": config["digest"], "layers": len(manifest["layers"])}


def clean_env():
    # Docker uses the existing local Unix socket, never a remote DOCKER_HOST or user config.
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/nonexistent",
            "DOCKER_CONFIG": "/nonexistent"}


def regular_tree(directory):
    directory = Path(directory).resolve(strict=True)
    require(directory.is_dir(), "not a staging directory")
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "symlink in staging tree")
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), "socket/device/special file in staging tree")
        if path.is_file():
            result[str(path.relative_to(directory))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def create_command(docker, image, name, inputs, outputs, work, limits, stage):
    require(stage in ("preflight", "build-a", "build-b", "runtime"), "unknown stage")
    require(re.fullmatch(r"kws-asr-[a-z0-9-]{1,48}", name), "invalid owned container name")
    require(re.fullmatch(r"docker.io/library/python@sha256:[0-9a-f]{64}", image), "image must be digest pinned")
    require(Path(docker).is_absolute(), "Docker client must be absolute")
    for p in (inputs, outputs, work):
        require(Path(p).is_absolute() and not any(c in str(p) for c in (",", ":", "\n")), "unsafe bind source")
    # /work is dedicated disk scratch for the complete CUDA-capable wheel closure.
    # Scratch/output binds have sampled caps; /tmp has a hard tmpfs byte cap.
    return [str(docker), "--host=unix:///var/run/docker.sock", "create", "--pull=never", "--name", name, "--label=com.kws.asr-runtime.owner=" + name,
            "--platform=linux/amd64", "--network=none", "--read-only", "--user=1000:1000",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--init", "--pids-limit=" + str(limits["container_pids"]),
            "--memory=" + str(limits["container_memory_bytes"]), "--memory-swap=" + str(limits["container_memory_bytes"]),
            "--cpus=" + str(limits["container_cpus"]), "--cgroupns=private", "--ipc=private",
            "--ulimit=nofile=1024:1024", "--ulimit=core=0:0", "--log-driver=local", "--log-opt=max-size=1m", "--log-opt=max-file=2",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=" + str(limits["container_tmpfs_bytes"]) + ",uid=1000,gid=1000,mode=0700",
            "--mount=type=bind,src=" + str(inputs) + ",dst=/inputs,readonly,bind-propagation=rprivate",
            "--mount=type=bind,src=" + str(outputs) + ",dst=/output,bind-propagation=rprivate",
            "--mount=type=bind,src=" + str(work) + ",dst=/work,bind-propagation=rprivate",
            "--workdir=/work", "--entrypoint=/usr/bin/env", image, "-i",
            "PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8", "HOME=/work/home", "TMPDIR=/tmp",
            "USER=kws-asr-runtime", "LOGNAME=kws-asr-runtime", "XDG_CACHE_HOME=/work/cache",
            "TORCHINDUCTOR_CACHE_DIR=/work/cache/torchinductor",
            "PYTHONDONTWRITEBYTECODE=1", "PYTHONHASHSEED=0", "SOURCE_DATE_EPOCH=1704067200",
            "PIP_CONFIG_FILE=/dev/null", "PIP_NO_INDEX=1", "PIP_DISABLE_PIP_VERSION_CHECK=1", "PIP_NO_CACHE_DIR=1",
            "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1",
            "HF_HOME=/work/hf", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
            "NUMBA_NUM_THREADS=1", "/usr/local/bin/python", "-I", "-S", "/inputs/container_stage.py", stage]


def verify_container_inspect(record, command, image_id, limits):
    require(isinstance(record, list) and len(record) == 1, "invalid container inspect")
    c = record[0]
    h = c["HostConfig"]
    require(c["Image"] == image_id, "container image ID changed")
    require(h["NetworkMode"] == "none" and h["ReadonlyRootfs"] and not h["Privileged"], "container isolation mismatch")
    require(h["CapDrop"] == ["ALL"] and not h.get("CapAdd") and h["SecurityOpt"] == ["no-new-privileges"], "capability policy mismatch")
    require(h["Memory"] == limits["container_memory_bytes"] and h["MemorySwap"] == h["Memory"] and
            h["NanoCpus"] == limits["container_cpus"] * 10**9 and h["PidsLimit"] == limits["container_pids"], "container resource mismatch")
    require(h["CgroupnsMode"] == "private" and h["IpcMode"] == "private" and not h.get("PidMode") and
            not h.get("Devices") and not h.get("DeviceRequests") and h.get("Init") is True, "namespace/device/init mismatch")
    require(c["Config"]["User"] == "1000:1000" and c["Config"]["Entrypoint"] == ["/usr/bin/env"], "entrypoint/user mismatch")
    label = next(x.split('=', 2)[2] for x in command if x.startswith('--label=com.kws.asr-runtime.owner='))
    require(c["Config"]["Labels"].get('com.kws.asr-runtime.owner') == label, 'container ownership label mismatch')
    image_index = next(i for i, arg in enumerate(command) if arg.startswith('docker.io/library/python@sha256:'))
    require(c['Config']['Cmd'] == command[image_index + 1:], 'container command mismatch')
    mounts = [m for m in c["Mounts"] if m["Type"] == "bind"]
    require(all(m["Type"] == "bind" or (m["Type"] == "tmpfs" and m["Destination"] == "/tmp") for m in c["Mounts"]), "unexpected mount type")
    require(len(mounts) == 3 and {x["Destination"] for x in mounts} == {"/inputs", "/output", "/work"}, "unexpected bind mount")
    for mount in mounts:
        expected = next(x for x in command if x.startswith("--mount=") and "dst=" + mount["Destination"] + "," in x)
        source = expected.split("src=", 1)[1].split(",", 1)[0]
        require(mount["Type"] == "bind" and mount["Source"] == source and
                mount["RW"] == (mount["Destination"] in ("/output", "/work")) and mount.get("Propagation") == "rprivate", "bind mount identity mismatch")
    require(set(h["Tmpfs"]) == {"/tmp"}, "tmpfs set mismatch")
    required_tmp = {'rw', 'noexec', 'nosuid', 'nodev', 'size=' + str(limits['container_tmpfs_bytes']), 'uid=1000', 'gid=1000', 'mode=0700'}
    require(set(h['Tmpfs']['/tmp'].split(',')) == required_tmp, 'tmpfs policy mismatch')
    return c


def summarize_state(c, stop_reason=None):
    state = c["State"]
    return {"exit_code": state.get("ExitCode"), "oom_killed": state.get("OOMKilled"),
            "running": state.get("Running"), "dead": state.get("Dead"), "stop_reason": stop_reason,
            "status": "pass" if stop_reason is None and state.get("ExitCode") == 0 and
            state.get("OOMKilled") is False and state.get("Running") is False else "failed"}


def bounded_command(args, timeout=30, maximum=MAX_LOG):
    """Absolute process-group deadline, bounded drain and prefix/tail diagnostics."""
    import selectors
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=clean_env(), stdin=subprocess.DEVNULL, start_new_session=True)
    start, prefix, tail, total = time.monotonic(), bytearray(), bytearray(), 0
    timed_out, killed_at = False, None
    stream_hash = hashlib.sha256()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    while selector.get_map():
        now = time.monotonic()
        if now - start > timeout and killed_at is None:
            timed_out, killed_at = True, now
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if killed_at is not None and now - killed_at > 1:
            break  # Descendant-held pipe cannot extend this absolute drain bound.
        for key, _ in selector.select(.05):
            chunk = os.read(key.fd, 65536)
            if not chunk:
                selector.unregister(key.fd)
                continue
            total += len(chunk)
            stream_hash.update(chunk)
            prefix.extend(chunk[:max(0, maximum // 2 - len(prefix))])
            tail.extend(chunk)
            if len(tail) > maximum // 2:
                del tail[:-maximum // 2]
    selector.close()
    process.stdout.close()
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    code = process.wait(timeout=2)
    # For short outputs combine non-overlapping bytes only.
    data = bytes(prefix) + (bytes(tail)[-(total - len(prefix)):] if total > len(prefix) else b'')
    if total > maximum:
        data = bytes(prefix) + b'\n[OUTPUT TRUNCATED; PREFIX AND TAIL RETAINED]\n' + bytes(tail)
    return {"returncode": code, "timeout": timed_out, "truncated": total > maximum,
            "bytes": total, "captured_stream_sha256": stream_hash.hexdigest(),
            "stream_complete": killed_at is None, "output": data.decode("utf-8", "replace")}


def docker_preflight(run=bounded_command):
    docker = shutil.which("docker", path="/usr/bin:/bin")
    require(docker is not None, "existing Docker client missing; no installation permitted")
    version = run([docker, "--host=unix:///var/run/docker.sock", "version", "--format={{json .}}"])
    require(version["returncode"] == 0 and not version["timeout"] and not version["truncated"], "existing Docker daemon access failed")
    v = json.loads(version["output"])
    info = run([docker, "--host=unix:///var/run/docker.sock", "info", "--format={{json .}}"])
    require(info["returncode"] == 0 and not info["timeout"] and not info["truncated"], "existing Docker info failed")
    i = json.loads(info["output"])
    require(i["OSType"] == "linux" and i["Architecture"] in ("x86_64", "amd64") and i["CgroupVersion"] == "2", "required Docker platform/cgroup2 unavailable")
    require(i.get("MemoryLimit") and i.get("SwapLimit") and i.get("CpuCfsQuota") and i.get("PidsLimit"), "required cgroup enforcement unavailable")
    # Do not publish raw info/config/auth/plugin, labels, host name, env or root paths.
    return {"docker": docker, "client_sha256": sha256_file(docker), "client_version": v["Client"]["Version"],
            "server_version": v["Server"]["Version"], "cgroup_version": i["CgroupVersion"],
            "cgroup_driver": i["CgroupDriver"], "storage_driver": i["Driver"], "security_options": i["SecurityOptions"]}


def host_runtime_snapshot():
    import importlib.metadata
    import site
    packages = sorted((d.metadata.get('Name', ''), d.version) for d in
                      importlib.metadata.Distribution.discover(path=site.getsitepackages()))
    return {'python_sha256': sha256_file(sys.executable),
            'system_distribution_metadata_sha256': hashlib.sha256(canonical(packages)).hexdigest()}


def candidate_status(root=ROOT):
    admission = load_json(root / "locks/admission.json")
    missing = [n for n in ("inventory.json", "build-plan.json", "image.manifest.json", "image.config.json")
               if not (root / "locks" / n).is_file()]
    return {"schema": SCHEMA, "evidence_class": "source_preparation_only", "execution_enabled": admission["execution_enabled"],
            "missing_required_inputs": missing,
            "activation_blockers": (["execution_not_approved"] if not admission["execution_enabled"] else []) +
                (["source_build_plan_independent_review_pending"] if (root / "locks/build-plan.json").is_file() and not load_json(root / "locks/build-plan.json").get("independently_reviewed") else []),
            "target_speech_model_weights_loaded": 0, "user_audio_loaded": 0, "dependency_installs": 0,
            "interpretation": "Candidate source/fixtures are not runtime, ASR, BF16 model, or device performance evidence."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true", help="report unresolved inputs without networking")
    parser.add_argument("--docker-readonly", action="store_true", help="read existing daemon only; never pull/create/start")
    args = parser.parse_args(argv)
    record = candidate_status()
    if args.docker_readonly:
        record["docker_preflight"] = docker_preflight()
    sys.stdout.buffer.write(canonical(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
