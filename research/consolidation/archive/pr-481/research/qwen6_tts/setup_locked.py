"""Future, single-attempt setup from local locks. Importing performs no I/O.

Run under the setup supervisor with CPython 3.12.14. No index lookup, cache,
download retry, dependency resolution, or model execution is implemented here.
Builds are offline-configured; this does not claim kernel network isolation.
"""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import venv
import zipfile

GIB = 1024 ** 3
DOWNLOAD_CAP = 4 * GIB
INSTALL_CAP = 3 * GIB
BUILD_CAP = 2 * GIB
WORKSPACE_CAP = 10 * GIB
INITIAL_FREE = 12 * GIB
SETUP_SECONDS = 1200
PYTHON_VERSION = "3.12.14"
TOOLS = {"build": "1.6.1", "packaging": "26.3", "pyproject-hooks": "1.3.3",
         "setuptools": "81.0.0", "wheel": "0.48.0"}
SDISTS = {"sox"}
SOURCE_MEMBER_COUNT = 17
MODEL_IDS = {"qwen_tts": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def normalized_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def safe_relative(value):
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or "\\" in value or
            path.is_absolute() or any(p in (".", "..", "") for p in value.split("/"))):
        raise ValueError("Unsafe relative path")
    return path


def file_identity(path):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"Expected unaliased regular file: {path.name}")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    after = path.lstat()
    stamp = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns, x.st_nlink)
    if stamp(before) != stamp(after) or size != after.st_size:
        raise ValueError("File changed during verification")
    return {"bytes": size, "sha256": digest.hexdigest()}


def require_file(path, size, digest):
    observed = file_identity(path)
    if observed != {"bytes": size, "sha256": digest}:
        raise ValueError(f"File size/SHA256 mismatch: {Path(path).name}")
    return observed


def validate_download(row):
    name = row["filename"]
    safe_relative(name)
    size = row.get("size_bytes", row.get("bytes"))
    if type(size) is not int or not 0 < size <= DOWNLOAD_CAP:
        raise ValueError("Invalid declared download size")
    if not isinstance(row["sha256"], str) or not SHA256.fullmatch(row["sha256"]):
        raise ValueError("Invalid declared SHA256")
    url = row.get("url", row.get("source_url"))
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in
            {"files.pythonhosted.org", "download-r2.pytorch.org", "download.pytorch.org", "huggingface.co"}
            or parsed.username or parsed.password or parsed.fragment or parsed.query):
        raise ValueError("Download must use the exact HTTPS artifact URL")
    if urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]) != PurePosixPath(name).name:
        raise ValueError("URL and filename disagree")
    return name, size, url


def validate_locks(runtime, models):
    rows = runtime["files"]
    if len(rows) != 92 or len({normalized_name(r["name"]) for r in rows}) != len(rows):
        raise ValueError("Expected 92 distinct locked package inputs")
    if any(len(safe_relative(r["filename"]).parts)!=1 for r in rows):
        raise ValueError("Package filename must be a basename")
    if len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Duplicate input filename")
    if {r["name"] for r in rows if r["kind"] == "sdist"} != SDISTS:
        raise ValueError("The one source input changed")
    if any(r["kind"] not in {"wheel", "sdist"} for r in rows):
        raise ValueError("Unknown package artifact kind")
    if {r["name"]: r["version"] for r in rows if r["build_tool"]} != TOOLS:
        raise ValueError("Build tool versions changed")
    torch = [r for r in rows if r["name"] == "torch"]
    if len(torch) != 1 or torch[0]["version"] != "2.11.0+cpu":
        raise ValueError("Exact Torch 2.11.0+cpu required; suffix is significant")
    if runtime["target_environment"]["python_full_version"] != PYTHON_VERSION:
        raise ValueError("Python lock changed")
    if set(models) != set(MODEL_IDS):
        raise ValueError("Expected one TTS model lock")
    all_inputs = list(rows)
    for name, plan in models.items():
        lock = plan["asset_lock"]
        if plan["model_id"] != MODEL_IDS[name] or lock["model_id"] != MODEL_IDS[name]:
            raise ValueError("Model identity changed")
        if canonical_sha(lock) != plan["asset_lock_sha256"]:
            raise ValueError("Model asset lock digest mismatch")
        if canonical_sha(plan["source_lock"]) != plan["source_lock_sha256"]:
            raise ValueError("Model source lock digest mismatch")
        names = [r["filename"] for r in lock["files"]]
        if len(set(names)) != len(names) or len(names) != 13:
            raise ValueError("Model asset inventory changed")
        for row in lock["files"]:
            expected = f"https://huggingface.co/{lock['model_id']}/resolve/{lock['revision']}/{row['filename']}"
            if row["source_url"] != expected or not re.fullmatch(r"[0-9a-f]{40}", lock["revision"]):
                raise ValueError("Model revision URL changed")
        source_keys = [(r["distribution"], r["relative_path"]) for r in plan["source_lock"]]
        if len(source_keys) != len(set(source_keys)) or len(source_keys) != SOURCE_MEMBER_COUNT:
            raise ValueError("Source member inventory changed")
        for row in plan["source_lock"]:
            safe_relative(row["relative_path"])
            if not SHA256.fullmatch(row["sha256"]):
                raise ValueError("Invalid source member digest")
        all_inputs.extend(lock["files"])
    total = sum(validate_download(r)[1] for r in all_inputs)
    if total > DOWNLOAD_CAP:
        raise ValueError("Declared package/model inputs exceed download cap")
    return total


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class Downloader:
    """Count response bytes consumed; never request bytes beyond the total cap."""
    def __init__(self, cap=DOWNLOAD_CAP, opener=None):
        self.cap = cap
        self.actual_bytes = 0
        self.attempted = set()
        self.receipts = []
        self.opener = opener or urllib.request.build_opener(NoRedirect())

    def _open(self, url):
        hosts = []
        for _ in range(9):
            if url in self.attempted:
                raise ValueError("Repeated URL/download attempt prohibited")
            self.attempted.add(url)
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != "https" or parsed.username or parsed.password:
                raise ValueError("Unsafe redirect")
            hosts.append(parsed.hostname)
            request = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "fixed30-locked-setup/1"})
            try:
                return self.opener.open(request, timeout=60), hosts
            except urllib.error.HTTPError as error:
                location = error.headers.get("Location")
                error.close()
                if error.code not in (301, 302, 303, 307, 308) or not location:
                    raise
                url = urllib.parse.urljoin(url, location)
        raise ValueError("Too many artifact redirects")

    def fetch(self, row, destination):
        name, expected_size, url = validate_download(row)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        if destination.exists() or partial.exists():
            raise FileExistsError("Download destination already exists; no cache/reuse")
        if self.actual_bytes + expected_size > self.cap:
            raise ValueError("Download would exceed remaining byte budget")
        record = {"filename": name, "url": url, "bytes": 0}
        self.receipts.append(record)
        digest = hashlib.sha256()
        try:
            response, hosts = self._open(url)
            record["redirect_hosts"] = hosts
            with response, partial.open("xb") as output:
                if response.status != 200:
                    raise ValueError("Artifact response must be HTTP 200")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Encoded response is not the exact locked artifact")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != expected_size:
                    raise ValueError("Response Content-Length differs from lock")
                while True:
                    remaining = self.cap - self.actual_bytes
                    if remaining <= 0:
                        if declared is not None and record["bytes"] == expected_size:
                            break
                        raise ValueError("Cannot confirm download EOF within byte budget")
                    block = response.read(min(1024 * 1024,
                                              expected_size - record["bytes"] + 1,
                                              remaining))
                    if not block:
                        break
                    self.actual_bytes += len(block)
                    record["bytes"] += len(block)
                    digest.update(block)
                    if self.actual_bytes > self.cap or record["bytes"] > expected_size:
                        raise ValueError("Actual download byte budget exceeded")
                    output.write(block)
            record["sha256"] = digest.hexdigest()
            if record["bytes"] != expected_size or record["sha256"] != row["sha256"]:
                raise ValueError("Downloaded artifact size/SHA256 mismatch")
            partial.rename(destination)
            return record
        finally:
            if partial.exists():
                partial.unlink()


def tree_bytes(root):
    """Logical file bytes; symlinks are measured but never followed."""
    if not Path(root).exists():
        return 0
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            total += (Path(base) / name).lstat().st_size
        for name in dirs:
            entry = Path(base) / name
            if entry.is_symlink():
                total += entry.lstat().st_size
    return total


def disk_measurements(root):
    root = Path(root)
    current = {"installed_bytes": tree_bytes(root / "venv"),
               "buildtmp_bytes": tree_bytes(root / "buildtmp"),
               "workspace_bytes": tree_bytes(root), "free_bytes": shutil.disk_usage(root).free}
    for key, cap in (("installed_bytes", INSTALL_CAP), ("buildtmp_bytes", BUILD_CAP), ("workspace_bytes", WORKSPACE_CAP)):
        if current[key] > cap:
            raise ValueError(f"{key} exceeds declared cap")
    return current


def require_python():
    observed = {"implementation": platform.python_implementation(), "version": platform.python_version(),
                "system": platform.system(), "machine": platform.machine(), "byteorder": sys.byteorder}
    if observed != {"implementation": "CPython", "version": PYTHON_VERSION,
                    "system": "Linux", "machine": "x86_64", "byteorder": "little"}:
        raise RuntimeError("Exact CPython 3.12.14 on little-endian Linux x86_64 required")
    return observed


def offline_environment(root):
    root = Path(root)
    environment = {"PATH": str(root / "venv/bin") + os.pathsep + os.defpath,
                   "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                   "TMPDIR": str(root / "buildtmp"), "XDG_CACHE_HOME": str(root / "buildtmp/cache"),
                   "PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1", "PIP_NO_DEPS": "1",
                   "PIP_NO_BUILD_ISOLATION": "1", "PIP_NO_CACHE_DIR": "1",
                   "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                   "HF_HOME": str(root / "buildtmp/hf"), "CUDA_VISIBLE_DEVICES": "",
                   "MAKEFLAGS": "-j2", "MAX_JOBS": "2", "CMAKE_BUILD_PARALLEL_LEVEL": "2",
                   "NPY_NUM_BUILD_JOBS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "RAYON_NUM_THREADS": "2"}
    return environment


def install_command(python, wheels):
    if not wheels or any(not Path(path).is_absolute() or Path(path).suffix != ".whl" for path in wheels):
        raise ValueError("Only explicit local wheels can be installed")
    return [str(python), "-I", "-m", "pip", "--isolated", "install", "--no-index", "--no-deps",
            "--no-cache-dir", "--disable-pip-version-check", "--no-compile", *map(str, wheels)]


class Commands:
    def __init__(self, root, receipt, started):
        self.root, self.receipt, self.started = Path(root), receipt, started

    def run(self, argv, *, stage, package=None, cwd=None, capture=False, input_text=None, metadata_diagnostic=None):
        from setup_diagnostics import STAGES, safe_exception, stderr_diagnostic
        if stage not in STAGES: raise ValueError('Unknown setup command stage')
        remaining=SETUP_SECONDS-(time.monotonic()-self.started)
        if remaining<=0: raise TimeoutError('Setup wall budget exhausted')
        index=len(self.receipt['commands']);log=self.root/'logs'/f'setup-{index:03d}.log'
        error_log=self.root/'logs'/f'setup-{index:03d}.stderr.log'
        record={'argv':list(map(str,argv)),'log':str(log.relative_to(self.root)),
                'stage':stage,'package':package,'command_role':STAGES[stage],'status':'started','returncode':None}
        self.receipt['commands'].append(record)
        self.receipt['phase']={'stage':stage,'package':package}
        _save_receipt(self.root/'setup-receipt.json',self.receipt)
        tick=time.monotonic()
        try:
            with log.open('xb') as output,error_log.open('xb') as errors:
                result=subprocess.run(argv,cwd=cwd or self.root,env=offline_environment(self.root),
                    input=input_text,text=True,stdout=subprocess.PIPE if capture else output,
                    stderr=errors,timeout=remaining,check=False)
            record.update(returncode=result.returncode,status='success' if result.returncode==0 else 'failed')
            self.receipt['disk_checkpoints'].append(disk_measurements(self.root))
            if result.returncode: raise RuntimeError('Setup subprocess failed; safe diagnostics retained')
            return result.stdout if capture else None
        except BaseException as error:
            record.update(status='failed',exception_type=safe_exception(error))
            raise
        finally:
            record['wall_seconds']=time.monotonic()-tick
            record['stderr_diagnostic']=stderr_diagnostic(error_log)
            if metadata_diagnostic is not None and Path(metadata_diagnostic).is_file():
                raw=Path(metadata_diagnostic).read_bytes()
                if len(raw)>65536: raise ValueError('Bounded metadata diagnostic exceeded')
                record['metadata_diagnostic']=json.loads(raw)
            _save_receipt(self.root/'setup-receipt.json',self.receipt)


class MetadataMismatch(ValueError):
    def __init__(self,expected,observed):
        from setup_diagnostics import metadata_difference
        self.diagnostic=metadata_difference(expected,observed)
        super().__init__('Built/installed normalized Requires-Dist differs from the saved lock')


def normalize_requirements(values):
    """Called only inside the locked venv (or pure unit tests)."""
    from packaging.requirements import Requirement
    normalized = []
    for text in values:
        requirement = Requirement(text)
        normalized.append((normalized_name(requirement.name),
                           tuple(sorted(normalized_name(x) for x in requirement.extras)),
                           str(requirement.specifier), requirement.url or "",
                           str(requirement.marker) if requirement.marker else ""))
    return sorted(normalized)


def check_metadata(raw, row, *, normalize=False):
    metadata = email.parser.BytesParser().parsebytes(raw)
    if metadata.defects or any(len(metadata.get_all(key,[]))!=1 for key in ("Name","Version")):
        raise ValueError("Metadata Name/Version must be unique and parse cleanly")
    if (normalized_name(metadata.get("Name", "")) != normalized_name(row["name"])
            or metadata.get("Version") != row["version"]):
        raise ValueError("Exact package metadata name/version mismatch")
    observed = {"name": row["name"], "version": metadata["Version"],
                "metadata_sha256": hashlib.sha256(raw).hexdigest(), "metadata_bytes": len(raw),
                "requires_dist": metadata.get_all("Requires-Dist", [])}
    if row["kind"] == "wheel" and observed["metadata_sha256"] != row["metadata_sha256"]:
        raise ValueError("Wheel METADATA hash differs from the saved lock")
    if normalize:
        observed["normalized_requires_dist"] = normalize_requirements(observed["requires_dist"])
        if observed["normalized_requires_dist"] != normalize_requirements(row["requires_dist"]):
            raise MetadataMismatch(row["requires_dist"],observed["requires_dist"])
    return observed


def inspect_built_metadata(path,row,diagnostic_path):
    from setup_diagnostics import safe_exception
    raw=wheel_metadata(path,row)
    diagnostic={'schema':'fixed30-built-metadata-diagnostic-v1','metadata_sha256':hashlib.sha256(raw).hexdigest(),
                'metadata_bytes':len(raw),'status':'started'}
    try:
        result=check_metadata(raw,row,normalize=True)
        diagnostic.update(status='success',code='full_requires_dist_match')
        return result
    except MetadataMismatch as error:
        diagnostic.update(status='failed',exception_type=safe_exception(error),code='requires_dist_mismatch',difference=error.diagnostic)
        raise
    except Exception as error:
        diagnostic.update(status='failed',exception_type=safe_exception(error),code='metadata_validation_failed')
        raise
    finally:
        Path(diagnostic_path).write_text(json.dumps(diagnostic,sort_keys=True,allow_nan=False))


def wheel_metadata(path, row=None):
    from wheel_identity import wheel_members, select_metadata
    with zipfile.ZipFile(path) as archive:
        members=wheel_members(archive)
        name=select_metadata(members,row['name'] if row else None,row['version'] if row else None)
        dist=name.split('/')[0]
        if any(n.split('/')[0].endswith('.dist-info') and n.split('/')[0]!=dist for n in members):
            raise ValueError('ZIP_FOREIGN_DIST_INFO')
        member=members[name]
        if member.file_size>2*1024*1024: raise ValueError('Oversized wheel METADATA')
        return archive.read(member)


def inspect_source_archive(path, destination):
    """Extract regular members only, rejecting traversal, aliases and devices."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Source extraction cannot be repeated")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names, roots, expanded = set(), set(), 0
        for member in members:
            relative = safe_relative(member.name.rstrip("/"))
            if str(relative) in names or not (member.isdir() or member.isfile()):
                raise ValueError("Source archive contains duplicate/link/device member")
            names.add(str(relative))
            roots.add(relative.parts[0])
            expanded += member.size
            if expanded > BUILD_CAP:
                raise ValueError("Expanded source exceeds build temporary cap")
        if len(roots) != 1:
            raise ValueError("Source archive must have one top-level directory")
        destination.mkdir(parents=True)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        source_root = destination / next(iter(roots))
        if not source_root.is_dir():
            raise ValueError("Source root is not a directory")
        config = source_root / "pyproject.toml"
        system = tomllib.loads(config.read_text())["build-system"] if config.exists() else {
            "requires": ["setuptools>=40.8.0"], "build-backend": "setuptools.build_meta:__legacy__"}
        if system.get("build-backend") not in {"setuptools.build_meta", "setuptools.build_meta:__legacy__"} or system.get("backend-path"):
            raise ValueError("Source requires an unreviewed build backend")
        provenance = {"archive": file_identity(path), "expanded_source_bytes": expanded,
                      "build_system": system,
                      "build_files": {name: file_identity(source_root / name) for name in
                                      ("pyproject.toml", "setup.py", "setup.cfg") if (source_root / name).is_file()}}
        return source_root, provenance


def verify_build_requirements(requirements):
    from packaging.requirements import Requirement
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        name = normalized_name(requirement.name)
        if (name not in TOOLS or requirement.url or requirement.extras or
                TOOLS[name] not in requirement.specifier):
            raise ValueError("Build requirement is not satisfied by locked build tools")
        if importlib.metadata.version(name) != TOOLS[name]:
            raise ValueError("Installed build tool version drift")


def _read_locks(lock_path=None, model_lock_path=None):
    public = Path(__file__).resolve().parent
    lock_path = Path(lock_path or public / "runtime-lock.json")
    model_lock_path = Path(model_lock_path or public / "model-locks.json")
    runtime, models = json.loads(lock_path.read_text()), json.loads(model_lock_path.read_text())
    validate_locks(runtime, models)
    identities = {"runtime_lock": file_identity(lock_path), "model_locks": file_identity(model_lock_path)}
    return runtime, models, identities


def _installed_metadata(distribution, row=None):
    from wheel_identity import select_metadata
    expected=row or {'name':distribution.metadata['Name'],'version':distribution.version}
    name=select_metadata([str(f) for f in distribution.files or []],expected['name'],expected['version'])
    return Path(distribution.locate_file(name))


def verify_setup(root, lock_path=None, model_lock_path=None, *, _during_setup=False):
    """Read-only gate. Call with runtime/venv/bin/python before model imports."""
    root = Path(root).resolve()
    require_python()
    if Path(sys.prefix).resolve() != root / "venv":
        raise RuntimeError("Verification must run with this setup's venv Python")
    runtime, models, identities = _read_locks(lock_path, model_lock_path)
    receipt = json.loads((root / "setup-receipt.json").read_text())
    if "exception" in receipt or (not _during_setup and "verification" not in receipt):
        raise ValueError("Setup has no completed verification receipt")
    if identities != receipt["locks"]:
        raise ValueError("Setup lock files changed")
    expected = {normalized_name(r["name"]): r for r in runtime["files"]}
    distributions = {}
    for dist in importlib.metadata.distributions():
        name = normalized_name(dist.metadata["Name"])
        if name in distributions:
            raise ValueError("Duplicate installed distribution")
        distributions[name] = dist
    if set(distributions) != set(expected) | {"pip"}:
        raise ValueError("Missing or extra installed distribution")
    installed, sources, assets = [], [], {}
    for name, row in expected.items():
        dist = distributions[name]
        if dist.version != row["version"]:
            raise ValueError("Exact installed version mismatch; local suffixes are significant")
        path = _installed_metadata(dist,row)
        if not path.resolve().is_relative_to(root / "venv"):
            raise ValueError("Distribution escapes the venv")
        raw = path.read_bytes()
        observed = check_metadata(raw, row, normalize=True)
        wheel_record = receipt["wheels"][name]
        wheel = root / wheel_record["path"]
        require_file(wheel, wheel_record["bytes"], wheel_record["sha256"])
        if hashlib.sha256(wheel_metadata(wheel,row)).hexdigest() != observed["metadata_sha256"]:
            raise ValueError("Installed METADATA differs from retained local wheel")
        if observed["metadata_sha256"] != wheel_record["metadata_sha256"]:
            raise ValueError("Retained built wheel METADATA digest changed")
        if row["kind"] == "sdist":
            require_file(root / wheel_record["retained_metadata"],
                         observed["metadata_bytes"], observed["metadata_sha256"])
        if row["kind"] == "wheel":
            require_file(wheel, row["size_bytes"], row["sha256"])
        installed.append(observed)
    pip = distributions["pip"]
    pip_hash = hashlib.sha256(_installed_metadata(pip).read_bytes()).hexdigest()
    if (pip.version != receipt["bootstrap_pip"]["version"] or
            pip_hash != receipt["bootstrap_pip"]["metadata_sha256"]):
        raise ValueError("Bundled ensurepip bootstrap provenance changed")
    for name, plan in models.items():
        directory = root / "models" / name
        rows = plan["asset_lock"]["files"]
        if directory.is_symlink() or {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()} != {r["filename"] for r in rows}:
            raise ValueError("Missing or extra model asset")
        assets[name] = [{"filename": row["filename"], **require_file(directory / row["filename"], row["bytes"], row["sha256"])} for row in rows]
        for row in plan["source_lock"]:
            dist = distributions[normalized_name(row["distribution"])]
            path = Path(dist.locate_file(row["relative_path"]))
            if not path.resolve().is_relative_to(root / "venv"):
                raise ValueError("Source member escapes the venv")
            observed = file_identity(path)
            if observed["sha256"] != row["sha256"]:
                raise ValueError("Installed source member differs from saved source pin")
            sources.append({**row, "bytes": observed["bytes"]})
    return {"installed": installed, "source_members": sources, "model_assets": assets,
            "disk": disk_measurements(root), "python": require_python()}


def _save_receipt(path, receipt):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _helper_command(python, script, body):
    return [str(python), "-I", "-B", "-c", "import json,pathlib,runpy,sys; sys.path.insert(0,str(pathlib.Path(sys.argv[1]).resolve().parent)); m=runpy.run_path(sys.argv[1]); " + body, str(script)]


def setup(root):
    started = time.monotonic()
    python_identity = require_python()
    runtime, models, identities = _read_locks()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "setup-receipt.json"
    managed = ("downloads", "models", "venv", "buildtmp", "built-wheels")
    if receipt_path.exists() or any((root / name).exists() for name in managed):
        raise FileExistsError("Setup already attempted; reuse/retry is prohibited")
    initial_free = shutil.disk_usage(root).free
    if initial_free < INITIAL_FREE:
        raise ValueError("At least 12 GiB initially free is required")
    receipt = {"schema": "fixed30-setup-receipt-v1", "python": python_identity,
               "python_executable": str(Path(sys.executable).resolve()), "locks": identities,
               "started_unix_seconds": time.time(), "initial_free_bytes": initial_free,
               "network_configuration": "offline_configured_not_kernel_isolated",
               "commands": [], "wheels": {}, "builds": [], "disk_checkpoints": []}
    with receipt_path.open("x") as output:
        json.dump(receipt, output)
    downloader = Downloader()
    commands = Commands(root, receipt, started)
    script = Path(__file__).resolve()
    python = root / "venv/bin/python"
    try:
        for name in managed:
            if name != "venv":
                (root / name).mkdir()
        (root / "logs").mkdir(exist_ok=True)
        for row in runtime["files"]:
            receipt['phase']={'stage':'download_package','package':row['name']}
            _save_receipt(receipt_path,receipt)
            downloader.fetch(row, root / "downloads" / row["filename"])
            receipt["disk_checkpoints"].append(disk_measurements(root))
        # Bootstrap only from the wheel bundled with the exact Python interpreter.
        import ensurepip
        bundled = list((Path(ensurepip.__file__).parent / "_bundled").glob("pip-*.whl"))
        if len(bundled) != 1:
            raise ValueError("Cannot identify exactly one bundled ensurepip wheel")
        pip_raw = wheel_metadata(bundled[0],{"name":"pip","version":ensurepip.version()})
        pip_meta = email.parser.BytesParser().parsebytes(pip_raw)
        receipt["bootstrap_pip"] = {"source": "exact_python_bundled_ensurepip_not_independent_lock",
                                     "filename": bundled[0].name, **file_identity(bundled[0]),
                                     "version": pip_meta["Version"], "metadata_sha256": hashlib.sha256(pip_raw).hexdigest()}
        venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(root / "venv")
        commands.run([str(python), "-I", "-m", "ensurepip", "--default-pip"],stage="bootstrap_pip",package="pip")
        wheels = []
        for row in runtime["files"]:
            if row["kind"] != "wheel":
                continue
            path = root / "downloads" / row["filename"]
            observed = check_metadata(wheel_metadata(path,row), row)
            receipt["wheels"][normalized_name(row["name"])] = {
                "path": str(path.relative_to(root)), **file_identity(path), **observed}
            wheels.append(path)
        tool_paths = [root / "downloads" / r["filename"] for r in runtime["files"] if r["build_tool"]]
        commands.run(install_command(python, tool_paths),stage="install_build_tools")
        # All remaining prebuilt runtime wheels are explicit local paths, once each.
        commands.run(install_command(python, [p for p in wheels if p not in tool_paths]),stage="install_runtime_wheels")
        built = []
        for row in runtime["files"]:
            if row["kind"] != "sdist":
                continue
            receipt['phase']={'stage':'extract_source','package':row['name']}
            _save_receipt(receipt_path,receipt)
            source, provenance = inspect_source_archive(root / "downloads" / row["filename"], root / "buildtmp" / row["name"])
            provenance["name"] = row["name"]
            receipt["builds"].append(provenance)
            commands.run(_helper_command(python, script,
                         "m['verify_build_requirements'](json.load(sys.stdin))"),
                         input_text=json.dumps(provenance["build_system"]["requires"]),stage="check_build_requirements",package=row["name"])
            out = root / "built-wheels" / row["name"]
            out.mkdir()
            commands.run([str(python), "-I", "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out), str(source)],stage="build_wheel",package=row["name"])
            generated = list(out.glob("*.whl"))
            if len(generated) != 1:
                raise ValueError("Build must produce exactly one wheel in its single attempt")
            path = generated[0]
            diagnostic_path=root/'logs'/(row['name']+'.metadata-diagnostic.json')
            body = "x=json.load(sys.stdin); print(json.dumps(m['inspect_built_metadata'](x['path'],x['row'],x['diagnostic_path'])))"
            observed = json.loads(commands.run(_helper_command(python, script, body),capture=True,
                input_text=json.dumps({'path':str(path),'row':row,'diagnostic_path':str(diagnostic_path)}),
                stage='verify_built_metadata',package=row['name'],metadata_diagnostic=diagnostic_path))
            metadata_path = out / "METADATA"
            metadata_path.write_bytes(wheel_metadata(path,row))
            receipt["wheels"][normalized_name(row["name"])] = {
                "path": str(path.relative_to(root)), **file_identity(path), **observed,
                "retained_metadata": str(metadata_path.relative_to(root))}
            provenance["wheel"] = dict(receipt["wheels"][normalized_name(row["name"])])
            built.append(path)
            shutil.rmtree(source.parent)
        commands.run(install_command(python, built),stage="install_built_wheels")
        for name, plan in models.items():
            directory = root / "models" / name
            directory.mkdir()
            for row in plan["asset_lock"]["files"]:
                receipt['phase']={'stage':'download_model_asset','package':name}
                _save_receipt(receipt_path,receipt)
                downloader.fetch(row, directory / row["filename"])
                receipt["disk_checkpoints"].append(disk_measurements(root))
        _save_receipt(receipt_path, receipt)
        body = "print(json.dumps(m['verify_setup'](sys.argv[2],_during_setup=True)))"
        receipt["verification"] = json.loads(commands.run(_helper_command(python, script, body) + [str(root)], capture=True,stage="verify_setup"))
    except BaseException as error:
        receipt["exception"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["downloads"] = downloader.receipts
        receipt["actual_download_bytes"] = downloader.actual_bytes
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["ended_unix_seconds"] = time.time()
        _save_receipt(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "runtime")
    parser.add_argument("--verify", action="store_true", help="Read-only recheck using the setup venv Python")
    arguments = parser.parse_args(argv)
    if arguments.verify:
        print(json.dumps(verify_setup(arguments.root), sort_keys=True))
    else:
        setup(arguments.root)


if __name__ == "__main__":
    main()
