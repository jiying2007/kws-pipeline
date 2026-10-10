"""Source-screen bindings for retained locked setup; stdlib-only at import time.

Run in the admitted official CPython 3.12.14 container, with /code read-only:
  python -I -B setup_adapter.py --profile tts --stage setup --runtime /runtime
  /runtime/venv/bin/python -I -B setup_adapter.py --profile tts --stage verify --runtime /runtime

The host owns image provenance, admission and hard limits. Setup allows bounded
network acquisition; verify requires network-none. This is not a host launcher.
No old runner or arbitrary helper body is executed by this adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import stat
import sys

ROOT = Path(__file__).resolve().parent
GIB = 1024 ** 3
IMAGE_TRANSFER_RESERVE = 128 * 1024 ** 2
TTS_TRANSFER_CAP = 6 * GIB
TTS_INPUT_CAP = TTS_TRANSFER_CAP - IMAGE_TRANSFER_RESERVE
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_REVISION = "5ecdb67327fd37bb2e042aab12ff7391903235d3"
MODEL_LOCK_SHA256 = "747562e07b4788bcae25c13004139048741e18f07f39eca291891010f459a25b"
CACHE_KEYS = ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME",
              "NUMBA_CACHE_DIR", "TORCH_HOME", "MPLCONFIGDIR", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR")
# These are the exact retained files needed in the read-only staged code tree.
RETAINED_PINS = {
    "tts": {
        "setup_locked.py": "aac8d8e66504444f2965aa8d95dcfef9e8c6fa63d74f2eade15ca1290bc6b4b3",
        "runtime-lock.json": "6a707c2fa2c2ae8f842176a7f9c8301186d51fddef167c413160c7f2816613de",
        "model-locks.json": "c206556a633b7d04c334494fc5a82934a19d572c125982c689f18f8b2d472110",
        "setup_diagnostics.py": "7c583ba0ed6f0e90f72ab3b5c107e3666a32d25359ce13ffc284d545f7d1dd5f",
        "wheel_identity.py": "623ee07a0d8f85ba1d7218a56fcd0e270df8c6e2c3e353dd52b3184ca915d8d5",
        "import_preflight.py": "30cfe60347ccc6074d98a21406fccee3e3be5c54822aa8d874aae3655693ae1c",
        "runtime_gate.py": "f4333b694bb1bc261e4bdeea6d586dc9f76b1afce6ce0af9dfcdb74d57d12815",
        "generate_six.py": "25530b89fa3f58f44ceb8d3c2a8f26f3b37651c7173f62414dc96142919ea03e",
    },
    "asr": {
        "setup_locked.py": "d7c580b59b0bd249bc62c9093b9de5f8842170510cf11e7e61c29057e6287af4",
        "runtime-lock.json": "2d496a76bd0e044a6812bfc9afdd8965474bf135bd21f3a24f71695638d1183f",
        "model-locks.json": "8b0813aba3c2447c7dc30b47e856e4ce9a619a98d32ed6dfa9f44dc6646714e4",
        "setup_diagnostics.py": "1b98e253d7054d0bfcd5ec128cb3a331bb05de48776330e84ea348694b3a1bf2",
        "wheel_identity.py": "623ee07a0d8f85ba1d7218a56fcd0e270df8c6e2c3e353dd52b3184ca915d8d5",
        "import_preflight.py": "30cfe60347ccc6074d98a21406fccee3e3be5c54822aa8d874aae3655693ae1c",
    },
}
HELPER_BODIES = {
    "m['verify_build_requirements'](json.load(sys.stdin))": "requirements",
    "x=json.load(sys.stdin); print(json.dumps(m['inspect_built_metadata'](x['path'],x['row'],x['diagnostic_path'])))": "metadata",
    "print(json.dumps(m['verify_setup'](sys.argv[2],_during_setup=True,_before_models=True)))": "verify_before_models",
    "print(json.dumps(m['verify_setup'](sys.argv[2],_during_setup=True)))": "verify_after_models",
    "m['execute'](sys.argv[2],m['pathlib'].Path(sys.argv[3]),m['pathlib'].Path(sys.argv[3])/'import-preflight.json')": "preflight",
}
BOOTSTRAP = ("import pathlib,runpy,sys; p=pathlib.Path(sys.argv[1]).resolve(); "
             "sys.path.insert(0,str(p.parent)); m=runpy.run_path(str(p)); "
             "m['_run_child'](sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5:])")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(path, expected=None):
    path = Path(path)
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "Unaliased regular code/lock file required")
    raw = path.read_bytes()
    after = path.lstat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
    _require(stamp(before) == stamp(after) and len(raw) == after.st_size, "Code/lock changed while reading")
    identity = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    _require(expected is None or identity["sha256"] == expected, "Retained code/lock pin changed: " + path.name)
    return identity


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _scope(stage):
    _require(stage in ("setup", "verify"), "Unknown setup stage")
    scope = _load("runtime_scope", ROOT / "runtime_scope.py")
    # Only the networked setup process tree uses this explicit exception.
    return scope.verify_runtime_scope(network_required=False) if stage == "setup" else scope.verify_runtime_scope()


def converted_tts_models(retained, source, canonical_sha):
    """Pure conversion. Source pins remain exactly the audited distribution pins."""
    _require(source["model_id"] == MODEL_ID and source["revision"] == MODEL_REVISION,
             "Source-screen VoiceDesign identity changed")
    _require(len(source["files"]) == 13, "Exactly 13 source-screen model assets required")
    _require(sum(row["size_bytes"] for row in source["files"]) == source["total_snapshot_bytes"],
             "Source-screen declared model bytes changed")
    _require(all(row["observed_repo_revision"] == MODEL_REVISION for row in source["files"]),
             "Mixed source-screen model revisions")
    lock = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
            "files": [{"filename": row["path"], "bytes": row["size_bytes"],
                       "sha256": row["sha256"], "source_url": row["source_url"]}
                      for row in sorted(source["files"], key=lambda r: r["path"])]}
    old = retained["qwen_tts"]
    _require(canonical_sha(old["source_lock"]) == old["source_lock_sha256"], "Retained source pin mismatch")
    return {"qwen_tts": {"model_id": MODEL_ID, "asset_lock": lock,
                         "asset_lock_sha256": canonical_sha(lock),
                         "source_lock": old["source_lock"], "source_lock_sha256": old["source_lock_sha256"]}}


def _child_command(python, profile, runtime, role):
    return [str(python), "-I", "-B", "-c", BOOTSTRAP, str(ROOT / "setup_adapter.py"),
            profile, str(runtime), role]


def bind(profile, runtime, stage):
    """Explicit in-process bindings, never modifying retained files on disk."""
    _require(profile in RETAINED_PINS, "Unknown setup profile")
    runtime = Path(runtime)
    _require(runtime == Path("/runtime") and runtime.resolve() == runtime, "Exact /runtime mount required")
    sys.dont_write_bytecode = True
    scope = _scope(stage)  # Before loading helpers or any third-party dependency.
    retained_root = ROOT.parent / ("qwen6_" + profile)
    identities = {name: _identity(retained_root / name, digest)
                  for name, digest in RETAINED_PINS[profile].items()}
    for name in ("setup_diagnostics", "wheel_identity", "import_preflight"):
        _load(name, retained_root / (name + ".py"))
    if profile == "tts":
        gate = _load("runtime_gate", retained_root / "runtime_gate.py")
        cache = runtime / "buildtmp/import-cache" if stage == "setup" else Path("/scratch")
        gate.cache_paths = lambda: {key: cache / key.lower() for key in CACHE_KEYS}
        # Only derivative_signal is used by native preflight; never run its CLI.
        _load("generate_six", retained_root / "generate_six.py")
    helper = _load("source_screen_locked_" + profile, retained_root / "setup_locked.py")
    helper.DOWNLOAD_CAP = TTS_INPUT_CAP if profile == "tts" else 4 * GIB
    helper.INSTALL_CAP, helper.BUILD_CAP = 3 * GIB, 2 * GIB
    helper.WORKSPACE_CAP, helper.INITIAL_FREE = 12 * GIB, 16 * GIB
    if profile == "tts":
        helper.MODEL_IDS = {"qwen_tts": MODEL_ID}

    def read_locks(lock_path=None, model_lock_path=None):
        _require(lock_path is None and model_lock_path is None, "Alternate lock paths prohibited")
        current = {name: _identity(retained_root / name, digest)
                   for name, digest in RETAINED_PINS[profile].items()}
        runtime_lock = json.loads((retained_root / "runtime-lock.json").read_bytes())
        models = json.loads((retained_root / "model-locks.json").read_bytes())
        locks = {"runtime_lock": current["runtime-lock.json"],
                 "retained_model_locks": current["model-locks.json"], "retained_files": current}
        if profile == "tts":
            locks["source_screen_model_lock"] = _identity(ROOT / "qwen-model-lock.json", MODEL_LOCK_SHA256)
            models = converted_tts_models(models, json.loads((ROOT / "qwen-model-lock.json").read_bytes()), helper.canonical_sha)
        locks["bound_models_sha256"] = helper.canonical_sha(models)
        locks["adapter"] = _identity(ROOT / "setup_adapter.py")
        locks["runtime_scope"] = _identity(ROOT / "runtime_scope.py")
        locks["profile"] = profile
        locks["limits"] = {"input_download_bytes": helper.DOWNLOAD_CAP, "installed_bytes": helper.INSTALL_CAP,
                           "buildtmp_bytes": helper.BUILD_CAP, "workspace_bytes": helper.WORKSPACE_CAP,
                           "initial_free_bytes": helper.INITIAL_FREE,
                           "image_transfer_reserve_bytes": IMAGE_TRANSFER_RESERVE if profile == "tts" else 0}
        helper.validate_locks(runtime_lock, models)
        return runtime_lock, models, locks

    original_downloader = helper.Downloader

    class BoundDownloader(original_downloader):
        def __init__(self, cap=None, opener=None):
            _require(cap is None or cap == helper.DOWNLOAD_CAP, "Alternate download budget prohibited")
            # The retained default argument captured its old 4 GiB value.
            super().__init__(cap=helper.DOWNLOAD_CAP, opener=opener)

    original_environment = helper.offline_environment

    def environment(root):
        _require(Path(root) == runtime, "Runtime cache root changed")
        value = original_environment(root)
        value.update(HOME=str(runtime / "buildtmp/home"), ORT_DISABLE_TELEMETRY="1",
                     HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        return value

    def helper_command(python, script, body):
        _require(stage == "setup", "Setup helper commands prohibited during read-only verify")
        role = HELPER_BODIES.get(body)
        _require(role is not None, "Unknown retained helper body")
        expected = retained_root / ("import_preflight.py" if role == "preflight" else "setup_locked.py")
        _require(Path(script) == expected and Path(python) == runtime / "venv/bin/python", "Retained helper target changed")
        return _child_command(python, profile, runtime, role)

    original_commands = helper.Commands

    class BoundCommands(original_commands):
        def run(self, argv, **kwargs):
            _require(stage == "setup", "Setup commands prohibited during verify")
            argv = list(map(str, argv))
            if argv[:3] == [str(runtime / "venv/bin/python"), "-I", "-m"]:
                module = argv[3]
                _require(module in ("ensurepip", "pip", "build"), "Unknown setup module")
                argv = _child_command(argv[0], profile, runtime, "module:" + module) + argv[4:]
            else:
                _require(argv[:8] == _child_command(runtime / "venv/bin/python", profile, runtime, "")[:8],
                         "Unbound retained command prohibited")
                _require(len(argv) > 8 and argv[8] in set(HELPER_BODIES.values()), "Unknown setup child")
            return super().run(argv, **kwargs)

    original_save_receipt = helper._save_receipt

    def save_receipt(path, receipt):
        _require(Path(path) == runtime / "setup-receipt.json", "Setup receipt destination changed")
        receipt["source_screen_binding"] = {"schema": "screen32-setup-binding-v1", "profile": profile,
                                            "runtime": str(runtime), "scope": scope,
                                            "model_id": MODEL_ID if profile == "tts" else None,
                                            "input_download_cap_bytes": helper.DOWNLOAD_CAP}
        return original_save_receipt(path, receipt)

    helper._read_locks, helper.Downloader = read_locks, BoundDownloader
    helper.offline_environment, helper._helper_command, helper.Commands = environment, helper_command, BoundCommands
    helper._save_receipt = save_receipt
    helper.source_screen_scope = scope
    helper.source_screen_retained_identities = identities
    return helper


def _run_child(profile, runtime, role, arguments):
    """Fixed setup-only subprocess dispatch; no executable text from argv/stdin."""
    root = Path(runtime)
    helper = bind(profile, root, "setup")
    if role.startswith("module:"):
        module = role.removeprefix("module:")
        _require(module in ("ensurepip", "pip", "build"), "Unknown setup module")
        sys.argv = [module, *arguments]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return
    expected = [profile, str(root)] if role == "preflight" else [str(root)] if role.startswith("verify_") else []
    _require(arguments == expected, "Retained child arguments changed")
    if role == "requirements":
        helper.verify_build_requirements(json.load(sys.stdin))
    elif role == "metadata":
        value = json.load(sys.stdin)
        _require(set(value) == {"path", "row", "diagnostic_path"}, "Metadata helper input shape")
        print(json.dumps(helper.inspect_built_metadata(value["path"], value["row"], value["diagnostic_path"])))
    elif role in ("verify_before_models", "verify_after_models"):
        print(json.dumps(helper.verify_setup(root, _during_setup=True, _before_models=role == "verify_before_models")))
    elif role == "preflight":
        sys.modules["import_preflight"].execute(profile, root, root / "import-preflight.json")
    else:
        raise ValueError("Unknown setup child role")


def verify_runtime(root, profile):
    """Read-only worker entry: hard offline scope, exact venv/packages/sources/assets.

    Call before importing any third-party inference dependency. The caller must
    already be using /runtime/venv/bin/python. TTS runtime_gate is then bound to
    /scratch caches; its retained prepare() still owns runtime import guards.
    """
    return bind(profile, root, "verify").verify_setup(Path(root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("tts", "asr"), required=True)
    parser.add_argument("--stage", choices=("setup", "verify"), required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args(argv)
    helper = bind(args.profile, args.runtime, args.stage)
    if args.stage == "verify":
        result = helper.verify_setup(args.runtime)
    else:
        _require(helper.shutil.disk_usage(args.runtime).free >= helper.INITIAL_FREE,
                 "At least 16 GiB initially free is required")
        result = helper.setup(args.runtime)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
