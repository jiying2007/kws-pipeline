#!/usr/bin/env python3
"""Install an explicitly selected research software candidate from pinned bytes.

The reviewed Git selection is the trust root. Its immutable manifest binds the
independent full original quality/engineering readback and four original
engineering artifacts. This tool checks that binding and transports exact
bytes; it is not another acoustic or engineering evaluator.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import tarfile
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
SELECTION = HERE / "selection.json"
PACK_BYTES = 216960
B0_SHA = "ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b"
B0_PATH = "models/B0.pack"
PIPELINE_COMMIT = "88a043c6c00bb3a1a62f45098a023c7601042ded"
LIMIT = 128_000_000
META_LIMIT = 4_000_000
MAX_MEMBERS = 8192
BACKENDS = ("gcc", "clang", "cortex-a32-qemu")
ORIGIN_ROLES = {"result", "producer", "consumed", "input_proof"}
SCOPE = dict(use="research_software_integration", physical_ssc305=False,
    product_promotion_allowed=False, cal_locked_access=False, field_frr=None,
    continuous_far_per_hour=None, word_end_p95_ms=None, product_decoder_settings=None)
REQUIRED_SDK = {"bin/heym43-live", "bin/heym43_host_cli", "lib/libheym43_stream.a",
    "include/heym43/mero43_stream.h", "include/heym43/mero_ctc.h",
    "lib/cmake/HeyMero43/HeyMero43Config.cmake",
    "lib/cmake/HeyMero43/HeyMero43ConfigVersion.cmake",
    "lib/cmake/HeyMero43/HeyMero43Targets.cmake"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def document(raw):
    need(len(raw) <= META_LIMIT, "bounded metadata")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            need(key not in value, "duplicate JSON field")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("nonfinite JSON value")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    need(type(value) is dict, "metadata must be an object")
    return value


def safe_name(name):
    need(type(name) is str and 0 < len(name) <= 1024
         and re.fullmatch(r"[A-Za-z0-9_./-]+", name), "safe relative member name")
    value = PurePosixPath(name)
    need(bool(value.parts) and not value.is_absolute() and value.as_posix() == name
         and all(part not in (".", "..") for part in value.parts), "canonical relative member path")
    return name


def regular(path):
    path = Path(path).absolute()
    need(all(not part.is_symlink() for part in (path, *path.parents)), "no symlink in file path")
    need(path.is_file() and stat.S_ISREG(path.stat().st_mode), "regular file required")
    return path


def read(path, maximum=LIMIT):
    path = regular(path)
    need(path.stat().st_size <= maximum, "file exceeds byte limit")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    need(len(raw) <= maximum, "file grew beyond byte limit")
    return raw


def valid_pin(pin, maximum=LIMIT, mode=False, allow_empty=False):
    keys = {"bytes", "sha256"} | ({"mode"} if mode else set())
    need(type(pin) is dict and set(pin) == keys
         and type(pin["bytes"]) is int and (0 if allow_empty else 1) <= pin["bytes"] <= maximum
         and type(pin["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]), "complete byte pin")
    if mode:
        need(type(pin["mode"]) is int and pin["mode"] in (0o644, 0o755), "reviewed file mode")


def matches(raw, pin):
    need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "original bytes or SHA mismatch")


def artifact_pin(pin, maximum=LIMIT):
    need(type(pin) is dict and set(pin) == {"id", "run_id", "run_head", "bytes", "sha256"}
         and type(pin["id"]) is type(pin["run_id"]) is int
         and pin["id"] > 0 and pin["run_id"] > 0
         and type(pin["run_head"]) is str and re.fullmatch(r"[0-9a-f]{40}", pin["run_head"]), "original artifact identity")
    valid_pin({key: pin[key] for key in ("bytes", "sha256")}, maximum)


def prescribed_mode(name):
    safe_name(name)
    if name in {"sdk-" + backend + "/bin/" + tool for backend in BACKENDS
                for tool in ("heym43-live", "heym43_host_cli")}:
        return 0o755
    need(not any(name.startswith("sdk-" + backend + "/bin/") for backend in BACKENDS),
         "no other executable is part of this SDK delivery")
    return 0o644


def selection():
    value = document(read(SELECTION, META_LIMIT))
    need(set(value) == {"schema", "selected", "manifest", "baseline", "scope"}
         and value["schema"] == "heym43-software-selection-v1"
         and value["baseline"] == dict(path=B0_PATH, bytes=PACK_BYTES, sha256=B0_SHA)
         and encoded(value["scope"]) == encoded(SCOPE), "fixed baseline and research-only selection")
    if value["selected"] is None:
        need(value["manifest"] is None, "closed selection has no manifest authority")
        return value
    need(type(value["selected"]) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value["selected"]), "candidate identifier")
    pin = value["manifest"]
    need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}, "complete reviewed manifest pin")
    valid_pin({key: pin[key] for key in ("bytes", "sha256")}, META_LIMIT)
    need(set(pin) == {"path", "bytes", "sha256"}
         and pin["path"] == "manifests/" + pin["sha256"] + ".json", "immutable manifest filename equals its SHA")
    return value


def manifest_contract(value):
    need(set(value) == {"schema", "candidate_id", "pipeline_commit", "candidate_pack", "files",
                        "engineering", "quality_locks", "readback", "scope"}
         and value["schema"] == "heym43-software-delivery-v1"
         and value["pipeline_commit"] == PIPELINE_COMMIT
         and encoded(value["scope"]) == encoded(SCOPE), "software delivery contract; no product or physical claim")
    files = value["files"]
    need(type(files) is dict and 0 < len(files) < MAX_MEMBERS,
         "complete finite payload inventory")
    for name, pin in files.items():
        safe_name(name); valid_pin(pin, mode=True)
        need(pin["mode"] == prescribed_mode(name), "mode is prescribed, never inherited from upload ZIP")
        need(name in ("LICENSE", "README.md") or name.startswith("models/")
             or any(name.startswith("sdk-" + backend + "/") for backend in BACKENDS), "only model, license and complete installed SDK payload")
    need(sum(p["bytes"] for p in files.values()) <= LIMIT, "finite total delivery bytes")
    for backend in BACKENDS:
        need({"sdk-" + backend + "/" + name for name in REQUIRED_SDK} <= set(files), "complete installed SDK for " + backend)
    pack = value["candidate_pack"]
    need(type(pack) is str and re.fullmatch(r"models/[A-Za-z0-9_-]+\.pack", pack)
         and pack != B0_PATH and pack in files and files[pack]["bytes"] == PACK_BYTES
         and files[pack]["sha256"] != B0_SHA, "one original distinct candidate pack")
    need(files.get(B0_PATH) == dict(bytes=PACK_BYTES, sha256=B0_SHA, mode=0o644), "fixed actual B0 rollback pack")
    models = [name for name in files if name.startswith("models/") and name not in (pack, B0_PATH)]
    need(len(models) == 3 and {Path(name).suffix for name in models} == {".pt", ".npz", ".onnx"}
         and len({Path(name).stem for name in models}) == 1, "three original candidate exports")
    need({"LICENSE", "README.md"} <= set(files), "original software handoff and license retained")
    engineering = value["engineering"]
    need(set(engineering) == {"repository", "artifacts", "result_sha256", "manifest_sha256"}
         and engineering["repository"] == "jiying2007/kws-data"
         and set(engineering["artifacts"]) == ORIGIN_ROLES, "four original engineering artifacts")
    for pin in engineering["artifacts"].values(): artifact_pin(pin)
    need(len({p["id"] for p in engineering["artifacts"].values()}) == 4
         and len({(p["run_id"], p["run_head"]) for p in engineering["artifacts"].values()}) == 1,
         "four distinct originals from one actual engineering run")
    for key in ("result_sha256", "manifest_sha256"):
        need(type(engineering[key]) is str and re.fullmatch(r"[0-9a-f]{64}", engineering[key]), "original result member hash")
    locks = value["quality_locks"]
    need(type(locks) is dict and set(locks) == {"primary", "replication"}, "two original quality locks")
    for panel, lock in locks.items():
        need(type(lock) is dict and set(lock) == {"schema", "panel", "result", "producer", "consumed",
             "evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256"}
             and lock["panel"] == panel and re.fullmatch(r"english-v6-b[0-9]+[a-z0-9]*-original-quality-lock-v1", lock["schema"]),
             "full original panel lock; a PASS summary has no authority")
        for role in ("result", "producer", "consumed"):
            artifact_pin(lock[role], 512_000_000 if panel == "replication" and role == "producer" else LIMIT)
        need(len({lock[k]["id"] for k in ("result", "producer", "consumed")}) == 3
             and len({(lock[k]["run_id"], lock[k]["run_head"]) for k in ("result", "producer", "consumed")}) == 1,
             "three distinct same-run quality originals")
        for key in ("evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256"):
            need(type(lock[key]) is str and re.fullmatch(r"[0-9a-f]{64}", lock[key]), "full original quality digest")
    need(locks["primary"]["schema"] == locks["replication"]["schema"]
         and locks["primary"]["result"]["run_id"] != locks["replication"]["result"]["run_id"], "two independent original panel executions")
    need(type(value["readback"]) is dict and set(value["readback"]) == {"report", "verifier"},
         "exact independent full original readback and its verifier are pinned")
    for pin in value["readback"].values(): valid_pin(pin, META_LIMIT)
    return value


def trusted_manifest(raw=None):
    selected = selection()
    need(selected["selected"] is not None, "INERT: no reviewed software candidate is selected")
    pin = selected["manifest"]
    if raw is None: raw = read(HERE / pin["path"], META_LIMIT)
    matches(raw, pin)
    value = manifest_contract(document(raw))
    need(value["candidate_id"] == selected["selected"], "same reviewed candidate identity")
    return value, raw


@contextmanager
def original_archive(path, pin):
    raw = read(path); matches(raw, pin)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        need(len(members) <= MAX_MEMBERS and len(members) == len({m.filename for m in members})
             and sum(m.file_size for m in members) <= LIMIT, "bounded unique complete original ZIP")
        for member in members:
            safe_name(member.filename)
            need(not member.is_dir() and stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG)
                 and member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "regular original ZIP members")
        need(archive.testzip() is None, "original ZIP CRC")
        yield archive


def pack_header(raw):
    need(len(raw) == PACK_BYTES and raw[:16] == struct.pack("<8sII", b"MERO43V1", 1, 54236), "exact Mero43 pack envelope")


def prepare_payload(originals, baseline):
    value, manifest_raw = trusted_manifest()  # Before caller-controlled paths.
    originals = Path(originals)
    # These files are authenticated by the reviewed manifest SHA. No field such
    # as a self-reported PASS is interpreted as an alternative permission.
    for role, filename in (("report", "completion.json"), ("verifier", "readback.py")):
        matches(read(originals / filename, META_LIMIT), value["readback"][role])
    pins = value["engineering"]["artifacts"]
    for role in sorted(ORIGIN_ROLES - {"result"}):
        with original_archive(originals / (role + ".zip"), pins[role]): pass
    with original_archive(originals / "result.zip", pins["result"]) as archive:
        raw = archive.read("MANIFEST.json")
        need(sha(raw) == value["engineering"]["manifest_sha256"], "exact original engineering manifest")
        original = document(raw)
        need(set(original["files"]) == set(archive.namelist()) - {"MANIFEST.json"}, "entire original artifact inventory, no omitted or added files")
        for name, pin in original["files"].items():
            valid_pin(pin, allow_empty=True); matches(archive.read(name), pin)
        raw = archive.read("RESULT.json")
        need(sha(raw) == value["engineering"]["result_sha256"], "exact original engineering RESULT")
        result = document(raw)
        need(re.fullmatch(r"PASS_B[0-9]+[A-Z0-9]*_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE", result["status"])
             and original["status"] == result["status"] and result["pipeline_commit"] == PIPELINE_COMMIT
             and result["pack_bytes"] == PACK_BYTES and result["pack_sha256"] == value["files"][value["candidate_pack"]]["sha256"]
             and result["entered_backend_calls"] == result["saved_backend_outputs"] == 140
             and result["consumed_unknown"] == [] and result["physical_ssc305"] is result["product_promotion_allowed"] is False,
             "original engineering completion agrees with the independently authenticated selection")
        pin = pins["result"]
        need(str(result["identity"]["run_id"]) == str(pin["run_id"])
             and result["identity"]["run_head"] == pin["run_head"] and result["identity"]["run_attempt"] == 1,
             "same original engineering identity")
        for panel, lock in value["quality_locks"].items():
            need(result["quality"][panel]["panel_gate"] == "PASS"
                 and result["quality"][panel]["original_result_sha256"] == lock["original_result_sha256"], "same already-read original quality result")
        selected = {name for name in archive.namelist() if name.startswith(("models/", *("sdk-" + b + "/" for b in BACKENDS)))}
        selected |= {"LICENSE", "README.md"}
        need(set(value["files"]) == selected | {B0_PATH}, "complete original models and installed SDK subset")
        payload = {name: archive.read(name) for name in sorted(selected)}
    payload[B0_PATH] = read(baseline, PACK_BYTES)
    for name, raw in payload.items(): matches(raw, value["files"][name])
    pack_header(payload[value["candidate_pack"]]); pack_header(payload[B0_PATH])
    payload["DELIVERY.json"] = manifest_raw
    return value, payload


def package(originals, baseline, output):
    value, payload = prepare_payload(originals, baseline)
    output = Path(output).absolute()
    need(output.parent.is_dir() and not output.exists() and not output.is_symlink(), "fresh delivery archive path")
    need(all(not p.is_symlink() for p in output.parents), "regular delivery parent")
    # ZIP executable bits are deliberately ignored. This reproducible tar
    # carries the reviewed 0755/0644 prescription, not claimed original modes.
    with output.open("xb") as target, gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            for name, raw in sorted(payload.items()):
                info = tarfile.TarInfo(name); info.size = len(raw)
                info.mode = 0o644 if name == "DELIVERY.json" else value["files"][name]["mode"]
                info.mtime = info.uid = info.gid = 0
                archive.addfile(info, io.BytesIO(raw))
    return dict(status="PACKAGED_VERIFIED_ORIGINAL_BYTES", bytes=output.stat().st_size,
                sha256=sha(read(output)), mode_authority="reviewed_prescription", scientific_calls=0)


def package_members(path):
    value, trusted_raw = trusted_manifest()
    raw = read(path)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        need(0 < len(members) <= MAX_MEMBERS and len(members) == len({m.name for m in members})
             and sum(m.size for m in members) <= LIMIT, "bounded unique delivery archive")
        need({m.name for m in members} == set(value["files"]) | {"DELIVERY.json"}, "all and only selected delivery members")
        payload = {}
        for member in members:
            safe_name(member.name)
            mode = 0o644 if member.name == "DELIVERY.json" else value["files"][member.name]["mode"]
            need(member.isfile() and member.mode == mode and not member.pax_headers, "regular member with exact prescribed mode")
            body = archive.extractfile(member).read()
            if member.name == "DELIVERY.json": need(body == trusted_raw, "same trusted immutable delivery manifest")
            else: matches(body, value["files"][member.name])
            payload[member.name] = body
    return value, trusted_raw, payload


def version_directory(prefix, manifest_raw):
    prefix = Path(prefix).absolute()
    need(all(not p.is_symlink() for p in (prefix, *prefix.parents)), "no symlink in install prefix")
    return prefix / "versions" / sha(manifest_raw)


def verify_installed(prefix):
    value, raw = trusted_manifest()
    destination = version_directory(prefix, raw)
    need(destination.is_dir() and not destination.is_symlink() and not destination.parent.is_symlink(),
         "selected immutable version has not been installed as regular directories")
    expected = set(value["files"]) | {"DELIVERY.json"}
    expected_directories = {parent.as_posix() for name in expected for parent in PurePosixPath(name).parents if parent.parts}
    observed, directories = set(), set()
    need(stat.S_IMODE(destination.stat().st_mode) == 0o755, "installed version directory mode")
    for path in destination.rglob("*"):
        need(not path.is_symlink(), "no installed symlinks")
        if path.is_dir():
            directories.add(path.relative_to(destination).as_posix())
            need(stat.S_IMODE(path.stat().st_mode) == 0o755, "installed directory mode")
            continue
        observed.add(path.relative_to(destination).as_posix())
    need(observed == expected and directories == expected_directories, "exact installed file and directory inventory")
    for name in sorted(expected):
        path = destination / name
        mode = 0o644 if name == "DELIVERY.json" else value["files"][name]["mode"]
        body = read(path)
        need(stat.S_IMODE(path.stat().st_mode) == mode, "post-install file mode mismatch")
        if name == "DELIVERY.json": need(body == raw, "installed immutable manifest matches selection")
        else: matches(body, value["files"][name])
    return value, destination


def install(archive, prefix):
    value, raw, payload = package_members(archive)  # Entire validation before writes.
    destination = version_directory(prefix, raw)
    if destination.exists():
        verify_installed(prefix)
        return dict(status="ALREADY_INSTALLED", version=str(destination), selected_process_changed=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    need(all(not p.is_symlink() for p in (destination.parent, *destination.parent.parents)), "regular install parent")
    staged = Path(tempfile.mkdtemp(prefix=".heym43-", dir=destination.parent))
    try:
        staged.chmod(0o755)
        for name, body in payload.items():
            target = staged / name; target.parent.mkdir(parents=True, exist_ok=True)
            for directory in target.parents:
                if directory == staged.parent: break
                directory.chmod(0o755)
            with target.open("xb") as handle: handle.write(body)
            target.chmod(0o644 if name == "DELIVERY.json" else value["files"][name]["mode"])
        need(not destination.exists(), "immutable version cannot be replaced")
        staged.rename(destination)
    finally:
        if staged.exists(): shutil.rmtree(staged)
    verify_installed(prefix)
    return dict(status="INSTALLED_VERIFIED_BYTES_AND_MODES", version=str(destination),
                selected_process_changed=False, scientific_calls=0)


def installed_paths(prefix, model, backend):
    need(model in ("candidate", "B0") and backend in BACKENDS, "explicit installed model and backend")
    value, directory = verify_installed(prefix)
    pack = directory / (value["candidate_pack"] if model == "candidate" else B0_PATH)
    return directory / ("sdk-" + backend), pack


def launch(prefix, model, backend, max_gap, cooldown):
    need(type(max_gap) is type(cooldown) is int and 0 < max_gap < 2**64 and 0 <= cooldown < 2**64,
         "caller supplies explicit uint64 decoder policy; no defaults or calibration")
    sdk, pack = installed_paths(prefix, model, backend)
    binary = sdk / "bin/heym43-live"
    # Replacing this new launcher process starts a clean model/state. Existing
    # consumer processes are neither killed nor silently switched.
    os.execv(str(binary), [str(binary), str(pack), str(max_gap), str(cooldown)])


def check_load(prefix, model, backend, checker):
    _, pack = installed_paths(prefix, model, backend)
    checker = regular(checker)
    result = subprocess.run([str(checker), str(pack)], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    observed = document(result.stdout.encode())
    need(observed == dict(schema="heymero43-delivery-load-check-v1", status="PASS_MODEL_LOAD_ZERO_FORWARD",
         pack_bytes=PACK_BYTES, model_storage_bytes=4 + 4 * 54236, model_load_calls=1,
         model_forwards=0, stream_init_calls=0, pcm_samples_consumed=0),
         "installed clean consumer must load the selected real pack without a forward")
    return dict(status="INSTALLED_MODEL_LOAD_OK", model=model, pack_sha256=sha(read(pack)),
                model_forwards=0, pcm_samples=0, physical_ssc305=False, product_promotion_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check")
    item = modes.add_parser("package")
    item.add_argument("--originals", type=Path, required=True)
    item.add_argument("--baseline-pack", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item = modes.add_parser("install")
    item.add_argument("--archive", type=Path, required=True); item.add_argument("--prefix", type=Path, required=True)
    item = modes.add_parser("verify-installed"); item.add_argument("--prefix", type=Path, required=True)
    for mode in ("launch", "check-load"):
        item = modes.add_parser(mode)
        item.add_argument("--prefix", type=Path, required=True)
        item.add_argument("--model", choices=("candidate", "B0"), required=True)
        item.add_argument("--backend", choices=BACKENDS, default="gcc")
        if mode == "launch":
            item.add_argument("--max-gap-samples", type=int, required=True)
            item.add_argument("--cooldown-samples", type=int, required=True)
        else: item.add_argument("--checker", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "check":
        chosen = selection()
        if chosen["selected"] is not None: trusted_manifest()
        result = dict(status="INERT" if chosen["selected"] is None else "REVIEWED_MANIFEST_SELECTED",
                      selected=chosen["selected"], scientific_calls=0, packs_created=0)
    elif args.mode == "package": result = package(args.originals, args.baseline_pack, args.output)
    elif args.mode == "install": result = install(args.archive, args.prefix)
    elif args.mode == "verify-installed":
        value, directory = verify_installed(args.prefix)
        result = dict(status="INSTALLED_BYTES_AND_MODES_VERIFIED", candidate=value["candidate_id"], version=str(directory))
    elif args.mode == "launch":
        launch(args.prefix, args.model, args.backend, args.max_gap_samples, args.cooldown_samples); return
    else: result = check_load(args.prefix, args.model, args.backend, args.checker)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
