#!/usr/bin/env python3
"""Deliver one reviewed B24R1 software candidate without another model forward.

The entry workflow lives in kws-data, where its existing read-only Actions
token can retrieve the original artifacts. This file executes from the exact
public kws-pipeline commit pinned by that entry's reviewed Git binding. The
selection and independent readback are never created or enabled here.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
definition = importlib.util.spec_from_file_location("reviewed_delivery", HERE / "delivery.py")
d = importlib.util.module_from_spec(definition)
definition.loader.exec_module(d)
DATA_REPO = "jiying2007/kws-data"
PIPELINE_REPO = "jiying2007/kws-pipeline"
BRANCH = "refs/heads/research/en-v6-b24r1-delivery-once-20261125"
WORKFLOW = ".github/workflows/english-v6-b24r1-delivery-once.yml"
TEMPLATE = "automation/english_v6_b24r1_delivery_workflow_template.yml"
BINDING = "automation/english_v6_b24r1_delivery_inputs.json"
TOKEN = b"__B24R1_REVIEWED_DELIVERY_PARENT__"
PACKER = "research/english_v5_mero/pack_mero43.py"
PACKER_SHA = "d58f19472ce00391fe0d24ec037ef898c29c9cf3418c7e041c25cad172665bc5"
# Preserve the entire original identity; the candidate uses its separate pins.
B0 = dict(id=14514228406, run_id=44495594627,
    run_head="d0b567fe37147aae6ab5d7bdec3721b879a0b22d", bytes=501276,
    sha256="91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702")
B0_NPZ = "02b7febf9f736fd70f4f9ae04042ffc6d609b15011d03d235b960f69b1d71696"
DELIVERY_FILES = {
    "research/english_v6_delivery/once.py", "research/english_v6_delivery/delivery.py",
    "research/english_v6_delivery/selection.json", "research/english_v6_delivery/loader_check.c",
    "research/english_v6_delivery/CMakeLists.txt", PACKER,
}
ENTRY_SCHEMA = "english-v6-b24r1-software-delivery-entry-v1"


def write(path, value):
    with Path(path).open("xb") as handle:
        handle.write(d.encoded(value))


def pin(raw):
    return dict(bytes=len(raw), sha256=d.sha(raw))


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], timeout=60)


def readback_paths(value):
    return {
        "completion.json": HERE / "readbacks" / value["readback"]["report"]["sha256"] / "completion.json",
        "readback.py": HERE / "readbacks" / (value["readback"]["verifier"]["sha256"] + ".py"),
    }


def reviewed_readback(value):
    """Authenticate reviewed Git bytes; never execute the saved verifier."""
    paths = readback_paths(value)
    saved = {name: d.read(path, d.META_LIMIT) for name, path in paths.items()}
    d.matches(saved["completion.json"], value["readback"]["report"])
    d.matches(saved["readback.py"], value["readback"]["verifier"])
    report = d.document(saved["completion.json"])
    d.need(report["schema"] == "english-v6-b24r1-verified-original-engineering-v1"
        and report["status"] == "VERIFIED_ORIGINAL_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE"
        and value["candidate_id"] == report["candidate_id"] == "B24R1"
        and report["baseline_pack_requires_separate_byte_authentication"] is True
        and report["readback_verifier"] == value["readback"]["verifier"],
        "actual complete independent B24R1 readback and exact verifier identity")
    for key in ("candidate_pack", "pipeline_commit", "files", "engineering", "quality_locks", "scope"):
        d.need(report[key] == value[key], "reviewed manifest agrees with original readback: " + key)
    return saved


def expected_pipeline_files(value, manifest_raw):
    paths = set(DELIVERY_FILES)
    paths.add("research/english_v6_delivery/manifests/" + d.sha(manifest_raw) + ".json")
    paths.update(path.relative_to(ROOT).as_posix() for path in readback_paths(value).values())
    return paths


def entry_contract(value):
    d.need(type(value) is dict and set(value) == {
        "schema", "enabled", "pipeline_repository", "pipeline_commit", "manifest_sha256", "pipeline_files"
    } and value["schema"] == ENTRY_SCHEMA and value["pipeline_repository"] == PIPELINE_REPO
        and type(value["enabled"]) is bool, "one exact reviewed delivery entry binding")
    if value["enabled"] is False:
        d.need(all(value[key] is None for key in ("pipeline_commit", "manifest_sha256", "pipeline_files")),
               "inert delivery entry has no invented selection or producer pins")
        return value
    d.need(type(value["pipeline_commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", value["pipeline_commit"])
        and type(value["manifest_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["manifest_sha256"])
        and type(value["pipeline_files"]) is dict and bool(value["pipeline_files"]), "complete selected pipeline identity")
    for name, member in value["pipeline_files"].items():
        d.safe_name(name)
        d.need(type(member) is dict and set(member) == {"bytes", "sha256", "git_blob_sha1", "mode"}
            and type(member["git_blob_sha1"]) is str and re.fullmatch(r"[0-9a-f]{40}", member["git_blob_sha1"])
            and member["mode"] in ("100644", "100755"), "exact regular pipeline Git file pin")
        d.valid_pin({key: member[key] for key in ("bytes", "sha256")}, d.META_LIMIT)
    return value


def pipeline_contract(binding, value, manifest_raw):
    d.need(binding["enabled"] is True and binding["manifest_sha256"] == d.sha(manifest_raw),
           "INERT: exact reviewed selection must be bound before delivery")
    names = expected_pipeline_files(value, manifest_raw)
    d.need(set(binding["pipeline_files"]) == names
        and git(ROOT, "rev-parse", "HEAD").decode().strip() == binding["pipeline_commit"]
        and not git(ROOT, "status", "--porcelain", "--untracked-files=no"),
        "complete exact public pipeline commit and clean reviewed producer")
    entries = {}
    for line in git(ROOT, "ls-tree", "-z", "HEAD", "--", *sorted(names)).split(b"\0"):
        if not line: continue
        fields, name = line.split(b"\t", 1)
        mode, kind, digest = fields.decode().split()
        d.need(kind == "blob", "regular pipeline Git blob")
        entries[name.decode()] = (mode, digest)
    d.need(set(entries) == names, "all delivery files tracked at exact selection commit")
    for name in sorted(names):
        raw = d.read(ROOT / name, d.META_LIMIT)
        member = binding["pipeline_files"][name]
        d.matches(raw, member)
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        d.need(entries[name] == (member["mode"], member["git_blob_sha1"])
            and blob == member["git_blob_sha1"], "original selected pipeline Git bytes: " + name)
    d.need(d.sha(d.read(ROOT / PACKER)) == PACKER_SHA, "unchanged original B0 container geometry source")


def guard(prepared, data_root, env=None, event=None):
    env = os.environ if env is None else env
    d.need(type(prepared) is str and re.fullmatch(r"[0-9a-f]{40}", prepared), "reviewed delivery parent")
    event = d.document(d.read(env["GITHUB_EVENT_PATH"], d.META_LIMIT)) if event is None else event
    d.need(env.get("GITHUB_ACTIONS") == "true" and env.get("GITHUB_REPOSITORY") == DATA_REPO
        and env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REF") == BRANCH
        and env.get("GITHUB_RUN_ATTEMPT") == "1" and re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", ""))
        and re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", ""))
        and event["repository"]["full_name"] == DATA_REPO and event["repository"]["private"] is False
        and event["before"] == prepared and event["after"] == env["GITHUB_SHA"]
        and event["ref"] == BRANCH, "first public fixed-parent delivery push only")
    head = git(data_root, "rev-parse", "HEAD").decode().strip()
    parents = git(data_root, "rev-list", "--parents", "-n", "1", "HEAD").decode().split()[1:]
    changed = git(data_root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").decode().splitlines()
    template = d.read(data_root / TEMPLATE, d.META_LIMIT)
    d.need(head == env["GITHUB_SHA"] and parents == [prepared] and changed == [WORKFLOW]
        and not git(data_root, "status", "--porcelain", "--untracked-files=no")
        and template.count(TOKEN) == 2 and d.read(data_root / WORKFLOW) == template.replace(TOKEN, prepared.encode()),
        "exact workflow-only child with two reviewed parent substitutions")
    return dict(repository=DATA_REPO, run_id=int(env["GITHUB_RUN_ID"]), run_head=head,
                run_attempt=1, run_branch=BRANCH, prepared_parent=prepared)


def api(path):
    d.need(type(path) is str and path.startswith("repos/" + DATA_REPO + "/")
        and ".." not in path and "\n" not in path, "fixed public read-only API origin")
    return subprocess.check_output(["gh", "api", "--allow-escape-sequences", "--hostname", "github.com",
                                    "--method", "GET", path], timeout=240)


def one_execution(identity):
    run = d.document(api("repos/" + DATA_REPO + "/actions/runs/" + str(identity["run_id"])))
    inventory = d.document(api("repos/" + DATA_REPO + "/actions/workflows/" + Path(WORKFLOW).name
        + "/runs?event=push&branch=" + BRANCH.removeprefix("refs/heads/") + "&per_page=100"))
    d.need(run["id"] == identity["run_id"] and run["head_sha"] == identity["run_head"]
        and run["run_attempt"] == 1 and run["event"] == "push" and run["path"] == WORKFLOW
        and run["head_branch"] == BRANCH.removeprefix("refs/heads/")
        and run["repository"]["full_name"] == DATA_REPO and run["repository"]["private"] is False
        and run["head_repository"]["full_name"] == DATA_REPO and run["head_repository"]["private"] is False
        and inventory["total_count"] == len(inventory["workflow_runs"]) == 1
        and inventory["workflow_runs"][0]["id"] == identity["run_id"], "one original delivery run, never replay")
    return dict(run=run, workflow_runs=inventory)


def artifact_name(role, run_id):
    if role == "B0": return None  # Immutable original artifact ID is the authority.
    stems = dict(result="software-sdk-evidence-", producer="engineering-producer-",
                 consumed="engineering-consumed", input_proof="engineering-input-proof-")
    return "english-v6-b24r1-" + stems[role] + ("" if role == "consumed" else str(run_id))


def provenance(pin_value, run, artifact, commit, role):
    d.artifact_pin(pin_value)
    d.need(run["id"] == pin_value["run_id"] and run["head_sha"] == commit["sha"] == pin_value["run_head"]
        and run["status"] == "completed" and run["conclusion"] == "success" and run["run_attempt"] == 1
        and run["repository"]["full_name"] == DATA_REPO and run["repository"]["private"] is False
        and run["head_repository"]["full_name"] == DATA_REPO and run["head_repository"]["private"] is False
        and artifact["id"] == pin_value["id"] and artifact["expired"] is False
        and artifact["size_in_bytes"] == pin_value["bytes"] and artifact["digest"] == "sha256:" + pin_value["sha256"]
        and artifact["workflow_run"]["id"] == pin_value["run_id"]
        and artifact["workflow_run"]["head_sha"] == pin_value["run_head"], "exact successful original public artifact")
    if role != "B0":
        d.need(artifact["name"] == artifact_name(role, pin_value["run_id"])
            and run["event"] == "push" and run["path"] == ".github/workflows/english-v6-b24r1-engineering-once.yml"
            and run["head_branch"] == "research/en-v6-b24r1-engineering-once-20261124",
            "actual original B24R1 engineering role")


def fetch_originals(value, originals, saved_readback):
    originals.mkdir()
    for name, raw in saved_readback.items():
        with (originals / name).open("xb") as handle: handle.write(raw)
    pins = dict(value["engineering"]["artifacts"], B0=B0)
    runs, commits, evidence = {}, {}, {}
    for role, source in sorted(pins.items()):
        if source["run_id"] not in runs:
            runs[source["run_id"]] = d.document(api("repos/" + DATA_REPO + "/actions/runs/" + str(source["run_id"])))
            commits[source["run_head"]] = d.document(api("repos/" + DATA_REPO + "/commits/" + source["run_head"]))
        artifact = d.document(api("repos/" + DATA_REPO + "/actions/artifacts/" + str(source["id"])))
        run, commit = runs[source["run_id"]], commits[source["run_head"]]
        provenance(source, run, artifact, commit, role)
        raw = api("repos/" + DATA_REPO + "/actions/artifacts/" + str(source["id"]) + "/zip")
        d.matches(raw, source)
        path = originals / (role + ".zip")
        with path.open("xb") as handle: handle.write(raw)
        with d.original_archive(path, source): pass
        evidence[role] = dict(run=run, artifact=artifact, commit=commit)
    return evidence


def npy_payload(raw, shape):
    d.need(raw[:6] == b"\x93NUMPY" and raw[6:8] in (b"\x01\x00", b"\x02\x00"), "original NPY version")
    width = 2 if raw[6:8] == b"\x01\x00" else 4
    offset = 8 + width
    size = int.from_bytes(raw[8:offset], "little")
    d.need(0 < size <= 65536 and len(raw) >= offset + size, "bounded complete original NPY header")
    header = ast.literal_eval(raw[offset:offset + size].decode("latin1"))
    d.need(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"}
        and header["descr"] == "<f4" and header["fortran_order"] is False and header["shape"] == shape,
        "same little-endian contiguous float32 tensor; no numeric conversion")
    payload = raw[offset + size:]
    d.need(len(payload) == 4 * math.prod(shape), "complete unchanged original float bytes")
    return payload


def baseline_container(original):
    """Serialize only original B0 tensor bytes in the existing frozen order."""
    raw_packer = d.read(ROOT / PACKER)
    d.need(d.sha(raw_packer) == PACKER_SHA, "unchanged original geometry implementation")
    definition = importlib.util.spec_from_file_location("original_b0_container_geometry", ROOT / PACKER)
    packer = importlib.util.module_from_spec(definition); definition.loader.exec_module(packer)
    packer.source_check()
    with d.original_archive(original, B0) as archive:
        expected = {"STARTED.json", "RESULT.json", "TRAINING.json", "ADMISSION.json",
                    *("models/mero-fsmn-ctc-expanded256." + suffix for suffix in ("pt", "npz", "onnx"))}
        d.need(set(archive.namelist()) == expected, "complete exact original B0 export archive")
        raw = archive.read("models/mero-fsmn-ctc-expanded256.npz")
        d.need(d.sha(raw) == B0_NPZ == packer.NPZ_SHA256, "fixed original B0 NPZ bytes")
        geometry = packer.geometry()
        with zipfile.ZipFile(io.BytesIO(raw)) as arrays:
            names = [name + ".npy" for name, _ in geometry]
            d.need(len(arrays.infolist()) == len(names) and set(arrays.namelist()) == set(names)
                and arrays.testzip() is None, "all and only original B0 tensors")
            payload = [npy_payload(arrays.read(name + ".npy"), shape) for name, shape in geometry]
    packed = struct.pack("<8sII", packer.MAGIC, packer.VERSION, packer.PACK_FLOAT_COUNT) + b"".join(payload)
    d.need(len(packed) == d.PACK_BYTES == 216960 and d.sha(packed) == d.B0_SHA,
           "the exact already-established B0 pack identity")
    d.pack_header(packed)
    return packed, dict(status="VERIFIED_ORIGINAL_B0_CONTAINER_SERIALIZATION", source_artifact=B0,
        original_npz_sha256=B0_NPZ, frozen_geometry_sha256=PACKER_SHA, pack=pin(packed),
        baseline_container_serialized=True, existing_downloaded_pack_claimed=False,
        weights_modified=False, model_forwards=0, optimizer_updates=0, numpy_imported=False)


def command(args, log, env=None):
    with Path(log).open("xb") as handle:
        subprocess.run([str(arg) for arg in args], check=True, stdout=handle, stderr=subprocess.STDOUT,
                       timeout=180, env=env)


def load_checks(prefix, output, receipts):
    value, installed = d.verify_installed(prefix)
    build = output / "loader-build"
    d.need(not build.exists(), "fresh clean consumer build")
    child_env = dict(os.environ)
    for key in ("CFLAGS", "CPPFLAGS", "LDFLAGS", "CC", "CXX", "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE"):
        child_env.pop(key, None)
    configure = ["/usr/bin/cmake", "-S", HERE, "-B", build, "-G", "Unix Makefiles",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=/usr/bin/gcc", "-DCMAKE_MAKE_PROGRAM=/usr/bin/make",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DHeyMero43_DIR=" + str(installed / "sdk-gcc/lib/cmake/HeyMero43")]
    compile_command = ["/usr/bin/cmake", "--build", build, "--parallel", "2"]
    command(configure, receipts / "configure.log", child_env)
    command(compile_command, receipts / "build.log", child_env)
    checker = build / "heym43_delivery_load_check"
    binary = d.read(checker)
    d.need(binary[:4] == b"\x7fELF" and os.access(checker, os.X_OK), "actual built executable consumer")
    calls = json.loads(d.read(build / "compile_commands.json", d.META_LIMIT))
    d.need(len(calls) == 1 and Path(calls[0]["file"]).resolve() == HERE / "loader_check.c"
        and Path(calls[0]["directory"]).resolve() == build.resolve(), "only the reviewed clean consumer was compiled")
    write(receipts / "loader_build.json", dict(configure=[str(arg) for arg in configure],
        build=[str(arg) for arg in compile_command], checker=pin(binary), compile_commands=calls,
        sdk_rebuilt=False, compiler="/usr/bin/gcc", cmake="/usr/bin/cmake",
        consumer_source=pin(d.read(HERE / "loader_check.c")), cmake_source=pin(d.read(HERE / "CMakeLists.txt"))))
    candidate = d.check_load(prefix, "candidate", "gcc", checker)
    write(receipts / "candidate_load.json", candidate)
    baseline = d.check_load(prefix, "B0", "gcc", checker)
    write(receipts / "B0_rollback_load.json", baseline)
    d.verify_installed(prefix)
    return dict(candidate=candidate, B0=baseline, model_load_calls=2, model_forwards=0,
                stream_init_calls=0, pcm_samples_consumed=0, existing_process_changed=False)


def execute(data_root, prepared, output):
    value, manifest_raw = d.trusted_manifest()  # Closed before external paths, network or compiler.
    saved = reviewed_readback(value)
    data_root = Path(data_root).resolve()
    binding = entry_contract(d.document(d.read(data_root / BINDING, d.META_LIMIT)))
    pipeline_contract(binding, value, manifest_raw)
    identity = guard(prepared, data_root)
    original_run = one_execution(identity)
    output = Path(output).absolute()
    d.need(output.parent.is_dir() and not output.exists()
        and all(not path.is_symlink() for path in (output, *output.parents)), "fresh immutable delivery output")
    output.mkdir(); receipts = output / "receipts"; receipts.mkdir()
    write(receipts / "STARTED.json", dict(identity=identity, pipeline_commit=binding["pipeline_commit"],
        manifest_sha256=d.sha(manifest_raw), entry_binding=pin(d.read(data_root / BINDING)),
        pipeline_files=binding["pipeline_files"], model_forwards=0, pcm_samples_consumed=0))
    write(receipts / "DELIVERY_RUN.json", original_run)
    stage = "original_artifact_authentication"
    try:
        originals = output / "originals"
        evidence = fetch_originals(value, originals, saved)
        write(receipts / "ORIGINAL_API_PROVENANCE.json", evidence)
        for name, raw in saved.items():
            with (receipts / name).open("xb") as handle: handle.write(raw)
        stage = "baseline_container_serialization"
        packed, baseline = baseline_container(originals / "B0.zip")
        baseline_path = output / "B0.pack"
        with baseline_path.open("xb") as handle: handle.write(packed)
        write(receipts / "baseline_container.json", baseline)
        stage = "exact_original_packaging_and_installation"
        archive = output / "heym43-B24R1-software-delivery.tar.gz"
        packaged = d.package(originals, baseline_path, archive)
        write(receipts / "package.json", packaged)
        prefix = output / "installed"
        installed = d.install(archive, prefix)
        write(receipts / "install.json", installed)
        _, version = d.verify_installed(prefix)
        write(receipts / "verify_installed.json", dict(status="INSTALLED_BYTES_AND_MODES_VERIFIED", version=str(version)))
        stage = "clean_installed_gcc_loader_checks"
        loads = load_checks(prefix, output, receipts)
        result = dict(schema="english-v6-b24r1-software-delivery-result-v1",
            status="DELIVERED_ORIGINAL_SOFTWARE_WITH_CANDIDATE_AND_B0_LOAD_VERIFIED", identity=identity,
            delivery_manifest_sha256=d.sha(manifest_raw), delivery_archive=packaged,
            pipeline_delivery_commit=binding["pipeline_commit"], original_sdk_commit=value["pipeline_commit"],
            original_engineering=value["engineering"], original_readback=value["readback"],
            candidate_pack_copied_unchanged=True, candidate_pack=value["files"][value["candidate_pack"]],
            baseline_container_serialized=True, weights_modified=False, loader_checks=loads,
            model_forwards=0, pcm_samples_consumed=0, stream_init_calls=0, optimizer_updates=0,
            sdk_rebuilt=False, existing_process_changed=False, release_published=False, scope=d.SCOPE)
        write(receipts / "RESULT.json", result)
        return result
    except BaseException as exc:
        write(receipts / "STOP.json", dict(status="DELIVERY_STOP", stage=stage,
            error_type=type(exc).__name__, message=str(exc), identity=identity,
            product_promotion_allowed=False, physical_ssc305=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check")
    item = modes.add_parser("execute")
    item.add_argument("--data-root", type=Path, required=True)
    item.add_argument("--prepared-parent", required=True)
    item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "check":
        chosen = d.selection()
        if chosen["selected"] is not None:
            value, _ = d.trusted_manifest(); reviewed_readback(value)
        result = dict(status="INERT" if chosen["selected"] is None else "REVIEWED_READBACK_SELECTED",
                      selected=chosen["selected"], model_forwards=0, packs_created=0)
    else:
        result = execute(args.data_root, args.prepared_parent, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
