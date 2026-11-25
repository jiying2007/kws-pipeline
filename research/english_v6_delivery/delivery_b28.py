#!/usr/bin/env python3
"""Separate B28 delivery contract over the exact frozen byte transport.

The B27 files and default selection stay unchanged. The private transport module
uses a separate B28 selection and requires actual pinned engineering readback
bytes before a selection can authorize any packaging or installation.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "delivery.py"
ORIGINAL_SHA = "b10643eb1f47ebcdafe5f9dc69e47c0617fcc1edf483ff82be1a40ad8fa0464d"
if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA:
    raise ValueError("unchanged original delivery byte transport")
definition = importlib.util.spec_from_file_location("b28_private_delivery_bytes", ORIGINAL)
base = importlib.util.module_from_spec(definition)
definition.loader.exec_module(base)
CONFIG = HERE / "b28_binding.json"
SELECTION = HERE / "selection_b28.json"
CONFIG_SCHEMA = "english-v6-b28-software-delivery-binding-v1"

# Shared byte and shape helpers keep their original implementations. The new
# manifest contract below changes only candidate/quality-lock specialization.
need, read, document, encoded = base.need, base.read, base.document, base.encoded
safe_name, valid_pin, artifact_pin = base.safe_name, base.valid_pin, base.artifact_pin
prescribed_mode = base.prescribed_mode
PACK_BYTES, B0_SHA, B0_PATH = base.PACK_BYTES, base.B0_SHA, base.B0_PATH
PIPELINE_COMMIT, LIMIT, META_LIMIT = base.PIPELINE_COMMIT, base.LIMIT, base.META_LIMIT
MAX_MEMBERS, BACKENDS, ORIGIN_ROLES = base.MAX_MEMBERS, base.BACKENDS, base.ORIGIN_ROLES
SCOPE, REQUIRED_SDK = base.SCOPE, base.REQUIRED_SDK
_original_selection = base.selection
_original_trusted_manifest = base.trusted_manifest


def configuration(require_enabled=False):
    value = document(read(CONFIG, META_LIMIT))
    need(set(value) == {"schema", "enabled", "engineering_report", "engineering_verifier"}
         and value["schema"] == CONFIG_SCHEMA and type(value["enabled"]) is bool,
         "exact separate B28 delivery proof binding")
    if value["enabled"] is False:
        need(value["engineering_report"] is value["engineering_verifier"] is None,
             "inert B28 delivery has no invented engineering report or verifier pin")
    else:
        for role in ("engineering_report", "engineering_verifier"):
            valid_pin(value[role], META_LIMIT)
    need(not require_enabled or value["enabled"] is True,
         "INERT: complete actual B28 engineering report and verifier remain unbound")
    return value


def selection():
    bound = configuration()
    value = _original_selection()
    if bound["enabled"] is False:
        need(value["selected"] is value["manifest"] is None,
             "INERT: B28 selection cannot precede complete original engineering readback")
    if value["selected"] is not None:
        need(value["selected"] == "B28", "separate B28 selection only")
    return value


def trusted_manifest(raw=None):
    bound = configuration(True)
    value, manifest_raw = _original_trusted_manifest(raw)
    need(value["readback"] == dict(report=bound["engineering_report"], verifier=bound["engineering_verifier"]),
         "selected B28 manifest uses the exact reviewed engineering report and verifier")
    return value, manifest_raw


def manifest_contract(value):
    need(set(value) == {"schema", "candidate_id", "pipeline_commit", "candidate_pack", "files",
                        "engineering", "quality_locks", "readback", "scope"}
         and value["schema"] == "heym43-software-delivery-v1"
         and value["candidate_id"] == "B28" and value["candidate_pack"] == "models/b28-challenger.pack"
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
    need(type(locks) is dict and set(locks) == {"primary", "replication"}, "two original B28 quality locks")
    for panel, lock in locks.items():
        fields = {"schema", "panel", "result", "producer", "consumed", "evaluator_sha256", "inputs_sha256",
                  "producer_manifest_sha256", "original_result_sha256", "external_proof_binding_sha256"}
        need(type(lock) is dict and set(lock) == fields and lock["panel"] == panel
             and lock["schema"] == "english-v6-b28-original-quality-lock-v1", "one complete original B28 quality lock")
        for role in ("result", "producer", "consumed"):
            artifact_pin(lock[role], LIMIT)
        need(len({lock[k]["id"] for k in ("result", "producer", "consumed")}) == 3
             and len({(lock[k]["run_id"], lock[k]["run_head"]) for k in ("result", "producer", "consumed")}) == 1,
             "three distinct same-run original B28 quality artifacts")
        for key in ("evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256",
                    "external_proof_binding_sha256"):
            need(type(lock[key]) is str and re.fullmatch(r"[0-9a-f]{64}", lock[key]), "exact complete original B28 proof digest")
    need(locks["primary"]["result"]["run_id"] != locks["replication"]["result"]["run_id"],
         "two independent original B28 panel executions")
    need(type(value["readback"]) is dict and set(value["readback"]) == {"report", "verifier"},
         "exact independent full original readback and its verifier are pinned")
    for pin in value["readback"].values(): valid_pin(pin, META_LIMIT)
    return value


# The private module's existing transport callables resolve these globals.
# No module imported by a B27 entry is modified, and no old file is rewritten.
base.SELECTION = SELECTION
base.selection = selection
base.manifest_contract = manifest_contract
base.trusted_manifest = trusted_manifest
