#!/usr/bin/env python3
"""Deliver a reviewed B27 software candidate through unchanged byte transport.

Selection stays closed until the complete actual dual quality and engineering
proofs have been independently read and committed. This B27 adapter reuses the
original B0 serialization, package installation and zero-forward loader checks.
It neither executes the saved verifier nor selects or evaluates a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "once.py"
ORIGINAL_SHA = "30cb080cc58b72ed9ed22ce50bb27081a51c85074a851be1e34c55f7257246dc"
if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA:
    raise ValueError("unchanged original zero-forward delivery implementation")
definition = importlib.util.spec_from_file_location("b27_original_delivery_once", ORIGINAL)
original = importlib.util.module_from_spec(definition)
definition.loader.exec_module(original)
d = original.d
ROOT = original.ROOT

BRANCH = "refs/heads/research/en-v6-b27-engineering-delivery-once-20261125"
WORKFLOW = ".github/workflows/english-v6-b27-engineering-delivery-once.yml"
TEMPLATE = "automation/english_v6_b27_engineering_delivery_workflow_template.yml"
BINDING = "automation/english_v6_b27_engineering_delivery_inputs.json"
TOKEN = b"__B27_REVIEWED_DELIVERY_PARENT__"
ENTRY_SCHEMA = "english-v6-b27-software-delivery-entry-v1"
REPORT_SCHEMA = "english-v6-b27-verified-original-engineering-v1"
ENGINEERING_WORKFLOW = ".github/workflows/english-v6-b27-engineering-once.yml"
ENGINEERING_BRANCH = "research/en-v6-b27-engineering-once-20261125"
DELIVERY_FILES = original.DELIVERY_FILES | {"research/english_v6_delivery/once_b27.py"}

# The imported module is private to this adapter. The original file, its B0
# identities and original callable implementations remain unchanged.
for name in ("BRANCH", "WORKFLOW", "TEMPLATE", "BINDING", "TOKEN", "ENTRY_SCHEMA", "DELIVERY_FILES"):
    setattr(original, name, globals()[name])


def reviewed_readback(value):
    """Authenticate exact reviewed report/verifier bytes without running them."""
    paths = original.readback_paths(value)
    saved = {name: d.read(path, d.META_LIMIT) for name, path in paths.items()}
    return validate_reviewed_readback(value, saved)


def validate_reviewed_readback(value, saved):
    """The same read-only contract is used when preparing an unselected draft."""
    d.matches(saved["completion.json"], value["readback"]["report"])
    d.matches(saved["readback.py"], value["readback"]["verifier"])
    report = d.document(saved["completion.json"])
    d.need(report["schema"] == REPORT_SCHEMA
        and report["status"] == "VERIFIED_ORIGINAL_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE"
        and value["candidate_id"] == report["candidate_id"] == "B27"
        and value["candidate_pack"] == "models/b27-challenger.pack"
        and report["baseline_pack_requires_separate_byte_authentication"] is True
        and report["readback_verifier"] == value["readback"]["verifier"],
        "actual complete independent B27 readback and exact verifier identity")
    for key in ("candidate_pack", "pipeline_commit", "files", "engineering", "quality_locks", "scope"):
        d.need(report[key] == value[key], "reviewed manifest agrees with original B27 readback: " + key)
    d.need(set(value["quality_locks"]) == {"primary", "replication"}
        and all(value["quality_locks"][panel]["schema"] == "english-v6-b27-original-quality-lock-v1"
            and value["quality_locks"][panel]["panel"] == panel for panel in ("primary", "replication")),
        "both complete B27 panel locks belong to this reviewed readback")
    d.need(report["checks"]["complete_original_TRAIN_verified"] is True
        and report["checks"]["complete_original_Q1_failure_verified"] is True
        and report["checks"]["external_originals_retained"] is True
        and report["original_B24Q1_failure_report_sha256"] == "9cd5df45c5b9ca22b0eb56f3fdda4788b2420e906a35bb754af9c23eb38be62f"
        and type(report["original_training_report_sha256"]) is str
        and d.re.fullmatch(r"[0-9a-f]{64}", report["original_training_report_sha256"]),
        "complete actual B27 training and original Q1 terminal qualification were independently verified")
    for panel in ("primary", "replication"):
        proof = report["quality"][panel]
        d.need(proof["schema"] == "english-v6-b27-verified-original-quality-v1"
            and proof["panel"] == panel and proof["panel_gate"] == "PASS"
            and proof["original_training_report_sha256"] == report["original_training_report_sha256"]
            and proof["original_B24Q1_failure_report_sha256"] == report["original_B24Q1_failure_report_sha256"]
            and proof["external_proof_binding_sha256"] == value["quality_locks"][panel]["external_proof_binding_sha256"]
            and proof["all_original_training_proof_verified"] is proof["complete_original_Q1_failure_verified"] is True,
            "same complete original B27 dual PASS and full external qualification chain")
    return saved


def artifact_name(role, run_id):
    if role == "B0":
        return None
    stems = dict(result="software-sdk-evidence-", producer="engineering-producer-",
                 consumed="engineering-consumed", input_proof="engineering-input-proof-")
    return "english-v6-b27-" + stems[role] + ("" if role == "consumed" else str(run_id))


def provenance(pin_value, run, artifact, commit, role):
    d.need(role in d.ORIGIN_ROLES | {"B0"}, "exact original delivery artifact role")
    d.artifact_pin(pin_value)
    d.need(run["id"] == pin_value["run_id"] and run["head_sha"] == commit["sha"] == pin_value["run_head"]
        and run["status"] == "completed" and run["conclusion"] == "success" and run["run_attempt"] == 1
        and run["repository"]["full_name"] == original.DATA_REPO and run["repository"]["private"] is False
        and run["head_repository"]["full_name"] == original.DATA_REPO and run["head_repository"]["private"] is False
        and artifact["id"] == pin_value["id"] and artifact["expired"] is False
        and artifact["size_in_bytes"] == pin_value["bytes"] and artifact["digest"] == "sha256:" + pin_value["sha256"]
        and artifact["workflow_run"]["id"] == pin_value["run_id"]
        and artifact["workflow_run"]["head_sha"] == pin_value["run_head"],
        "exact successful original public artifact")
    if role == "B0":
        d.need(pin_value == original.B0, "unchanged complete original B0 artifact identity")
    else:
        d.need(artifact["name"] == artifact_name(role, pin_value["run_id"])
            and run["event"] == "push" and run["path"] == ENGINEERING_WORKFLOW
            and run["head_branch"] == ENGINEERING_BRANCH, "actual original B27 engineering role")


original.reviewed_readback = reviewed_readback
original.artifact_name = artifact_name
original.provenance = provenance


def execute(data_root, prepared, output):
    value, manifest_raw = d.trusted_manifest()
    saved = reviewed_readback(value)
    data_root = Path(data_root).resolve()
    binding = original.entry_contract(d.document(d.read(data_root / BINDING, d.META_LIMIT)))
    original.pipeline_contract(binding, value, manifest_raw)
    identity = original.guard(prepared, data_root)
    original_run = original.one_execution(identity)
    output = Path(output).absolute()
    d.need(output.parent.is_dir() and not output.exists()
        and all(not path.is_symlink() for path in (output, *output.parents)), "fresh immutable delivery output")
    output.mkdir(); receipts = output / "receipts"; receipts.mkdir()
    original.write(receipts / "STARTED.json", dict(identity=identity, pipeline_commit=binding["pipeline_commit"],
        manifest_sha256=d.sha(manifest_raw), entry_binding=original.pin(d.read(data_root / BINDING)),
        pipeline_files=binding["pipeline_files"], model_forwards=0, pcm_samples_consumed=0))
    original.write(receipts / "DELIVERY_RUN.json", original_run)
    stage = "original_artifact_authentication"
    try:
        originals = output / "originals"
        evidence = original.fetch_originals(value, originals, saved)
        original.write(receipts / "ORIGINAL_API_PROVENANCE.json", evidence)
        for name, raw in saved.items():
            with (receipts / name).open("xb") as handle:
                handle.write(raw)
        stage = "baseline_container_serialization"
        packed, baseline = original.baseline_container(originals / "B0.zip")
        baseline_path = output / "B0.pack"
        with baseline_path.open("xb") as handle:
            handle.write(packed)
        original.write(receipts / "baseline_container.json", baseline)
        stage = "exact_original_packaging_and_installation"
        archive = output / "heym43-B27-software-delivery.tar.gz"
        packaged = d.package(originals, baseline_path, archive)
        original.write(receipts / "package.json", packaged)
        prefix = output / "installed"
        installed = d.install(archive, prefix)
        original.write(receipts / "install.json", installed)
        _, version = d.verify_installed(prefix)
        original.write(receipts / "verify_installed.json", dict(status="INSTALLED_BYTES_AND_MODES_VERIFIED", version=str(version)))
        stage = "clean_installed_gcc_loader_checks"
        loads = original.load_checks(prefix, output, receipts)
        result = dict(schema="english-v6-b27-software-delivery-result-v1",
            status="DELIVERED_ORIGINAL_SOFTWARE_WITH_CANDIDATE_AND_B0_LOAD_VERIFIED", identity=identity,
            delivery_manifest_sha256=d.sha(manifest_raw), delivery_archive=packaged,
            pipeline_delivery_commit=binding["pipeline_commit"], original_sdk_commit=value["pipeline_commit"],
            original_engineering=value["engineering"], original_readback=value["readback"],
            candidate_pack_copied_unchanged=True, candidate_pack=value["files"][value["candidate_pack"]],
            baseline_container_serialized=True, weights_modified=False, loader_checks=loads,
            model_forwards=0, pcm_samples_consumed=0, stream_init_calls=0, optimizer_updates=0,
            sdk_rebuilt=False, existing_process_changed=False, release_published=False, scope=d.SCOPE)
        original.write(receipts / "RESULT.json", result)
        return result
    except BaseException as exc:
        original.write(receipts / "STOP.json", dict(status="DELIVERY_STOP", stage=stage,
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
            value, _ = d.trusted_manifest()
            reviewed_readback(value)
        result = dict(status="INERT" if chosen["selected"] is None else "REVIEWED_READBACK_SELECTED",
                      selected=chosen["selected"], model_forwards=0, packs_created=0)
    else:
        result = execute(args.data_root, args.prepared_parent, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
