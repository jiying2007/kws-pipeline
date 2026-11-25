#!/usr/bin/env python3
"""Deliver a reviewed B28 software candidate through unchanged byte transport.

Selection stays closed until the complete actual dual quality and engineering
proofs have been independently read and committed. This B28 adapter reuses the
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
contract_spec = importlib.util.spec_from_file_location("b28_delivery_contract", HERE / "delivery_b28.py")
contract = importlib.util.module_from_spec(contract_spec)
contract_spec.loader.exec_module(contract)
definition = importlib.util.spec_from_file_location("b28_original_delivery_once", ORIGINAL)
original = importlib.util.module_from_spec(definition)
definition.loader.exec_module(original)
original.d = contract.base
d = original.d
ROOT = original.ROOT

BRANCH = "refs/heads/research/en-v6-b28-engineering-delivery-once-20261125"
WORKFLOW = ".github/workflows/english-v6-b28-engineering-delivery-once.yml"
TEMPLATE = "automation/english_v6_b28_engineering_delivery_workflow_template.yml"
BINDING = "automation/english_v6_b28_engineering_delivery_inputs.json"
TOKEN = b"__B28_REVIEWED_DELIVERY_PARENT__"
ENTRY_SCHEMA = "english-v6-b28-software-delivery-entry-v1"
REPORT_SCHEMA = "english-v6-b28-verified-original-engineering-v1"
ENGINEERING_WORKFLOW = ".github/workflows/english-v6-b28-engineering-once.yml"
ENGINEERING_BRANCH = "research/en-v6-b28-engineering-once-20261125"
DELIVERY_FILES = (original.DELIVERY_FILES - {"research/english_v6_delivery/selection.json"}) | {
    "research/english_v6_delivery/once_b28.py", "research/english_v6_delivery/delivery_b28.py",
    "research/english_v6_delivery/selection_b28.json", "research/english_v6_delivery/b28_binding.json",
}

# The imported module is private to this adapter. The original file, its B0
# identities and original callable implementations remain unchanged.
for name in ("BRANCH", "WORKFLOW", "TEMPLATE", "BINDING", "TOKEN", "ENTRY_SCHEMA", "DELIVERY_FILES"):
    setattr(original, name, globals()[name])


def reviewed_readback(value):
    """Authenticate only the specifically bound report and saved verifier bytes."""
    bound = contract.configuration(True)
    d.need(value["readback"] == dict(report=bound["engineering_report"], verifier=bound["engineering_verifier"]),
           "same actual B28 engineering report and verifier as the reviewed binding")
    paths = original.readback_paths(value)
    saved = {name: d.read(path, d.META_LIMIT) for name, path in paths.items()}
    return validate_reviewed_readback(value, saved)


def validate_reviewed_readback(value, saved):
    """Metadata-only intake; actual proof must already be independently read."""
    bound = contract.configuration(True)
    d.manifest_contract(value)
    d.need(set(saved) == {"completion.json", "readback.py"}
        and value["readback"] == dict(report=bound["engineering_report"], verifier=bound["engineering_verifier"]),
        "same complete actual B28 report and verifier binding")
    d.matches(saved["completion.json"], bound["engineering_report"])
    d.matches(saved["readback.py"], bound["engineering_verifier"])
    report = d.document(saved["completion.json"])
    d.need(report["schema"] == REPORT_SCHEMA
        and report["status"] == "VERIFIED_ORIGINAL_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE"
        and value["candidate_id"] == report["candidate_id"] == "B28"
        and value["candidate_pack"] == "models/b28-challenger.pack"
        and report["baseline_pack_requires_separate_byte_authentication"] is True
        and report["readback_verifier"] == bound["engineering_verifier"],
        "actual complete independent B28 readback and exact verifier identity")
    for key in ("candidate_pack", "pipeline_commit", "files", "engineering", "quality_locks", "scope"):
        d.need(d.encoded({key: report[key]}) == d.encoded({key: value[key]}),
               "reviewed manifest agrees with original B28 readback: " + key)
    counts = dict(original_engineering_artifacts=4, raw_arrays=140, event_csvs=84, original_conditions=28,
        complete_original_quality_passes=2, frozen_pipeline_files=13, natural_objective_returns_verified=384,
        external_training_originals=71, replication_external_total_originals=89)
    checks = report["checks"]
    d.need(all(type(checks[key]) is int and checks[key] == expected for key, expected in counts.items())
        and type(checks["original_producer_files"]) is int and checks["original_producer_files"] > 0
        and all(checks[key] is True for key in ("complete_original_TRAIN_verified", "complete_original_Q1_failure_verified",
            "complete_original_B27_failure_verified", "external_originals_retained", "original_pack_tensor_bytes_verified",
            "unchanged_original_exports", "installed_sdk_bytes_verified", "every_saved_array_and_event_recomputed",
            "original_source_admission_recomputed", "original_api_and_git_provenance_verified", "natural_no_event_terms_recomputed")),
        "complete original14x2x5 engineering, dual quality, TRAIN and prior failure were independently verified")
    d.need(all(type(report[key]) is int and report[key] == 0 for key in
        ("new_model_calls", "new_source_calls", "new_observer_calls", "compiler_calls", "packs_created"))
        and report["candidate_selection_changed"] is False,
        "independent engineering readback created no new scientific or delivery evidence")
    for key in ("original_training_report_sha256", "original_B27_failure_report_sha256"):
        d.need(type(report[key]) is str and d.re.fullmatch(r"[0-9a-f]{64}", report[key]), "complete actual B28 qualification digest")
    certificate = report["original_B27_failure_certification"]
    d.need(type(certificate) is dict
        and set(certificate) == {"schema", "original_reader_path", "original_reader_sha256",
            "original_report_sha256", "adapter_path", "adapter_sha256", "adapter_report_schema",
            "adapter_report_sha256", "original_standalone_continuation_succeeded",
            "new_model_calls", "new_source_calls", "new_observer_calls", "new_optimizer_updates"}
        and certificate["schema"] == "english-v6-b28-original-b27-failure-certification-binding-v1"
        and certificate["original_report_sha256"] == report["original_B27_failure_report_sha256"]
        and certificate["original_reader_path"] == "automation/english_v6_b27_quality_readback.py"
        and certificate["original_reader_sha256"] == "87bc31725380cbf70d70d6b46929a724dc6bff656d421c5ae63b54c0724e31aa"
        and certificate["adapter_path"] == "automation/english_v6_b27_primary_no_promotion_readback.py"
        and certificate["adapter_report_schema"] == "english-v6-b27-primary-rooted-no-promotion-readback-v1"
        and certificate["original_standalone_continuation_succeeded"] is False
        and all(type(certificate[key]) is int and certificate[key] == 0 for key in
            ("new_model_calls", "new_source_calls", "new_observer_calls", "new_optimizer_updates")),
        "same complete original B27 failure certification with no new scientific calls")
    for key in ("original_reader_sha256", "adapter_sha256", "adapter_report_sha256"):
        d.need(type(certificate[key]) is str and d.re.fullmatch(r"[0-9a-f]{64}", certificate[key]),
               "exact original reader and import-scope adapter qualification identity")
    d.need(type(report["quality"]) is dict and set(report["quality"]) == {"primary", "replication"},
           "both complete original B28 quality readbacks are retained")
    model_files = [name for name in value["files"] if name.startswith("models/") and name.endswith(".pt")]
    d.need(len(model_files) == 1, "one exact original candidate PT export")
    model_hashes = report["quality"]["primary"]["model_pt_sha256"]
    d.need(type(model_hashes) is dict and set(model_hashes) == {"B0", "B9", "control", "challenger"}
        and all(type(digest) is str and d.re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in model_hashes.values())
        and len(set(model_hashes.values())) == 4
        and model_hashes["challenger"] == value["files"][model_files[0]]["sha256"],
        "the exact exported candidate PT is the same challenger that passed both original panels")
    for panel in ("primary", "replication"):
        proof, lock = report["quality"][panel], value["quality_locks"][panel]
        d.need(proof["schema"] == "english-v6-b28-verified-original-quality-v1"
            and proof["panel"] == panel and proof["panel_gate"] == "PASS"
            and proof["model_pt_sha256"] == model_hashes
            and proof["original_training_report_sha256"] == report["original_training_report_sha256"]
            and proof["original_B27_failure_report_sha256"] == report["original_B27_failure_report_sha256"]
            and d.encoded(proof["original_B27_failure_certification"]) == d.encoded(certificate)
            and proof["external_proof_binding_sha256"] == lock["external_proof_binding_sha256"]
            and proof["original_result_sha256"] == lock["original_result_sha256"]
            and proof["original_lock_sha256"] == d.sha(json.dumps(lock, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode())
            and all(proof[key] is True for key in ("all_original_arrays_verified", "original_source_admission_recomputed",
                "original_producer_git_verified", "all_original_training_proof_verified", "complete_original_Q1_failure_verified",
                "complete_original_B27_failure_verified", "natural_no_event_terms_recomputed",
                "captured_code_matches_pinned_local_reader_root", "old_stops_and_reserves_preserved"))
            and type(proof["natural_objective_returns_verified"]) is int and proof["natural_objective_returns_verified"] == 384
            and type(proof["external_training_originals"]) is int and proof["external_training_originals"] == 71
            and type(proof["external_total_originals"]) is int and proof["external_total_originals"] == (71 if panel == "primary" else 89)
            and proof["primary_independently_verified"] is (panel == "replication")
            and proof["replication_open_allowed"] is (panel == "primary")
            and proof["candidate_engineering_validation_allowed"] is (panel == "replication")
            and all(type(proof[key]) is int and proof[key] == 0 for key in
                ("new_model_calls", "new_source_calls", "new_observer_calls", "new_optimizer_updates"))
            and proof["promotion_allowed"] is proof["product_released"] is proof["automatic_next_assessment_allowed"] is False,
            "same complete actual B28 dual PASS and original external qualification chain")
    return saved


def artifact_name(role, run_id):
    if role == "B0":
        return None
    stems = dict(result="software-sdk-evidence-", producer="engineering-producer-",
                 consumed="engineering-consumed", input_proof="engineering-input-proof-")
    return "english-v6-b28-" + stems[role] + ("" if role == "consumed" else str(run_id))


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
            and run["head_branch"] == ENGINEERING_BRANCH, "actual original B28 engineering role")


original.reviewed_readback = reviewed_readback
original.artifact_name = artifact_name
original.provenance = provenance


def execute(data_root, prepared, output):
    contract.configuration(True)
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
        archive = output / "heym43-B28-software-delivery.tar.gz"
        packaged = d.package(originals, baseline_path, archive)
        original.write(receipts / "package.json", packaged)
        prefix = output / "installed"
        installed = d.install(archive, prefix)
        original.write(receipts / "install.json", installed)
        _, version = d.verify_installed(prefix)
        original.write(receipts / "verify_installed.json", dict(status="INSTALLED_BYTES_AND_MODES_VERIFIED", version=str(version)))
        stage = "clean_installed_gcc_loader_checks"
        loads = original.load_checks(prefix, output, receipts)
        result = dict(schema="english-v6-b28-software-delivery-result-v1",
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
