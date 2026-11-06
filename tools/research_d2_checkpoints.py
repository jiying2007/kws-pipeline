#!/usr/bin/env python3
"""Private per-phase evidence capsules and externally acknowledged barriers.

This module does not upload anything or certify a storage provider. The caller
must save the exact ZIP and retain the actual provider response. A launch intent
belongs to one live driver invocation: a new invocation cannot reuse it merely
because a restored queue has no claim. Completed phases remain reusable after
their original queue receipts have been independently authenticated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_queue as q

SCHEMA = "kws-d2-durable-checkpoint-v1"
SAVE = "kws-persistent-checkpoint-save-v1"
PHASES = ("before-generation", "generation-complete", "before-asr", "chunk-complete")
INTENTS = {"before-generation", "before-asr"}


def read(path):
    return q.loads(q.read_bytes(path))


def pin(path):
    return {"path": str(q.safe_path(path)), **q.file_record(path)}


def regular_tree(root):
    """Immutable phase files only; callers choose the phase's precise roots."""
    root = q.safe_path(root)
    if root.is_file():
        return [root]
    q.require(root.is_dir(), "D2_CHECKPOINT_SOURCE_MISSING:" + str(root))
    result = []
    for path in sorted(root.rglob("*")):
        q.require(not path.is_symlink(), "D2_CHECKPOINT_PATH_ALIAS")
        if path.is_file() and not (path.name.endswith(".lock") or path.name == "heartbeat.json" or
                                   "__pycache__" in path.parts):
            result.append(path)
    return result


def create_capsule(folder, context, source_paths):
    folder = q.safe_path(folder)
    q.require(not folder.exists(), "D2_CHECKPOINT_CREATE_ONLY")
    folder.mkdir(mode=0o700, parents=True)
    # This marker is inside the saved ZIP. Restoring only phase source files
    # therefore leaves an explicit old intent and cannot look like a fresh run.
    marker = folder / "phase-marker.json"
    q.write_json(marker, {"schema": SCHEMA, "context": context})
    source_paths = list(source_paths) + [marker]
    originals = sorted({str(q.safe_path(p)) for p in source_paths})
    q.require(originals and len(originals) <= 100000, "D2_CHECKPOINT_FILE_SCOPE")
    files = {f"files/{index:06d}": pin(path) for index, path in enumerate(originals)}
    q.require(sum(row["bytes"] for row in files.values()) <= 2 * 1024 ** 3,
              "D2_CHECKPOINT_SIZE_BOUND")
    manifest = {"schema": SCHEMA, "context": context, "checkpoint_root": str(folder), "files": files,
                "new_model_calls": 0, "local_capsule_is_remote_persistence_proof": False}
    raw = q.canonical(manifest)
    capsule = folder / ("checkpoint-" + q.sha(q.canonical(context))[:24] + ".zip")
    with zipfile.ZipFile(capsule, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, row in files.items():
            before = q.file_record(row["path"])
            q.require(before == {k: row[k] for k in ("bytes", "sha256")}, "D2_CHECKPOINT_SOURCE_CHANGED")
            archive.write(row["path"], name)
            q.require(q.file_record(row["path"]) == before, "D2_CHECKPOINT_SOURCE_CHANGED_DURING_SAVE")
        archive.writestr("MANIFEST.json", raw)
    descriptor = {"schema": SCHEMA, "context": context, "capsule": pin(capsule),
                  "member_manifest_sha256": q.sha(raw), "files": len(files),
                  "provider_save_required": True}
    q.write_json(folder / "capsule.json", descriptor)
    verify_capsule(folder)
    return descriptor


def verify_capsule(folder, *, current_sources=False):
    folder = q.safe_path(folder)
    value = read(folder / "capsule.json")
    capsule = folder / Path(value["capsule"]["path"]).name
    q.require(value["schema"] == SCHEMA and q.file_record(capsule) ==
              {k: value["capsule"][k] for k in ("bytes", "sha256")}, "D2_CHECKPOINT_CAPSULE_CHANGED")
    with zipfile.ZipFile(capsule) as archive:
        names = archive.namelist()
        q.require(len(names) == len(set(names)) and "MANIFEST.json" in names, "D2_CHECKPOINT_MEMBERS")
        q.require(archive.getinfo("MANIFEST.json").file_size <= 16 * 1024 ** 2, "D2_CHECKPOINT_MANIFEST_SIZE")
        manifest_bytes = archive.read("MANIFEST.json")
        q.require(q.sha(manifest_bytes) == value["member_manifest_sha256"], "D2_CHECKPOINT_MANIFEST_CHANGED")
        manifest = q.loads(manifest_bytes)
        q.require(manifest["schema"] == SCHEMA and manifest["context"] == value["context"] and
                  set(names) == set(manifest["files"]) | {"MANIFEST.json"} and
                  len(manifest["files"]) == value["files"], "D2_CHECKPOINT_MEMBER_SCOPE")
        for name, row in manifest["files"].items():
            q.relative(name)
            q.require(archive.getinfo(name).file_size == row["bytes"], "D2_CHECKPOINT_MEMBER_SIZE")
            with archive.open(name) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            q.require(digest == row["sha256"], "D2_CHECKPOINT_MEMBER_HASH")
            if current_sources:
                q.require(q.file_record(row["path"]) == {k: row[k] for k in ("bytes", "sha256")},
                          "D2_CHECKPOINT_LIVE_SOURCE_CHANGED:" + row["path"])
    return value


def restore_capsule(capsule, expected_sha256, allowed_roots):
    """Restore authenticated phase bytes create-only, keeping original paths.

    The explicit allowlist is required because source paths came from a saved
    artifact. Existing changed files, aliases and traversal fail before writes.
    Remote acknowledgements are never invented. A restored launch intent still
    belongs to its old invocation and cannot authorize new inference.
    """
    capsule = q.safe_path(capsule)
    capsule_record = q.file_record(capsule)
    q.require(q.SHA.fullmatch(expected_sha256) and capsule_record["sha256"] == expected_sha256,
              "D2_CHECKPOINT_RESTORE_SHA_REQUIRED")
    allowed = [q.safe_path(root) for root in allowed_roots]
    q.require(allowed and all(root != Path("/") for root in allowed), "D2_CHECKPOINT_RESTORE_ROOTS_REQUIRED")
    def destination(value):
        q.require(type(value) is str and value.startswith("/") and ".." not in Path(value).parts,
                  "D2_CHECKPOINT_RESTORE_ABSOLUTE_PATH")
        path = q.safe_path(value)
        q.require(any(root == path or root in path.parents for root in allowed), "D2_CHECKPOINT_RESTORE_OUTSIDE_ROOT")
        return path
    with zipfile.ZipFile(capsule) as archive:
        names = archive.namelist()
        q.require(len(names) == len(set(names)) and "MANIFEST.json" in names, "D2_CHECKPOINT_MEMBERS")
        q.require(archive.getinfo("MANIFEST.json").file_size <= 16 * 1024 ** 2, "D2_CHECKPOINT_MANIFEST_SIZE")
        raw = archive.read("MANIFEST.json")
        manifest = q.loads(raw)
        q.require(manifest.get("schema") == SCHEMA and set(names) == set(manifest["files"]) | {"MANIFEST.json"},
                  "D2_CHECKPOINT_RESTORE_MANIFEST")
        q.require(0 < len(manifest["files"]) <= 100000 and
                  sum(row["bytes"] for row in manifest["files"].values()) <= 2 * 1024 ** 3,
                  "D2_CHECKPOINT_RESTORE_SIZE_BOUND")
        folder = destination(manifest["checkpoint_root"])
        target_zip = folder / ("checkpoint-" + q.sha(q.canonical(manifest["context"]))[:24] + ".zip")
        descriptor = {"schema": SCHEMA, "context": manifest["context"],
                      "capsule": {"path": str(target_zip), **capsule_record},
                      "member_manifest_sha256": q.sha(raw), "files": len(manifest["files"]),
                      "provider_save_required": True}
        targets = []
        seen_targets = set()
        for name, row in manifest["files"].items():
            q.relative(name)
            target = destination(row["path"])
            q.require(target not in seen_targets and target not in (target_zip, folder / "capsule.json"),
                      "D2_CHECKPOINT_RESTORE_PATH_COLLISION")
            seen_targets.add(target)
            q.require(0 <= row["bytes"] <= 256 * 1024 ** 2 and archive.getinfo(name).file_size == row["bytes"],
                      "D2_CHECKPOINT_RESTORE_MEMBER_SIZE")
            with archive.open(name) as stream:
                q.require(hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"],
                          "D2_CHECKPOINT_RESTORE_MEMBER_CHANGED")
            record = {k: row[k] for k in ("bytes", "sha256")}
            q.require(not target.exists() or q.file_record(target) == record, "D2_CHECKPOINT_RESTORE_WOULD_OVERWRITE")
            targets.append((name, target))
        q.require(not target_zip.exists() or q.file_record(target_zip) == capsule_record,
                  "D2_CHECKPOINT_RESTORE_ZIP_CONFLICT")
        descriptor_path = folder / "capsule.json"
        q.require(not descriptor_path.exists() or q.read_bytes(descriptor_path) == q.canonical(descriptor),
                  "D2_CHECKPOINT_RESTORE_DESCRIPTOR_CONFLICT")
        # No mutation occurs before every member/path/conflict has passed.
        for name, target in targets:
            if not target.exists():
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                q.write_new(target, archive.read(name))
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not target_zip.exists():
            q.write_new(target_zip, q.read_bytes(capsule, maximum=2 * 1024 ** 3))
        if not descriptor_path.exists():
            q.write_json(descriptor_path, descriptor)
    verify_capsule(folder)
    return {"status": "restored_original_phase_bytes", "checkpoint": str(folder), "files": len(targets),
            "new_model_calls": 0, "acknowledgement_created": False,
            "restored_launch_intent_reusable": False, "original_paths_rewritten": False}


def contains(value, needle):
    if isinstance(value, dict):
        return any(contains(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(contains(item, needle) for item in value)
    return value == needle


def provider_failed(value):
    if isinstance(value, dict):
        if value.get("isError") is True or value.get("success") is False or value.get("ok") is False:
            return True
        if value.get("status") in ("failed", "error", "rejected", "cancelled") or value.get("error") or value.get("errors"):
            return True
        return any(provider_failed(item) for item in value.values())
    if isinstance(value, list):
        return any(provider_failed(item) for item in value)
    return False


def acknowledge(folder, save_receipt):
    """Bind actual caller-supplied provider output; do not invent a remote proof."""
    folder, save_receipt = q.safe_path(folder), q.safe_path(save_receipt)
    descriptor = verify_capsule(folder)
    saved = read(save_receipt)
    q.require(saved.get("schema") == SAVE and saved.get("status") == "saved" and
              type(saved.get("provider")) is str and saved["provider"] and
              type(saved.get("object_id")) is str and saved["object_id"] and
              type(saved.get("provider_response")) is dict and saved["provider_response"] and
              not provider_failed(saved["provider_response"]) and
              contains(saved["provider_response"], saved["object_id"]) and
              saved.get("artifact") == descriptor["capsule"] and
              saved.get("member_manifest_sha256") == descriptor["member_manifest_sha256"],
              "D2_CHECKPOINT_PROVIDER_SAVE_RECEIPT_REQUIRED")
    # Retain the actual response. If this process previously stopped between
    # the two metadata writes, only the exact already-recorded save may finish
    # the acknowledgement. This repeats no upload and grants no new intent.
    saved_path = folder / "provider-save.json"
    if saved_path.exists():
        q.require(q.read_bytes(saved_path) == q.canonical(saved), "D2_CHECKPOINT_PROVIDER_SAVE_CONFLICT")
    else:
        q.write_json(saved_path, saved)
    value = {"schema": SCHEMA, "descriptor": pin(folder / "capsule.json"),
             "capsule": descriptor["capsule"], "member_manifest_sha256": descriptor["member_manifest_sha256"],
             "provider_save": pin(folder / "provider-save.json"), "provider": saved["provider"],
             "object_id": saved["object_id"], "version": saved.get("version"),
             "provider_response_supplied_by_operator": True, "remote_bytes_independently_retrieved": False}
    ack_path = folder / "ack.json"
    if ack_path.exists():
        q.require(q.read_bytes(ack_path) == q.canonical(value), "D2_CHECKPOINT_ACK_CONFLICT")
    else:
        q.write_json(ack_path, value)
    return verify_ack(folder)


def verify_ack(folder):
    folder = q.safe_path(folder)
    descriptor = verify_capsule(folder)
    ack = read(folder / "ack.json")
    q.require(ack["schema"] == SCHEMA and ack["descriptor"] == pin(folder / "capsule.json") and
              ack["capsule"] == descriptor["capsule"] and
              ack["member_manifest_sha256"] == descriptor["member_manifest_sha256"] and
              ack["provider_save"] == pin(folder / "provider-save.json"), "D2_CHECKPOINT_ACK_CHANGED")
    saved = read(folder / "provider-save.json")
    q.require(saved.get("schema") == SAVE and saved.get("status") == "saved" and
              saved.get("artifact") == descriptor["capsule"] and
              saved.get("member_manifest_sha256") == descriptor["member_manifest_sha256"] and
              ack["provider"] == saved["provider"] and ack["object_id"] == saved["object_id"] and
              not provider_failed(saved["provider_response"]) and
              contains(saved["provider_response"], saved["object_id"]), "D2_CHECKPOINT_SAVE_CHANGED")
    return ack


class Gate:
    """One live invocation owns intents; acknowledgements never grant retries."""

    def __init__(self, *, wait_seconds=1, max_wait_seconds=900, on_wait=None):
        q.require(.01 <= wait_seconds <= 30 and 0 <= max_wait_seconds <= 3600, "D2_CHECKPOINT_WAIT_BOUND")
        self.owner = uuid.uuid4().hex
        self.wait_seconds, self.max_wait_seconds, self.on_wait = wait_seconds, max_wait_seconds, on_wait

    def barrier(self, campaign, stage, job_id, phase, source_paths, *, progress=None):
        q.require(phase in PHASES, "D2_CHECKPOINT_PHASE")
        root = Path(campaign["root"])
        folder = root / "checkpoints" / job_id / phase
        index = campaign["job_ids"].index(job_id)
        context = {"campaign": pin(root / "campaign.json"), "stage": campaign["stage"],
                   "job_id": job_id, "phase": phase, "cell_ids": stage["chunks"][index],
                   "generation_queue": campaign["generation_queue"],
                   "resume_pending_intent_after_restart": False, "automatic_retries": 0}
        # Each capsule binds previous durable acknowledgements. Actual ZIPs are
        # separate immutable objects, not recursively embedded copies.
        previous = sorted((root / "checkpoints").glob("*/*/ack.json")) if (root / "checkpoints").exists() else []
        parent_files = [file for ack in previous for file in
                        (ack, ack.parent / "provider-save.json", ack.parent / "capsule.json")]
        def report():
            if progress:
                progress(job_id, "awaiting_durable_save:" + phase)
        return self.checkpoint(folder, context, list(source_paths) + parent_files,
                               intent=phase in INTENTS, on_progress=report)

    def checkpoint(self, folder, scope, source_paths, *, intent, on_progress=None):
        """Generic phase gate for an explicitly declared bounded queue operation.

        `scope` must include that operation's immutable IDs/recipe/queue pins.
        A caller can use this for a new context or training phase without a
        second model runner. `intent=True` is required before every model call.
        """
        q.require(type(scope) is dict and scope and type(intent) is bool and
                  "owner_invocation" not in scope and "launch_intent" not in scope,
                  "D2_CHECKPOINT_EXPLICIT_SCOPE_REQUIRED")
        folder = q.safe_path(folder)
        context = {**scope, "owner_invocation": self.owner, "launch_intent": intent}
        if folder.exists():
            descriptor = verify_capsule(folder)
            old = descriptor["context"]
            q.require({k: v for k, v in old.items() if k != "owner_invocation"} ==
                      {k: v for k, v in context.items() if k != "owner_invocation"}, "D2_CHECKPOINT_CONTEXT_CHANGED")
            if intent:
                q.require(old["owner_invocation"] == self.owner, "D2_CHECKPOINT_AMBIGUOUS_INTENT_AFTER_RESTART")
        else:
            descriptor = create_capsule(folder, context, source_paths)
        start = time.monotonic()
        while not (folder / "ack.json").exists():
            if on_progress:
                on_progress()
            if self.on_wait:
                self.on_wait(folder, descriptor)
            if (folder / "ack.json").exists():
                break
            q.require(time.monotonic() - start < self.max_wait_seconds, "D2_CHECKPOINT_SAVE_NOT_ACKNOWLEDGED")
            time.sleep(self.wait_seconds)
        ack = verify_ack(folder)
        if intent:
            verify_capsule(folder, current_sources=True)
        return ack


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ack = commands.add_parser("acknowledge")
    ack.add_argument("--checkpoint", type=Path, required=True)
    ack.add_argument("--save-receipt", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--checkpoint", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--capsule", type=Path, required=True)
    restore.add_argument("--sha256", required=True)
    restore.add_argument("--allowed-root", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "acknowledge":
            value = acknowledge(args.checkpoint, args.save_receipt)
        elif args.command == "restore":
            value = restore_capsule(args.capsule, args.sha256, args.allowed_root)
        else:
            value = verify_ack(args.checkpoint)
        print(json.dumps(value, sort_keys=True, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, OSError, TypeError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "invalid", "reason": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
