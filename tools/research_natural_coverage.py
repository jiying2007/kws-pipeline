#!/usr/bin/env python3
"""Plan fixed natural windows and summarize verified, source-bound coverage.

Standard library only. Preparation, sealed recognition and machine-weak text
coverage are separate interval unions. No decoder, training or seal creation.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prepare_natural_asr_windows as natural
import research_queue as q
import research_runtime

PLAN = "kws-natural-fixed-pair-plan-v1"
REQUEST = "kws-natural-coverage-request-v1"
REPORT = "kws-natural-coverage-report-v1"
VIEW = "kws-natural-asr-evidence-view-v1"
VERIFIER = "automation/active_natural_asr_weak_view.py"
PREPARATION_VERIFIER = "tools/prepare_natural_asr_windows.py"
PREPARATION_IMPORTS = frozenset({"tools/research_d2_checkpoints.py", "tools/research_job_lock.py",
    "tools/research_no_network.py", "training/research_acoustic_augment.py"})
PREPARATION_TIMEOUT_SECONDS = 60
RATE = 16000
WINDOW, ADVANCE, PAIR = 30 * RATE, 25 * RATE, 55 * RATE
# Preserve native receipt IDs (not display names such as "Qwen").
MODELS = frozenset(natural.selective.batch.MODELS)
BOUNDARY = {
    "target_acoustic_presence": "unknown", "accepted_negative_seconds": 0,
    "natural_far_available": False, "human_gold": False, "independent_test": False,
    "new_model_calls": 0, "training_performed": False, "new_seals_written": 0,
    "duration_semantics": "interval unions within each parent PCM; not negative exposure or independent speakers",
}


def require(ok, reason):
    q.require(ok, "NATURAL_COVERAGE_" + reason)


def pinned_json(path, expected_sha, maximum=16 * 1024 ** 2):
    require(natural.is_sha(expected_sha), "EXTERNAL_SHA_REQUIRED")
    raw = q.read_bytes(path, maximum=maximum)
    require(q.sha(raw) == expected_sha, "FILE_HASH_CHANGED")
    return q.loads(raw)


class Parents:
    """Keep one exact mono identity per dataset/session and one split per family."""
    def __init__(self):
        self.by_pcm, self.ids, self.sessions, self.families = {}, {}, {}, {}

    def add(self, parent):
        pcm = parent["pcm_sha256"]
        if pcm in self.by_pcm:
            require(parent == self.by_pcm[pcm], "PARENT_ALIAS_OR_IDENTITY_CONFLICT")
            return
        require(parent["parent_id"] not in self.ids, "PARENT_IDENTITY_CONFLICT")
        session = (parent["dataset_id"], parent["session_id"])
        require(session not in self.sessions, "SECOND_CHANNEL_OR_PARENT_FOR_SESSION")
        family = parent["source_family_id"]
        require(self.families.setdefault(family, parent["split"]) == parent["split"],
                "SOURCE_FAMILY_SPLIT_CONFLICT")
        self.by_pcm[pcm] = parent
        self.ids[parent["parent_id"]] = pcm
        self.sessions[session] = pcm


def overlaps(left, right):
    return max(left[0], right[0]) < min(left[1], right[1])


def fixed_plan(declarations):
    """Use only parent lengths and excluded declaration intervals, never outputs."""
    require(type(declarations) is list and 1 <= len(declarations) <= 64, "PRIOR_DECLARATION_ROSTER")
    registry, excluded = Parents(), {}
    for value in declarations:
        parents = natural.validate_declaration(value)
        for parent in value["parents"]:
            registry.add(parent)
        for window in value["windows"]:
            pcm = parents[window["parent_id"]]["pcm_sha256"]
            excluded.setdefault(pcm, []).append((window["start_sample"], window["end_sample"]))
    require(1 <= len(registry.by_pcm) <= natural.MAX_PARENTS, "BOUNDED_PARENT_ROSTER")
    ordered = sorted(registry.by_pcm.values(), key=lambda p: (p["session_id"], p["dataset_id"], p["parent_id"]))
    full, shards = [], []
    for parent in ordered:
        size = parent["sample_count"]
        require(size > PAIR, "PARENT_TOO_SHORT_FOR_FIXED_RULE")
        groups, windows = [], []
        for index in range(1, 5):
            start = (size - PAIR) * index // 5
            span = (start, start + PAIR)
            require(all(not overlaps(span, old) for old in excluded[parent["pcm_sha256"]]),
                    "FIXED_GROUP_INTERSECTS_PRIOR_DECLARATION")
            require(all(not overlaps(span, old) for old in groups), "FIXED_GROUPS_OVERLAP")
            groups.append(span)
            for side, offset in (("a", 0), ("b", ADVANCE)):
                windows.append({"window_id": f"natural-pair-{parent['pcm_sha256'][:24]}-{index}-{side}",
                    "parent_id": parent["parent_id"], "start_sample": start + offset,
                    "end_sample": start + offset + WINDOW})
        full.extend(windows)
        for offset in (0, 4):
            value = {"schema": natural.DECLARATION, "status": "declared_not_materialized",
                "selection_reason": "Frozen length-only rule: anchor=floor((N-880000)*i/5), i=1..4; A=[anchor,anchor+480000), B=[anchor+400000,anchor+880000). Prior declared intervals and other groups may not overlap; no output-based selection or adaptive relocation.",
                "parents": [parent], "windows": windows[offset:offset + 4]}
            natural.validate_declaration(value)
            shards.append(value)
    combined = {"schema": natural.DECLARATION, "status": "declared_not_materialized",
                "selection_reason": shards[0]["selection_reason"], "parents": ordered, "windows": full}
    natural.validate_declaration(combined)
    return {"schema": PLAN, "status": "declared_not_materialized", "selection_uses_model_outputs": False,
        "rule": {"anchor": "floor((N-880000)*i/5)", "indices": [1, 2, 3, 4],
                 "window_samples": WINDOW, "second_window_advance_samples": ADVANCE,
                 "pair_samples": PAIR, "adaptive_relocation": False},
        "coverage": natural.coverage(combined), "combined_declaration": combined,
        "declarations": shards, "excluded_prior_ranges": {k: natural.merged_ranges(v) for k, v in sorted(excluded.items())},
        "boundary": BOUNDARY}


def write_plan(paths, hashes, output):
    require(len(paths) == len(hashes) and len(set(hashes)) == len(hashes), "UNIQUE_PINNED_DECLARATIONS")
    declarations = [pinned_json(path, digest) for path, digest in zip(paths, hashes)]
    report = fixed_plan(declarations)
    output = q.safe_path(output)
    require(not output.exists() and output.parent.is_dir() and ROOT not in output.parents and output != ROOT,
            "NEW_PRIVATE_OUTPUT_REQUIRED")
    files = {}
    for index, declaration in enumerate(report.pop("declarations")):
        files[f"declaration-{index:03d}.json"] = q.canonical(declaration)
    files["combined-declaration.json"] = q.canonical(report.pop("combined_declaration"))
    report["input_declarations"] = [{"path": str(q.safe_path(path)), "sha256": digest}
                                    for path, digest in zip(paths, hashes)]
    report["files"] = {name: {"sha256": q.sha(raw), "bytes": len(raw)} for name, raw in files.items()}
    report["planner_source"] = q.file_record(Path(__file__))
    files["plan.json"] = q.canonical(report)
    output.mkdir(mode=0o700)
    for name, raw in files.items():
        q.write_new(output / name, raw)
    return report


def intervals(rows, parents):
    by_parent = []
    for pcm, parent in sorted(parents.items(), key=lambda item: (item[1]["session_id"], item[0])):
        spans = [(row["start_sample"], row["end_sample"]) for row in rows if row["parent_pcm_sha256"] == pcm]
        union = natural.merged_ranges(spans)
        processed = sum(end - start for start, end in spans)
        unique = sum(end - start for start, end in union)
        by_parent.append({key: parent[key] for key in ("dataset_id", "session_id", "source_family_id", "split")} | {
            "parent_pcm_sha256": pcm, "sample_rate_hz": RATE, "windows": len(spans),
            "union_sample_ranges": union, "window_input_samples": processed,
            "unique_parent_samples": unique, "overlap_samples": processed - unique})
    processed = sum(row["window_input_samples"] for row in by_parent)
    unique = sum(row["unique_parent_samples"] for row in by_parent)
    return {"windows": len(rows), "sample_rate_hz": RATE, "window_input_samples": processed,
        "unique_parent_samples": unique, "overlap_samples": processed - unique,
        "window_input_seconds": processed / RATE, "unique_parent_seconds": unique / RATE,
        "overlap_seconds": (processed - unique) / RATE, "by_parent": by_parent}


def source_checks(spec):
    require(type(spec) is dict and set(spec) == {"path", "source_files_sha256"}, "VERIFIER_FIELDS")
    path = q.safe_path(spec["path"])
    require(path.is_absolute() and path.name == Path(VERIFIER).name and path.parent.name == "automation", "VERIFIER_PATH")
    root = path.parents[1]
    expected = spec["source_files_sha256"]
    require(type(expected) is dict and 1 <= len(expected) <= 64 and VERIFIER in expected, "VERIFIER_SOURCE_ROSTER")
    for name, digest in expected.items():
        relative = natural.relative(name)
        require(natural.is_sha(digest) and q.file_record(root / relative)["sha256"] == digest, "VERIFIER_SOURCE_CHANGED")
    return path, root, expected


def load_verifier(spec):
    path, root, expected = source_checks(spec)
    for name, module in list(sys.modules.items()):
        if name in ("active_natural_asr_weak_view", "active_asr_weak_view", "automation", "kws_data_contract") or \
                name.startswith(("automation.", "kws_data_contract.")):
            origin = getattr(module, "__file__", None)
            origins = [origin] if origin else list(getattr(module, "__path__", ()))
            require(origins and all(Path(item).resolve().is_relative_to(root) for item in origins),
                    "MIXED_DATA_MODULES")
    module_spec = importlib.util.spec_from_file_location("active_natural_asr_weak_view", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    require(len(module.PRODUCER_FILES) == len(expected) and set(module.PRODUCER_FILES) == set(expected),
            "VERIFIER_SOURCE_ROSTER_CHANGED")
    source_checks(spec)
    return module


def preparation_source_checks(spec, binding, *, verify_git=True):
    """Pin the same preparation algorithm with a recorded historical dependency set."""
    require(type(spec) is dict and set(spec) == {
        "path", "git_repository", "git_commit", "source_files_sha256"}, "PREPARATION_VERIFIER_FIELDS")
    path, git_repository = q.safe_path(spec["path"]), q.safe_path(spec["git_repository"])
    require(path.is_absolute() and path.name == Path(PREPARATION_VERIFIER).name and
            path.parent.name == "tools" and git_repository.is_absolute() and git_repository.is_dir(),
            "PREPARATION_VERIFIER_PATH")
    require(type(spec["git_commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", spec["git_commit"]),
            "PREPARATION_GIT_COMMIT")
    root, expected = path.parents[1], spec["source_files_sha256"]
    producer_names = set(natural.producer_sources())
    require(type(expected) is dict and set(expected) == producer_names | PREPARATION_IMPORTS,
            "PREPARATION_SOURCE_ROSTER")
    actual = set()
    for item in root.rglob("*"):
        q.safe_path(item)
        require(item.is_file() or item.is_dir(), "PREPARATION_REGULAR_SOURCE_FILES")
        if item.is_file():
            actual.add(item.relative_to(root).as_posix())
    require(actual == set(expected), "PREPARATION_SNAPSHOT_ROSTER")
    records = {}
    for name in sorted(expected):
        require(natural.is_sha(expected[name]), "PREPARATION_SOURCE_PIN")
        records[name] = q.file_record(root / name)
        require(0 < records[name]["bytes"] <= 1024 ** 2 and records[name]["sha256"] == expected[name],
                "PREPARATION_SOURCE_CHANGED:" + name)
    require(records[PREPARATION_VERIFIER] == q.file_record(Path(natural.__file__)),
            "PREPARATION_ALGORITHM_CHANGED")
    require(binding["preparation_sources"] == {name: records[name] for name in producer_names},
            "PREPARATION_BINDING_SOURCE_MISMATCH")
    if verify_git:
        environment = os.environ.copy(); environment["GIT_NO_REPLACE_OBJECTS"] = "1"
        queries = b"".join((spec["git_commit"] + ":" + name + "\n").encode() for name in sorted(expected))
        try:
            result = subprocess.run(["git", "--no-pager", "-C", str(git_repository), "cat-file", "--batch"],
                input=queries, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
                timeout=20, check=False)
        except subprocess.TimeoutExpired as error:
            raise ValueError("NATURAL_COVERAGE_PREPARATION_GIT_TIMEOUT") from error
        require(result.returncode == 0, "PREPARATION_GIT_SOURCE_UNAVAILABLE")
        remaining = result.stdout
        for name in sorted(expected):
            header, separator, remaining = remaining.partition(b"\n")
            parts = header.split()
            require(separator and len(parts) == 3 and parts[1] == b"blob" and
                    parts[2].isdigit() and int(parts[2]) == records[name]["bytes"],
                    "PREPARATION_GIT_SOURCE_CHANGED:" + name)
            size = records[name]["bytes"]
            require(len(remaining) >= size + 1 and remaining[size:size+1] == b"\n" and
                    q.sha(remaining[:size]) == expected[name], "PREPARATION_GIT_SOURCE_CHANGED:" + name)
            remaining = remaining[size+1:]
        require(not remaining, "PREPARATION_GIT_EXTRA_OUTPUT")
    return path, records


def verify_prepared(entry, bundle, audio_root):
    """Only the fixed read-only prepare verifier CLI may use an explicit old closure."""
    raw = q.read_bytes(bundle / "natural-window-bindings.json")
    binding = q.loads(raw)
    if "preparation_verifier" not in entry:
        summary = natural.verify(bundle, audio_root, entry["declaration_sha256"])
        receipt = {"mode": "current-source"}
    else:
        spec = entry["preparation_verifier"]
        path, records = preparation_source_checks(spec, binding)
        flags = ["-I", "-S", "-B"] + (["-" + "O" * sys.flags.optimize] if sys.flags.optimize else [])
        command = [sys.executable, *flags, str(path), "verify", "--bundle", str(bundle),
            "--audio-root", str(audio_root), "--declaration-sha256", entry["declaration_sha256"]]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=path.parents[1], timeout=PREPARATION_TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired as error:
            raise ValueError("NATURAL_COVERAGE_PREPARATION_VERIFY_TIMEOUT") from error
        require(result.returncode == 0, "PREPARATION_VERIFY_FAILED:" + result.stderr.decode(errors="replace")[-2048:])
        summary = q.loads(result.stdout)
        require(summary == q.loads(q.read_bytes(bundle / "summary.json")), "PREPARATION_VERIFIER_SUMMARY_DIFFERS")
        preparation_source_checks(spec, binding, verify_git=False)
        receipt = {"mode": "pinned-prepare-cli", "verifier": spec,
            "source_files": records, "stdout_sha256": q.sha(result.stdout),
            "timeout_seconds": PREPARATION_TIMEOUT_SECONDS}
    require(q.read_bytes(bundle / "natural-window-bindings.json") == raw, "PREPARATION_BINDINGS_CHANGED_DURING_VERIFY")
    return {"bundle": str(bundle), "bindings_sha256": q.sha(raw),
            "summary_sha256": q.sha(q.canonical(summary)), **receipt}


def summarize(request):
    require(type(request) is dict and set(request) == {"schema", "prepared", "views", "verifier"} and
            request["schema"] == REQUEST, "REQUEST_FIELDS")
    packages, views = request["prepared"], request["views"]
    require(type(packages) is list and 1 <= len(packages) <= 128 and
            type(views) is list and len(views) <= 128, "BOUNDED_PACKAGE_ROSTER")
    registry, prepared, bundles, package_ids = Parents(), {}, {}, set()
    preparation_verifications = []
    for entry in packages:
        require(type(entry) is dict and set(entry) in (
            {"bundle", "audio_root", "declaration_sha256"},
            {"bundle", "audio_root", "declaration_sha256", "preparation_verifier"}), "PREPARED_FIELDS")
        bundle, audio_root = q.safe_path(entry["bundle"]), q.safe_path(entry["audio_root"])
        require(bundle.is_absolute() and audio_root.is_absolute(), "ABSOLUTE_INPUT_PATHS")
        # The existing verifier reads each complete parent and reconstructs the
        # slices. It does not trust a summary or mutate a bundle.
        verified = verify_prepared(entry, bundle, audio_root)
        bindings_raw = q.read_bytes(bundle / "natural-window-bindings.json")
        bindings, identity = q.loads(bindings_raw), q.sha(bindings_raw)
        require(identity == verified["bindings_sha256"], "PREPARATION_BINDINGS_CHANGED_AFTER_VERIFY")
        preparation_verifications.append(verified)
        require(identity not in package_ids, "DUPLICATE_PREPARED_PACKAGE")
        package_ids.add(identity)
        for parent in bindings["parents"]:
            registry.add(parent)
        for row in bindings["rows"]:
            require(row["source_id"] not in prepared, "DUPLICATE_PREPARED_WINDOW")
            prepared[row["source_id"]] = {**row, "sample_rate_hz": RATE, "bindings_sha256": identity}
        bundles[identity] = {"declaration_sha256": entry["declaration_sha256"],
            "inputs_manifest_sha256": bindings["inputs_manifest_sha256"],
            "rows": {row["source_id"]: row for row in bindings["rows"]}}
    require(len(prepared) <= 8192, "BOUNDED_WINDOW_ROSTER")
    observed, quorum, view_ids, view_records = {}, {}, set(), []
    if views:
        module = load_verifier(request["verifier"])
    else:
        require(request["verifier"] is None, "UNUSED_VERIFIER")
    for entry in views:
        require(type(entry) is dict and set(entry) == {"path", "manifest_sha256"} and
                natural.is_sha(entry["manifest_sha256"]), "VIEW_FIELDS")
        path = q.safe_path(entry["path"])
        require(path.is_absolute() and entry["manifest_sha256"] not in view_ids, "DUPLICATE_OR_RELATIVE_VIEW")
        view_ids.add(entry["manifest_sha256"])
        report = module.verify_view(path, expected_manifest_sha256=entry["manifest_sha256"])
        require(report["schema"] == VIEW and report["boundary"]["accepted_negative_seconds"] == 0 and
                report["boundary"]["human_gold"] is False, "VIEW_SCOPE")
        batches, row_counts = {}, {}
        for batch in report["batches"]:
            key = batch["batch_id"]
            require(key not in batches and batch["status"] == "complete", "BATCH_NOT_UNIQUELY_SEALED")
            binding = batch["bindings_sha256"]
            require(binding in bundles and batch["declaration_sha256"] == bundles[binding]["declaration_sha256"] and
                    batch["input_manifest_sha256"] == bundles[binding]["inputs_manifest_sha256"], "VIEW_INPUT_BINDING")
            for field in ("asr_request_sha256", "final_audit_sha256", "inventory_sha256"):
                require(natural.is_sha(batch[field]), "SEALED_NATIVE_IDENTITY")
            batches[key], row_counts[key] = batch, set()
        for row in report["rows"]:
            source, batch_id = row["source_id"], row["batch_id"]
            require(source in prepared and source not in observed and batch_id in batches, "UNBOUND_OR_DUPLICATE_OBSERVED_WINDOW")
            baseline = prepared[source]
            require(baseline["bindings_sha256"] == batches[batch_id]["bindings_sha256"], "ROW_BATCH_BINDING")
            for field in ("clip_id", "source_id", "parent_pcm_sha256", "parent_wav_sha256", "dataset_id",
                          "session_id", "source_family_id", "split", "sample_rate_hz", "start_sample", "end_sample"):
                require(type(row[field]) is type(baseline[field]) and row[field] == baseline[field], "ROW_SOURCE_BINDING:" + field)
            observations = row["observations"]
            require(type(observations) is list and len(observations) == 3 and
                    {item["model"] for item in observations} == MODELS and
                    all(item["status"] == "complete" and item["requested"] is True for item in observations),
                    "THREE_SEALED_NATIVE_OBSERVATIONS_REQUIRED")
            row_counts[batch_id].add(source)
            observed[source] = baseline
            decision = row["decision"]
            if decision["full_text_quorum"] is True and decision["ctc_mask"] is True:
                quorum[source] = baseline
        for key, batch in batches.items():
            require(row_counts[key] == set(bundles[batch["bindings_sha256"]]["rows"]) and
                    type(batch["requested_native_results"]) is int and batch["requested_native_results"] == 3 * len(row_counts[key]),
                    "OBSERVED_BATCH_ROSTER_OR_COUNT")
        view_records.append({**entry, "observed_windows": len(report["rows"]), "batches": len(batches)})
    if views:
        source_checks(request["verifier"])
    return {"schema": REPORT, "prepared": intervals(list(prepared.values()), registry.by_pcm),
        "observed": intervals(list(observed.values()), registry.by_pcm),
        "weak_fulltext_quorum": intervals(list(quorum.values()), registry.by_pcm),
        "prepared_without_complete_observation": sorted(set(prepared) - set(observed)),
        "preparation_verifications": preparation_verifications,
        "views": view_records, "boundary": BOUNDARY,
        "observed_semantics": "three actual native results from fully replayed existing sealed evidence; usability and transcript agreement are separate",
        "weak_fulltext_quorum_semantics": "machine-weak full-text candidate coverage only; no automatic training, acoustic truth, event labels or FAR denominator"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Create a fixed length-only declaration plan; no audio or model calls")
    plan.add_argument("--prior-declaration", type=Path, action="append", required=True)
    plan.add_argument("--prior-declaration-sha256", action="append", required=True)
    plan.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize", help="Read-only verification of prepared and already-sealed evidence coverage")
    summary.add_argument("--request", type=Path, required=True)
    summary.add_argument("--request-sha256", required=True)
    args = parser.parse_args(argv)
    research_runtime.check_interpreter(research_runtime.load_manifest())
    if args.command == "plan":
        report = write_plan(args.prior_declaration, args.prior_declaration_sha256, args.output)
    else:
        report = summarize(pinned_json(args.request, args.request_sha256))
        report["request_sha256"] = args.request_sha256
    print(q.canonical(report).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
