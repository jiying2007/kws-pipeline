"""Aggregate retained D1 weak views without relabeling or changing old loaders.

The existing loader replays *every* source/ASR/native/PCM row before yielding
its vocabulary-filtered projection. Materializing that iterator is mandatory,
even if it yields zero rows. Only after that complete replay do we project the
hash-bound report's confirmed rows into a newly declared vocabulary.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re
import sys

from research_d1_vocab import digest, require

DATA_SOURCE_PATHS = (
    "automation/active_asr_weak_view.py", "automation/research_weak_sampling.py",
    "kws_data_contract/__init__.py", "kws_data_contract/identity.py",
    "kws_data_contract/reference_text.py", "kws_data_contract/source_registry.py",
    "research/auto-tts-voice-clone-20261102/bulk_pool.py",
)
RETAINED_SAMPLER_SHA256 = "5a6f5447b1f4e618f38e9f1bde0b79b1ba3b21db3fbab3fd0e237c622f39e288"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_data_modules(loader_path):
    path = Path(loader_path).resolve()
    root = path.parents[1]
    require(path == root / DATA_SOURCE_PATHS[0], "D1_DATA_LOADER_PATH")
    for name, module in list(sys.modules.items()):
        if name == "kws_data_contract" or name.startswith("kws_data_contract."):
            origin = getattr(module, "__file__", None)
            require(origin and Path(origin).resolve().is_relative_to(root),
                    "D1_MIXED_DATA_SOURCE_MODULES")
    before = {name: sha((root / name).read_bytes()) for name in DATA_SOURCE_PATHS}
    require(before[DATA_SOURCE_PATHS[1]] == RETAINED_SAMPLER_SHA256,
            "D1_RETAINED_SAMPLER_VERSION_CHANGED")
    modules = []
    for name, relative in (("d1_verified_weak_loader", DATA_SOURCE_PATHS[0]),
                           ("d1_retained_exposure_sampler", DATA_SOURCE_PATHS[1])):
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    require(before == {name: sha((root / name).read_bytes()) for name in DATA_SOURCE_PATHS},
            "D1_DATA_SOURCE_CHANGED_DURING_IMPORT")
    return *modules, {"root": str(root), "files": before}


def aggregate_batches(packages, loader, *, expected_plan_sha256):
    """Return all verified rows, including uncertain and legacy-CTC-OOV rows.

    Package-local clip IDs may repeat across <=64-item batches. The global key
    is (frozen plan hash, source/cell ID); a repeated cell is rejected. Split
    checks include every original/derived PCM, reference ancestor and speaker,
    including uncertain rows that will contribute no training loss.
    """
    require(type(packages) is list and 1 <= len(packages) <= 256 and
            type(expected_plan_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256),
            "D1_BATCH_DECLARATION")
    rows, identities, pcm_identities, parents = [], set(), set(), []
    package_identities = set()
    roles = loader.SourceRoles()
    plan_cells = None
    for declaration in packages:
        require(type(declaration) is dict and set(declaration) == {"path", "manifest_sha256"},
                "D1_PACKAGE_DECLARATION")
        package = Path(declaration["path"]).absolute()
        manifest_raw = loader.read_bytes(package / "MANIFEST.json")
        manifest_sha = sha(manifest_raw)
        require(manifest_sha == declaration["manifest_sha256"] and manifest_sha not in package_identities,
                "D1_PACKAGE_IDENTITY_OR_DUPLICATE")
        package_identities.add(manifest_sha)
        manifest = loader.loads(manifest_raw)
        report_raw = loader.read_bytes(package / "report.json", 32 * 1024 * 1024)
        inputs_raw = loader.read_bytes(package / "evidence/inputs.json")
        require(manifest.get("report.json") == sha(report_raw) and
                manifest.get("evidence/inputs.json") == sha(inputs_raw), "D1_PARENT_REPORT_BINDING")
        report, inputs = loader.loads(report_raw), loader.loads(inputs_raw)
        adapter = report.get("source_adapter")
        require(type(adapter) is dict and adapter.get("schema") == "kws-active-d1-frozen-plan-source-v1" and
                adapter.get("plan_sha256") == expected_plan_sha256, "D1_COMMON_FROZEN_PLAN_REQUIRED")
        # Do not replace this with next(...), a count from a saved summary, or
        # direct report loading: OOV and uncertain rows must also be replayed.
        list(loader.load_training_rows(package, "rnn-four-token"))
        require(loader.read_bytes(package / "MANIFEST.json") == manifest_raw and
                loader.read_bytes(package / "report.json", 32 * 1024 * 1024) == report_raw and
                loader.read_bytes(package / "evidence/inputs.json") == inputs_raw,
                "D1_PACKAGE_CHANGED_DURING_REPLAY")
        plan_raw = loader.read_bytes(package / "evidence/source/d1/plan.json")
        require(manifest.get("evidence/source/d1/plan.json") == sha(plan_raw), "D1_RETAINED_PLAN_BINDING")
        plan = loader.loads(plan_raw)
        require(sha(loader.canonical(plan)) == expected_plan_sha256 and len(plan["cells"]) == 256,
                "D1_PLAN_IDENTITY_OR_SIZE")
        current_cells = {row["cell_id"] for row in plan["cells"]}
        require(len(current_cells) == 256 and (plan_cells is None or plan_cells == current_cells),
                "D1_PLAN_CELL_ROSTER")
        plan_cells = current_cells
        require(type(report["items"]) is list and 1 <= len(report["items"]) <= 64 and
                [row["item_id"] for row in report["items"]] == [row["id"] for row in inputs["clips"]],
                "D1_VERIFIED_BATCH_ROSTER")
        for raw, clip in zip(report["items"], inputs["clips"]):
            source, measured = raw["source"], raw["audio_derivation"]
            require(source["source_kind"] == "synthetic-reference-clone" and
                    source["source_plan_sha256"] == expected_plan_sha256 and source["source_id"] in plan_cells,
                    "D1_SOURCE_CELL_BINDING")
            key = (expected_plan_sha256, source["source_id"])
            require(key not in identities, "D1_DUPLICATE_GLOBAL_CELL")
            require(clip["original_pcm"]["sha256"] not in pcm_identities, "D1_DUPLICATE_ORIGINAL_PCM")
            identities.add(key)
            pcm_identities.add(clip["original_pcm"]["sha256"])
            loader.add_source_role(roles, source, clip, measured)
            row = loader.training_row(raw, "fsmn-fulltext")
            row.update(
                audio=str(package / loader.relative(raw["audio"])),
                audio_path=str(package / loader.relative(raw["audio"])),
                consumer="d1-verified-full-transcript-v2",
                label_status=raw["decision"]["status"], label_reason=raw["decision"]["reason"],
                parent_manifest_sha256=manifest_sha, parent_report_sha256=sha(report_raw),
                parent_inputs_sha256=sha(inputs_raw), global_source_key=list(key),
                source_role=source["role"], parent_source_sha256=source["parent_source_sha256"],
                intended_keyword_id=source.get("intended_keyword_id"),
                reference_wav_sha256=source["reference_wav_sha256"],
                reference_pcm_sha256=source["reference_pcm_sha256"],
                asr_transcripts=raw["asr"], retained_row_sha256=digest(raw),
            )
            rows.append(row)
        parents.append({"path": str(package), "manifest_sha256": manifest_sha,
                        "report_sha256": sha(report_raw), "inputs_sha256": sha(inputs_raw),
                        "observed_items": len(report["items"])})
    require(len(rows) <= 256, "D1_AGGREGATE_SIZE")
    rows.sort(key=lambda row: (row["source_family_id"], row["source_id"]))
    return rows, {
        "schema": "kws-d1-verified-batch-aggregate-v2", "plan_sha256": expected_plan_sha256,
        "parents": parents, "planned_items": 256, "observed_items": len(rows),
        "complete_plan_coverage": set(row["source_id"] for row in rows) == plan_cells,
        "global_identity": "frozen-plan-sha256 plus cell/source-id; local clip IDs may repeat",
        "all_rows_replayed_before_vocabulary_projection": True,
        "human_gold": False, "independent_human_evidence": False,
    }
