"""Synthetic metadata only: no private rows, audio, weights, or model imports."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import research_controlled_chinese_plan as plan


def raw(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"


def h(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def canonical_sha(value):
    return h(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def label_sha(row):
    return canonical_sha({key: value for key, value in row.items() if key not in ("audio", "audio_path")})


def info(body):
    return {"bytes": len(body), "sha256": h(body)}


class Fixture:
    def __init__(self, root):
        self.root = root
        self.isolation = root / "isolation"
        self.isolation.mkdir()
        self.vocabulary = {"schema": "kws-d1-fulltext-vocabulary-v2", "output_count": 2601,
            "blank_id": 0, "observed_development_text_used": False,
            "token_to_id": {char: i + 1 for i, char in enumerate("你好小窝我吃饭屋")}}
        self.current = {
            "increment_train_candidate": [self.row("new-train-positive", "train", "你好小窝", "full-positive", .5),
                                           self.row("new-train-speech", "train", "我吃饭", "ordinary-negative", 1.0)],
            "increment_development": [self.row("new-dev", "development", "小窝小窝", "full-positive", .5)],
            "increment_nearphone_quarantine": [self.row("new-near", "train", "你好小屋", "near-negative", .5)],
            "increment_unknown_zero_loss": [self.row("new-unknown", "train", "你好小窝", "full-positive", 0.0)],
        }
        self.old_train = [self.row("old-train", "train", "你好小窝", "full-positive", 1.0),
                          self.row("old-near", "train", "你好小屋", "near-negative", .5)]
        self.old_dev = [self.row("old-dev", "development", "小窝小窝", "full-positive", .5)]
        self.old_unknown = [self.row("old-unknown", "train", None, "partial-negative", 0.0)]
        self.old = self.old_train + self.old_dev + self.old_unknown
        self.audio_files = {}
        self.audio_manifest = {}
        for ordinal, row in enumerate(self.old):
            row["audio"] = row["audio_path"] = "audio/" + row["source_id"] + ".wav"
            row["file_sha256"] = h("wav:" + row["source_id"])
            record = {"path": "/synthetic-retained/audio/" + row["source_id"] + ".wav",
                      "bytes": 32044, "sha256": row["file_sha256"]}
            self.audio_files[row["source_id"]] = record
            self.audio_manifest["files/audio-" + str(ordinal)] = record
        self.recipe = {"schema": "kws-c1-source-augmentation-v1", "human_gold": False, "independent_test": False,
            "generation_intent_is_supervision": False, "development_optimizer_draws": 0,
            "vocabulary_sha256": canonical_sha(self.vocabulary), "train": self.old_train,
            "development": self.old_dev, "evaluation_rows": self.old,
            "new_zero_loss_rows": [row["source_id"] for row in self.old_unknown],
            "audio_files": self.audio_files, "excluded_original_rows": [],
            "optimizer_steps": 999, "batch_size": 99, "seed": 123}
        for key, rows in (("train_binding_sha256", self.old_train), ("development_binding_sha256", self.old_dev),
                          ("evaluation_binding_sha256", self.old)):
            self.recipe[key] = canonical_sha([label_sha(row) for row in rows])
        self.original_old_identities, self.original_bindings = [], []
        for row in self.old:
            role = "train" if row in self.old_train else "development" if row in self.old_dev else "unknown_zero_loss"
            self.original_old_identities.append({"r1_effective_role": role, **{key: row[key] for key in plan.DIMENSIONS}})
            self.original_bindings.append({"source_id": row["source_id"], "r1_effective_role": role,
                "split": row["split"], "pcm_sha256": row["pcm_sha256"], "computed_label_binding_sha256": label_sha(row),
                "saved_label_binding_sha256": label_sha(row), "feature_sha256": h("feature:" + row["source_id"]),
                "feature_shape": [32, 400], "match": True})
        self.original_current_identities = [{"current_role": plan.AUDITED_ROLES[group],
                **{key: row[key] for key in plan.DIMENSIONS}}
            for group, rows in self.current.items() if group in plan.AUDITED_ROLES for row in rows]
        self.no_audio = [{"source_id": "synthetic-consumed-no-audio", "has_audio": False, "may_retry": False,
            "ctc_mask": False, "ctc_weight": 0.0, "training_admitted": False, "observed_text": None,
            "status": "unobservable-consumed", "claim_pin": {"sha256": h("synthetic-claim")}}]
        self.extra_audit_member = None
        self.write()

    def row(self, name, split, text, category, weight):
        active = weight > 0
        targets = [self.vocabulary["token_to_id"][char] for char in text] if active else None
        minimum = len(targets) + sum(a == b for a, b in zip(targets, targets[1:])) if active else None
        result = {key: (h(name + ":" + key) if key.endswith("_sha256") else "synthetic-" + name + ":" + key)
                  for key in plan.DIMENSIONS}
        result.update(source_id="synthetic-" + name, split=split,
            source_role="train-candidate" if split == "train" else "development-only", full_text=text,
            target_ids=targets, ctc_mask=active, ctc_weight=weight,
            keyword_targets=[int(word in text) for word in plan.KEYWORDS] if active else [None, None],
            keyword_mask=[active, active], keyword_weight=[weight, weight] if active else [0.0, 0.0],
            samples=16000, minimum_ctc_frames=minimum, r1_output_frames=32,
            vocabulary_sha256=canonical_sha(self.vocabulary), generation_category=category,
            human_gold=False, independent_test=False,
            label_status="machine_candidate" if active else "uncertain")
        return result

    def write(self):
        recipe_raw = raw(self.recipe)
        self.recipe_path = self.root / "recipe.json"
        self.recipe_path.write_bytes(recipe_raw)
        cap = {"schema": "kws-d2-durable-checkpoint-v1", "files": {
            **self.audio_manifest, "files/recipe": {"path": "/synthetic-retained/recipe.json", **info(recipe_raw)}}}
        cap_raw = raw(cap)
        self.cap_path = self.root / "capsule-manifest.json"
        self.cap_path.write_bytes(cap_raw)
        capref = {"name": "synthetic-capsule.zip", "sha256": h("synthetic-capsule-archive"),
            "manifest_sha256": h(cap_raw), "r1_recipe_member": "files/recipe", "r1_recipe_sha256": h(recipe_raw),
            "r1_recipe_original_path": "/synthetic-retained/recipe.json"}
        modelref = {"name": "synthetic-model.zip", "bytes": 123456, "sha256": h("synthetic-model-archive"),
            "actual_feature_bindings_sha256": h("synthetic-original-feature-bindings")}
        references = {"first64_original_evidence_and_r1_source_recipe_archive": capref,
                      "r1_model_owner_archive": modelref}
        package_manifest = {"report.json": h("synthetic-native-report")}
        for rows in self.current.values():
            for row in rows:
                for prefix in ("audio/", "evidence/original/"):
                    name = prefix + row["source_id"] + ".wav"
                    package_manifest[name] = h(name)
        package_raw = raw(package_manifest)
        package = {"id": "synthetic-package", "archives": ["synthetic-source.zip"],
            "manifest_sha256": h(package_raw), "report_sha256": package_manifest["report.json"],
            "original_manifest_saved_as": "original_manifests/synthetic.json",
            "source_archive_member_prefix": "synthetic-view/"}
        bodies = {"original_frozen_vocabulary.json": raw(self.vocabulary),
            "source_archives.json": raw([{"name": "synthetic-source.zip", "bytes": 12345, "sha256": h("source-archive")}]),
            "source_packages.json": raw([package]), "original_manifests/synthetic.json": package_raw,
            "r1_and_first64_source_references.json": raw(references), "consumed_unknown_no_retry.json": raw(self.no_audio)}
        for group, (member, usage, _) in plan.TABLES.items():
            for row in self.current[group]:
                row.update(schema="kws-v4-restored-existing-input-row-v1", prepared_usage=usage,
                    formal_admission=False, new_admission_granted=False,
                    optimizer_loss_authorized_by_this_preparation=False, public_executor_ready=False,
                    globally_unused=False, previously_used_in_historical_head_experiments=False,
                    wav_pcm_source_labels_verified=True, original_projection_sha256=h("synthetic-projection"),
                    original_projection_row_canonical_sha256=h("original-row:" + row["source_id"]),
                    included_in_non_near_candidate_set=group == "increment_train_candidate",
                    source_package_id=package["id"], parent_manifest_sha256=h(package_raw),
                    parent_report_sha256=package["report_sha256"])
                for field, prefix in (("normalized_audio_locator", "audio/"), ("original_audio_locator", "evidence/original/")):
                    member_name = prefix + row["source_id"] + ".wav"
                    row.setdefault(field, {"archive": "synthetic-source.zip", "member": "synthetic-view/" + member_name,
                                           "sha256": package_manifest[member_name]})
            bodies[member] = b"".join(raw(row).replace(b"\n", b" ").rstrip() + b"\n" for row in self.current[group])
        summary = {"schema": "kws-v4-chinese-existing-input-preparation-v1", "completed": True,
            "formal_admission": False, "human_gold": False, "independent_test": False, "new_admission_granted": False,
            "public_audio_upload_allowed": False, "public_executor_ready": False,
            "source_projection_sha256": h("synthetic-projection"),
            "frozen_vocabulary_file_sha256": h(bodies["original_frozen_vocabulary.json"]),
            "r1_recipe_sha256": h(recipe_raw), "source_archives": 1, "restored_native_packages": 1,
            "consumed_unknown_no_audio_no_retry": len(self.no_audio),
            "prepared_usage_counts": {plan.TABLES[group][1]: len(rows) for group, rows in self.current.items()},
            "actual_observed_16k_wav_pcm_verified": sum(map(len, self.current.values())),
            "actual_observed_original_wav_pcm_verified": sum(map(len, self.current.values()))}
        bodies["summary.json"] = raw(summary)
        manifest = {"schema": "kws-v4-chinese-increment-metadata-manifest-v1",
                    "files": {name: info(body) for name, body in bodies.items()}}
        bodies["FILE_MANIFEST.json"] = raw(manifest)
        if self.extra_audit_member:
            bodies[self.extra_audit_member] = b"unindexed"
        self.audit_path = self.root / "audit.zip"
        with zipfile.ZipFile(self.audit_path, "w") as z:
            for name, body in sorted(bodies.items()):
                z.writestr(name, body)
        self.audit_sha = h(self.audit_path.read_bytes())
        model_member = {"member": "synthetic-r1/model.pt", "bytes": 987, "sha256": h("synthetic-checkpoint")}
        plan_member = {"member": "synthetic-r1/training/plan.json", "bytes": 123, "sha256": h("synthetic-original-plan")}
        report = {"result": "PASS_NO_IDENTITY_OVERLAP", "inherited_overlap_flags_used_as_evidence": False,
            "archive_identity_and_bytes_verified": True, "identity_dimensions": list(plan.DIMENSIONS),
            "nonzero_field_intersections": 0, "comparison_groups": 6, "field_comparisons": 54,
            "inputs": {"current_audit": info(self.audit_path.read_bytes()),
                "r1_recipe_capsule": {"name": capref["name"], "bytes": 999999, "sha256": capref["sha256"]},
                "r1_model_owner_archive": modelref},
            "capsule_manifest_sha256": h(cap_raw), "recipe_member_verified": {"member": "files/recipe", **info(recipe_raw)},
            "r1_actual_feature_bindings_sha256": modelref["actual_feature_bindings_sha256"],
            "r1_checkpoint_sha256": model_member["sha256"], "r1_plan_sha256": plan_member["sha256"],
            "r1_selected_members_verified": [model_member, plan_member],
            "r1_binding_checks": {"rows": len(self.old),
                "effective_role_counts": {"train": len(self.old_train), "development": len(self.old_dev),
                                         "unknown_zero_loss": len(self.old_unknown)}}}
        saved = {"independent_overlap_audit.json": report, "current_identity_rows.json": self.original_current_identities,
            "r1_complete_identity_rows.json": self.original_old_identities, "r1_actual_binding_checks.json": self.original_bindings,
            "verified_input_members.json": {"recipe": {"member": "files/recipe", **info(recipe_raw)}}}
        files = []
        for name, value in saved.items():
            body = raw(value)
            (self.isolation / name).write_bytes(body)
            files.append({"name": name, **info(body)})
        body = raw({"schema": "private-independent-audit-evidence-v1", "files": files})
        (self.isolation / "FILE_MANIFEST.json").write_bytes(body)
        self.isolation_sha = h(body)

    def build(self):
        return plan.build_plan(self.audit_path, self.audit_sha, self.isolation, self.isolation_sha,
                               self.recipe_path, self.cap_path)

    def argv(self, mode, path):
        return [mode, "--audit", str(self.audit_path), "--audit-sha256", self.audit_sha,
            "--isolation-dir", str(self.isolation), "--isolation-manifest-sha256", self.isolation_sha,
            "--r1-recipe", str(self.recipe_path), "--capsule-manifest", str(self.cap_path), "--plan", str(path)]


class ControlledPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="controlled-plan-", dir=os.environ.get("KWS_CONTROLLED_PLAN_TEST_ROOT"))
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        (base / "inputs").mkdir()
        self.output = base / "output"
        self.output.mkdir()
        self.fx = Fixture(base / "inputs")

    def test_complete_projection_preserves_old_and_increment_roles(self):
        result = self.fx.build()
        self.assertEqual(result["counts"], {"increment_train_candidate": 2, "increment_development": 1,
            "increment_nearphone_quarantine": 1, "increment_unknown_zero_loss": 1,
            "r1_train_reference": 2, "r1_development_reference": 1, "r1_unknown_zero_loss": 1})
        self.assertEqual(len(result["rows"]), 9)
        self.assertEqual(result["r1_binding"]["retained_checkpoint_member"]["sha256"], h("synthetic-checkpoint"))
        self.assertEqual(len({row["key"] for row in result["rows"]}), 9)
        self.assertTrue(all(not row["scope"]["optimizer_authorized"] and not row["scope"]["effective_optimizer_mask"] for row in result["rows"]))
        self.assertIsNone(result["execution"]["optimizer_steps"])
        self.assertIsNone(result["execution"]["batch_size"])
        self.assertIsNone(result["execution"]["seed"])
        self.assertFalse(result["r1_binding"]["historical_recipe_execution_parameters_inherited"])

    def test_current_near_and_development_keep_labels_without_training_mask(self):
        result = self.fx.build()
        for group in ("increment_nearphone_quarantine", "increment_development", "r1_development_reference"):
            row = next(row for row in result["rows"] if row["scope"]["group"] == group)
            self.assertTrue(row["labels"]["ctc_mask"])
            self.assertEqual(row["labels"]["ctc_weight"], .5)
            self.assertFalse(row["scope"]["candidate_ctc_mask"])
            self.assertEqual(row["scope"]["candidate_ctc_weight"], 0)

    def test_historical_r1_near_role_not_rewritten_by_new_quarantine(self):
        result = self.fx.build()
        row = next(row for row in result["rows"] if row["source_identity"]["source_id"] == "synthetic-old-near")
        self.assertEqual(row["scope"]["group"], "r1_train_reference")
        self.assertTrue(row["scope"]["candidate_ctc_mask"])
        self.assertEqual(row["labels"]["ctc_weight"], .5)
        self.assertFalse(row["scope"]["optimizer_authorized"])

    def test_unknown_retained_text_does_not_become_target_or_negative(self):
        result = self.fx.build()
        row = next(row for row in result["rows"] if row["scope"]["group"] == "increment_unknown_zero_loss")
        self.assertEqual(row["labels"]["full_text"], "你好小窝")
        self.assertIsNone(row["labels"]["target_ids"])
        self.assertEqual(row["labels"]["keyword_targets"], [None, None])
        self.assertEqual(row["labels"]["keyword_weight"], [0, 0])
        self.assertFalse(row["scope"]["candidate_ctc_mask"])

    def test_original_old_audio_paths_resolve_to_exact_capsule_members(self):
        result = self.fx.build()
        old = [row for row in result["rows"] if row["key"].startswith("r1/")]
        self.assertEqual([row["audio"]["normalized_audio_locator"]["member"] for row in old],
                         ["files/audio-" + str(i) for i in range(4)])
        self.assertTrue(all(row["audio"]["normalized_audio_locator"]["member_sha256"] ==
                            row["source_observation"]["file_sha256"] for row in old))

    def test_fulltext_target_tamper_rejected(self):
        self.fx.current["increment_train_candidate"][0]["target_ids"] = [1]
        self.fx.write()
        with self.assertRaisesRegex(ValueError, "full-text CTC"):
            self.fx.build()

    def test_unknown_loss_or_keyword_promotion_rejected(self):
        row = self.fx.current["increment_unknown_zero_loss"][0]
        row["keyword_targets"] = [0, 0]
        self.fx.write()
        with self.assertRaisesRegex(ValueError, "unknown is not"):
            self.fx.build()

    def test_ctc_repeated_target_geometry_rejected_without_text_trimming(self):
        row = self.fx.row("short", "train", "小小", "ordinary-negative", .5)
        row["samples"] = 1200
        row["r1_output_frames"] = 2
        with self.assertRaisesRegex(ValueError, "geometry infeasible"):
            plan.validate_labels(row, self.fx.vocabulary)

    def test_new_near_training_candidate_rejected(self):
        row = self.fx.current["increment_train_candidate"][1]
        row["generation_category"] = "near-negative"
        self.fx.write()
        with self.assertRaisesRegex(ValueError, "quarantined"):
            self.fx.build()

    def test_unknown_source_alias_cannot_cross_development_boundary(self):
        self.fx.current["increment_unknown_zero_loss"][0]["speaker_id"] = self.fx.old_dev[0]["speaker_id"]
        self.fx.write()
        with self.assertRaisesRegex(ValueError, "crosses train/development"):
            self.fx.build()

    def test_duplicate_pcm_in_unknown_row_rejected(self):
        self.fx.current["increment_unknown_zero_loss"][0]["pcm_sha256"] = self.fx.old_train[0]["pcm_sha256"]
        self.fx.write()
        with self.assertRaisesRegex(ValueError, "duplicate source or PCM"):
            self.fx.build()

    def test_audio_locator_must_match_original_manifest_and_archive(self):
        self.fx.current["increment_train_candidate"][0]["normalized_audio_locator"]["sha256"] = h("other-wav")
        self.fx.write()
        with self.assertRaisesRegex(ValueError, "audio locator differs"):
            self.fx.build()

    def test_r1_original_label_bindings_cannot_be_replaced(self):
        self.fx.old_train[0]["ctc_weight"] = .5
        self.fx.write()
        with self.assertRaisesRegex(ValueError, "cohort binding"):
            self.fx.build()

    def test_r1_audio_filename_match_alone_is_insufficient(self):
        sid = self.fx.old_train[0]["source_id"]
        self.fx.recipe["audio_files"] = copy.deepcopy(self.fx.audio_files)
        self.fx.recipe["audio_files"][sid]["sha256"] = h("wrong-audio")
        self.fx.write()
        with self.assertRaisesRegex(ValueError, "audio path/hash"):
            self.fx.build()

    def test_audit_zip_must_be_closed(self):
        self.fx.extra_audit_member = "unindexed.json"
        self.fx.write()
        with self.assertRaisesRegex(ValueError, "not closed"):
            self.fx.build()

    def test_consumed_no_audio_cannot_retry(self):
        self.fx.no_audio[0]["may_retry"] = True
        self.fx.write()
        with self.assertRaisesRegex(ValueError, "no-audio terminal"):
            self.fx.build()

    def test_stale_input_hash_and_path_traversal_rejected(self):
        self.fx.audit_sha = h("stale")
        with self.assertRaisesRegex(ValueError, "archive hash"):
            self.fx.build()
        for member in ("../audio.wav", "/audio.wav", "a/../audio.wav", "a\\audio.wav"):
            with self.subTest(member=member), self.assertRaises(ValueError):
                plan.safe_member(member)

    def test_duplicate_json_key_and_nonfinite_number_rejected(self):
        for value in (b'{"weight":0,"weight":1}', b'{"weight":NaN}', b'{"weight":Infinity}', b'{"weight":1e999}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plan.strict_json(value)

    def test_cli_create_only_then_rebuild_verification_rejects_saved_plan_mutation(self):
        path = self.output / "plan.json"
        with redirect_stdout(io.StringIO()):
            plan.main(self.fx.argv("prepare", path))
            plan.main(self.fx.argv("verify", path))
        with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
            plan.main(self.fx.argv("prepare", path))
        saved = json.loads(path.read_bytes())
        saved["rows"][0]["scope"]["optimizer_authorized"] = True
        path.write_bytes(raw(saved))
        with self.assertRaisesRegex(ValueError, "saved projection differs"), redirect_stdout(io.StringIO()):
            plan.main(self.fx.argv("verify", path))

    def test_private_output_refuses_repository_and_input_evidence(self):
        for destination in (ROOT / "must-not-create-private-plan.json", self.fx.root / "must-not-create.json"):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "outside repository"), redirect_stdout(io.StringIO()):
                plan.main(self.fx.argv("prepare", destination))
            self.assertFalse(destination.exists())

    def test_private_output_refuses_symlink_parent(self):
        alias = Path(self.tmp.name) / "linked-output"
        alias.symlink_to(self.output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "unsafe private output"), redirect_stdout(io.StringIO()):
            plan.main(self.fx.argv("prepare", alias / "must-not-create.json"))
        self.assertFalse((self.output / "must-not-create.json").exists())

    def test_private_output_has_owner_only_mode(self):
        destination = self.output / "plan.json"
        with redirect_stdout(io.StringIO()):
            plan.main(self.fx.argv("prepare", destination))
        self.assertEqual(destination.stat().st_mode & 0o777, 0o600)

    def test_saved_plan_verification_preserves_json_types(self):
        destination = self.output / "plan.json"
        with redirect_stdout(io.StringIO()):
            plan.main(self.fx.argv("prepare", destination))
        original = json.loads(destination.read_bytes())
        mutations = (
            lambda value: value["rows"][0]["labels"]["target_ids"].__setitem__(0, True),
            lambda value: value["rows"][0]["source_observation"].__setitem__("samples", 16000.0),
            lambda value: value["execution"].__setitem__("optimizer_authorized", 0),
        )
        for mutate in mutations:
            changed = copy.deepcopy(original)
            mutate(changed)
            destination.write_bytes(raw(changed))
            with self.assertRaisesRegex(ValueError, "saved projection differs"), redirect_stdout(io.StringIO()):
                plan.main(self.fx.argv("verify", destination))


if __name__ == "__main__":
    unittest.main()
