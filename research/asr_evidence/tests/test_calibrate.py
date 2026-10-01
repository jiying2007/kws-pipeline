#!/usr/bin/env python3
"""Offline, invented evidence only; none of these tests runs an ASR model."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import calibrate as c

FIXTURES = Path(__file__).parent / "fixtures"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        shutil.copytree(FIXTURES, self.inputs)
        self.paths = {key: self.inputs / (key + ".json") for key in c.FILE_LIMITS}
        self.data = {key: load(path) for key, path in self.paths.items()}
        self.hashes = load(self.inputs / "hashes.json")
        self.output = self.root / "readout.json"

    def refresh(self):
        for name in ("characters", "phrases"):
            self.write(name)
            self.data["rules"]["lexicons"][name]["sha256"] = self.hashes[name]
        for name in ("rules", "manifest"):
            self.write(name)
        for name in ("model_a", "model_b"):
            self.data[name]["manifest_sha256"] = self.hashes["manifest"]
            self.data[name]["rules_sha256"] = self.hashes["rules"]
            self.write(name)

    def write(self, name):
        data = (json.dumps(self.data[name], ensure_ascii=False, indent=2) + "\n").encode()
        self.paths[name].write_bytes(data)
        self.hashes[name] = digest(data)

    def args(self):
        args = []
        for key, path in self.paths.items():
            args += ["--" + key.replace("_", "-"), str(path)]
        return args + ["--manifest-sha256", self.hashes["manifest"], "--rules-sha256",
                       self.hashes["rules"], "--output", str(self.output)]

    def cli(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = c.main(self.args())
        return status, out.getvalue(), err.getvalue()

    def result(self):
        status, out, err = self.cli()
        self.assertEqual((status, err), (0, ""))
        self.receipt = json.loads(out)
        return load(self.output)

    def rejected(self):
        status, out, err = self.cli()
        self.assertEqual(status, 2, err)
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("REJECTED:"), err)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".asr-evidence-*")))

    def derive(self, text, **overrides):
        rules = c.Rules(self.data["rules"], self.hashes["rules"], self.data["characters"],
                        self.data["phrases"], self.data["manifest"]["target_order"])
        row = dict(raw_text=text, status="success", completeness="complete", quality_flags=[])
        row.update(overrides)
        return rules.derive(row)

    def change_kind(self, kind):
        manifest = self.data["manifest"]
        manifest["execution_kind"] = kind
        for i, name in enumerate(("model_a", "model_b")):
            manifest["models"][i]["model_id"] = "external-" + name
            self.data[name]["model"] = copy.deepcopy(manifest["models"][i])
            self.data[name]["execution_kind"] = kind
        self.refresh()

    def test_fixture_byte_locks(self):
        for key, path in self.paths.items():
            self.assertEqual(digest(path.read_bytes()), self.hashes[key])

    def test_cli_receipt_and_readout_identity(self):
        result = self.result()
        self.assertEqual(self.receipt["readout_sha256"], digest(self.output.read_bytes()))
        for obj in (result, self.receipt):
            for key in ("manifest", "rules"):
                self.assertEqual(obj[key + "_sha256"], self.hashes[key])
            self.assertEqual(obj["rule_version"], c.RULE_VERSION)
            self.assertEqual(obj["input_sha256"], self.hashes)
            self.assertEqual(obj["predeclared_models"], self.data["manifest"]["models"])
        self.assertFalse(result["inference_performed_by_this_tool"])
        self.assertFalse(result["wav_bytes_verified_by_this_tool"])
        self.assertEqual(result["evidence_level"], "synthetic")

    def test_cli_subprocess_success_and_missing_sha_option(self):
        good = subprocess.run([sys.executable, str(ROOT / "calibrate.py"), *self.args()],
                              capture_output=True, text=True, timeout=10)
        self.assertEqual(good.returncode, 0, good.stderr)
        args = self.args()
        start = args.index("--manifest-sha256")
        del args[start:start + 2]
        bad = subprocess.run([sys.executable, str(ROOT / "calibrate.py"), *args],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(bad.stdout, "")

    def test_known_metrics_and_coerrors_hand_counted(self):
        r = self.result()
        a, b = r["per_model_known_bit_readout"]
        expected = [(1, 1, 2, 1, 1, 0), (2, 0, 4, 0, 1, 0),
                    (1, 1, 2, 1, 1, 0), (1, 1, 4, 0, 1, 0)]
        keys = ("true_positive", "false_negative", "true_negative", "false_positive",
                "unknown_on_positive", "unknown_on_negative")
        for values, target in zip(expected, a["per_target"] + b["per_target"]):
            self.assertEqual(tuple(target[k] for k in keys), values)
            self.assertEqual(sum(values), target["human_known_bits"])
            self.assertEqual(sum(values[:4]), target["covered_known_bits"])
        self.assertEqual([x["human_known_bits"] for x in a["per_target"]], [6, 7])
        self.assertEqual([x["human_unknown_bits"] for x in a["per_target"]], [2, 1])
        paired = r["paired_known_bit_readout"]
        self.assertEqual([x["recording_id"] for x in paired[0]["both_wrong_known_bits"]],
                         ["synthetic-04", "synthetic-05"])
        self.assertEqual(len(paired[1]["definite_disagreements_known_bits"]), 1)

    def test_unknown_gold_and_conflicting_human_labels_preserved(self):
        result = self.result()
        for record, declared in zip(result["records"], self.data["manifest"]["records"]):
            self.assertEqual([v["presence"] for v in record["final_target_presence"]],
                             declared["human_target_presence"])
            self.assertTrue(all(not v["gold_changed_by_models"] for v in record["final_target_presence"]))
            self.assertTrue(all(not v["promoted_to_human_gold"] for v in record["weak_model_consensus"]))
        unknown = result["records"][2]
        self.assertEqual([x["presence"] for x in unknown["weak_model_consensus"]], ["positive", "positive"])
        self.assertEqual([x["presence"] for x in unknown["final_target_presence"]], ["unknown", "unknown"])
        self.assertEqual(result["records"][4]["final_target_presence"][0]["human_conflict_models"], [0, 1])

    def test_all_human_unknown_has_null_coverage(self):
        for row in self.data["manifest"]["records"]:
            row["human_target_presence"] = ["unknown", "unknown"]
        self.refresh()
        for model in self.result()["per_model_known_bit_readout"]:
            for t in model["per_target"]:
                self.assertEqual(t["known_bit_coverage"], {"numerator": 0, "denominator": 0, "fraction": None})

    def test_failed_rows_are_unknown_not_false_negative(self):
        for row in self.data["model_a"]["records"]:
            row.update(status="error", raw_text="星河开灯", completeness="unknown")
        self.refresh()
        for t in self.result()["per_model_known_bit_readout"][0]["per_target"]:
            self.assertEqual(t["false_negative"], 0)
            self.assertEqual(t["covered_known_bits"], 0)
            self.assertEqual(t["unknown_on_positive"] + t["unknown_on_negative"], t["human_known_bits"])

    def test_declared_output_never_claims_verified_execution(self):
        self.change_kind("declared_model_output")
        r = self.result()
        self.assertEqual(r["evidence_level"], "external-declarations-only")
        self.assertFalse(r["inference_performed_by_this_tool"])
        self.assertFalse(r["wav_bytes_verified_by_this_tool"])
        self.assertIn("not actual WAV consumption", r["limitations"][0])

    def test_not_run_template_zero_coverage(self):
        self.change_kind("not_run_template")
        for name in ("model_a", "model_b"):
            for row in self.data[name]["records"]:
                row.update(status="not_run", raw_text=None, completeness="unknown", quality_flags=[])
        self.refresh()
        r = self.result()
        self.assertTrue(all(t["covered_known_bits"] == 0 for m in r["per_model_known_bit_readout"] for t in m["per_target"]))

    def test_all_not_run_cannot_claim_execution(self):
        self.change_kind("declared_model_output")
        for row in self.data["model_a"]["records"]:
            row.update(status="not_run", raw_text=None, completeness="unknown")
        self.refresh()
        self.rejected()

    def test_template_cannot_contain_successful_prediction(self):
        self.change_kind("not_run_template")
        self.rejected()

    def test_missing_mandatory_marker_cannot_weaken_rule(self):
        self.data["rules"]["uncertainty_markers"].remove("[unk]")
        self.refresh()
        self.rejected()

    def test_nontarget_lexicon_oov_or_ambiguity_abstains(self):
        for value in (None, "tiān,tián"):
            key = str(ord("天"))
            if value is None:
                del self.data["characters"][key]
            else:
                self.data["characters"][key] = value
            self.data["rules"]["lexicons"]["characters"]["entries"] = len(self.data["characters"])
            self.assertEqual([t["presence"] for t in self.derive("今天晴朗")["targets"]],
                             ["unknown", "unknown"])

    def test_transcript_at_length_limit_retained(self):
        text = "星" * c.MAX_TEXT
        self.data["model_a"]["records"][0]["raw_text"] = text
        self.refresh()
        self.assertEqual(self.result()["records"][0]["model_evidence"][0]["raw_text"], text)

    def test_cli_bad_declared_sha_type_shape(self):
        for bad in ("a" * 63, "G" * 64, "A" * 64, "main", ""):
            self.hashes["manifest"] = bad
            self.rejected()

    def test_no_model_evidence_fields_can_override_state(self):
        self.data["model_a"]["human_labels"] = ["positive"]
        self.refresh()
        self.rejected()

    def test_model_rows_reordered_still_id_sha_joined(self):
        expected = self.result()
        self.output = self.root / "permuted.json"
        self.data["model_a"]["records"].reverse()
        self.refresh()
        actual = self.result()
        actual.pop("input_sha256")
        expected.pop("input_sha256")
        self.assertEqual(actual, expected)

    def test_sha_predeclaration_drift(self):
        for name in ("manifest", "rules"):
            with self.subTest(name=name):
                before = self.paths[name].read_bytes()
                self.paths[name].write_bytes(before + b" ")
                self.rejected()
                self.paths[name].write_bytes(before)

    def test_model_identity_drift(self):
        for key, value in (("model_id", "fixture-only-other"), ("revision", "c" * 40),
                           ("run_id", "different"), ("revision", "main"), ("run_id", 1)):
            with self.subTest(key=key, value=value):
                original = self.data["model_a"]["model"][key]
                self.data["model_a"]["model"][key] = value
                self.refresh()
                self.rejected()
                self.data["model_a"]["model"][key] = original

    def test_model_header_drift(self):
        for key, value in (("manifest_sha256", "0" * 64), ("rules_sha256", "0" * 64),
                           ("execution_kind", "declared_model_output"), ("schema_version", "legacy")):
            with self.subTest(key=key):
                original = self.data["model_a"][key]
                self.data["model_a"][key] = value
                self.write("model_a")
                self.rejected()
                self.data["model_a"][key] = original

    def test_record_validation_matrix(self):
        cases = [("recording_id", "unlisted"), ("recording_id", []), ("wav_sha256", "0" * 64),
                 ("wav_sha256", True), ("status", True), ("status", "maybe"),
                 ("raw_text", []), ("raw_text", None), ("completeness", True),
                 ("quality_flags", ["ambiguous", "ambiguous"]), ("quality_flags", [True]),
                 ("quality_flags", "ambiguous"), ("quality_flags", ["unlisted"]),
                 ("status", "not_run"), ("raw_text", "a" * (c.MAX_TEXT + 1))]
        base = copy.deepcopy(self.data["model_a"]["records"][0])
        for key, value in cases:
            with self.subTest(key=key, value=str(value)[:40]):
                self.data["model_a"]["records"][0] = dict(base, **{key: value})
                self.refresh()
                self.rejected()

    def test_duplicate_missing_extra_and_id_swapped_records(self):
        rows = self.data["model_a"]["records"]
        variants = [rows[:-1], rows + [copy.deepcopy(rows[0])],
                    [copy.deepcopy(rows[1])] + rows[1:],
                    [dict(rows[0], recording_id=rows[1]["recording_id"]), dict(rows[1], recording_id=rows[0]["recording_id"])] + rows[2:]]
        for variant in variants:
            with self.subTest(length=len(variant)):
                self.data["model_a"]["records"] = variant
                self.refresh()
                self.rejected()

    def test_manifest_validation_matrix(self):
        original = copy.deepcopy(self.data["manifest"])
        cases = [("target_order", []), ("target_order", ["星 河"]), ("target_order", ["a" * 17]),
                 ("target_order", ["星河", "星河"]), ("models", [original["models"][0]]),
                 ("records", []), ("records", [original["records"][0]] * (c.MAX_RECORDS + 1)),
                 ("execution_kind", "actual_model_output"), ("probe_set_id", "bad\nID")]
        for key, value in cases:
            with self.subTest(key=key):
                self.data["manifest"] = dict(original, **{key: value})
                self.refresh()
                self.rejected()

    def test_manifest_duplicate_ids_bool_labels_and_shared_models(self):
        original = copy.deepcopy(self.data["manifest"])
        variants = []
        bad = copy.deepcopy(original); bad["records"][1] = copy.deepcopy(bad["records"][0]); variants.append(bad)
        bad = copy.deepcopy(original); bad["records"][0]["human_target_presence"] = [True, "unknown"]; variants.append(bad)
        bad = copy.deepcopy(original); bad["models"][1] = copy.deepcopy(bad["models"][0]); variants.append(bad)
        bad = copy.deepcopy(original); bad["models"][0]["model_id"] = "claimed-real-brand"; variants.append(bad)
        for bad in variants:
            self.data["manifest"] = bad
            self.refresh()
            self.rejected()

    def test_generic_single_target_manifest_supported(self):
        self.data["manifest"]["target_order"].pop()
        self.data["rules"]["target_order"].pop()
        for row in self.data["manifest"]["records"]:
            row["human_target_presence"].pop()
        self.refresh()
        self.assertEqual(len(self.result()["paired_known_bit_readout"]), 1)

    def test_model_total_text_bound(self):
        manifest, a, b = (self.data[k] for k in ("manifest", "model_a", "model_b"))
        manifest["records"], a["records"], b["records"] = [], [], []
        for i in range(c.MAX_TOTAL_TEXT // c.MAX_TEXT + 1):
            row = {"recording_id": f"long-{i}", "wav_sha256": f"{i:064x}"}
            manifest["records"].append(dict(row, human_target_presence=["unknown", "unknown"]))
            for run in (a, b):
                run["records"].append(dict(row, status="success", raw_text="星" * c.MAX_TEXT,
                                           completeness="complete", quality_flags=[]))
        self.refresh()
        self.rejected()

    def test_strict_json_duplicate_nan_float_unicode_depth(self):
        payloads = [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1.0}',
                    b'{"a":1e9999}', b'\xff', b'"\\ud800"', b'"\\udfff"',
                    b'{"a":"\\u0000"}', b'[' * 1000 + b']' * 1000,
                    b'[' * 10 + b'0' + b']' * 10, b'{"a":' + b'1' * 5000 + b'}']
        for payload in payloads:
            with self.subTest(payload=payload[:25]):
                self.paths["model_a"].write_bytes(payload)
                self.rejected()

    def test_lexicon_hash_tamper_rejected(self):
        self.paths["characters"].write_bytes(self.paths["characters"].read_bytes() + b" ")
        self.rejected()

    def test_lexicon_and_rule_type_length_limits(self):
        original = copy.deepcopy(self.data)
        variants = [("characters", {"001": "a"}), ("characters", {"55296": "a"}),
                    ("characters", {"65": True}), ("characters", {"65": "a" * 33}),
                    ("phrases", {"星河": [["xīng"]]}),
                    ("phrases", {"星" * (c.MAX_PHRASE_LENGTH + 1): [["xīng"]] * (c.MAX_PHRASE_LENGTH + 1)})]
        for key, values in variants:
            self.data = copy.deepcopy(original)
            self.data[key] = values
            self.data["rules"]["lexicons"][key]["entries"] = len(values)
            self.refresh()
            self.rejected()
        for field, value in (("rule_version", "legacy"), ("uncertainty_markers", []),
                             ("uncertainty_markers", ["a" * 65]),
                             ("uncertainty_markers", ["?"] * (c.MAX_RULE_MARKERS + 1))):
            self.data = copy.deepcopy(original)
            self.data["rules"][field] = value
            self.refresh()
            self.rejected()
        self.data = copy.deepcopy(original)
        self.data["rules"]["lexicons"]["characters"]["entries"] = True
        self.refresh()
        self.rejected()

    def test_partial_or_ambiguous_target_lexicon_rejected(self):
        self.data["characters"][str(ord("灯"))] = "dēng,dèng"
        self.refresh()
        self.rejected()

    def test_file_limits_regular_files_and_symlinks(self):
        original = self.paths["model_a"].read_bytes()
        for payload in (b"", b" " * (c.FILE_LIMITS["model_a"] + 1)):
            self.paths["model_a"].write_bytes(payload)
            self.rejected()
        self.paths["model_a"].unlink()
        self.paths["model_a"].mkdir()
        self.rejected()
        self.paths["model_a"].rmdir()
        real = self.inputs / "real.json"; real.write_bytes(original)
        self.paths["model_a"].symlink_to(real)
        self.rejected()
        self.paths["model_a"].unlink()
        os.mkfifo(self.paths["model_a"])
        self.rejected()  # Must not block waiting for a writer.

    def test_no_overwrite_existing_input_hardlink_or_symlink(self):
        self.output.write_bytes(b"KEEP")
        self.assertEqual(self.cli()[0], 2)
        self.assertEqual(self.output.read_bytes(), b"KEEP")
        self.output.unlink()
        for mode in ("hardlink", "symlink"):
            with self.subTest(mode=mode):
                original = self.paths["model_a"].read_bytes()
                if mode == "hardlink": os.link(self.paths["model_a"], self.output)
                else: self.output.symlink_to(self.paths["model_a"])
                self.assertEqual(self.cli()[0], 2)
                self.assertEqual(self.paths["model_a"].read_bytes(), original)
                self.output.unlink()
        self.output = self.paths["manifest"]
        before = self.output.read_bytes()
        self.assertEqual(self.cli()[0], 2)
        self.assertEqual(self.output.read_bytes(), before)

    def test_racing_output_creation_and_serialization_limit(self):
        original_link = os.link
        def race(src, dst):
            Path(dst).write_bytes(b"RACE")
            original_link(src, dst)
        with mock.patch.object(c.os, "link", side_effect=race):
            self.assertEqual(self.cli()[0], 2)
        self.assertEqual(self.output.read_bytes(), b"RACE")
        self.assertFalse(list(self.root.glob(".asr-evidence-*")))
        self.output.unlink()
        with mock.patch.object(c, "MAX_OUTPUT_BYTES", 1):
            self.rejected()

    def test_duplicate_input_inode(self):
        self.paths["model_b"].unlink()
        os.link(self.paths["model_a"], self.paths["model_b"])
        self.rejected()

    def test_text_rule_matrix(self):
        cases = [("星河开灯", ["positive", "negative"]), ("今天晴朗", ["negative", "negative"]),
                 ("星河开等", ["unknown", "negative"]), ("星，河开灯", ["unknown", "negative"]),
                 ("星河开", ["unknown", "negative"]), ("心合开灯", ["unknown", "negative"]),
                 ("星河凯灯", ["unknown", "negative"]), ("", ["unknown", "unknown"]),
                 (" \n\t", ["unknown", "unknown"]), ("...", ["unknown", "unknown"]),
                 ("星河开灯[unk]", ["unknown", "unknown"]), ("星河开灯不确定", ["unknown", "unknown"]),
                 ("<|zh|>星河开灯", ["unknown", "unknown"]),
                 ("language Chinese<asr_text>星河开灯", ["unknown", "unknown"]),
                 ("星河开灯\u202e", ["unknown", "unknown"]), ("星河\u200b开灯", ["unknown", "unknown"]),
                 ("🌙", ["unknown", "unknown"])]
        for text, expected in cases:
            with self.subTest(text=text):
                evidence = self.derive(text)
                self.assertEqual([t["presence"] for t in evidence["targets"]], expected)
                self.assertEqual(evidence["raw_text"], text)
                self.assertTrue(all(not t["rule_evidence"]["pinyin_is_acoustic_measurement"] for t in evidence["targets"]))

    def test_status_completeness_quality_abstain_even_literal(self):
        overrides = [{"status": x} for x in ("error", "timeout", "not_run")]
        overrides += [{"completeness": x} for x in ("incomplete", "unknown")]
        overrides += [{"quality_flags": [x]} for x in c.FLAGS]
        for flags in overrides:
            with self.subTest(flags=flags):
                self.assertEqual([t["presence"] for t in self.derive("星河开灯", **flags)["targets"]],
                                 ["unknown", "unknown"])

    def test_unicode_raw_preservation_normalization_and_tones(self):
        raw = " \t星河开灯，\n星河开灯e\u0301"
        derived = self.derive(raw)
        self.assertEqual(derived["raw_text"], raw)
        self.assertEqual(derived["normalized_text"], "星河开灯， 星河开灯é")
        self.assertEqual(c.normalize("Ａ"), "Ａ")
        self.assertNotEqual(c.toneless("lǜ"), c.toneless("lù"))
        self.assertEqual(c.toneless("lǜ"), c.toneless("lü4"))
        self.assertEqual(c.toneless("lǜ"), c.toneless("lu:4"))

    def test_no_reliability_accuracy_or_transcript_metrics(self):
        forbidden = {"cer", "wer", "accuracy", "reliability", "occurrence_count", "word_timestamps"}
        pending = [self.result()]
        while pending:
            item = pending.pop()
            if type(item) is dict:
                self.assertFalse(forbidden & item.keys())
                pending.extend(item.values())
            elif type(item) is list:
                pending.extend(item)


if __name__ == "__main__":
    unittest.main(verbosity=2)
