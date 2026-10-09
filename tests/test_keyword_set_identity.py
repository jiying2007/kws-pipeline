#!/usr/bin/env python3
"""Synthetic, offline keyword identity regressions; no model/audio operations."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compile_keywords as compiler
import keyword_set_identity as identity
from kws_vocab import load_tokens
from validate_shipping_keywords import validate_shipping_keywords


class IdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parameters = json.loads((ROOT / "configs/parameter-contract.json").read_text())
        self.write_parameters()
        self.tokens = "<blk> 0\na 1\nb 2\nunused 3\n"
        self.rows = ["1\talpha\t0.55\ta b\t0\t0\tlongest\t0",
                     "2\tbeta\t0.6\tb a\t0\t1\tgrace\t0"]
        self.write("tokens.txt", self.tokens)
        self.write_rows()

    def write(self, name, text):
        (self.root / name).write_text(text, encoding="utf-8")

    def write_rows(self):
        self.write("keywords.tsv", "\n".join(self.rows) + "\n")

    def write_parameters(self):
        self.write("parameters.json", json.dumps(self.parameters))

    def build(self, **changes):
        arguments = dict(root=self.root, tokens_path="tokens.txt",
                         keywords_path="keywords.tsv", parameter_contract_path="parameters.json")
        return identity.build_keyword_set_identity(**(arguments | changes))

    def save_contract(self, value=None):
        value = self.build() if value is None else value
        self.write("identity.json", json.dumps(value))
        return value

    def verify(self):
        return identity.verify_keyword_set_contract("identity.json", root=self.root)

    def test_canonical_parsers_and_every_effective_field(self):
        with patch.object(compiler, "parse_keywords", wraps=compiler.parse_keywords) as parse, \
                patch.object(compiler, "load_contract", wraps=compiler.load_contract) as load, \
                patch.object(compiler, "text_to_pinyin", side_effect=AssertionError("implicit conversion")):
            built = self.build()
        parse.assert_called_once()
        load.assert_called_once()
        expected = compiler.parse_keywords(self.root / "keywords.tsv",
                                          load_tokens(self.root / "tokens.txt"), self.parameters)
        self.assertEqual(built["semantics"]["keywords"], expected)
        self.assertEqual(set(expected[0]), {"id", "text", "threshold", "tokens", "token_ids",
                                         "min_trailing_blanks", "priority", "prefix_policy",
                                         "prefix_policy_id", "grace_frames"})
        self.assertEqual(expected[0]["min_trailing_blanks"], 1)
        self.assertEqual(expected[1]["grace_frames"], 3)
        self.assertEqual(built["semantic_sha256"], hashlib.sha256(
            identity.canonical_bytes(built["semantics"])).hexdigest())

    def test_every_tsv_field_affects_identity(self):
        baseline = self.build()["semantic_sha256"]
        original = self.rows[0]
        changes = {0: "3", 1: "renamed", 2: "0.56", 3: "a unused", 4: "2", 5: "4",
                   6: "immediate", 7: "4"}
        for column, value in changes.items():
            with self.subTest(column=column):
                row = original.split("\t")
                row[column] = value
                self.rows[0] = "\t".join(row)
                self.write_rows()
                self.assertNotEqual(baseline, self.build()["semantic_sha256"])
        self.rows[0] = original
        self.rows.reverse()
        self.write_rows()
        self.assertNotEqual(baseline, self.build()["semantic_sha256"])

    def test_entire_vocabulary_and_token_map_are_bound(self):
        baseline = self.build()["semantic_sha256"]
        for changed in (self.tokens.replace("unused", "other"), self.tokens + "extra 4\n",
                        "<blk> 0\nb 1\na 2\nunused 3\n"):
            with self.subTest(tokens=changed):
                self.write("tokens.txt", changed)
                self.assertNotEqual(baseline, self.build()["semantic_sha256"])

    def test_defaults_ranges_and_policy_defaults_are_bound(self):
        baseline = self.build()["semantic_sha256"]
        original = copy.deepcopy(self.parameters)
        mutations = [("keyword_pack", "min_trailing_blanks", "default", 2),
                     ("keyword_pack", "priority", "default", 2),
                     ("keyword_pack", "grace_frames", "default", 2),
                     ("keyword_pack", "threshold", "min", 0.1),
                     ("keyword_pack", "priority", "max", 14),
                     ("policy_defaults", "longest", "min_trailing_blanks", 2),
                     ("policy_defaults", "grace", "grace_frames", 4)]
        for table, name, field, value in mutations:
            with self.subTest(table=table, name=name, field=field):
                self.parameters = copy.deepcopy(original)
                self.parameters[table][name][field] = value
                self.write_parameters()
                self.assertNotEqual(baseline, self.build()["semantic_sha256"])

    def test_omitted_values_use_current_canonical_defaults(self):
        self.rows = ["1\talpha\t0.55\ta b"]
        self.write_rows()
        baseline = self.build()["semantics"]["keywords"][0]
        self.assertEqual([baseline[k] for k in ("min_trailing_blanks", "priority", "grace_frames")], [0, 0, 0])
        for key in ("min_trailing_blanks", "priority", "grace_frames"):
            self.parameters["keyword_pack"][key]["default"] = 2
        self.write_parameters()
        changed = self.build()["semantics"]["keywords"][0]
        self.assertEqual([changed[k] for k in ("min_trailing_blanks", "priority", "grace_frames")], [2, 2, 2])

    def test_formatting_comments_and_explicit_defaults(self):
        baseline = self.build()
        self.write("tokens.txt", "# comment\nunused 3\nb 2\n a 1\n<blk> 0\n\n")
        self.write("keywords.tsv", "# comment\n\n1\t alpha \t5.500e-1\t a   b \t1\t0\tLONGEST\t0\n"
                   "2\tbeta\t0.600000\tb a\t0\t1\tgrace\t3\n")
        self.parameters["note"] = "changed commentary"
        self.parameters["keyword_pack"]["threshold"]["summary"] = "changed commentary"
        self.parameters["keyword_pack"]["threshold"]["min"] = 0
        self.write("parameters.json", json.dumps(self.parameters, indent=4, sort_keys=True))
        changed = self.build()
        self.assertEqual(baseline["semantic_sha256"], changed["semantic_sha256"])
        for key in identity.SOURCES:
            self.assertNotEqual(baseline["sources"][key]["sha256"], changed["sources"][key]["sha256"])

    def test_runtime_parameters_are_outside_keyword_semantics(self):
        baseline = self.build()
        self.parameters["runtime"]["state_retention"]["default"] = 0.93
        self.write_parameters()
        changed = self.build()
        self.assertEqual(baseline["semantic_sha256"], changed["semantic_sha256"])
        self.assertNotEqual(baseline["sources"]["parameter_contract"], changed["sources"]["parameter_contract"])

    def test_verification_and_identity_json_formatting(self):
        baseline = self.save_contract()
        self.assertEqual(self.verify(), baseline)
        self.write("identity.json", json.dumps(baseline, indent=4, sort_keys=True))
        self.assertEqual(self.verify(), baseline)

    def test_raw_changes_rejected_even_when_semantics_match(self):
        self.save_contract()
        self.write("keywords.tsv", "# new comment\n" + "\n".join(self.rows) + "\n")
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.verify()

    def test_contract_all_unknown_fields_and_wrong_types_rejected(self):
        original = self.build()
        mutations = [((), "unknown", 1), ((), "schema_version", True), ((), "schema_version", "2"),
                     ((), "schema_version", 1), ((), "policy", "kws-keyword-set-identity-v1"),
                     ((), "semantic_sha256", "0" * 64), ((), "semantics", []),
                     (("sources",), "unexpected", {}),
                     (("sources", "tokens"), "sha256", "A" * 64),
                     (("sources", "tokens"), "bytes", True),
                     (("sources", "tokens"), "bytes", -1),
                     (("sources", "tokens"), "unknown", True),
                     (("semantics",), "unknown", True),
                     (("semantics", "keywords", 0), "unknown", True),
                     (("semantics", "keywords", 0), "id", True),
                     (("semantics", "keywords", 0), "threshold", "0.55"),
                     (("semantics", "tokens", 0), "id", False),
                     (("semantics", "parameter_policy", "keyword_pack", "priority"), "max", True)]
        for path, key, value in mutations:
            with self.subTest(path=path, key=key, value=value):
                changed = copy.deepcopy(original)
                target = changed
                for part in path:
                    target = target[part]
                target[key] = value
                self.save_contract(changed)
                with self.assertRaises(ValueError):
                    self.verify()
        for key in original:
            changed = copy.deepcopy(original)
            del changed[key]
            self.save_contract(changed)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.verify()

    def test_invalid_json_is_rejected_in_both_contracts(self):
        valid = self.build()
        invalid = [b"[]", b"null", b"true", b"{", b'{"a":1,"a":2}',
                   b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e9999}', b"\xff"]
        for name in ("parameters.json", "identity.json"):
            for raw in invalid:
                with self.subTest(name=name, raw=raw):
                    self.write_parameters()
                    self.save_contract(valid)
                    (self.root / name).write_bytes(raw)
                    with self.assertRaises((ValueError, UnicodeError)):
                        self.build() if name == "parameters.json" else self.verify()

    def test_malformed_parameter_contracts(self):
        original = copy.deepcopy(self.parameters)
        mutations = [((), "unknown", {}), ((), "schema_version", True), ((), "schema_version", 2),
                     ((), "contract_id", "new-unsupported-contract"), ((), "layers", []),
                     (("layers",), "L4", "unknown"), (("runtime",), "unknown", {}),
                     (("algorithm_constants",), "KWS_UNKNOWN", {}),
                     (("keyword_pack",), "unknown", {}),
                     (("keyword_pack", "threshold"), "typo", 0),
                     (("keyword_pack", "threshold"), "min_exclusive", "false"),
                     (("keyword_pack", "threshold"), "min", True),
                     (("keyword_pack", "threshold"), "max", "1"),
                     (("keyword_pack", "threshold"), "max", 1e100),
                     (("keyword_pack", "threshold"), "min", None),
                     (("keyword_pack", "priority"), "default", 1.5),
                     (("keyword_pack", "priority"), "default", -1),
                     (("keyword_pack", "priority"), "default", 16),
                     (("keyword_pack", "priority"), "default", None),
                     (("keyword_pack", "priority"), "max", 256),
                     (("keyword_pack", "priority"), "type", "float"),
                     (("keyword_pack", "priority"), "layer", "L2"),
                     (("keyword_pack", "priority"), "effective", False),
                     (("keyword_pack", "priority"), "min_exclusive", True),
                     (("policy_defaults",), "unknown", {}),
                     (("policy_defaults", "longest"), "extra", 1),
                     (("policy_defaults", "longest"), "min_trailing_blanks", True),
                     (("policy_defaults", "longest"), "min_trailing_blanks", 0),
                     (("policy_defaults", "longest"), "min_trailing_blanks", 9),
                     (("policy_defaults", "grace"), "grace_frames", 1.5),
                     (("runtime", "state_retention"), "default", 1.5),
                     (("algorithm_constants", "KWS_MEL_HIGH_HZ"), "default", 9000)]
        for path, key, value in mutations:
            with self.subTest(path=path, key=key, value=value):
                self.parameters = copy.deepcopy(original)
                target = self.parameters
                for part in path:
                    target = target[part]
                target[key] = value
                self.write_parameters()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.build()

    def test_invalid_keyword_rows_fail_closed(self):
        rows = ["", "# only comment", "1\t中文\t0.55", "1\t中文\t0.55\t",
                "1\t\t0.55\ta b", "1\talpha\tNaN\ta b", "1\talpha\tinf\ta b",
                "1\talpha\t0\ta b", "1\talpha\t1\ta b", "1\talpha\t0.55\tmissing",
                "1\talpha\t0.55\t<blk>", "-1\talpha\t0.55\ta b",
                "4294967296\talpha\t0.55\ta b", "true\talpha\t0.55\ta b",
                "1\talpha\t0.55\ta b\t9", "1\talpha\t0.55\ta b\t0\t16",
                "1\talpha\t0.55\ta b\t0\t0\tunknown",
                "1\talpha\t0.55\ta b\t0\t0\tgrace\t33",
                "1\talpha\t0.55\ta b\t0\t0\tgrace\t3\textra",
                "1\talpha\t0.55\t" + " ".join(["a"] * 17),
                self.rows[0] + "\n1\tsecond\t0.6\tb a", self.rows[0] + "\n2\tsecond\t0.6\ta b"]
        with patch.object(compiler, "text_to_pinyin", side_effect=AssertionError("must not convert")):
            for row in rows:
                with self.subTest(row=row):
                    self.write("keywords.tsv", row)
                    with self.assertRaises(ValueError):
                        self.build()

    def test_keyword_count_limit(self):
        self.write("tokens.txt", "<blk> 0\n" + "".join(f"t{i} {i}\n" for i in range(1, 18)))
        self.write("keywords.tsv", "".join(f"{i}\tword{i}\t0.55\tt{i}\n" for i in range(1, 18)))
        with self.assertRaisesRegex(ValueError, "at most"):
            self.build()

    def test_invalid_vocabulary_fails_closed(self):
        for tokens in ("", "a 0\nb 1", "<blk> 0\na 1\na 2", "<blk> 0\na 1\nb 1",
                       "<blk> 0\na 2", "<blk> 0\na -1", "<blk> 0\na 512", "<blk> 0\na True",
                       "<blk> 0\na 1 extra"):
            with self.subTest(tokens=tokens):
                self.write("tokens.txt", tokens)
                with self.assertRaises(ValueError):
                    self.build()

    def test_paths_reject_escape_noncanonical_missing_directory_symlink(self):
        for name in ("", ".", "..", "../tokens.txt", "/etc/passwd", str(self.root / "tokens.txt"),
                     "a/../tokens.txt", "a//tokens.txt", "./tokens.txt", "a\\tokens.txt",
                     "missing.txt", "bad\x00name", True, None):
            with self.subTest(path=name), self.assertRaises((ValueError, OSError)):
                self.build(tokens_path=name)
        (self.root / "directory").mkdir()
        (self.root / "link").symlink_to(self.root / "tokens.txt")
        (self.root / "out").symlink_to(self.root.parent, target_is_directory=True)
        for name in ("directory", "link", "out/anything"):
            with self.subTest(path=name), self.assertRaises((ValueError, OSError)):
                self.build(tokens_path=name)
        with self.assertRaises((ValueError, OSError)):
            self.build(root=self.root / "tokens.txt")

    def test_contract_path_and_every_source_confined_to_root(self):
        original = self.save_contract()
        for key in identity.SOURCES:
            changed = copy.deepcopy(original)
            changed["sources"][key]["path"] = "../outside"
            self.save_contract(changed)
            with self.subTest(source=key), self.assertRaises(ValueError):
                self.verify()
        with self.assertRaises(ValueError):
            identity.verify_keyword_set_contract("../outside", root=self.root)
        (self.root / "identity-link.json").symlink_to(self.root / "identity.json")
        with self.assertRaises((ValueError, OSError)):
            identity.verify_keyword_set_contract("identity-link.json", root=self.root)

    def test_input_snapshots_match_digests_even_if_original_changes(self):
        original = compiler.parse_keywords
        before = (self.root / "keywords.tsv").read_bytes()
        def mutate_then_parse(*args, **kwargs):
            self.write("keywords.tsv", "1\tchanged\t0.7\ta\n")
            return original(*args, **kwargs)
        with patch.object(compiler, "parse_keywords", side_effect=mutate_then_parse):
            built = self.build()
        self.assertEqual(built["sources"]["keywords"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(built["semantics"]["keywords"][0]["text"], "alpha")

    def test_cli_build_verify_and_failure_without_traceback(self):
        command = [sys.executable, "-B", str(ROOT / "tools/keyword_set_identity.py")]
        built = subprocess.run(command + ["build", "--root", str(self.root), "--tokens", "tokens.txt",
                               "--keywords", "keywords.tsv", "--parameter-contract", "parameters.json"],
                               check=True, capture_output=True, text=True)
        self.write("identity.json", built.stdout)
        verified = subprocess.run(command + ["verify", "--root", str(self.root), "--contract", "identity.json"],
                                  check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(built.stdout), json.loads(verified.stdout))
        self.write("identity.json", "[]")
        failed = subprocess.run(command + ["verify", "--root", str(self.root), "--contract", "identity.json"],
                                capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1)
        self.assertEqual(failed.stdout, "")
        self.assertNotIn("Traceback", failed.stderr)

    def exercise_open_swap(self, target, mode):
        """Swap exactly between name selection and the kernel open operation."""
        names = {"tokens": "tokens.txt", "keywords": "keywords.tsv",
                 "parameter_contract": "parameters.json", "contract": "identity.json"}
        baseline = self.save_contract()
        nested = self.root / "nested"
        nested.mkdir()
        leaf = names[target]
        (nested / leaf).write_bytes((self.root / leaf).read_bytes())
        kwargs = {"tokens_path": "tokens.txt", "keywords_path": "keywords.tsv",
                  "parameter_contract_path": "parameters.json"}
        if target != "contract":
            kwargs[{"tokens": "tokens_path", "keywords": "keywords_path",
                    "parameter_contract": "parameter_contract_path"}[target]] = "nested/" + leaf
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other)
            (outside / leaf).write_bytes(b"OUTSIDE ROOT: MUST NEVER BE READ")
            boundary = nested / leaf if mode == "leaf" else nested
            backup = self.root / "held"
            real_open = os.open
            triggered = False
            def swap_open(name, flags, *args, **kw):
                nonlocal triggered
                match = (name == leaf and not flags & os.O_DIRECTORY) if mode == "leaf" else name == "nested"
                if not triggered and match:
                    triggered = True
                    fd = real_open(name, flags, *args, **kw) if mode == "opened-parent" else None
                    boundary.rename(backup)
                    boundary.symlink_to(outside / leaf if mode == "leaf" else outside,
                                        target_is_directory=mode != "leaf")
                    if fd is not None:
                        return fd
                return real_open(name, flags, *args, **kw)
            def action():
                if target == "contract":
                    return identity.verify_keyword_set_contract("nested/identity.json", root=self.root)
                return self.build(**kwargs)
            try:
                with patch.object(os, "open", swap_open), \
                        patch.object(os, "supports_dir_fd", os.supports_dir_fd | {swap_open}):
                    if mode == "opened-parent":
                        built = action()
                        self.assertEqual(built["semantic_sha256"], baseline["semantic_sha256"])
                        for key in identity.SOURCES:
                            self.assertEqual(built["sources"][key]["sha256"], baseline["sources"][key]["sha256"])
                    else:
                        with self.assertRaises(OSError):
                            action()
                self.assertTrue(triggered)
            finally:
                if triggered:
                    boundary.unlink()
                    backup.rename(boundary)
                (nested / leaf).unlink()
                nested.rmdir()

    def test_leaf_swap_race_rejected_for_every_input(self):
        for target in (*identity.SOURCES, "contract"):
            with self.subTest(target=target):
                self.exercise_open_swap(target, "leaf")

    def test_parent_swap_race_rejected_for_every_input(self):
        for target in (*identity.SOURCES, "contract"):
            with self.subTest(target=target):
                self.exercise_open_swap(target, "parent")

    def test_parent_swap_after_open_keeps_original_descriptor_for_every_input(self):
        for target in (*identity.SOURCES, "contract"):
            with self.subTest(target=target):
                self.exercise_open_swap(target, "opened-parent")

    def test_verification_keeps_one_root_descriptor_after_contract_read(self):
        baseline = self.save_contract()
        real_read = identity.read_input
        backup = self.root.with_name(self.root.name + "-held")
        with tempfile.TemporaryDirectory() as other:
            def swap_root(fd, name):
                raw = real_read(fd, name)
                if name == "identity.json":
                    self.root.rename(backup)
                    self.root.symlink_to(other, target_is_directory=True)
                return raw
            try:
                with patch.object(identity, "read_input", side_effect=swap_root):
                    self.assertEqual(self.verify(), baseline)
            finally:
                if backup.exists():
                    self.root.unlink()
                    backup.rename(self.root)

    def test_root_symlink_swap_before_open_rejected(self):
        real_open = os.open
        backup = self.root.with_name(self.root.name + "-held")
        with tempfile.TemporaryDirectory() as other:
            def swap_open(name, flags, *args, **kwargs):
                if name == self.root.name:
                    self.root.rename(backup)
                    self.root.symlink_to(other, target_is_directory=True)
                return real_open(name, flags, *args, **kwargs)
            try:
                with patch.object(os, "open", swap_open), \
                        patch.object(os, "supports_dir_fd", os.supports_dir_fd | {swap_open}):
                    with self.assertRaises(OSError):
                        self.build()
            finally:
                if backup.exists():
                    self.root.unlink()
                    backup.rename(self.root)

    def test_fifo_rejected_without_blocking_and_unsupported_access_fails_closed(self):
        os.mkfifo(self.root / "fifo")
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.build(tokens_path="fifo")
        with patch.object(os, "supports_dir_fd", set()):
            with self.assertRaisesRegex(RuntimeError, "descriptor-relative"):
                self.build()

    def test_float32_edge_thresholds_are_identity_not_binary_validation(self):
        for threshold, rounded in (("1e-100", 0.0), ("0.999999999", 1.0)):
            with self.subTest(threshold=threshold):
                self.write("keywords.tsv", f"1\talpha\t{threshold}\ta b\n")
                built = self.save_contract()
                self.assertEqual(self.verify(), built)
                self.assertEqual(built["semantics"]["keywords"][0]["threshold"], float(threshold))
                compiler.emit_pack(built["semantics"]["keywords"], load_tokens(self.root / "tokens.txt"),
                                   self.root / "fixture.kwk")
                emitted = struct.unpack_from("<f", (self.root / "fixture.kwk").read_bytes(), 28)[0]
                self.assertEqual(emitted, rounded)
                with self.assertRaises(ValueError):
                    compiler.bounded_float(emitted, "wire threshold", self.parameters["keyword_pack"]["threshold"])
                self.assertNotIn("shipping_approved", built)
                self.assertNotIn("binary_validated", built)

    def test_current_shipping_tuple_unchanged(self):
        expected_sha = {
            "configs/parameter-contract.json": "3171898c0c76367ca2fecfd31279ba2d25cfcd2029ebf1d9495337a5d5fbef06",
            "keywords/zh_cn_example.tsv": "1d17ed101c950bbc55327dd56c7e0a03faa6ab1a6ef36cf73ede55d7d9b88db5",
            "keywords/tokens.example.txt": "af113e57eb6375b3c364b3845c6698ea460e29f718c8303657777183d6eda8c7",
            "configs/shipping.xiaowo.json": "4f39810a362096f2345c8e2e3326c3899e0d30b0b5e0632a6f2622a1364d5b21",
            "configs/nightly.xiaowo-frozen-model.json": "a515749b2987e1df23f588a1db8f1e9a7f710662ed6f8b4df92c09aacf40f5f6",
        }
        shipping = json.loads((ROOT / "configs/shipping.xiaowo.json").read_text())
        self.assertFalse(shipping["shipping_approved"])
        self.assertEqual(shipping["model"]["release_tag"], "model-749187ec1d66")
        calibration = shipping["threshold_calibration"]
        self.assertEqual(calibration["parameter_contract_sha256"],
                         "eb124d432382f989aa9e40951d8561c1c1448fd74f167324f2383fa6e6f73a3b")
        self.assertEqual(calibration["required_parameter_contract_sha256"],
                         expected_sha["configs/parameter-contract.json"])
        self.assertTrue(calibration["recalibration_required"])
        self.assertEqual([entry["value"] for entry in calibration["thresholds"]], [0.55, 0.55])
        self.assertEqual(shipping["runtime"], {
            "min_speech_dbfs": -55.0, "token_boost": 1.5, "state_retention": 0.94,
            "refractory_ms": 1200, "external_vad_threshold": 0.45,
        })
        for key, value in {
            "model_sha256": "ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402",
            "checkpoint_sha256": "d005bfe74188c0e24e9e665c10251fd6feea5c879ebcbe96cb8c85f6a85ea14b",
            "keyword_pack_sha256": "370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723",
            "keyword_tsv_sha256": "d04023ccaeeafc6938fd8620762edd739e4760c35321be6edc8b32631b982ab4",
        }.items():
            self.assertEqual(shipping["model"][key], value)
        self.assertEqual(shipping["vocabulary"]["size"], 5)
        for path, digest in expected_sha.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        validate_shipping_keywords(ROOT / "keywords/zh_cn_example.tsv")
        before = {p: (ROOT / p).read_bytes() for p in expected_sha}
        built = identity.build_keyword_set_identity(root=ROOT, tokens_path="keywords/tokens.example.txt",
                                                   keywords_path="keywords/zh_cn_example.tsv",
                                                   parameter_contract_path="configs/parameter-contract.json")
        self.assertEqual(built["semantic_sha256"], "1da8b5e4356080f163fe79301c4664f96d8339b8df4e2d86df2e38f8819bfba1")
        self.assertEqual([row["id"] for row in built["semantics"]["keywords"]], [1, 2])
        self.assertEqual([row["text"] for row in built["semantics"]["keywords"]], ["你好小窝", "小窝小窝"])
        self.assertEqual([row["threshold"] for row in built["semantics"]["keywords"]], [0.55, 0.55])
        self.assertEqual(before, {p: (ROOT / p).read_bytes() for p in expected_sha})


if __name__ == "__main__":
    unittest.main(verbosity=2)
