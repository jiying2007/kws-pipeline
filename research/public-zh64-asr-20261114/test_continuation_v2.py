"""Source-only tests: no network, Docker, package install, cached-code execution or models."""
from pathlib import Path
import ast
import copy
import importlib.util
import json
import marshal
import os
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import continuation_v2 as c
import cache_verify_v2 as k
from test_continuation import composition_fixture

CACHE = "pip/__pycache__/__pip-runner__.cpython-312.pyc"
SOURCE = "pip/__pip-runner__.py"


def inventory_fixture():
    old = {"schema": "whisper-screen-installed-runtime-v1", "files": {
        CACHE: {"bytes": 128, "sha256": "1" * 64},
        SOURCE: {"bytes": 16, "sha256": "2" * 64},
        "pip-25.0.1.dist-info/LICENSE.txt": {"bytes": 12, "sha256": "3" * 64}}}
    fresh = copy.deepcopy(old)
    fresh["files"][CACHE]["sha256"] = "4" * 64
    stable = {name: value for name, value in old["files"].items() if name != CACHE}
    contract = dict(installed_file_count=3, cache_file_count=1, unchanged_file_count=2,
        baseline_inventory=k.identity(k.canonical(old)),
        unchanged_records_sha256=k.sha(k.canonical(stable)),
        source_records_sha256=k.sha(k.canonical({SOURCE: old["files"][SOURCE]})),
        cache_records={CACHE: dict(original_cache=old["files"][CACHE], source_path=SOURCE,
                                  source_identity=old["files"][SOURCE])},
        license_records={"pip-25.0.1.dist-info/LICENSE.txt": old["files"]["pip-25.0.1.dist-info/LICENSE.txt"]})
    return old, fresh, contract


def native_fixture():
    old, fresh, contract = inventory_fixture()
    old_raw, fresh_raw = k.canonical(old), k.canonical(fresh)
    source = dict(schema="retained-asr-source-lock-v1", executed_head_sha="b" * 40,
        files=[dict(path=c.INVENTORY, **k.identity(old_raw)), dict(path="recognizer.py", bytes=9, sha256="9" * 64)],
        source_inventory_scope="unchanged")
    changed = copy.deepcopy(source)
    changed["executed_head_sha"] = "a" * 40
    changed["files"][0].update(k.identity(fresh_raw))
    config = dict(source_lock_sha256=k.sha(k.canonical(source)), asset_lock_sha256="d" * 64,
        effective_config=dict(language="zh", temperature=0.0, sample_len=224))
    identity = dict(model_family="whisper", model_sha256="e" * 64, native_decode_config=config,
                    native_decode_config_sha256=k.sha(k.canonical(config)))
    actual = copy.deepcopy(identity)
    actual["native_decode_config"]["source_lock_sha256"] = k.sha(k.canonical(changed))
    actual["native_decode_config_sha256"] = k.sha(k.canonical(actual["native_decode_config"]))
    return identity, actual, source, changed, old_raw, fresh_raw, contract


def pyc(source=b"value = 3\n", filename="/runtime/venv/lib/python3.12/site-packages/pip/example.py", mtime=123, flags=0, code=None):
    code = compile(source, filename, "exec", dont_inherit=True, optimize=0) if code is None else code
    suffix = struct.pack("<II", mtime & 0xffffffff, len(source)) if flags == 0 else importlib.util.source_hash(source)
    return importlib.util.MAGIC_NUMBER + struct.pack("<I", flags) + suffix + marshal.dumps(code)


class RequestAndLineageTests(unittest.TestCase):
    def setUp(self):
        self.request, self.digest = c.frozen_request()

    def test_original_controls_and_exact_sixteen_roles_preserved(self):
        c.verify_control(self.request)
        old, _ = c.prior.frozen_request()
        self.assertEqual(self.request["slots"], old["slots"])
        self.assertEqual(self.request["max_asr_calls"], 16)
        self.assertEqual(self.request["already_complete_excluded"], {"qwen06": 64, "sensevoice": 64, "whisper": 48})
        self.assertEqual(self.request["minimum_host_available_bytes"], 14 * 1024 ** 3)
        self.assertFalse(self.request["retry"])
        self.assertFalse(self.request["training_admitted"])
        self.assertEqual(self.request["cache_contract"]["cache_file_count"], 411)
        self.assertEqual(self.request["cache_contract"]["unchanged_file_count"], 17277)

    def test_formatting_cannot_reset_permanent_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "request.json"
            path.write_bytes(c.canonical(self.request))
            _, reformatted = c.frozen_request(path)
            self.assertNotEqual(reformatted, self.digest)
            self.assertEqual(c.claim_ref(reformatted), c.claim_ref(self.digest))
            self.assertNotEqual(c.claim_ref(reformatted), c.prior.claim_ref(self.digest))
            self.assertNotEqual(c.claim_ref(reformatted), c.host.claim_ref(self.digest))

    def test_scope_head_and_cache_list_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "request.json"
            for key, value in (("max_asr_calls", 17), ("retry", True), ("family", "qwen06"), ("training_admitted", True)):
                path.write_bytes(c.canonical(dict(self.request, **{key: value})))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    c.frozen_request(path)
            changed = copy.deepcopy(self.request)
            changed["cache_contract"]["cache_records"].pop(CACHE)
            path.write_bytes(c.canonical(changed))
            with self.assertRaises(ValueError):
                c.frozen_request(path)

    def test_both_closed_failed_runs_and_live_claims_required(self):
        old, _ = c.prior.frozen_request()
        failed = c.failed_parent_request(self.request)
        values = []
        for request in (old, failed):
            claim = dict(ref=request["parent_claim_ref"], object=dict(type="commit", sha=request["parent_head_sha"]))
            run = dict(id=request["parent_run_id"], run_attempt=1, head_sha=request["parent_head_sha"],
                status="completed", conclusion="failure", event="push", path=".github/workflows/public-zh64-asr-once.yml",
                repository=dict(full_name="jiying2007/kws-pipeline", private=False))
            values.extend((claim, run))
        c.verify_live_parents(*values, self.request)
        for index in (1, 3):
            changed = copy.deepcopy(values)
            changed[index]["status"] = "in_progress"
            with self.assertRaises(ValueError):
                c.verify_live_parents(*changed, self.request)
        changed = copy.deepcopy(values)
        changed[2]["object"]["sha"] = "0" * 40
        with self.assertRaises(ValueError):
            c.verify_live_parents(*changed, self.request)

    def test_public_first_attempt_new_head_and_no_host_token(self):
        head = "a" * 40
        env = dict(GITHUB_REPOSITORY="jiying2007/kws-pipeline", GITHUB_EVENT_NAME="push", GITHUB_REF="refs/heads/main",
            GITHUB_RUN_ATTEMPT="1", GITHUB_ACTIONS="true", RUNNER_ENVIRONMENT="github-hosted", RUNNER_OS="Linux",
            RUNNER_ARCH="X64", GITHUB_SHA=head, GITHUB_RUN_ID="123", GITHUB_WORKFLOW_SHA=head,
            GITHUB_WORKFLOW_REF="jiying2007/kws-pipeline/.github/workflows/public-zh64-asr-once.yml@refs/heads/main")
        event = dict(after=head, ref="refs/heads/main", deleted=False, repository=dict(private=False, full_name="jiying2007/kws-pipeline"))
        claim = dict(ref=c.claim_ref(self.digest), object=dict(sha=head, type="commit"))
        self.assertEqual(c.validate_execution(env, event, claim, self.request, self.digest), (head, "123"))
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_EVENT_NAME", "pull_request"), ("GH_TOKEN", "credential"),
                           ("GITHUB_SHA", self.request["failed_parent"]["head_sha"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_execution(dict(env, **{key: value}), event, claim, self.request, self.digest)

    def test_wrong_second_zip_rejected_before_parse(self):
        with self.assertRaisesRegex(ValueError, "exact second failed"):
            c.failed_archive(b"unrelated-archive", self.request)

    def test_composition_preserves_176_objects_and_only_replaces_absent_sixteen(self):
        parent, fresh = composition_fixture()
        saved = copy.deepcopy(parent)
        combined = c.prior.combine(parent, fresh)
        self.assertEqual(parent, saved)
        self.assertEqual(combined["providers"]["qwen06"], saved["providers"]["qwen06"])
        self.assertEqual(combined["providers"]["sensevoice"], saved["providers"]["sensevoice"])
        self.assertEqual(combined["providers"]["whisper"][:48], saved["providers"]["whisper"][:48])
        self.assertFalse(combined["single_run_three_model_execution"])
        parent["providers"]["whisper"][48]["attempt_consumed"] = True
        with self.assertRaises(ValueError):
            c.prior.combine(parent, fresh)


class InventoryIdentityTests(unittest.TestCase):
    def test_only_declared_cache_hash_may_vary(self):
        old, fresh, contract = inventory_fixture()
        k.compare_inventories(old, fresh, contract)
        for kind in ("add", "remove", "size", "source", "license", "cache-fields"):
            changed = copy.deepcopy(fresh)
            if kind == "add": changed["files"]["pip/new.py"] = {"bytes": 1, "sha256": "a" * 64}
            if kind == "remove": changed["files"].pop(SOURCE)
            if kind == "size": changed["files"][CACHE]["bytes"] += 1
            if kind == "source": changed["files"][SOURCE]["sha256"] = "b" * 64
            if kind == "license": changed["files"]["pip-25.0.1.dist-info/LICENSE.txt"]["sha256"] = "c" * 64
            if kind == "cache-fields": changed["files"][CACHE]["accepted"] = True
            with self.subTest(kind=kind), self.assertRaises((ValueError, KeyError)):
                k.compare_inventories(old, changed, contract)

    def test_all_411_real_declared_paths_including_hyphen_are_reachable(self):
        request, _ = c.frozen_request()
        contract = copy.deepcopy(request["cache_contract"])
        self.assertIn(CACHE, contract["cache_records"])
        files = dict(contract["license_records"])
        for name, value in contract["cache_records"].items():
            files[name], files[value["source_path"]] = value["original_cache"], value["source_identity"]
        stable = {name: value for name, value in files.items() if name not in contract["cache_records"]}
        contract.update(installed_file_count=len(files), unchanged_file_count=len(stable),
                        unchanged_records_sha256=k.sha(k.canonical(stable)))
        old = dict(schema="whisper-screen-installed-runtime-v1", files=files)
        fresh = copy.deepcopy(old)
        for name, value in contract["cache_records"].items():
            fresh["files"][name] = value["failed_prepare_cache"]
        k.compare_inventories(old, fresh, contract)

    def test_allowed_source_identity_propagates_true_head_and_inventory(self):
        old, fresh, old_source, source, old_raw, fresh_raw, contract = native_fixture()
        c.source_identity(old, fresh, old_source, source, old_raw, fresh_raw, "a" * 40, contract)
        for kind in ("model", "decoder", "asset", "source", "head", "unrelated-field"):
            changed, changed_source = copy.deepcopy(fresh), copy.deepcopy(source)
            if kind == "model": changed["model_sha256"] = "0" * 64
            if kind == "decoder": changed["native_decode_config"]["effective_config"]["temperature"] = 0.1
            if kind == "asset": changed["native_decode_config"]["asset_lock_sha256"] = "0" * 64
            if kind == "source": changed_source["files"][1]["sha256"] = "0" * 64
            if kind == "head": changed_source["executed_head_sha"] = old_source["executed_head_sha"]
            if kind == "unrelated-field": changed_source["source_inventory_scope"] = "expanded"
            changed["native_decode_config"]["source_lock_sha256"] = k.sha(k.canonical(changed_source))
            changed["native_decode_config_sha256"] = k.sha(k.canonical(changed["native_decode_config"]))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                c.source_identity(old, changed, old_source, changed_source, old_raw, fresh_raw, "a" * 40, contract)

    def test_runtime_tree_is_closed_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "x.py").write_bytes(b"pass\n")
            files = {"x.py": k.identity(b"pass\n")}
            k.verify_files(root, files)
            (root / "extra.py").write_bytes(b"")
            with self.assertRaises(ValueError):
                k.verify_files(root, files)
            (root / "extra.py").unlink()
            (root / "x.py").write_bytes(b"nope\n")
            with self.assertRaises(ValueError):
                k.verify_files(root, files)


class CachePayloadTests(unittest.TestCase):
    source = b"value = 3\n"
    filename = "/runtime/venv/lib/python3.12/site-packages/pip/example.py"

    def test_timestamp_and_both_legal_hash_headers(self):
        for flags in (0, 1, 3):
            with self.subTest(flags=flags):
                result = k.verify_cache(pyc(self.source, self.filename, flags=flags), self.source, self.filename, 123)
                self.assertTrue(result["recursive_code_equal"])
                self.assertFalse(result["cached_code_executed"])
                self.assertEqual(result["header"]["flags"], flags)

    def test_bad_magic_flags_timestamp_size_or_hash_rejected(self):
        good = pyc(self.source, self.filename)
        for raw in (b"nope" + good[4:], good[:4] + struct.pack("<I", 2) + good[8:],
                    good[:8] + struct.pack("<II", 124, len(self.source)) + good[16:],
                    good[:12] + struct.pack("<I", len(self.source) + 1) + good[16:],
                    good[:4] + struct.pack("<I", 3) + b"bad_hash" + good[16:]):
            with self.subTest(header=raw[:16].hex()), self.assertRaises(ValueError):
                k.verify_cache(raw, self.source, self.filename, 123)

    def test_marshal_trailing_truncated_or_noncode_rejected(self):
        good = pyc(self.source, self.filename)
        for raw in (good + b"extra", good[:-7], good[:16] + marshal.dumps(("not", "code"))):
            with self.assertRaises((ValueError, EOFError, TypeError)):
                k.verify_cache(raw, self.source, self.filename, 123)

    def test_filename_bytecode_constants_and_debug_fields_are_bound(self):
        code = compile(self.source, self.filename, "exec", dont_inherit=True, optimize=0)
        changes = (dict(co_filename="/wrong.py"), dict(co_consts=(4, None)), dict(co_names=("different",)),
                   dict(co_firstlineno=2), dict(co_linetable=b""), dict(co_exceptiontable=b"x"), dict(co_qualname="wrong"))
        for change in changes:
            with self.subTest(field=next(iter(change))), self.assertRaises(ValueError):
                k.verify_cache(pyc(self.source, self.filename, code=code.replace(**change)), self.source, self.filename, 123)

    def test_nested_code_fields_are_compared(self):
        source = b"def inner():\n    return 'fixed'\n"
        code = compile(source, self.filename, "exec", dont_inherit=True, optimize=0)
        changed = code.replace(co_consts=tuple(item.replace(co_name="changed") if type(item) is types.CodeType else item for item in code.co_consts))
        with self.assertRaises(ValueError):
            k.verify_cache(pyc(source, self.filename, code=changed), source, self.filename, 123)

    def test_surrogate_emoji_and_unequal_strings_are_lossless(self):
        source = b"values = ('\\ud800', '\\U0001f642', '\\udfff')\n"
        k.verify_cache(pyc(source, self.filename), source, self.filename, 123)
        for value in ("\ud800", "\udfff", "🙂", "plain", "\ud800🙂"):
            tree = k.code_tree(value)
            self.assertEqual(tree[0], "str-utf8-surrogatepass-hex")
            self.assertEqual(bytes.fromhex(tree[1]).decode("utf-8", "surrogatepass"), value)
            k.canonical(tree)
        self.assertNotEqual(k.code_tree("\ud800"), k.code_tree("\ud801"))
        self.assertNotEqual(k.code_tree("🙂"), k.code_tree("🙃"))
        self.assertNotEqual(k.code_tree("\ud83d\ude42"), k.code_tree("🙂"))

    def test_typed_constants_preserve_float_bits_and_frozenset_values(self):
        self.assertNotEqual(k.code_tree(True), k.code_tree(1))
        self.assertNotEqual(k.code_tree(0.0), k.code_tree(-0.0))
        self.assertNotEqual(k.code_tree(complex(0.0, 0.0)), k.code_tree(complex(-0.0, 0.0)))
        self.assertEqual(k.code_tree(frozenset(("a", 1))), k.code_tree(frozenset((1, "a"))))
        with self.assertRaises(ValueError):
            k.code_tree({"unsupported": 1})

    def test_compile_does_not_execute_payload(self):
        source = b"raise RuntimeError('this code must never execute')\n"
        result = k.verify_cache(pyc(source, self.filename), source, self.filename, 123)
        self.assertTrue(result["recursive_code_equal"])
        self.assertFalse(result["cached_code_executed"])

    def test_rejected_cache_preserves_raw_source_actual_stat_and_current_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site, output = root / "site", root / "output"
            (site / CACHE).parent.mkdir(parents=True)
            (site / SOURCE).write_bytes(self.source)
            raw = b"BAD!" + pyc(self.source, str(site / SOURCE))[4:]
            (site / CACHE).write_bytes(raw)
            output.mkdir()
            k.put(output / "read-observations.jsonl", b"")
            declared = dict(source_path=SOURCE, source_identity=k.identity(self.source), original_cache=k.identity(raw))
            state = {}
            with mock.patch.object(k, "SITE", site), self.assertRaisesRegex(ValueError, "actual pinned interpreter"):
                k.verify_observed_cache(output, CACHE, declared, {"files": {CACHE: k.identity(raw)}}, state)
            self.assertEqual((output / "cache" / CACHE).read_bytes(), raw)
            self.assertEqual((output / "source" / SOURCE).read_bytes(), self.source)
            observed = json.loads((output / "read-observations.jsonl").read_bytes())
            self.assertEqual(observed["source_stat_before"]["st_mtime_ns"], (site / SOURCE).stat().st_mtime_ns)
            self.assertEqual(state["cache_path"], CACHE)
            self.assertEqual(state["phase"], "cache-header-and-compiled-code")
            self.assertFalse((output / "cache-verification.json").exists())

    def test_failure_wrapper_preserves_exception_chain_without_verified_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            def fail(out, state):
                state.update(phase="cache-header-and-compiled-code", cache_path=CACHE, source_path=SOURCE)
                try:
                    raise ValueError("inner")
                except ValueError as error:
                    raise RuntimeError("outer \ud800") from error
            with mock.patch.object(k, "Path", return_value=output), mock.patch.object(k, "check", side_effect=fail):
                self.assertEqual(k.main(), 2)
            failure = json.loads((output / "failure.json").read_bytes())
            self.assertEqual([row["error_type"] for row in failure["exceptions"]], ["RuntimeError", "ValueError"])
            self.assertFalse(failure["verified"])
            self.assertFalse((output / "cache-verification.json").exists())


class SourceBoundaryTests(unittest.TestCase):
    def test_no_assert_no_model_import_and_checker_does_not_use_old_shim(self):
        for name in ("continuation_v2.py", "cache_verify_v2.py"):
            tree = ast.parse((HERE / name).read_bytes())
            self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
        self.assertFalse({"pip", "torch", "numpy", "scipy", "funasr", "qwen_asr", "whisper"} & set(sys.modules))
        source = (HERE / "continuation_v2.py").read_text()
        self.assertIn('"-cache-check", ["/runtime/venv/bin/python", "-I", "-B",', source)
        self.assertIn('command = host.shim(["/runtime/venv/bin/python"', source)
        self.assertIn('worker_command("whisper", "infer", head', source)
        self.assertNotIn("setup_adapter.py", source)
        self.assertNotIn("--derive", source)

    def test_new_actual_job_has_no_write_permission_and_parents_precede_setup(self):
        workflow = (HERE.parents[1] / ".github/workflows/public-zh64-asr-once.yml").read_text()
        actual = workflow.split("  actual-first-whisper16-v2:", 1)[1].split("\n  actual-first-whisper16:", 1)[0]
        self.assertIn("contents: read", actual)
        self.assertNotIn("contents: write", actual)
        self.assertIn("--verify-parents", actual)
        self.assertIn("--failed-artifact", actual)
        self.assertIn("test -z \"${GH_TOKEN-}\"", actual)
        self.assertIn("persist-credentials: false", actual)
        self.assertIn("public-zh64-first-whisper16-v2-${{ github.run_id }}", actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
