"""One-shot delivery contracts use invented bytes and mocked process boundaries.

These tests do not download archives, compile a consumer, load weights, run a
verifier, or touch real audio. The committed software selection stays null.
"""
import ast
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location("delivery_once_contract", HERE / "once.py")
m = importlib.util.module_from_spec(definition); definition.loader.exec_module(m)
d = m.d


def npy(raw, shape=(2,), descr="<f4", fortran=False):
    header = (repr(dict(descr=descr, fortran_order=fortran, shape=shape)) + "\n").encode()
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + raw


def closed():
    return dict(schema=m.ENTRY_SCHEMA, enabled=False, pipeline_repository=m.PIPELINE_REPO,
                pipeline_commit=None, manifest_sha256=None, pipeline_files=None)


class OnceTests(unittest.TestCase):
    def test_null_selection_stops_before_caller_paths_network_or_compiler(self):
        d.selection()  # A later real reviewed selection must not disable closed-gate regression tests.
        with patch.object(d, "selection", return_value=dict(selected=None, manifest=None)), \
             patch.object(m, "reviewed_readback", side_effect=AssertionError("no report path")), \
             patch.object(m, "api", side_effect=AssertionError("no network")), \
             patch.object(m, "git", side_effect=AssertionError("no Git process")), \
             patch.object(m, "baseline_container", side_effect=AssertionError("no container")), \
             patch.object(m, "load_checks", side_effect=AssertionError("no compiler or loader")):
            with self.assertRaisesRegex(ValueError, "INERT"):
                m.execute(Path("/not-a-real-path"), "0" * 40, Path("/not-an-output"))

    def test_inert_binding_cannot_carry_unreviewed_authority(self):
        self.assertEqual(m.entry_contract(closed()), closed())
        for key in ("pipeline_commit", "manifest_sha256", "pipeline_files"):
            value = closed(); value[key] = "invented"
            with self.assertRaises(ValueError): m.entry_contract(value)
        value = closed(); value["enabled"] = 0
        with self.assertRaises(ValueError): m.entry_contract(value)
        value = closed(); value["enabled"] = True
        with self.assertRaises(ValueError): m.entry_contract(value)

    def test_complete_producer_requires_exact_set_and_manifest(self):
        binding = closed(); binding.update(enabled=True, pipeline_commit="a" * 40,
            manifest_sha256="b" * 64, pipeline_files={"unrelated.py": dict(bytes=1, sha256="c" * 64,
            git_blob_sha1="d" * 40, mode="100644")})
        m.entry_contract(binding)
        with patch.object(m, "git", side_effect=AssertionError("stop before Git")):
            with self.assertRaisesRegex(ValueError, "INERT"):
                m.pipeline_contract(binding, {}, b"not the reviewed manifest")

    def test_every_selected_pipeline_file_and_git_blob_is_authenticated(self):
        packer = (m.ROOT / m.PACKER).read_bytes()
        with tempfile.TemporaryDirectory() as temp, patch.object(m, "ROOT", Path(temp)), \
             patch.object(m, "HERE", Path(temp) / "research/english_v6_delivery"):
            value = dict(readback=dict(report=dict(sha256="1" * 64), verifier=dict(sha256="2" * 64)))
            raw_manifest = b"invented selected manifest bytes"
            names = m.expected_pipeline_files(value, raw_manifest)
            self.assertEqual(len(names), 9)
            files, rows = {}, []
            for name in sorted(names):
                raw = packer if name == m.PACKER else ("invented " + name).encode()
                path = m.ROOT / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
                blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                files[name] = dict(**m.pin(raw), git_blob_sha1=blob, mode="100644")
                rows.append(b"100644 blob " + blob.encode() + b"\t" + name.encode() + b"\0")
            binding = dict(schema=m.ENTRY_SCHEMA, enabled=True, pipeline_repository=m.PIPELINE_REPO,
                pipeline_commit="a" * 40, manifest_sha256=d.sha(raw_manifest), pipeline_files=files)
            tree = b"".join(rows)
            def git_output(_, *args):
                if args[0] == "rev-parse": return b"a" * 40 + b"\n"
                if args[0] == "status": return b""
                if args[0] == "ls-tree": return tree
                raise AssertionError("no other process")
            with patch.object(m, "git", side_effect=git_output):
                m.pipeline_contract(binding, value, raw_manifest)
                (m.ROOT / "research/english_v6_delivery/once.py").write_bytes(b"changed code")
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    m.pipeline_contract(binding, value, raw_manifest)
            incomplete = copy.deepcopy(binding); del incomplete["pipeline_files"][m.PACKER]
            with patch.object(m, "git", side_effect=AssertionError("reject missing producer before process")):
                with self.assertRaisesRegex(ValueError, "complete exact"):
                    m.pipeline_contract(incomplete, value, raw_manifest)

    def report_fixture(self, root):
        source = b"# exact saved verifier; executing it is prohibited\n"
        value = dict(candidate_id="B24R1", candidate_pack="models/b24r1-challenger.pack",
            pipeline_commit=d.PIPELINE_COMMIT, files={"invented": {}}, engineering={"invented": 1},
            quality_locks={"primary": {"invented": 2}, "replication": {"invented": 3}},
            readback=dict(verifier=m.pin(source)), scope=copy.deepcopy(d.SCOPE))
        report = dict(schema="english-v6-b24r1-verified-original-engineering-v1",
            status="VERIFIED_ORIGINAL_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE",
            baseline_pack_requires_separate_byte_authentication=True, readback_verifier=m.pin(source),
            **{key: value[key] for key in ("candidate_id", "candidate_pack", "pipeline_commit", "files", "engineering", "quality_locks", "scope")})
        raw = d.encoded(report); value["readback"]["report"] = m.pin(raw)
        self.save_readback(root, value, raw, source)
        return value, report, source

    def save_readback(self, root, value, raw, source):
        directory = root / "readbacks" / value["readback"]["report"]["sha256"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "completion.json").write_bytes(raw)
        (root / "readbacks" / (value["readback"]["verifier"]["sha256"] + ".py")).write_bytes(source)

    def test_exact_report_and_verifier_are_git_bytes_never_executed(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(m, "HERE", Path(temp)):
            value, _, source = self.report_fixture(Path(temp))
            with patch.object(m.subprocess, "run", side_effect=AssertionError("no verifier process")), \
                 patch.object(m.importlib.util, "spec_from_file_location", side_effect=AssertionError("no verifier import")):
                saved = m.reviewed_readback(value)
            self.assertEqual(saved["readback.py"], source)
            path = m.readback_paths(value)["completion.json"]; path.write_bytes(b'{"status":"PASS"}')
            with self.assertRaisesRegex(ValueError, "SHA mismatch"): m.reviewed_readback(value)

    def test_report_cannot_authorize_different_engineering_models_or_quality_locks(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(m, "HERE", Path(temp)):
            value, report, source = self.report_fixture(Path(temp))
            for key in ("candidate_pack", "pipeline_commit", "files", "engineering", "quality_locks", "scope"):
                changed = copy.deepcopy(value); changed[key] = "unrelated"
                with self.assertRaises(ValueError): m.reviewed_readback(changed)
            changed = copy.deepcopy(report); changed["baseline_pack_requires_separate_byte_authentication"] = False
            raw = d.encoded(changed); value["readback"]["report"] = m.pin(raw)
            self.save_readback(Path(temp), value, raw, source)
            with self.assertRaises(ValueError): m.reviewed_readback(value)

    def execution(self):
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=m.DATA_REPO, GITHUB_EVENT_NAME="push",
            GITHUB_REF=m.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_ID="100", GITHUB_SHA="a" * 40)
        event = dict(repository=dict(full_name=m.DATA_REPO, private=False), before="b" * 40,
                     after="a" * 40, ref=m.BRANCH)
        return env, event

    def test_wrong_event_fork_private_or_retry_stops_before_git(self):
        env, event = self.execution()
        cases = []
        for key, value in (("GITHUB_EVENT_NAME", "workflow_dispatch"), ("GITHUB_RUN_ATTEMPT", "2"),
                           ("GITHUB_REPOSITORY", "other/public"), ("GITHUB_REF", "refs/heads/main")):
            changed = dict(env); changed[key] = value; cases.append((changed, event))
        for change in (lambda x: x["repository"].update(private=True), lambda x: x.update(before="c" * 40)):
            changed = copy.deepcopy(event); change(changed); cases.append((env, changed))
        with patch.object(m, "git", side_effect=AssertionError("guard before Git")):
            for environment, payload in cases:
                with self.assertRaises(ValueError): m.guard("b" * 40, Path("/unused"), environment, payload)

    def test_only_exact_two_substitution_workflow_child_is_accepted(self):
        env, event = self.execution()
        template = m.TOKEN + b"\n" + m.TOKEN + b"\n"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / m.TEMPLATE).parent.mkdir(); (root / m.WORKFLOW).parent.mkdir(parents=True)
            (root / m.TEMPLATE).write_bytes(template)
            (root / m.WORKFLOW).write_bytes(template.replace(m.TOKEN, b"b" * 40))
            outputs = [b"a" * 40 + b"\n", b"a" * 40 + b" " + b"b" * 40 + b"\n",
                       m.WORKFLOW.encode() + b"\n", b""]
            with patch.object(m, "git", side_effect=outputs):
                self.assertEqual(m.guard("b" * 40, root, env, event)["run_attempt"], 1)
            (root / m.WORKFLOW).write_bytes(template.replace(m.TOKEN, b"b" * 40) + b"changed")
            with patch.object(m, "git", side_effect=outputs):
                with self.assertRaisesRegex(ValueError, "workflow-only"): m.guard("b" * 40, root, env, event)

    def run_metadata(self, pin_value):
        return dict(id=pin_value["run_id"], head_sha=pin_value["run_head"], status="completed", conclusion="success", run_attempt=1,
            repository=dict(full_name=m.DATA_REPO, private=False), head_repository=dict(full_name=m.DATA_REPO, private=False),
            event="push", path=".github/workflows/english-v6-b24r1-engineering-once.yml",
            head_branch="research/en-v6-b24r1-engineering-once-20261124")

    def test_metadata_rejects_wrong_role_failed_run_expired_or_mutated_original(self):
        p = dict(id=55, run_id=66, run_head="a" * 40, bytes=77, sha256="b" * 64)
        run = self.run_metadata(p); commit = dict(sha=p["run_head"])
        artifact = dict(id=p["id"], expired=False, size_in_bytes=p["bytes"], digest="sha256:" + p["sha256"],
            workflow_run=dict(id=p["run_id"], head_sha=p["run_head"]), name=m.artifact_name("result", p["run_id"]))
        m.provenance(p, run, artifact, commit, "result")
        for change in (lambda r, a: r.update(conclusion="failure"), lambda r, a: r.update(run_attempt=2),
                       lambda r, a: a.update(expired=True), lambda r, a: a.update(name="lookalike"),
                       lambda r, a: a.update(digest="sha256:" + "c" * 64)):
            r, a = copy.deepcopy(run), copy.deepcopy(artifact); change(r, a)
            with self.assertRaises(ValueError): m.provenance(p, r, a, commit, "result")

    def test_duplicate_original_delivery_run_is_not_a_retry_route(self):
        identity = dict(run_id=100, run_head="a" * 40)
        run = dict(id=100, head_sha="a" * 40, run_attempt=1, event="push", path=m.WORKFLOW,
            head_branch=m.BRANCH.removeprefix("refs/heads/"), repository=dict(full_name=m.DATA_REPO, private=False),
            head_repository=dict(full_name=m.DATA_REPO, private=False))
        with patch.object(m, "api", side_effect=[d.encoded(run), d.encoded(dict(total_count=2, workflow_runs=[dict(id=99), dict(id=100)]))]):
            with self.assertRaisesRegex(ValueError, "never replay"): m.one_execution(identity)

    def test_original_zip_transport_preserves_binary_and_control_bytes(self):
        raw = b"PK\x03\x04\x00\xff\x1b[31m\x80\x07\r\noriginal binary bytes"
        endpoint = "repos/" + m.DATA_REPO + "/actions/artifacts/55/zip"
        with patch.object(m.subprocess, "check_output", return_value=raw) as request:
            self.assertEqual(m.api(endpoint), raw)
        command = request.call_args.args[0]
        self.assertIn("--allow-escape-sequences", command)
        self.assertEqual(command[command.index("--method") + 1], "GET")
        self.assertEqual(command[-1], endpoint)
        self.assertNotIn("shell", request.call_args.kwargs)

    def test_baseline_npy_preserves_raw_bits_and_rejects_transposition_conversion_or_truncation(self):
        raw = bytes.fromhex("0000803f00000080")  # Includes negative zero; no float round trip.
        self.assertEqual(m.npy_payload(npy(raw), (2,)), raw)
        for bad in (npy(raw, descr=">f4"), npy(raw, fortran=True), npy(raw, shape=(1, 2)), npy(raw[:-1]),
                    npy(raw) + b"x", b"not NPY"):
            with self.assertRaises(ValueError): m.npy_payload(bad, (2,))

    def test_baseline_rejects_changed_archive_before_extracting_or_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invented.zip"; path.write_bytes(b"not the original")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"): m.baseline_container(path)
            self.assertEqual(list(Path(temp).iterdir()), [path])
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_loader_path_builds_only_clean_consumer_and_loads_candidate_then_b0(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp); receipts = output / "receipts"; receipts.mkdir()
            installed = output / "installed/version"; installed.mkdir(parents=True)
            build = output / "loader-build"
            calls = []
            def fake_command(args, log, env=None):
                calls.append(args)
                if "--build" in args:
                    binary = build / "heym43_delivery_load_check"; binary.write_bytes(b"\x7fELF invented"); binary.chmod(0o755)
                    (build / "compile_commands.json").write_text(json.dumps([dict(file=str(HERE / "loader_check.c"), directory=str(build))]))
                else: build.mkdir()
            with patch.object(d, "verify_installed", return_value=({}, installed)), \
                 patch.object(m, "command", side_effect=fake_command), \
                 patch.object(d, "check_load", side_effect=lambda _, model, backend, checker: dict(model=model, model_forwards=0)) as load:
                result = m.load_checks(output / "prefix", output, receipts)
            self.assertEqual([call.args[1] for call in load.call_args_list], ["candidate", "B0"])
            self.assertEqual(result["model_forwards"], 0)
            self.assertEqual(result["model_load_calls"], 2)
            self.assertIn("-DHeyMero43_DIR=" + str(installed / "sdk-gcc/lib/cmake/HeyMero43"), calls[0])
            self.assertEqual(calls[0][calls[0].index("-S") + 1], HERE)
            self.assertFalse(json.loads((receipts / "loader_build.json").read_bytes())["sdk_rebuilt"])

    def test_no_scientific_runtime_or_candidate_repacker_in_delivery_entry(self):
        tree = ast.parse((HERE / "once.py").read_bytes())
        imports = {a.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
        self.assertFalse(imports & {"torch", "numpy", "onnxruntime", "piper", "soundfile", "transformers"})
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse(any(node.func.attr in ("launch", "execv", "forward")
            or (node.func.attr == "pack" and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "struct"))
            for node in calls))
        headers = [node for node in calls if node.func.attr == "pack"]
        self.assertEqual(len(headers), 1)
        self.assertEqual(ast.literal_eval(headers[0].args[0]), "<8sII")
        self.assertEqual(m.PACKER_SHA, d.sha((m.ROOT / m.PACKER).read_bytes()))


if __name__ == "__main__": unittest.main()
