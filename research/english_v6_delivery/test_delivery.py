"""Invented byte-transport fixtures only: no compiler, model, PCM or network.

Temporary trusted selections are patched within tests. No fixture manifest or
pack is a real quality permission, and the committed selection stays inert.
"""
from contextlib import ExitStack
import copy
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location("delivery_contract", HERE / "delivery.py")
d = importlib.util.module_from_spec(definition); definition.loader.exec_module(d)


def byte_pin(raw, mode=None):
    value = dict(bytes=len(raw), sha256=d.sha(raw))
    if mode is not None: value["mode"] = mode
    return value


def archive_bytes(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            item = zipfile.ZipInfo(name); item.create_system = 3
            item.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(item, raw)
    return stream.getvalue()


def invented_pack(first):
    values = [0.] * 43876 + [.5] * (257 * 40) + [0.] * 40 + [1.] * 40
    values[0] = first
    return struct.pack("<8sII", b"MERO43V1", 1, 54236) + struct.pack("<54236f", *values)


def artifact(number, raw=None, run=500):
    return dict(id=number, run_id=run, run_head="%040x" % run,
                **(byte_pin(raw) if raw is not None else dict(bytes=100, sha256="%064x" % number)))


class DeliveryTests(unittest.TestCase):
    def fixture(self, stack):
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        origins = root / "originals"; origins.mkdir()
        baseline = invented_pack(.125); candidate = invented_pack(.25)
        stack.enter_context(patch.object(d, "B0_SHA", d.sha(baseline)))
        stack.enter_context(patch.object(d, "HERE", root))
        stack.enter_context(patch.object(d, "SELECTION", root / "selection.json"))
        payload = {"models/invented.pack": candidate,
                   **{"models/invented." + ext: ("invented/" + ext).encode() for ext in ("pt", "npz", "onnx")},
                   "LICENSE": b"invented license", "README.md": b"invented software-only handoff"}
        for backend in d.BACKENDS:
            for name in d.REQUIRED_SDK | {"lib/cmake/HeyMero43/HeyMero43Targets-release.cmake"}:
                payload["sdk-" + backend + "/" + name] = ("invented SDK " + backend + "/" + name).encode()
        locks = {}
        for offset, panel in enumerate(("primary", "replication"), 1):
            locks[panel] = dict(schema="english-v6-b999-original-quality-lock-v1", panel=panel,
                **{role: artifact(10 * offset + i, run=offset) for i, role in enumerate(("result", "producer", "consumed"), 1)},
                **{key: "%064x" % (100 + offset) for key in
                   ("evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256")})
        result = dict(status="PASS_B999_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE", pipeline_commit=d.PIPELINE_COMMIT,
            pack_bytes=d.PACK_BYTES, pack_sha256=d.sha(candidate), entered_backend_calls=140,
            saved_backend_outputs=140, consumed_unknown=[], physical_ssc305=False, product_promotion_allowed=False,
            identity=dict(run_id="500", run_head="%040x" % 500, run_attempt=1),
            quality={panel: dict(panel_gate="PASS", original_result_sha256=lock["original_result_sha256"]) for panel, lock in locks.items()})
        original = dict(payload, **{"RESULT.json": d.encoded(result), "logs/empty-consumer-log.txt": b""})
        original["MANIFEST.json"] = d.encoded(dict(status=result["status"], files={k: byte_pin(v) for k, v in original.items()}))
        archives = {"result": archive_bytes(original), "producer": archive_bytes({"PRODUCER.json": b"{}"}),
                    "consumed": archive_bytes({"CONSUMPTION.json": b"{}"}),
                    "input_proof": archive_bytes({"FETCH.json": b"{}", "CONSUMPTION.json": b"{}"})}
        for role, raw in archives.items(): (origins / (role + ".zip")).write_bytes(raw)
        report, verifier = b"invented original-readback fixture; no permission", b"# invented verifier; never executed\n"
        (origins / "completion.json").write_bytes(report); (origins / "readback.py").write_bytes(verifier)
        (root / "baseline.pack").write_bytes(baseline)
        manifest = dict(schema="heym43-software-delivery-v1", candidate_id="invented-fixture", pipeline_commit=d.PIPELINE_COMMIT,
            candidate_pack="models/invented.pack", files={name: byte_pin(raw, d.prescribed_mode(name)) for name, raw in payload.items()},
            engineering=dict(repository="jiying2007/kws-data", artifacts={role: artifact(100 + i, raw) for i, (role, raw) in enumerate(archives.items())},
                result_sha256=d.sha(original["RESULT.json"]), manifest_sha256=d.sha(original["MANIFEST.json"])),
            quality_locks=locks, readback=dict(report=byte_pin(report), verifier=byte_pin(verifier)), scope=copy.deepcopy(d.SCOPE))
        manifest["files"][d.B0_PATH] = byte_pin(baseline, 0o644)
        self.bind_fixture(root, manifest)
        return root, origins, manifest

    def bind_fixture(self, root, manifest):
        raw = d.encoded(manifest); digest = d.sha(raw)
        target = root / "manifests" / (digest + ".json"); target.parent.mkdir(exist_ok=True); target.write_bytes(raw)
        selected = dict(schema="heym43-software-selection-v1", selected=manifest["candidate_id"],
            manifest=dict(path="manifests/" + digest + ".json", **byte_pin(raw)),
            baseline=dict(path=d.B0_PATH, bytes=d.PACK_BYTES, sha256=d.B0_SHA), scope=copy.deepcopy(d.SCOPE))
        (root / "selection.json").write_bytes(d.encoded(selected))

    def test_committed_selection_is_inert_or_reviewed_and_preserves_actual_b0(self):
        chosen = d.selection()
        if chosen["selected"] is None:
            self.assertIsNone(chosen["manifest"])
        else:
            value, raw = d.trusted_manifest()
            self.assertEqual(value["candidate_id"], chosen["selected"])
            self.assertEqual(d.sha(raw), chosen["manifest"]["sha256"])
        self.assertEqual(d.B0_SHA, "ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b")
        self.assertEqual(d.PIPELINE_COMMIT, "88a043c6c00bb3a1a62f45098a023c7601042ded")
        self.assertEqual(chosen["baseline"], dict(path=d.B0_PATH, bytes=d.PACK_BYTES, sha256=d.B0_SHA))
        with patch.object(d, "selection", return_value={**chosen, "selected": None, "manifest": None}), \
             patch.object(d, "original_archive", side_effect=AssertionError("no originals opened")), \
             patch.object(d.os, "execv", side_effect=AssertionError("no runtime")), \
             patch.object(d.subprocess, "run", side_effect=AssertionError("no checker")):
            for call in (lambda: d.package("/unused", "/unused", "/unused"),
                         lambda: d.install("/unused", "/unused"),
                         lambda: d.launch("/unused", "candidate", "gcc", 10000, 0),
                         lambda: d.check_load("/unused", "candidate", "gcc", "/unused")):
                with self.assertRaisesRegex(ValueError, "INERT"): call()

    def test_proof_strings_cannot_replace_reviewed_manifest_and_original_pins(self):
        with ExitStack() as stack:
            root, origins, manifest = self.fixture(stack)
            selected = d.selection(); path = root / selected["manifest"]["path"]
            changed = copy.deepcopy(manifest); changed["readback"]["report"]["sha256"] = "f" * 64
            path.write_bytes(d.encoded(changed))
            with self.assertRaisesRegex(ValueError, "SHA mismatch"): d.trusted_manifest()
            path.write_bytes(d.encoded(manifest))
            (origins / "completion.json").write_bytes(b'{"panel_gate":"PASS","engineering":"PASS"}')
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                d.package(origins, root / "baseline.pack", root / "not-created.tar.gz")
            self.assertFalse((root / "not-created.tar.gz").exists())

    def test_quality_summaries_and_product_claims_are_not_valid_manifests(self):
        with ExitStack() as stack:
            root, _, manifest = self.fixture(stack)
            for change in (lambda m: m.update(quality_locks={"primary": {"panel_gate": "PASS"}, "replication": {"panel_gate": "PASS"}}),
                           lambda m: m["scope"].update(product_promotion_allowed=True),
                           lambda m: m["engineering"]["artifacts"]["producer"].update(run_head="a" * 40),
                           lambda m: m["files"]["sdk-gcc/bin/heym43-live"].update(mode=0o644)):
                bad = copy.deepcopy(manifest); change(bad)
                with self.assertRaises((ValueError, KeyError)): d.manifest_contract(bad)

    def test_whole_original_artifact_identity_is_required_before_packaging(self):
        with ExitStack() as stack:
            root, origins, _ = self.fixture(stack)
            path = origins / "producer.zip"; path.write_bytes(path.read_bytes() + b"changed origin")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                d.package(origins, root / "baseline.pack", root / "not-created.tar.gz")
            self.assertFalse((root / "not-created.tar.gz").exists())

    def test_complete_original_manifest_inventory_cannot_hide_an_extra_member(self):
        with ExitStack() as stack:
            root, origins, manifest = self.fixture(stack)
            path = origins / "result.zip"
            with zipfile.ZipFile(path) as archive: contents = {n: archive.read(n) for n in archive.namelist()}
            contents["unlisted.txt"] = b"not in original MANIFEST"
            raw = archive_bytes(contents); path.write_bytes(raw)
            manifest["engineering"]["artifacts"]["result"].update(byte_pin(raw)); self.bind_fixture(root, manifest)
            with self.assertRaisesRegex(ValueError, "entire original artifact inventory"):
                d.prepare_payload(origins, root / "baseline.pack")

    def test_reproducible_package_restores_reviewed_modes_and_omits_raw_proof(self):
        with ExitStack() as stack:
            root, origins, manifest = self.fixture(stack)
            first, second = root / "one.tar.gz", root / "two.tar.gz"
            a = d.package(origins, root / "baseline.pack", first)
            b = d.package(origins, root / "baseline.pack", second)
            self.assertEqual(a["sha256"], b["sha256"])
            with tarfile.open(first, "r:gz") as archive:
                self.assertEqual(set(archive.getnames()), set(manifest["files"]) | {"DELIVERY.json"})
                self.assertNotIn("completion.json", archive.getnames()); self.assertNotIn("readback.py", archive.getnames())
                for item in archive.getmembers():
                    self.assertEqual(item.mode, 0o644 if item.name == "DELIVERY.json" else d.prescribed_mode(item.name))

    def test_install_is_immutable_idempotent_and_verifies_post_install_modes(self):
        with ExitStack() as stack:
            root, origins, manifest = self.fixture(stack)
            archive, prefix = root / "bundle.tar.gz", root / "installed"
            d.package(origins, root / "baseline.pack", archive)
            result = d.install(archive, prefix)
            self.assertFalse(result["selected_process_changed"])
            self.assertEqual(d.install(archive, prefix)["status"], "ALREADY_INSTALLED")
            _, installed = d.verify_installed(prefix)
            for name, pin in manifest["files"].items():
                self.assertEqual(stat.S_IMODE((installed / name).stat().st_mode), pin["mode"])
            binary = installed / "sdk-gcc/bin/heym43-live"; binary.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "mode mismatch"): d.verify_installed(prefix)
            with patch.object(d.os, "execv", side_effect=AssertionError("must not start damaged SDK")):
                with self.assertRaises(ValueError): d.launch(prefix, "candidate", "gcc", 10000, 0)

    def test_launch_uses_exact_candidate_or_b0_path_and_caller_policy(self):
        with ExitStack() as stack:
            root, origins, manifest = self.fixture(stack)
            archive, prefix = root / "bundle.tar.gz", root / "installed"
            d.package(origins, root / "baseline.pack", archive); d.install(archive, prefix)
            _, installed = d.verify_installed(prefix)
            for model, member in (("candidate", manifest["candidate_pack"]), ("B0", d.B0_PATH)):
                with patch.object(d.os, "execv") as execute:
                    d.launch(prefix, model, "gcc", 54321, 123)
                    binary = str(installed / "sdk-gcc/bin/heym43-live")
                    execute.assert_called_once_with(binary, [binary, str(installed / member), "54321", "123"])
            for gap, cooldown in ((0, 0), (-1, 0), (1, -1), (2**64, 0)):
                with self.assertRaises(ValueError): d.launch(prefix, "candidate", "gcc", gap, cooldown)

    def test_pack_tampering_and_symlinks_refuse_before_runtime(self):
        with ExitStack() as stack:
            root, origins, manifest = self.fixture(stack)
            archive, prefix = root / "bundle.tar.gz", root / "installed"
            d.package(origins, root / "baseline.pack", archive); d.install(archive, prefix)
            _, installed = d.verify_installed(prefix)
            pack = installed / manifest["candidate_pack"]; body = pack.read_bytes()
            pack.write_bytes(body[:-1] + bytes([body[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "SHA mismatch"): d.verify_installed(prefix)
            pack.unlink(); pack.symlink_to(root / "baseline.pack")
            with self.assertRaisesRegex(ValueError, "symlink"): d.verify_installed(prefix)

    def test_unsafe_tar_members_and_wrong_modes_never_create_install_prefix(self):
        with ExitStack() as stack:
            root, origins, _ = self.fixture(stack)
            _, payload = d.prepare_payload(origins, root / "baseline.pack")
            for kind in ("traversal", "symlink", "hardlink", "duplicate", "mode"):
                target = root / (kind + ".tar.gz")
                with tarfile.open(target, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
                    for i, (name, body) in enumerate(payload.items()):
                        info = tarfile.TarInfo("../outside" if kind == "traversal" and i == 0 else name)
                        info.size = len(body); info.mode = 0o644 if name == "DELIVERY.json" else d.prescribed_mode(name)
                        if i == 0 and kind in ("symlink", "hardlink"):
                            info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                            info.linkname = "elsewhere"; info.size = 0
                        if i == 0 and kind == "mode": info.mode = 0o777
                        archive.addfile(info, io.BytesIO(body))
                        if i == 0 and kind == "duplicate": archive.addfile(info, io.BytesIO(body))
                prefix = root / (kind + "-prefix")
                with self.assertRaises(ValueError): d.install(target, prefix)
                self.assertFalse(prefix.exists())

    def test_zero_forward_loader_is_separate_from_model_execution(self):
        with ExitStack() as stack:
            root, origins, manifest = self.fixture(stack)
            archive, prefix = root / "bundle.tar.gz", root / "installed"
            d.package(origins, root / "baseline.pack", archive); d.install(archive, prefix)
            checker = root / "invented-checker"; checker.write_bytes(b"never executed")
            response = dict(schema="heymero43-delivery-load-check-v1", status="PASS_MODEL_LOAD_ZERO_FORWARD",
                pack_bytes=d.PACK_BYTES, model_storage_bytes=216948, model_load_calls=1,
                model_forwards=0, stream_init_calls=0, pcm_samples_consumed=0)
            with patch.object(d.subprocess, "run") as run:
                run.return_value.stdout = json.dumps(response)
                result = d.check_load(prefix, "candidate", "gcc", checker)
                self.assertEqual(result["model_forwards"], 0)
                self.assertTrue(run.call_args.args[0][1].endswith(manifest["candidate_pack"]))
                response["model_forwards"] = 1; run.return_value.stdout = json.dumps(response)
                with self.assertRaises(ValueError): d.check_load(prefix, "candidate", "gcc", checker)

    def test_c_consumer_only_uses_installed_model_loader(self):
        source = (HERE / "loader_check.c").read_text()
        code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
        calls = re.findall(r"\b(en_v5_m43_[a-z_]+)\s*\(", code)
        self.assertEqual(calls, ["en_v5_m43_model_bytes", "en_v5_m43_model_load"])
        self.assertIn("extra != EOF", code); self.assertIn("model_bytes != sizeof(model)", code)
        cmake = (HERE / "CMakeLists.txt").read_text()
        self.assertIn("NO_DEFAULT_PATH", cmake); self.assertIn("HeyMero43::stream IMPORTED", cmake)
        self.assertNotIn("add_subdirectory", cmake); self.assertNotIn("mero43_stream.c", cmake)


if __name__ == "__main__":
    unittest.main()
