#!/usr/bin/env python3
"""SDK SPDX profile and release gate regression, using synthetic SDK bytes."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_sbom import document_namespace, validate  # noqa: E402


class SbomTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = pathlib.Path(self.temp.name)
        self.sdk = self.directory / "sdk"
        (self.sdk / "lib").mkdir(parents=True)
        (self.sdk / "include").mkdir()
        # Deliberately include duplicate content: verification code retains both.
        (self.sdk / "lib/core.a").write_bytes(b"synthetic archive\0\xff")
        (self.sdk / "include/api.h").write_bytes(b"public API\n")
        (self.sdk / "include/copy.h").write_bytes(b"public API\n")
        self.output = self.directory / "sdk.spdx.json"
        self.generate()
        self.document = json.loads(self.output.read_text())

    def generate(self, root=None, output=None, **changes):
        arguments = dict(root=root or self.sdk, output=output or self.output,
                         name="kws-pipeline", version="0.3.0", source_sha="a" * 40)
        arguments.update(changes)
        command = [sys.executable, str(ROOT / "tools/generate_sbom.py")]
        for key, value in arguments.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        return subprocess.run(command, capture_output=True, text=True, check=True)

    def rejected(self, document, message):
        # Recompute namespace so corruption must be caught by actual semantic
        # rules, not solely by the whole-document digest.
        document["documentNamespace"] = document_namespace(document)
        with self.assertRaisesRegex(ValueError, message):
            validate(document, self.sdk)

    def test_valid_checksums_and_verification_code(self):
        validate(self.document, self.sdk)
        sha1s = []
        for file in self.document["files"]:
            content = (self.sdk / file["fileName"]).read_bytes()
            expected = {"SHA1": hashlib.sha1(content, usedforsecurity=False).hexdigest(),
                        "SHA256": hashlib.sha256(content).hexdigest()}
            self.assertEqual({c["algorithm"]: c["checksumValue"] for c in file["checksums"]}, expected)
            sha1s.append(expected["SHA1"])
        expected = hashlib.sha1("".join(sorted(sha1s)).encode("ascii"), usedforsecurity=False).hexdigest()
        self.assertEqual(self.document["packages"][0]["packageVerificationCode"],
                         {"packageVerificationCodeValue": expected})

    def test_reproducible_across_relocation_and_mtime(self):
        import shutil
        original = self.output.read_bytes()
        self.generate()
        self.assertEqual(original, self.output.read_bytes())
        relocated = self.directory / "relocated"
        shutil.copytree(self.sdk, relocated)
        for path in relocated.rglob("*"):
            os.utime(path, (123, 123))
        self.generate(root=relocated)
        self.assertEqual(original, self.output.read_bytes())

    def test_namespace_identifies_content_paths_and_metadata(self):
        previous = self.document["documentNamespace"]
        (self.sdk / "lib/core.a").write_bytes(b"different compiler/architecture bytes")
        self.generate()
        changed = json.loads(self.output.read_text())
        self.assertNotEqual(previous, changed["documentNamespace"])
        validate(changed, self.sdk)
        previous = changed["documentNamespace"]
        (self.sdk / "include/api.h").rename(self.sdk / "include/renamed.h")
        self.generate()
        self.assertNotEqual(previous, json.loads(self.output.read_text())["documentNamespace"])
        previous = self.output.read_bytes()
        self.generate(source_sha="b" * 64)
        self.assertNotEqual(json.loads(previous)["documentNamespace"],
                            json.loads(self.output.read_text())["documentNamespace"])

    def test_missing_required_fields(self):
        for key in self.document:
            if key == "documentNamespace":
                continue
            with self.subTest(field=key):
                document = copy.deepcopy(self.document)
                del document[key]
                self.rejected(document, "missing")
        for section in ("packages", "files"):
            for key in self.document[section][0]:
                with self.subTest(section=section, field=key):
                    document = copy.deepcopy(self.document)
                    del document[section][0][key]
                    self.rejected(document, "missing")

    def test_missing_sha1_sha256_and_malformed_checksums(self):
        for algorithm in ("SHA1", "SHA256"):
            document = copy.deepcopy(self.document)
            document["files"][0]["checksums"] = [
                c for c in document["files"][0]["checksums"] if c["algorithm"] != algorithm]
            self.rejected(document, "requires SHA1 and SHA256")
        for value in ("0" * 39, "X" * 40, "ABCDEF" * 10, None, 0):
            document = copy.deepcopy(self.document)
            document["files"][0]["checksums"][0]["checksumValue"] = value
            self.rejected(document, "lowercase hex")
        document = copy.deepcopy(self.document)
        document["files"][0]["checksums"].append(document["files"][0]["checksums"][0])
        self.rejected(document, "duplicate checksum")

    def test_verification_code_and_analyzed_semantics(self):
        document = copy.deepcopy(self.document)
        document["packages"][0]["packageVerificationCode"]["packageVerificationCodeValue"] = "0" * 40
        self.rejected(document, "verification code mismatch")
        for value in (False, "true", 1, None):
            document = copy.deepcopy(self.document)
            document["packages"][0]["filesAnalyzed"] = value
            self.rejected(document, "must analyze")

    def test_identifiers_paths_and_graph(self):
        for key, value, message in (
            ("SPDXID", "bad:id", "invalid file SPDXID"),
            ("SPDXID", "SPDXRef-Package", "duplicate SPDXID"),
            ("fileName", "../outside", "package-relative"),
            ("fileName", "/absolute", "package-relative"),
            ("fileName", "a//b", "package-relative"),
            ("fileName", "a\\b", "package-relative"),
        ):
            document = copy.deepcopy(self.document)
            document["files"][0][key] = value
            self.rejected(document, message)
        for key in ("SPDXID", "fileName"):
            document = copy.deepcopy(self.document)
            document["files"][1][key] = document["files"][0][key]
            self.rejected(document, "duplicate")
        document = copy.deepcopy(self.document)
        document["relationships"].pop()
        self.rejected(document, "incomplete")
        document = copy.deepcopy(self.document)
        document["relationships"][0]["relatedSpdxElement"] = "SPDXRef-Missing"
        self.rejected(document, "dangling")
        document = copy.deepcopy(self.document)
        document["relationships"].append(document["relationships"][0])
        self.rejected(document, "duplicate relationship")

    def test_metadata_namespace_and_unknown_fields(self):
        for mutate, message in (
            (lambda d: d.update(spdxVersion="SPDX-2.2"), "SPDX-2.3"),
            (lambda d: d.update(dataLicense="NONE"), "CC0"),
            (lambda d: d.update(name=""), "single-line"),
            (lambda d: d["creationInfo"].update(creators=[]), "creators"),
            (lambda d: d["creationInfo"].update(created="2026-99-99T00:00:00Z"), "does not match"),
            (lambda d: d["creationInfo"].update(creators=["not a creator"]), "creator"),
            (lambda d: d["packages"][0]["externalRefs"][0].update(referenceLocator="a" * 41), "source-sha"),
            (lambda d: d.update(uncheckedMetadata="anything"), "unsupported"),
        ):
            document = copy.deepcopy(self.document)
            mutate(document)
            self.rejected(document, message)
        document = copy.deepcopy(self.document)
        del document["documentNamespace"]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate(document)
        for namespace in ("https://example.invalid/stale", self.document["documentNamespace"] + "#fragment"):
            document = copy.deepcopy(self.document)
            document["documentNamespace"] = namespace
            with self.assertRaisesRegex(ValueError, "namespace"):
                validate(document)

    def test_sdk_bytes_inventory_and_output_boundary(self):
        (self.sdk / "lib/core.a").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate(self.document, self.sdk)
        (self.sdk / "extra").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            validate(self.document, self.sdk)
        with self.assertRaises(subprocess.CalledProcessError):
            self.generate(output=self.sdk / "self.spdx.json")
        (self.sdk / "link").symlink_to(self.output)
        with self.assertRaises(subprocess.CalledProcessError):
            self.generate()

    def test_release_commands_run_semantics_and_fail_closed(self):
        for workflow in ("release.yml", "deployment-release.yml"):
            source = (ROOT / ".github/workflows" / workflow).read_text()
            match = re.search(r"^\s*python3 tools/validate_sbom.py[^\n]*(?:\\\n[^\n]*)*", source, re.M)
            self.assertIsNotNone(match, workflow)
            command = match.group(0).strip()
            self.assertNotIn("json.tool", command)
            # Execute the literal workflow command with its SDK root and output
            # variables bound to our synthetic fixture, before corrupting JSON.
            import shutil
            runner = self.directory / workflow
            (runner / "dist").mkdir(parents=True)
            for suffix in ("sdk", "sdk-a"):
                shutil.copytree(self.sdk, runner / suffix)
            destination = runner / "dist/kws-pipeline-0.3.0.spdx.json"
            destination.write_bytes(self.output.read_bytes())
            env = dict(os.environ, RUNNER_TEMP=str(runner), VERSION="0.3.0", stem="kws-pipeline-0.3.0")
            (runner / "tools").symlink_to(ROOT / "tools", target_is_directory=True)
            result = subprocess.run(["bash", "-e", "-c", command], cwd=runner, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            bad = copy.deepcopy(self.document)
            bad["files"][0]["checksums"] = bad["files"][0]["checksums"][1:]
            destination.write_text(json.dumps(bad))
            result = subprocess.run(["bash", "-e", "-c", command], cwd=runner, env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, workflow)
            self.assertIn("requires SHA1", result.stderr)

    def test_duplicate_json_key_cli_rejected(self):
        self.output.write_text(self.output.read_text().replace('"spdxVersion":', '"spdxVersion":"SPDX-2.2","spdxVersion":'))
        result = subprocess.run([sys.executable, ROOT / "tools/validate_sbom.py", self.output,
                                 "--root", self.sdk], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate JSON key", result.stderr)


if __name__ == "__main__":
    unittest.main()
