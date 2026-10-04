"""Pure-stdlib technical fixtures only: 2–3-frame opaque byte markers, no TTS.

No model, decoder, resampler, NumPy, network, or public publication is invoked.
All files are temporary header/byte fixtures; none are listening deliverables.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import packager as p


def write_json(path, value):
    path.write_bytes(p.json_bytes(value))


def npy_fixture(frames=3):
    header = repr({"descr": "<f4", "fortran_order": False, "shape": (frames,)}).encode("ascii") + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\0" * 12


def wav_fixture(rate, frames, marker):
    payload = bytes((marker, 0)) * frames
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(payload), b"WAVE", b"fmt ",
                       16, 1, 1, rate, rate * 2, 2, 16, b"data", len(payload)) + payload


class PackagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cosy30-header-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = self.root / "job"
        self.output = self.job / "outputs"
        self.output.mkdir(parents=True)
        self.destination = self.root / "artifacts"
        self.config = {"rows": copy.deepcopy(list(p.EXPECTED_ROWS)),
                       "references": [{"stock_voice": voice, "pcm_sha256": digest}
                                      for voice, digest in p.REFERENCE_PCM.items()]}

    def generated(self, index):
        row = p.EXPECTED_ROWS[index]
        clip = row["clip_id"]
        write_json(self.output / (clip + ".attempt.json"), {"clip_id": clip, "seed": row["seed"], "attempt": 1})
        data = {".native.npy": npy_fixture(), ".wav": wav_fixture(24000, 3, index + 1),
                ".16k.wav": wav_fixture(16000, 2, index + 1)}
        for suffix, raw in data.items():
            (self.output / (clip + suffix)).write_bytes(raw)
        result = dict(row, status="generated", frames=3, frames24k=3, sample_rate=24000,
                      duration_seconds=3 / 24000, peak=0.1, rms=0.01,
                      api_return_completed=True, eos_proved=False,
                      files={suffix: hashlib.sha256(raw).hexdigest() for suffix, raw in data.items()},
                      error="secret /private/path https://private.example/ token=private",
                      raw_semantic_output="DO NOT PUBLISH")
        write_json(self.output / (clip + ".result.json"), result)
        return result

    def change_result(self, index, **changes):
        path = self.output / (p.EXPECTED_ROWS[index]["clip_id"] + ".result.json")
        result = json.loads(path.read_text())
        result.update(changes)
        write_json(path, result)

    def replace_audio(self, index, suffix, data):
        clip = p.EXPECTED_ROWS[index]["clip_id"]
        (self.output / (clip + suffix)).write_bytes(data)
        path = self.output / (clip + ".result.json")
        result = json.loads(path.read_text())
        result["files"][suffix] = hashlib.sha256(data).hexdigest()
        write_json(path, result)

    def pack(self, **kwargs):
        return p.pack(self.job, self.config, self.destination, **kwargs)

    def successful_receipts(self):
        return {"controller":{"status":"completed"},
                "generate_supervisor":{"status":"passed","returncode":0},
                "generate_worker":{"status":"complete","generated_clips":30,"network_attempts":[]}}

    def test_all30_files_do_not_override_failed_or_missing_terminal_receipts(self):
        for i in range(30):
            self.generated(i)
        cases = [{}]
        for name in ("controller","generate_supervisor","generate_worker"):
            receipts=self.successful_receipts();receipts[name]["status"]="failed";cases.append(receipts)
        receipts=self.successful_receipts();receipts["generate_worker"]["generated_clips"]=29;cases.append(receipts)
        receipts=self.successful_receipts();receipts["generate_worker"]["network_attempts"]=["blocked"];cases.append(receipts)
        receipts=self.successful_receipts();receipts["generate_supervisor"]["returncode"]=1;cases.append(receipts)
        for index,receipts in enumerate(cases):
            self.destination=self.root/("failed-terminal-%d" % index)
            result=self.pack(technical_receipts=receipts)
            self.assertEqual(result["counts"]["generated"],30)
            self.assertEqual(result["status"],"incomplete")
            self.assertFalse(result["successful_execution_receipts"])
            self.assertFalse(result["initial_listening_enabled"])
            self.assertFalse((self.destination/p.ARTIFACTS[0]/"initial-listening").exists())

    def test_complete_partition_initial_four_and_manifest_hashes(self):
        for i in range(30):
            self.generated(i)
        for name in p.INTERNAL_JSON:
            write_json(self.output / name, {"private": "/private/path", "secret": "DO NOT PUBLISH"})
        summary = self.pack(technical_receipts=self.successful_receipts())
        self.assertEqual(summary["counts"], {"generated": 30, "invalid": 0, "not_attempted": 0})
        self.assertEqual({x.name for x in self.destination.iterdir()}, set(p.ARTIFACTS))
        for artifact, indices in zip(p.ARTIFACTS, (range(24), range(24, 30))):
            base = self.destination / artifact
            expected_audio = {p.EXPECTED_ROWS[i]["clip_id"] + suffix for i in indices for suffix in p.SUFFIXES}
            self.assertEqual({x.name for x in (base / "audio").iterdir()}, expected_audio)
            manifest = json.loads((base / "artifact-manifest.json").read_text())
            names = set()
            for member in manifest["files"]:
                path = base / member["path"]
                names.add(member["path"])
                self.assertEqual(path.stat().st_size, member["bytes"])
                self.assertEqual(p.sha256(path), member["sha256"])
            actual = {str(path.relative_to(base)) for path in base.rglob("*") if path.is_file()}
            self.assertEqual(actual, names | {"artifact-manifest.json"})
            for path in base.rglob("*.json"):
                text = path.read_text()
                self.assertNotIn("/private/path", text)
                self.assertNotIn("DO NOT PUBLISH", text)
                self.assertNotIn("private.example", text)
        train = self.destination / p.ARTIFACTS[0]
        listening = json.loads((train / "technical/initial-listening-manifest.json").read_text())
        self.assertTrue(listening["enabled"])
        self.assertEqual([row["clip_id"] for row in listening["rows"]], list(p.LISTENING_ORDER))
        self.assertEqual(set(p.LISTENING_ORDER), set(p.INITIAL_FOUR))
        self.assertNotIn("intended_text", json.dumps(listening))
        self.assertNotIn("stock_reference_voice", json.dumps(listening))
        for row in listening["rows"]:
            self.assertEqual(row["sample_rate"], 16000)
            self.assertEqual(p.sha256(train / row["path"]), p.sha256(self.output / (row["clip_id"] + ".16k.wav")))
        sealed = self.destination / p.ARTIFACTS[1]
        self.assertFalse((sealed / "initial-listening").exists())
        for path in sealed.rglob("*.json"):
            text = path.read_text()
            for forbidden in ("Dylan", "intended_text", "phrase_id", "peak", "rms", "human_label", "raw_semantic_output"):
                self.assertNotIn(forbidden, text)

    def test_partial_retains_valid_no_listening_and_all30_statuses(self):
        self.generated(0)
        second = self.generated(1)
        (self.output / (second["clip_id"] + ".result.json")).unlink()
        summary = self.pack()
        self.assertEqual(summary["counts"], {"generated": 1, "invalid": 1, "not_attempted": 28})
        self.assertEqual(len(summary["outcomes"]), 30)
        self.assertFalse(summary["initial_listening_enabled"])
        self.assertTrue((self.destination / p.ARTIFACTS[0] / "audio" / (p.EXPECTED_ROWS[0]["clip_id"] + ".wav")).exists())
        self.assertFalse((self.destination / p.ARTIFACTS[0] / "initial-listening").exists())

    def test_missing_output_is_all_not_attempted(self):
        self.output.rmdir()
        self.assertEqual(self.pack()["counts"], {"generated": 0, "invalid": 0, "not_attempted": 30})

    def test_missing_audio_and_hash_mismatch_are_invalid(self):
        first = self.generated(0)
        second = self.generated(1)
        (self.output / (first["clip_id"] + ".wav")).unlink()
        (self.output / (second["clip_id"] + ".16k.wav")).write_bytes(b"corrupt")
        result = self.pack()
        self.assertEqual([r["invalid_reason"] for r in result["outcomes"][:2]], ["missing_audio", "file_hash_mismatch"])

    def test_max20seconds_without_allocating_waveform(self):
        self.generated(0)
        self.generated(1)
        self.replace_audio(1, ".native.npy", npy_fixture(480001))
        result = self.pack()
        self.assertEqual(result["outcomes"][0]["status"], "generated")
        self.assertEqual(result["outcomes"][1]["invalid_reason"], "duration_limit")

    def test_duplicate_pcm_invalidates_both_owners(self):
        self.generated(0)
        self.generated(1)
        self.replace_audio(1, ".wav", wav_fixture(24000, 3, 1))
        result = self.pack()
        self.assertEqual(result["counts"]["invalid"], 2)
        self.assertEqual({row.get("invalid_reason") for row in result["outcomes"][:2]}, {"duplicate_or_exposed_pcm"})

    def test_duplicate_pcm_against_exposed_hash(self):
        self.generated(0)
        digest = hashlib.sha256(bytes((1, 0)) * 3).hexdigest()
        result = self.pack(exposed_pcm_sha256=[digest])
        self.assertEqual(result["outcomes"][0]["invalid_reason"], "duplicate_or_exposed_pcm")

    def test_attempt_contract_and_gap(self):
        self.generated(0)
        self.generated(2)
        path = self.output / (p.EXPECTED_ROWS[0]["clip_id"] + ".attempt.json")
        write_json(path, {"clip_id": p.EXPECTED_ROWS[0]["clip_id"], "seed": 610401, "attempt": 2})
        result = self.pack()
        self.assertEqual(result["outcomes"][0]["invalid_reason"], "attempt_contract")
        self.assertEqual(result["outcomes"][2]["invalid_reason"], "attempt_after_gap")

    def test_missing_attempt_does_not_admit_output(self):
        result = self.generated(0)
        (self.output / (result["clip_id"] + ".attempt.json")).unlink()
        self.assertEqual(self.pack()["outcomes"][0]["invalid_reason"], "missing_attempt")

    def test_api_return_and_eos_are_not_conflated(self):
        self.generated(0)
        self.generated(1)
        self.change_result(0, api_return_completed=False)
        self.change_result(1, eos_proved=True)
        result = self.pack()
        self.assertEqual([r["invalid_reason"] for r in result["outcomes"][:2]], ["api_return_unconfirmed", "unsupported_eos_claim"])

    def test_extra_audio_or_crosssplit_directory_rejected(self):
        (self.output / "unapproved.wav").write_bytes(b"header")
        with self.assertRaisesRegex(p.GateError, "unexpected_output_entry"):
            self.pack()
        self.assertFalse(self.destination.exists())
        (self.output / "unapproved.wav").unlink()
        (self.output / "sealed-six").mkdir()
        with self.assertRaisesRegex(p.GateError, "unexpected_output_entry"):
            self.pack()

    def test_traversal_or_changed_split_rejected(self):
        self.config["rows"][0]["clip_id"] = "../secret"
        with self.assertRaisesRegex(p.GateError, "frozen_row_mismatch"):
            self.pack()
        self.config["rows"] = copy.deepcopy(list(p.EXPECTED_ROWS))
        self.config["rows"][24]["split"] = "train"
        with self.assertRaisesRegex(p.GateError, "frozen_row_mismatch"):
            self.pack()

    def test_receipt_cannot_redirect_audio(self):
        result = self.generated(0)
        hashes = dict(result["files"], **{"../secret.wav": "a" * 64})
        self.change_result(0, files=hashes)
        self.assertEqual(self.pack()["outcomes"][0]["invalid_reason"], "file_hash_contract")

    def test_symlink_and_hardlink_rejected(self):
        self.generated(0)
        audio = self.output / (p.EXPECTED_ROWS[0]["clip_id"] + ".wav")
        original = self.root / "original"
        audio.rename(original)
        audio.symlink_to(original)
        with self.assertRaisesRegex(p.GateError, "symlink"):
            self.pack()
        audio.unlink()
        audio.hardlink_to(original)
        with self.assertRaisesRegex(p.GateError, "nonregular_file"):
            self.pack()

    def test_root_symlink_and_overlap_rejected(self):
        link = self.root / "link"
        link.symlink_to(self.job, target_is_directory=True)
        with self.assertRaisesRegex(p.GateError, "symlink"):
            p.pack(link, self.config, self.destination)
        with self.assertRaisesRegex(p.GateError, "overlapping_roots"):
            p.pack(self.job, self.config, self.job / "artifacts")

    def test_combined_cap_checked_before_destination_creation(self):
        self.generated(0)
        with patch.object(p, "LIMIT", 1024):
            with self.assertRaisesRegex(p.GateError, "combined_artifact_limit|retained_output_limit"):
                self.pack()
        self.assertFalse(self.destination.exists())
        with patch.object(p, "LIMIT", 100000):
            with self.assertRaisesRegex(p.GateError, "combined_publication_envelope"):
                self.pack()
        self.assertFalse(self.destination.exists())

    def test_receipts_allowlist_sanitization_and_provenance(self):
        controller = {"status": "failed", "run_id": "123456", "head_sha": "a" * 40,
                      "error": "https://private.example /secret password=secret", "elapsed_seconds": 5,
                      "steps": [{"phase": "generate-fixed30", "status": "failed", "elapsed_seconds": 4,
                                 "command": ["/secret"], "cleanup_verified": True, "log_sha256": "b" * 64}]}
        write_json(self.job / "controller-state.json", controller)
        runtime = self.job / "receipts" / "runtime"
        runtime.mkdir(parents=True)
        install = {"status": "installed_pending_offline_qualification", "python_version": "3.12.14",
                   "lock_sha256": "c" * 64, "events": [{"event": "verified_download", "bytes": 12, "url": "secret"},
                                                        {"event": "verified_download", "bytes": 30}]}
        write_json(runtime / "install-receipt.json", install)
        self.pack()
        clean = json.loads((self.destination / p.ARTIFACTS[0] / "technical/controller-resources.json").read_text())
        self.assertEqual(clean["controller"]["receipt_sha256"], p.sha256(self.job / "controller-state.json"))
        self.assertEqual(clean["controller"]["steps"][0]["phase"], "generate-fixed30")
        self.assertEqual(clean["runtime_install"]["runtime_download_bytes"], 42)
        text = json.dumps(clean)
        for forbidden in ("secret", "command", "error", "url", "events"):
            self.assertNotIn(forbidden, text)
        self.assertFalse((self.destination / p.ARTIFACTS[1] / "technical/controller-resources.json").exists())

    def test_invalid_json_and_status_are_bounded(self):
        self.generated(0)
        self.generated(1)
        self.change_result(0, status="https://private.example/token")
        path = self.output / (p.EXPECTED_ROWS[1]["clip_id"] + ".result.json")
        path.write_text('{"secret":"secret","status":NaN}')
        result = self.pack()
        self.assertEqual(result["outcomes"][0]["invalid_reason"], "result_status")
        self.assertEqual(result["outcomes"][1]["invalid_reason"], "nonfinite_json")
        self.assertNotIn("secret", json.dumps(result))

    def test_native_and_wav_headers_reject_hidden_metadata(self):
        self.generated(0)
        self.generated(1)
        self.replace_audio(0, ".wav", wav_fixture(24000, 3, 1) + b"private trailing metadata")
        header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (3,), 'private': 'secret'}\n"
        self.replace_audio(1, ".native.npy", b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\0" * 12)
        result = self.pack()
        self.assertEqual(result["outcomes"][0]["invalid_reason"], "wav_length")
        self.assertEqual(result["outcomes"][1]["invalid_reason"], "native_header")

    def test_destination_is_never_overwritten(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(p.GateError, "destination_exists"):
            self.pack()


if __name__ == "__main__":
    unittest.main()
