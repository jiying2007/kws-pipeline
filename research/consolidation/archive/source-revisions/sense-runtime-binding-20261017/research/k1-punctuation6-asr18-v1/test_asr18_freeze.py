"""Source inventory tests only. Synthetic temporary closure, no runtime imports."""
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_freeze as freeze


class SourceFreezeTests(unittest.TestCase):
    def test_exact_blind_inference_closure_excludes_campaign_and_host_data(self):
        rosters = freeze.declared_rosters()
        common = rosters["inference_code"]
        self.assertIn(freeze.NEW + "campaign_contract.py", common)
        self.assertIn(freeze.NEW + "native_worker.py", common)
        self.assertIn(freeze.NEW + "execution-release.json", common)
        for name in common:
            self.assertNotIn("qwen12-native-admission-v1", name)
            self.assertFalse(name.startswith("host-tts-verifier-source/"))
        for name in ("input-freeze.json", "tts-source-authority.json", "input_binding.py", "raw_data_reader.py", "hosted_run.py",
                     "claim_admission.py", "claim-admission-control.json"):
            self.assertNotIn(freeze.NEW + name, common)
        for name in ("claim_admission.py", "claim-admission-control.json"):
            self.assertIn(freeze.NEW + name, rosters["host_code"])
            self.assertNotIn(freeze.NEW + name, rosters["setup_code"])
        self.assertNotIn(freeze.OLD + "hosted_run.py", rosters["setup_code"])
        self.assertNotIn(freeze.OLD + "contract.py", rosters["setup_code"])
        self.assertEqual(len(freeze.HOST_TTS_VERIFIER), 43)
        self.assertEqual(len(set(freeze.HOST_TTS_VERIFIER)), 43)
        self.assertTrue(set(freeze.HOST_TTS_VERIFIER) <= rosters["host_code"])
        for names in rosters.values():
            self.assertNotIn(freeze.NEW + "execution-freeze.json", names)

    def test_complete_measured_self_excluding_snapshot_and_retained_pin(self):
        rosters = freeze.declared_rosters()
        with tempfile.TemporaryDirectory(prefix="k1-synthetic-closure-") as name:
            root = Path(name)
            for relative in rosters["host_code"]:
                src, target = freeze.ROOT / relative, root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                # Missing new host files are explicit synthetic test bodies.
                # This fixture is never persisted as a genuine source freeze.
                target.write_bytes(src.read_bytes() if src.is_file() else b"synthetic test member\n")
            (root / freeze.NEW / "execution-freeze.json").write_bytes(b"ignored self")
            extra = root / "research/qwen12-native-admission-v1/campaign_contract.py"
            extra.parent.mkdir(parents=True); extra.write_bytes(b"not selected")
            value = freeze.build(root)
            self.assertEqual(set(value), {"schema", *rosters})
            self.assertEqual(value["schema"], "k1-asr18-source-freeze-v1")
            for role, members in rosters.items():
                self.assertEqual(set(value[role]), members)
                self.assertTrue(all(set(row) == {"bytes", "sha256"} for row in value[role].values()))
            retained = next(iter(freeze.retained_pins(root / freeze.OLD / "asr_worker.py")))
            (root / "research/qwen6_asr" / retained).write_bytes(b"synthetic pin drift")
            with self.assertRaisesRegex(ValueError, "pin"):
                freeze.build(root)

    def test_no_arbitrary_cli_paths(self):
        with self.assertRaises(ValueError):
            freeze.main(["/unreviewed/source"])

    def test_literal_pin_roster_only_no_import(self):
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / "invented.py"
            source.write_text("raise RuntimeError('must not execute')\nRETAINED_PINS = {'safe.py': '" + "a" * 64 + "'}\n")
            self.assertEqual(freeze.retained_pins(source), {"safe.py": "a" * 64})
            source.write_text("RETAINED_PINS = {'../outside.py': '" + "a" * 64 + "'}\n")
            with self.assertRaises(ValueError):
                freeze.retained_pins(source)


if __name__ == "__main__":
    unittest.main()
