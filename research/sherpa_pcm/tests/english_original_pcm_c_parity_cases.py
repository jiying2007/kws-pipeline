"""No model, audio or external binary required: C research source identity tests."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_original_pcm_c_parity as research


class EnglishNativeMeroSourceCases(unittest.TestCase):
    def test_native_event_abi_is_opt_in_mero_only(self):
        src=(HERE.parent/"pcm_kws.cc").read_text()
        head=(HERE.parent/"pcm_kws.h").read_text()
        self.assertIn('!std::strcmp(r->keyword,"HEY_MERO")?3:0',src)
        self.assertIn('!std::strcmp(r->keyword,"你好小窝")?1',src)
        self.assertIn('!std::strcmp(r->keyword,"小窝小窝")?2',src)
        self.assertIn("HEY_MERO",head)
        self.assertEqual(research.POLICY,
                         "english63-native-c-mero-unpadded-event-parity-v1")
        self.assertEqual(research.KEYWORD,"HEY_MERO")
        self.assertEqual(set(research.PATS),{"fixed320","network1600","uneven"})
        self.assertEqual(research.REFERENCE_JSON_SHA,
                         "d5cfbd295299ff8c0ee7524538e44c9755ad25a943f5178455816bc691dbe521")

    def test_unverified_original_results_never_reach_native_model(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"source.zip"
            path.write_bytes(b"invented-not-Actions-source")
            with self.assertRaisesRegex(ValueError,"EN_C_BASELINE_ZIP_DRIFT"):
                research.verify_saved_baseline(path)

    def test_unsafe_runtime_or_missing_build_receipt_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/"libsherpa-onnx-c-api.so"
            fake.write_bytes(b"invented-not-actual")
            missing=Path(d)/"build.json"
            with self.assertRaisesRegex(ValueError,"EN_C_NATIVE_PATH"):
                research.verify_build_receipt(fake,missing)


if __name__=="__main__":
    unittest.main()
