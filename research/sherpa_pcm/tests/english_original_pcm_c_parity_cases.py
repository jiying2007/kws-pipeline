"""No model, audio or external binary required: C research source identity tests."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_original_pcm_c_parity as research
from build_official_x64 import ARCHIVE_SHA, ARCHIVE_BYTES, LIBRARIES, verify_original_archive


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


    def test_original_official_x64_runtime_has_distinct_immutable_sherpa_hash(self):
        self.assertEqual(ARCHIVE_SHA,
                         "c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4")
        self.assertEqual(ARCHIVE_BYTES,28156791)
        self.assertEqual(LIBRARIES["libsherpa-onnx-c-api.so"],(
            5124192,"b8351ca1632571ac108adbb317bcc4bf7cfe84b72690e3017316b0da3e1e344f"))
        self.assertEqual(LIBRARIES["libonnxruntime.so"],(
            27026609,"4b3607aebd1784b26b6f9b20e4bd974c7ab8287043e4d095cb7d2cb40b5e566e"))
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/"official.tar.bz2"
            fake.write_bytes(b"not-original-runtime")
            with self.assertRaisesRegex(ValueError,"EN_C_OFFICIAL_HOST_ARCHIVE_CHANGED"):
                verify_original_archive(fake)


if __name__=="__main__":
    unittest.main()
