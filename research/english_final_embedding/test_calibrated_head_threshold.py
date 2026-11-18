"""Default 0.80 C behavior unchanged; frozen 0.90 research binary differs.

No real audio, model weights, internet, work quota or user data. The one
calibration value 0.90 is adopted from an immutable external actual training
receipt, NOT inferred/tuned from a locked test.
"""
from __future__ import annotations
import ctypes as C
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from test_portable_head import Model,State,Output,Row,invented_model

BASELINE=HERE/"english_causal_head.c"
CANONICAL=HERE.parent/"english_calibrated_head/english_causal_head_090.c"
FROZEN_MODEL_BIN_SHA256="49a73ad2c9856b8c8d44fe07846de74a83ace80d15f8fe22ed58aff4d404b040"
CALIBRATED_THRESHOLD=0.90

class SourceOnlyRuntimeThreshold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix="english-research-calibrated-c-")
        cls.base=Path(cls.tmp.name)
        cls.libraries={}
        for flavor,flags in (("default",[]),("canonical",[
                "-DEHEAD_DECISION_THRESHOLD=0.90f"])):
            dest=cls.base/(flavor+".so")
            subprocess.run(["cc","-std=c99","-O2","-ffp-contract=off",
                "-fno-fast-math","-Wall","-Wextra","-Werror","-fPIC",
                "-shared",*flags,str(BASELINE if flavor=="default" else CANONICAL),
                "-I",str(HERE),"-lm","-o",str(dest)],check=True)
            lib=C.CDLL(str(dest))
            lib.ehead_step.argtypes=[C.POINTER(Model),C.POINTER(State),
                C.POINTER(C.c_float),C.c_uint64,C.POINTER(Output)]
            lib.ehead_step.restype=C.c_int
            cls.libraries[flavor]=lib

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def sequence(lib,probability):
        model=invented_model()
        # With logits 0, L, 0 the preferred class posterior is probability.
        model.b2[:]=[0.0,math.log(probability/(1-probability)*2.0),0.0]
        state=State();rows=[]
        for i in range(32):
            sample=16000+1280*i
            out=Output()
            rc=lib.ehead_step(C.byref(model),C.byref(state),Row(),sample,C.byref(out))
            if rc!=0:raise AssertionError(f"native step failed: {rc}")
            rows.append((sample,out.ready,out.keyword,
                         tuple(out.logits),tuple(out.probabilities)))
        return rows

    def test_default_080_unchanged_and_090_policy_applied(self):
        default=self.sequence(self.libraries["default"],0.85)
        calibrated=self.sequence(self.libraries["canonical"],0.85)
        self.assertEqual([x[:2] for x in default],[x[:2] for x in calibrated])
        self.assertEqual(default[17][2],1)
        self.assertEqual([x[2] for x in calibrated],[0]*len(calibrated))
        # Numeric scores MUST be unchanged by the threshold selection.
        for baseline,variant in zip(default,calibrated):
            for a,b in zip(baseline[3],variant[3]):self.assertEqual(a,b)
            for a,b in zip(baseline[4],variant[4]):self.assertEqual(a,b)

    def test_calibrated_real_threshold_does_not_break_strong_events(self):
        baseline=self.sequence(self.libraries["default"],0.96)
        canonical=self.sequence(self.libraries["canonical"],0.96)
        self.assertEqual([r[2] for r in canonical],[r[2] for r in baseline])
        self.assertEqual(canonical[17][2],1)
        self.assertEqual(canonical[31][0],16000+31*1280)

    def test_canonical_research_copy_is_exactly_one_literal_from_frozen_source(self):
        import hashlib
        original=BASELINE.read_bytes()
        native=CANONICAL.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         "e1e4a4cffd1758677ef820bd2183ec2ab3ab65d450af4d7eea35f565ebefa551")
        old=b"o->probabilities[selected] < 0.80f"
        self.assertEqual(original.count(old),1)
        self.assertEqual(native,original.replace(old,
                          b"o->probabilities[selected] < 0.90f"))
        self.assertEqual(CALIBRATED_THRESHOLD,0.9)
        self.assertEqual(len(FROZEN_MODEL_BIN_SHA256),64)

if __name__=="__main__":
    unittest.main()
