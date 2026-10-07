#!/usr/bin/env python3
"""Adapter-only synthetic tests. Never imports a DP or opens saved logits."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def load(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(original_path,recovery_path):
    old = load(original_path,"original_adapter_review")
    new = load(recovery_path,"recovery_adapter_review")
    checks = []
    for values in ([0]*6,[3.14159274,-.123456789,0,1,-2,4],[12,-20,9,0,1,-200],[0,-600,-601,-602,-603,-604]):
        assert new.softmax6(values) == old.softmax6(values)
    checks.append("four ordinary/positive-tail rows preserve original exact binary64 outputs")

    actual = new.softmax6([0,-1000,0,0,0,0])
    assert len(actual)==6 and actual[1]==0 and all(value>0 for index,value in enumerate(actual) if index!=1)
    checks.append("arithmetic exponential underflow keeps six positions and permits zero")

    positive = new.softmax6([0,-600,0,0,0,0])
    assert positive[1] > 0
    checks.append("positive term below -512 is retained; no threshold pruning")

    assert math.exp(-745)>0 and math.exp(-745)/5 == 0
    divided = new.softmax6([0,-745,0,0,0,0])
    assert divided[1] == 0 and sum(value>0 for value in divided)==5
    checks.append("division-only underflow allowed after exact-difference validation")

    exp = math.exp
    def manufactured_zero(value):
        return 0. if value<0 else exp(value)
    for difference,allowed in ((-512.,True),(-511.999969482421875,False),(-1.,False)):
        with patch.object(new.math,"exp",side_effect=manufactured_zero):
            try:
                values = new.softmax6([0,difference,0,0,0,0])
            except ValueError:
                assert not allowed
            else:
                assert allowed and values[1]==0
    checks.append("exact -512 guard boundary and next FP32 value above it; unexpected near-zero exp rejected")

    invalid = ([0]*5,[0]*7,[False]+[0]*5,[math.nan]+[0]*5,[math.inf]+[0]*5,[1e300]+[0]*5)
    for values in invalid:
        try:
            new.softmax6(values)
        except (ValueError,OverflowError):
            pass
        else:
            raise AssertionError("invalid row accepted")
    checks.append("six invalid shape/type/nonfinite/FP32-overflow rows remain rejected")

    counts = {key:0 for key in ("exponential_zeros","division_only_zeros","rows_converted","rows_with_zero_weight")}
    for values in ([0]*6,[0,-1000,0,0,0,0],[0,-745,0,0,0,0]):
        new.softmax6(values,counts)
    assert counts == {"exponential_zeros":1,"division_only_zeros":1,"rows_converted":3,"rows_with_zero_weight":2}
    callback = new.softmax_callback([[0]*6,[0]*5],"MOCK",2)
    assert len(next(callback)) == 6
    try:
        next(callback)
    except ValueError as exc:
        assert "observation=MOCK callback_index=2 row_index=1" in str(exc)
    else:
        raise AssertionError("bad callback row accepted")
    checks.append("stage-separated underflow counts and indexed failure context")

    # Tiny exact finite product example illustrates TV accumulation, with no DP.
    from fractions import Fraction
    from itertools import product
    probabilities = (Fraction(1,64),Fraction(1,128),Fraction(1,256))
    ideal = {bits:math.prod(p if bit else 1-p for bit,p in zip(bits,probabilities)) for bits in product((0,1),repeat=3)}
    tv = sum(abs(mass-(1 if bits==(0,0,0) else 0)) for bits,mass in ideal.items())/2
    assert tv == 1-math.prod(1-p for p in probabilities)
    assert tv <= sum(probabilities)
    checks.append("exact finite-product TV tail inequality illustration")
    return {"status":"PASS_ADAPTER_MOCKS_ONLY","checks":checks,"dp_runs":0,"saved_values_read":False,
            "original_runner_sha256":hashlib.sha256(original_path.read_bytes()).hexdigest(),
            "recovery_runner_sha256":hashlib.sha256(recovery_path.read_bytes()).hexdigest(),
            "test_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--recovery",type=Path,required=True)
    args = parser.parse_args()
    result = run(args.original,args.recovery)
    (HERE/"underflow-mock-results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
