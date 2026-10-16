"""Static source/plan pins only. Never imports Torch or executes the numerical runner."""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lr_runner_preflight',HERE/'run_lr_stability.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def main():
    p=r._contract.strict_json((HERE/'plan.json').read_bytes());r.validate_plan(p,False)
    for name,digest in p['diagnostic_code_sha256'].items():r.read_checked(HERE/name,digest)
    for name,digest in p['shared_source_sha256'].items():r.read_checked(r._contract.SHARED/name,digest)
    r.read_checked(HERE/'historical-input-pins.json',p['historical_input_pins_sha256'])
    print('Static pins PASS; zero model/optimizer/ASR/TTS calls.')
if __name__=='__main__':main()
