"""Static preflight only. No model/torch imports or numerical execution."""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('anchor_run',HERE/'run_anchor.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def main():
    plan=runner._contract.strict_json((HERE/'plan.json').read_bytes())
    runner.validate_plan(plan,require_enabled=False)
    for name,digest in plan['diagnostic_code_sha256'].items():runner.read_checked(HERE/name,digest)
    runner.read_checked(HERE/'control-baseline.json',plan['control_baseline_sha256'])
    for name,digest in plan['energy_sha256'].items():runner.read_checked(HERE/name,digest)
    runner.read_checked(HERE/'runtime-candidate.lock.json',plan['runtime_lock_sha256'])
    print('Static pins verified. No model, optimizer or experimental calls.')
if __name__=='__main__':main()
