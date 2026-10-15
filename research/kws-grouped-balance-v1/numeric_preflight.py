"""Static preflight. Never imports torch or calls model/optimizer."""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('grouped_runner',HERE/'run_grouped_balance.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def main():
    plan=runner._contract.strict_json((HERE/'plan.json').read_bytes())
    runner.validate_plan(plan,require_enabled=False)
    for name,digest in plan['diagnostic_code_sha256'].items():runner.read_checked(HERE/name,digest)
    runner.read_checked(HERE/'cohort-prefix-freeze.json',plan['cohort_sha256'])
    runner.read_checked(HERE/'runtime-candidate.lock.json',plan['runtime_lock_sha256'])
    print('Static pins verified; zero model, optimizer, TTS or ASR calls.')
if __name__=='__main__':main()
