"""Prepare source metadata only. Never activates, publishes, or runs the experiment."""
from pathlib import Path
import json
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common import EXPERIMENT, file_hash, canonical


def main():
    local = [p for p in HERE.iterdir() if p.is_file() and p.name not in
             ('execution-freeze.json', 'execution-release.json')]
    old = ROOT / 'research/source-screen32-v1'
    closure = [old / name for name in ('hosted_run.py', 'contract.py', 'tts_worker.py', 'qwen_adapter.py',
               'runtime_scope.py', 'sense_continuation.py', 'continuation-lock.json', 'container-lock.json')]
    retained = [p for p in (old / 'evidence/run-38018029787').rglob('*') if p.is_file()]
    helpers = [ROOT / 'tools/asr_native_contracts.py', ROOT / 'tests/asr_native_fixtures.py']
    files = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(local + closure + retained + helpers)}
    blind = ['research/source-screen32-whisper-v1/' + name for name in
             ('common.py', 'adapter.py', 'worker.py', 'setup_runtime.py', 'runtime-lock.json', 'source-lock.json')]
    blind += ['tools/asr_native_contracts.py', 'research/source-screen32-v1/runtime_scope.py']
    freeze = {'schema': 'whisper-screen-source-freeze-v1', 'files': files, 'blind_code': sorted(blind)}
    (HERE / 'execution-freeze.json').write_bytes(canonical(freeze))
    release = {'schema': 'whisper-screen-execution-release-v1', 'experiment': EXPERIMENT,
               'approved': False, 'source_freeze_sha256': file_hash(HERE / 'execution-freeze.json')}
    (HERE / 'execution-release.json').write_bytes(canonical(release))
    print(json.dumps({'source_files': len(files), 'source_freeze_sha256': release['source_freeze_sha256'],
                      'approved': False, 'model_calls': 0}))


if __name__ == '__main__':
    main()
