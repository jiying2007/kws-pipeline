"""Create a disabled source freeze after review; never changes historical bytes."""
from pathlib import Path
import json
import sys
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))
from common import read_json, file_hash, canonical, regular, require


def source_paths():
    original = read_json(BASE / 'attempt-2/execution-freeze.json')
    require(len(original['files']) == 205, 'original 205-file closure')
    paths = {ROOT / p for p in original['files']}
    for directory in (BASE, BASE / 'attempt-2'):
        paths.update(directory / p for p in ('execution-freeze.json', 'execution-release.json'))
    for run in (38105471896, 38107706348):
        paths.update(p for p in (BASE / ('evidence/run-' + str(run))).rglob('*') if p.is_file())
    paths.update(p for p in HERE.iterdir() if p.is_file() and p.name not in
                 ('execution-freeze.json', 'execution-release.json'))
    for path in paths: regular(path)
    return sorted(paths)


def main():
    from continue_once import EXPERIMENT, PRIOR, INFERENCE_CODE, SETUP_CODE, verify_prior
    verify_prior()
    files = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_paths()}
    freeze = {'schema': 'whisper-screen-third-source-freeze-v1', 'files': files,
              'inference_code': INFERENCE_CODE, 'setup_code': SETUP_CODE}
    (HERE / 'execution-freeze.json').write_bytes(canonical(freeze))
    release = {'schema': 'whisper-screen-third-execution-release-v1', 'experiment': EXPERIMENT,
               'campaign_attempt': 3, 'prior_run_ids': [p[0] for p in PRIOR], 'approved': False,
               'source_freeze_sha256': file_hash(HERE / 'execution-freeze.json')}
    (HERE / 'execution-release.json').write_bytes(canonical(release))
    print(json.dumps({'source_files': len(files), 'source_freeze_sha256': release['source_freeze_sha256'],
                      'campaign_attempt': 3, 'approved': False, 'model_calls': 0}))


if __name__ == '__main__':
    main()
