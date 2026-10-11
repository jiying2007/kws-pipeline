"""Freeze second-attempt source only; preserve all first-attempt freeze/release bytes."""
from pathlib import Path
import json
import sys
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(HERE))
from continue_once import EXPERIMENT, PRIOR_RUN, verify_prior
from common import read_json, file_hash, canonical


def main():
    verify_prior()
    original = read_json(BASE / 'execution-freeze.json')
    paths = {ROOT / p for p in original['files']}
    paths.update(BASE / p for p in ('execution-freeze.json', 'execution-release.json'))
    paths.update(p for p in (BASE / 'evidence/run-38105471896').rglob('*') if p.is_file())
    paths.update(p for p in HERE.iterdir() if p.is_file() and p.name not in
                 ('execution-freeze.json', 'execution-release.json'))
    files = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}
    freeze = {'schema': 'whisper-screen-second-source-freeze-v1', 'files': files,
              'blind_code': original['blind_code']}
    (HERE / 'execution-freeze.json').write_bytes(canonical(freeze))
    release = {'schema': 'whisper-screen-second-execution-release-v1', 'experiment': EXPERIMENT,
               'campaign_attempt': 2, 'prior_run_id': PRIOR_RUN, 'approved': False,
               'source_freeze_sha256': file_hash(HERE / 'execution-freeze.json')}
    (HERE / 'execution-release.json').write_bytes(canonical(release))
    print(json.dumps({'source_files': len(files), 'source_freeze_sha256': release['source_freeze_sha256'],
                      'campaign_attempt': 2, 'approved': False, 'model_calls': 0}))


if __name__ == '__main__':
    main()
