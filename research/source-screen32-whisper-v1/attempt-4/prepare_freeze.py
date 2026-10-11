"""Freeze fourth source locally, disarmed; never rewrite prior campaign records."""
from pathlib import Path
import importlib.util
import json
import sys
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from common import read_json, file_hash, canonical, regular, require

THIRD_FREEZE_SHA = '16901d0d47d10486a8754568cb03020a3a155a9034ccc4e522fb796929a48ad4'
FOURTH_SOURCE_NAMES = ('README.md', 'continue_once.py', 'prepare_freeze.py',
                       'test_fourth_campaign.py', 'workflow.yml')


def source_paths():
    # Preserve the complete original closure, including original WAVs and all
    # first/two-route evidence. Read old freezes, never regenerate them.
    previous = BASE / 'attempt-3/execution-freeze.json'
    require(file_hash(previous) == THIRD_FREEZE_SHA, 'immutable third freeze')
    original = read_json(previous)
    require(original['schema'] == 'whisper-screen-third-source-freeze-v1' and
            len(original['files']) == 234, 'original 234-file closure')
    paths = {ROOT / name for name in original['files']}
    for directory in (BASE, BASE / 'attempt-2', BASE / 'attempt-3'):
        paths.update(directory / name for name in ('execution-freeze.json', 'execution-release.json'))
    # Current compatibility policy, source audit and complete source-only tests
    # are setup/review context; they never enter the blind inference mount.
    paths.update(BASE / 'attempt-3' / name for name in
                 ('pip_compatibility.py', 'test_pip_compatibility.py'))
    paths.update(BASE / 'pip-compatibility' / name for name in
                 ('README.md', 'upstream-source-manifest.json'))
    paths.add(BASE / 'FAIL_FAST_PREPARATION.md')
    for run in (38105471896, 38107706348, 38112558973):
        directory = BASE / ('evidence/run-' + str(run))
        paths.update(path for path in directory.rglob('*') if path.is_file())
    paths.update(HERE / name for name in FOURTH_SOURCE_NAMES)
    for path in paths:
        require(path.is_relative_to(ROOT), 'source remains inside repository')
        regular(path)
    return sorted(paths)


def main():
    spec = importlib.util.spec_from_file_location('fourth_freeze_gate', HERE / 'continue_once.py')
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    gate.verify_prior()
    files = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in source_paths()}
    freeze = {'schema': 'whisper-screen-fourth-source-freeze-v1', 'files': files,
              'inference_code': gate.INFERENCE_CODE, 'setup_code': gate.SETUP_CODE}
    (HERE / 'execution-freeze.json').write_bytes(canonical(freeze))
    release = {'schema': 'whisper-screen-fourth-execution-release-v1', 'experiment': gate.EXPERIMENT,
               'campaign_attempt': 4, 'prior_run_ids': [row[0] for row in gate.PRIOR], 'approved': False,
               'source_freeze_sha256': file_hash(HERE / 'execution-freeze.json')}
    (HERE / 'execution-release.json').write_bytes(canonical(release))
    print(json.dumps({'source_files': len(files), 'source_freeze_sha256': release['source_freeze_sha256'],
                      'campaign_attempt': 4, 'approved': False, 'model_calls': 0}))


if __name__ == '__main__':
    main()
