"""Rebuilt stdlib-only frozen planning contract. Historical execution stays blocked."""
import argparse
import json
from pathlib import Path
import unicodedata
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker import frozen_plan, validate_plan, canonical, decode, read_regular, require, sha


def normalize(text):
    require(type(text) is str, 'transcript text required')
    return ''.join(c for c in unicodedata.normalize('NFKC', text)
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def coverage_only(candidate_ids):
    slots = {c['slot_id']: c for c in frozen_plan()['slots']}
    require(type(candidate_ids) is list and len(candidate_ids) == len(set(candidate_ids)) and
        all(type(i) is str and i in slots for i in candidate_ids), 'candidate identity set')
    kw1 = {'Serena', 'Vivian'} | {slots[i]['group_id'] for i in candidate_ids}
    kw2 = {'Serena', 'Vivian', 'Uncle_Fu', 'Dylan'}
    common = sorted(kw1 & kw2)
    return dict(kw1_groups=sorted(kw1), kw2_groups=sorted(kw2), common_groups=common,
        common_group_count=len(common), three_role_coverage_necessary_condition=len(common) >= 3,
        roles_assigned=False, group_independence_proven=False, human_gold=False,
        training_admitted=False, calibration_gold=False, selection_gold=False,
        admission_status='BLOCKED_NO_CALIBRATION_SELECTION_GOLD')


def blind_projection(rows):
    keys = {'opaque_id','file','sha256','bytes','frames','sample_rate_hz','channels','sample_format'}
    result = []
    for row in rows:
        require(type(row) is dict and keys <= set(row), 'audio transport fields')
        result.append({k: row[k] for k in sorted(keys)})
    return result


def check_execution(*_):
    raise ValueError('PREPARED_NOT_EXECUTED: historical planning gate always blocks execution')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path)
    parser.add_argument('--check-execution', action='store_true')
    args = parser.parse_args(argv)
    validate_plan(decode(read_regular(Path(__file__).with_name('plan.json'))))
    if args.base:
        bindings = decode(read_regular(Path(__file__).with_name('source-bindings.json')))
        files = bindings.get('files')
        require(type(files) is dict and files, 'explicit reference source files required')
        for name, digest in files.items():
            require(type(name) is str and not name.startswith('/') and '..' not in Path(name).parts and
                sha(read_regular(args.base/name)) == digest, 'reference source drift')
    if args.check_execution: check_execution()
    print(json.dumps(dict(status='PASS', slots=16, tts_calls=0, asr_calls=0)))


if __name__ == '__main__': main()
