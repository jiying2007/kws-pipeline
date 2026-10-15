#!/usr/bin/env python3
"""Independent, read-only verification of the bounded public evidence package."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import stat
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from execution_common import canonical, document, require, sha
from blind_pcm import read_regular, unaliased
from public_evidence import (CELL_FILES, DIGEST, ENGINES, MAX_FILE_BYTES, MAX_TOTAL_BYTES, RECOGNIZERS,
                             STAGE_FILES, assemble_report, assess_host_inputs, digest, fields,
                             json_bytes, ledger, safe_name, same, validate_admission,
                             validate_asr_archive, validate_tts_archive, assess_resources, enforce_stage_order,
                             validate_host_stage_binding)


def _inventory(root):
    files, directories = {}, set()
    count = 0
    for current, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(current) / name
            require(not path.is_symlink() and stat.S_ISDIR(path.lstat().st_mode), 'unaliased public directory')
            directories.add(path.relative_to(root).as_posix())
        for name in names:
            path = Path(current) / name
            relative = path.relative_to(root).as_posix(); safe_name(relative)
            raw = read_regular(path, MAX_FILE_BYTES)
            count += len(raw)
            require(count <= MAX_TOTAL_BYTES, 'public total byte cap')
            files[relative] = raw
        require(len(files) <= 2048 and len(directories) <= 512, 'public inventory cap')
    return files, directories, count


def _receipt_metadata(rows, allowed):
    require(type(rows) is list and len(rows) <= 256, 'bounded unverified receipt metadata')
    names = []
    for row in rows:
        fields(row, ('file', 'sha256', 'bytes'), 'unverified metadata fields only')
        require(type(row['file']) is str and allowed(row['file']), 'fixed unverified receipt name')
        if row['sha256'] is None:
            require(row['bytes'] is None, 'unreadable receipt bytes unknown')
        else:
            digest(row['sha256'])
            require(type(row['bytes']) is int and 0 <= row['bytes'] <= MAX_FILE_BYTES, 'receipt bytes type/cap')
        names.append(row['file'])
    require(names == sorted(set(names)), 'unique sorted receipt metadata')


def _unknown_stage(value, allowed):
    fields(value, ('verified', 'status', 'attempts', 'reserved_no_retry', 'receipts'), 'stage projection fields')
    same({k: value[k] for k in ('verified', 'status', 'attempts', 'reserved_no_retry')},
         dict(verified=False, status='UNVERIFIED', attempts=None, reserved_no_retry=True), 'unknown stage reserve')
    _receipt_metadata(value['receipts'], allowed)
    return dict(value, outcomes=[], files={})


def verify(path, expected_admission=None):
    from plan_contract import validate_plan
    root = unaliased(path)
    require(root.is_dir(), 'complete public directory required')
    files, directories, total = _inventory(root)
    required = {'manifest.json', 'admission.json', 'plan.json', 'evidence-index.json', 'report.json', 'ledger.json'}
    require(required <= files.keys(), 'incomplete public artifact')
    manifest = json_bytes(files['manifest.json'])
    fields(manifest, ('schema', 'admission_sha256', 'plan_sha256', 'payload_bytes', 'files'), 'manifest fields')
    require(manifest['schema'] == 'kw1-public-package-v1' and type(manifest['files']) is dict, 'manifest schema')
    same(set(manifest['files']), set(files) - {'manifest.json'}, 'exact public inventory')
    for name, identity in manifest['files'].items():
        fields(identity, ('sha256', 'bytes'), 'file identity fields')
        same(identity, dict(sha256=sha(files[name]), bytes=len(files[name])), 'public file identity')
    same(manifest['payload_bytes'], total - len(files['manifest.json']), 'payload byte count')
    same(manifest['admission_sha256'], sha(files['admission.json']), 'manifest admission binding')
    same(manifest['plan_sha256'], sha(files['plan.json']), 'manifest plan binding')
    admission = validate_admission(json_bytes(files['admission.json']))
    if expected_admission is not None:
        same(admission, validate_admission(expected_admission), 'public admission identity mismatch')
    same(admission['source_files'].get('research/kw1-preset-supplement-v1/plan.json'), sha(files['plan.json']),
         'admission reviewed plan source hash')
    plan = json_bytes(files['plan.json']); validate_plan(plan)
    index = json_bytes(files['evidence-index.json'])
    fields(index, ('schema', 'host', 'stages', 'resources'), 'public evidence index fields')
    require(index['schema'] == 'kw1-public-evidence-index-v1', 'public evidence index schema')
    keys = {'tts/' + x for x in ENGINES} | {'asr/' + x for x in RECOGNIZERS}
    require(type(index['stages']) is dict and set(index['stages']) == keys, 'fixed stage inventory')
    allowed_files = set(required)
    allowed_directories = {'evidence'}
    stages = {}
    validated_bytes = {}
    evidence = root / 'evidence'
    for engine in ENGINES:
        key = 'tts/' + engine; projection = index['stages'][key]
        require(type(projection) is dict and type(projection.get('verified')) is bool, 'typed stage verified')
        ids = {c['slot_id'] for c in plan['slots'] if c['engine'] == engine}
        names = set(STAGE_FILES) | {sid + '.raw-float.wav' for sid in ids} | {f'cells/{sid}/{n}' for sid in ids for n in CELL_FILES}
        if projection['verified']:
            stage = validate_tts_archive(evidence / key / 'output', engine, plan)
            same(projection, {k: stage[k] for k in ('verified', 'status', 'attempts', 'reserved_no_retry', 'receipts')},
                 'TTS evidence projection')
            allowed_files.update('evidence/' + key + '/output/' + n for n in stage['files'])
            validated_bytes.update({'evidence/' + key + '/output/' + n: raw for n, raw in stage['files'].items()})
            allowed_directories.add('evidence/' + key + '/output')
            stages[key] = stage
        else:
            stages[key] = _unknown_stage(projection, lambda n: n in names)
    fields(index['host'], ('verified', 'receipts'), 'host projection fields')
    require(type(index['host']['verified']) is bool, 'typed host verification')
    if index['host']['verified']:
        host = assess_host_inputs(evidence, plan)
        require(host['verified'], 'public host evidence failed')
        same(index['host']['receipts'], [], 'verified host metadata empty')
        allowed_files.update('evidence/' + n for n in host['files'])
        validated_bytes.update({'evidence/' + n: raw for n, raw in host['files'].items()})
    else:
        _receipt_metadata(index['host']['receipts'], lambda n: n in ('identity-map.json', 'blind-input/job.json') or
                          re.fullmatch(r'blind-input/[0-9a-f]{64}\.pcm', n) is not None)
        host = dict(verified=False, mapping=[], files={}, receipts=index['host']['receipts'])
    for recognizer in RECOGNIZERS:
        key = 'asr/' + recognizer; projection = index['stages'][key]
        require(type(projection) is dict and type(projection.get('verified')) is bool, 'typed ASR verified')
        if projection['verified']:
            require(host['verified'] and len(host['mapping']) == 16, 'ASR host evidence parent')
            _, stage = validate_asr_archive(evidence / 'blind-input', evidence / key / 'output', recognizer)
            same(projection, {k: stage[k] for k in ('verified', 'status', 'attempts', 'reserved_no_retry', 'receipts')},
                 'ASR evidence projection')
            allowed_files.update('evidence/' + key + '/output/' + n for n in stage['files'])
            validated_bytes.update({'evidence/' + key + '/output/' + n: raw for n, raw in stage['files'].items()})
            allowed_directories.add('evidence/' + key + '/output')
            stages[key] = stage
        else:
            stages[key] = _unknown_stage(projection, lambda n: n in STAGE_FILES or
                re.fullmatch(r'cells/[0-9a-f]{64}/(?:started|decoder-evidence|completed|failed)\.json', n) is not None)
    fields(index['resources'], ('verified', 'receipts'), 'resource projection fields')
    require(type(index['resources']['verified']) is bool, 'typed resource verification')
    if index['resources']['verified']:
        resources = assess_resources(evidence)
        require(resources['verified'], 'public resource evidence failed')
        same(index['resources']['receipts'], [], 'verified resource metadata empty')
        allowed_files.update('evidence/' + n for n in resources['files'])
        validated_bytes.update({'evidence/' + n: raw for n, raw in resources['files'].items()})
    else:
        _receipt_metadata(index['resources']['receipts'], lambda n: n == 'backend/summary.json')
        resources = dict(verified=False, body=None, receipts=index['resources']['receipts'], files={})
    enforce_stage_order(stages)
    validate_host_stage_binding(host, stages)
    expected_report = assemble_report(plan, host, stages, resources)
    same(json_bytes(files['report.json']), expected_report, 'strict independently rebuilt report')
    same(json_bytes(files['ledger.json']), ledger(expected_report), 'strict independently rebuilt fixed ledger')
    same(set(files), allowed_files, 'no unreviewed public bodies')
    for name, raw in validated_bytes.items():
        same(raw, files[name], 'public file changed across independent validation')
    for name in allowed_files:
        allowed_directories.update(p.as_posix() for p in Path(name).parents if p.as_posix() != '.')
    for name in list(allowed_directories):
        allowed_directories.update(p.as_posix() for p in Path(name).parents if p.as_posix() != '.')
    require(directories <= allowed_directories, 'unexpected public directories')
    return dict(schema='kw1-public-verification-v1', verified=True, files=len(files), bytes=total,
                admission_sha256=manifest['admission_sha256'], source_sha=admission['source_sha'],
                report_status=expected_report['status'], ledger_slots=64,
                identity_authenticated=expected_admission is not None,
                verification_scope='structure_and_supplied_admission_identity' if expected_admission is not None else 'internal_consistency_only',
                model_execution_performed_by_verifier=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--expected-admission', type=Path, help='Trusted local admission receipt; otherwise internal consistency only')
    args = parser.parse_args(argv)
    print(canonical(verify(args.artifact, document(args.expected_admission) if args.expected_admission else None)).decode())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
