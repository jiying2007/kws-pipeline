#!/usr/bin/env python3
"""Consume pinned kws-data exports for observed, whole-clip C-runner readback only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from corpus_identity import inspect_pcm16_wav


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_export(path, data_root, *, expected_receipt_sha256, expected_commit,
                expected_catalog_sha256, expected_datasets):
    """Check transport/pins and consumed bytes; kws-data owns archive validation."""
    require(sha(path) == expected_receipt_sha256, 'export receipt SHA mismatch')
    receipt = json.loads(path.read_text())
    require(type(receipt.get('schema_version')) is int and receipt['schema_version'] == 1 and receipt.get('evidence_class') ==
            'kws-data-native-consumer-receipt-v1', 'unexpected native export schema')
    require(receipt.get('data_repository_commit') == expected_commit and
            receipt.get('catalog_sha256') == expected_catalog_sha256, 'data identity mismatch')
    require(receipt.get('qualification_allowed') is False and receipt.get('labels') ==
            'native-text-no-token-or-event-alignment', 'unsupported evidence boundary')
    head = subprocess.check_output(['git', '-C', str(data_root), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == expected_commit and sha(data_root/'catalog.json') == expected_catalog_sha256,
            'data checkout identity mismatch')
    dirty = subprocess.check_output(['git', '-C', str(data_root), 'status', '--porcelain', '--untracked-files=all'], text=True)
    require(not dirty, 'consumer requires clean data checkout')
    datasets = receipt['datasets']
    ids = [d['dataset_id'] for d in datasets]
    require(len(ids) == len(set(ids)) and set(ids) == set(expected_datasets), 'dataset selection mismatch')
    rows, seen = [], set()
    for dataset in datasets:
        require(dataset['content_id'] == 'sha256:'+digest(dataset['identity']), 'content identity mismatch')
        require(dataset['identity']['dataset_id'] == dataset['dataset_id'], 'dataset identity mismatch')
        require(dataset.get('qualification_allowed') is False, 'dataset qualification forbidden')
        require(digest(dataset['recordings']) == dataset['recordings_sha256'], 'export row digest mismatch')
        for native in dataset['recordings']:
            row = dict(native)
            require(not ({'expected', 'tokens', 'token_ids', 'start_s', 'end_s',
                          'match_not_before_s', 'start_sample', 'end_sample'} & set(row)),
                    'token/event annotations are not supported')
            require(row['recording'] not in seen, 'duplicate recording')
            seen.add(row['recording'])
            relative = pathlib.Path(row['path'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe export path')
            audio = (data_root/relative).resolve(strict=True)
            require(audio.is_relative_to(data_root.resolve()), 'audio escapes data root')
            measured = inspect_pcm16_wav(audio)
            require(all(row[k] == measured[k] for k in ('file_sha256','pcm_sha256','frames','duration_s')),
                    'consumed WAV identity mismatch')
            require(row.get('observed_development') is True and row['split'] in
                    {'train','development_a','development_b'}, 'unsupported native role')
            require(row['review_method'] in {'human','asr'}, 'unknown review method')
            require(row['kind'] in {'positive','confusable'}, 'unsupported weak clip label')
            keyword = row['keyword_id']
            require((row['kind'] == 'positive' and type(keyword) is int and 0 <= keyword <= 0xFFFFFFFF) or
                    (row['kind'] == 'confusable' and keyword is None), 'invalid clip keyword label')
            row.update(dataset_id=dataset['dataset_id'], dataset_content_id=dataset['content_id'],
                       evidence_class='observed-clip-execution-input-v1', event_annotations_available=False)
            rows.append(row)
    require(bool(rows), 'empty exported corpus')
    return receipt, rows


def summarize(rows, detections):
    by_id = {r['recording']: [] for r in rows}
    require(len(by_id) == len(rows), 'duplicate recording')
    native = {r['recording']: r for r in rows}
    for event in detections:
        recording = event.get('recording')
        require(recording in by_id, 'detection for unknown recording')
        require(type(event.get('keyword_id')) is int and 0 <= event['keyword_id'] <= 0xFFFFFFFF, 'invalid detected keyword')
        for name in ('time_s', 'confidence'):
            v = event.get(name)
            require(type(v) in (int,float) and math.isfinite(v), 'invalid detection '+name)
        require(0 <= event['confidence'] <= 1 and 0 <= event['time_s'] <= native[recording]['duration_s'],
                'detection outside clip/confidence bounds')
        by_id[recording].append(event)
    details, groups = [], {}
    for row in rows:
        events = by_id[row['recording']]
        positive = row['kind'] == 'positive'
        target = sum(e['keyword_id'] == row['keyword_id'] for e in events) if positive else 0
        wrong = len(events)-target if positive else 0
        detail = {**row, 'detections': events, 'event_count': len(events),
                  'target_hit': bool(target) if positive else None,
                  'target_miss': not bool(target) if positive else None,
                  'target_event_count': target if positive else None,
                  'additional_target_events': max(0,target-1) if positive else None,
                  'wrong_keyword_events': wrong if positive else None,
                  'confusable_event_count': len(events) if not positive else None}
        details.append(detail)
        key = (row['dataset_id'],row['split'],row['review_method'],row['keyword_id'])
        group = groups.setdefault(key, dict(dataset_id=key[0], split=key[1], review_method=key[2], keyword_id=key[3],
                    recordings=0, positives=0, target_hits=0,target_misses=0,additional_target_events=0,
                    wrong_keyword_events=0,confusables=0,confusable_clips_with_events=0,confusable_event_count=0))
        group['recordings'] += 1
        if positive:
            group['positives'] += 1; group['target_hits'] += bool(target); group['target_misses'] += not bool(target)
            group['additional_target_events'] += max(0,target-1); group['wrong_keyword_events'] += wrong
        else:
            group['confusables'] += 1; group['confusable_clips_with_events'] += bool(events)
            group['confusable_event_count'] += len(events)
    return {'schema_version':1,'evidence_class':'observed-weak-clip-readback-v1',
            'qualification_allowed':False,'event_annotations_available':False,
            'execution':'reset C engine for each unmodified WAV; original clip duration only',
            'limits':['Target hit means target keyword appeared somewhere in the positive clip, not event alignment.',
                      'No endpoint latency, continuous FAR, fresh holdout, or product/generalization claim.',
                      'Human and ASR evidence and all native development roles remain separate.'],
            'groups':list(groups.values()),'recordings':details}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('receipt','data-root','runner','model','keywords','output-root','runner-build-receipt'):
        parser.add_argument('--'+name, type=pathlib.Path, required=True)
    for name in ('expected-receipt-sha256','expected-data-commit','expected-catalog-sha256','expected-model-sha256','expected-keywords-sha256','expected-runner-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--dataset',action='append',required=True)
    args=parser.parse_args()
    receipt,rows=load_export(args.receipt,args.data_root,expected_receipt_sha256=args.expected_receipt_sha256,
                    expected_commit=args.expected_data_commit,expected_catalog_sha256=args.expected_catalog_sha256,
                    expected_datasets=args.dataset)
    for path, expected in ((args.model,args.expected_model_sha256),(args.keywords,args.expected_keywords_sha256),(args.runner,args.expected_runner_sha256)):
        require(sha(path) == expected, 'execution asset SHA mismatch')
    build_receipt_sha256 = sha(args.runner_build_receipt)
    require(not args.output_root.exists(),'output root already exists; refusing overwrite')
    args.output_root.mkdir(parents=True)
    inputs=args.output_root/'execution-inputs.jsonl'
    inputs.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for r in rows))
    events=args.output_root/'detections.jsonl'; provenance=args.output_root/'run-provenance.json'
    subprocess.run([sys.executable,str(ROOT/'eval/run_corpus.py'),'--runner',str(args.runner.resolve()),
                    '--model',str(args.model.resolve()),'--keywords',str(args.keywords.resolve()),
                    '--references',str(inputs),'--audio-root',str(args.data_root.resolve()),
                    '--detections',str(events),'--provenance',str(provenance)],check=True)
    detections=[json.loads(line) for line in events.read_text().splitlines() if line.strip()]
    execution=json.loads(provenance.read_text())
    require(execution['model_sha256'] == args.expected_model_sha256 and execution['keyword_pack_sha256'] == args.expected_keywords_sha256 and execution['runner_sha256'] == args.expected_runner_sha256, 'execution assets changed during run')
    measured={r['recording']:r for r in execution['audio_files']}
    require(len(measured) == len(rows), 'execution coverage mismatch')
    for row in rows:
        require(row['recording'] in measured and all(row[k] == measured[row['recording']][k] for k in ('file_sha256','pcm_sha256','frames')), 'execution audio changed after admission')
    report=summarize(rows,detections)
    report['data_export_identity']={k:receipt[k] for k in ('data_repository_commit','catalog_sha256')}
    report['data_export_identity'].update(receipt_sha256=args.expected_receipt_sha256,
                                          datasets=[{k:d[k] for k in ('dataset_id','content_id','identity','recordings_sha256')} for d in receipt['datasets']])
    report['execution_provenance_sha256']=sha(provenance)
    require(sha(args.runner_build_receipt) == build_receipt_sha256, 'build receipt changed during run')
    report['runner_build_receipt_sha256']=build_receipt_sha256
    report['consumer_sha256']=sha(pathlib.Path(__file__))
    report['run_corpus_sha256']=sha(ROOT/'eval/run_corpus.py')
    (args.output_root/'clip-readback.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'recordings':len(rows),'detections':len(detections),'report':str(args.output_root/'clip-readback.json')}))


if __name__=='__main__':
    try: main()
    except (ValueError,KeyError,TypeError,OSError,subprocess.CalledProcessError) as exc:
        print('readback failed: '+str(exc),file=sys.stderr)
        raise SystemExit(2)
