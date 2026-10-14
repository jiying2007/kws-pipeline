#!/usr/bin/env python3
"""One-shot, source-bound diagnostic trigger replay. No negative labels or FAR."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import wave

SOURCE_COMMIT = '11b93abe92427c3ece2a01f9a497cdb74bab0062'
SOURCE_TREE = 'b8ae29e82097c38e92dbb1bb1cd4db10ade6cdc6'
AUDIT_SHA = '95b2f7b6abd409fe0a17e597f49a01aa0bbd97f86f4ec807762c609fd5a0aca3'
MODEL_SHA = 'ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402'
PACK_SHA = '370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723'
EXPECTED_SAMPLES = 242211440
RUNNER_SHA = '53a5dcb50c599d88bb72b2cfbaf6ff5d6e7a379ecef3011f28f5820261d155a9'
BUILD_RECEIPT_SHA = '78b2988e60a4bf6b875934222fd5e732633ca34de64e470c433e74289f57ac8a'


def check(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def frame_count(samples):
    return max(0, 1+(samples-400)//320)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_outputs(recording, samples, events, stats):
    check(isinstance(stats,dict), 'stats must be an object')
    for key, expected in {'processed_samples':samples, 'processed_frames':frame_count(samples),
                          'detections':len(events), 'discontinuities':0,
                          'lost_samples':0, 'external_vad_frames':0}.items():
        check(integer(stats.get(key)) and stats[key]==expected, 'stats mismatch: '+key)
    previous=-1.0
    for event in events:
        check(isinstance(event,dict) and event.get('recording')==recording, 'event recording mismatch')
        check(integer(event.get('keyword_id')) and event['keyword_id'] in (1,2), 'invalid keyword ID')
        for key in ('time_s','confidence'):
            value=event.get(key)
            check(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value), 'nonfinite/invalid event '+key)
        check(previous<=event['time_s']<=samples/16000+0.0000005 and event['time_s']>=0, 'event time out of range/order')
        check(0<=event['confidence']<=1, 'confidence outside [0,1]')
        check('timing_contract' not in event, 'unexpected AFE metadata in plain-input replay')
        previous=event['time_s']


def prepare(audit_path, receipt_path):
    audit_path, receipt_path=Path(audit_path).resolve(), Path(receipt_path).resolve()
    check(sha(audit_path)==AUDIT_SHA, 'audit hash differs from fixed intake')
    audit=json.loads(audit_path.read_text())
    check(sha(receipt_path)==BUILD_RECEIPT_SHA, 'build receipt differs from reviewed frozen build')
    receipt=json.loads(receipt_path.read_text())
    check(receipt['source_revision']==SOURCE_COMMIT and receipt['source_tree']==SOURCE_TREE, 'wrong frozen source')
    for key, digest in (('model',MODEL_SHA),('keyword_pack',PACK_SHA)):
        check(receipt[key]['sha256']==digest and sha(receipt[key]['path'])==digest, 'wrong '+key)
    check(receipt['runner']['sha256']==RUNNER_SHA and sha(receipt['runner']['path'])==RUNNER_SHA, 'runner hash mismatch')
    rows=audit['rows']
    check(len(rows)==8 and sum(r['sample_count'] for r in rows)==EXPECTED_SAMPLES, 'wrong fixed corpus')
    check(audit['accepted_negative_duration_s']==0, 'unexpected negative admission')
    checks={str(audit_path):AUDIT_SHA, str(receipt_path):sha(receipt_path), str(Path(__file__).resolve()):sha(__file__)}
    for key in ('runner','model','keyword_pack'):
        checks[str(Path(receipt[key]['path']).resolve())]=receipt[key]['sha256']
    for source in receipt['source_inputs']:
        check(sha(source['path'])==source['sha256'], 'source input changed since build')
        checks[source['path']]=source['sha256']
    root=audit_path.parent
    ids=set(); pcms=set()
    for row in rows:
        check(row['recording'] not in ids and row['pcm_sha256'] not in pcms, 'duplicate session/input')
        ids.add(row['recording']);pcms.add(row['pcm_sha256'])
        wav=(root/row['wav']).resolve()
        check(wav.is_relative_to(root) and not (root/row['wav']).is_symlink(), 'unsafe WAV path')
        check(sha(wav)==row['wav_sha256'], 'WAV hash mismatch')
        with wave.open(str(wav),'rb') as source:
            check(source.getnchannels()==1 and source.getsampwidth()==2 and source.getframerate()==16000 and source.getcomptype()=='NONE', 'wrong mono WAV format')
            check(source.getnframes()==row['sample_count'], 'sample count mismatch')
        checks[str(wav)]=row['wav_sha256']
        textgrid=root/'textgrid'/(row['session_id']+'.TextGrid')
        check(sha(textgrid)==row['textgrid']['sha256'], 'TextGrid hash mismatch')
        checks[str(textgrid.resolve())]=row['textgrid']['sha256']
    return audit, receipt, checks


def verify(checks):
    for path, digest in checks.items():
        check(sha(path)==digest, 'input identity changed: '+path)


def run(audit_path, receipt_path, out, execute=False):
    check(execute, 'diagnostic execution requires --execute; no model was run')
    out=Path(out)
    check(not out.exists(), 'output directory exists; replay will not overwrite or resume')
    audit, receipt, checks=prepare(audit_path,receipt_path)
    out.mkdir(parents=False,exist_ok=False)
    started=time.monotonic()
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    records=[]; all_events=[]
    save(out/'inputs.json',{'schema':'alimeeting-diagnostic-inputs-v1','sha256_by_path':checks,
                          'source_revision':SOURCE_COMMIT,'source_tree':SOURCE_TREE,
                          'planned_samples':EXPECTED_SAMPLES,'planned_sessions':8,
                          'block_samples':160,'state_reset':'only at real meeting boundaries',
                          'accepted_negative_duration_s':0,'confirmed_far_available':False})
    try:
        for row in audit['rows']:
            verify({path:digest for path,digest in checks.items() if path not in [str((Path(audit_path).resolve().parent/r['wav']).resolve()) for r in audit['rows'] if r is not row]})
            rec=row['recording']; stats_path=out/(rec+'.stats.json')
            stdout_path=out/(rec+'.detections.jsonl'); stderr_path=out/(rec+'.stderr.log')
            wav=(Path(audit_path).resolve().parent/row['wav']).resolve()
            argv=[receipt['runner']['path'],receipt['model']['path'],receipt['keyword_pack']['path'],str(wav),rec,
                  '--block-samples','160','--stats-json',str(stats_path.resolve())]
            remaining=3600-(time.monotonic()-started)
            check(remaining>0, 'one-hour total diagnostic budget exceeded')
            t0=time.monotonic()
            save(out/(rec+'.invocation.json'),{'argv':argv,'input_samples':row['sample_count'],
                                              'source_wav_sha256':row['source_wav_sha256'],
                                              'wav_sha256':row['wav_sha256'],'pcm_sha256':row['pcm_sha256'],
                                              'textgrid_sha256':row['textgrid']['sha256']})
            with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
                result=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=min(600,remaining),check=False)
            wall=time.monotonic()-t0
            check(result.returncode==0, f'{rec}: runner exit {result.returncode}; no automatic retry')
            events=[json.loads(line) for line in stdout_path.read_text().splitlines() if line.strip()]
            stats=json.loads(stats_path.read_text())
            validate_outputs(rec,row['sample_count'],events,stats)
            check(sha(wav)==row['wav_sha256'], 'WAV changed during processing')
            records.append({'recording':rec,'sample_count':row['sample_count'],'duration_s':row['duration_s'],
                            'processed_frames':stats['processed_frames'],'event_count':len(events),
                            'events_per_input_hour':len(events)/(row['duration_s']/3600),
                            'wall_seconds':wall,'exit_code':0,
                            'outputs':{p.name:sha(p) for p in (stdout_path,stderr_path,stats_path)},
                            'audio_after_textgrid_end_s':row['audio_after_textgrid_end_s']})
            all_events.extend(events)
            save(out/(rec+'.completion.json'),records[-1])
            print(json.dumps({'completed':len(records),'recording':rec,'events':len(events),'input_seconds':row['duration_s']},ensure_ascii=False),flush=True)
        verify(checks)
        with (out/'events.jsonl').open('x',encoding='utf-8') as stream:
            for event in all_events: stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        hours=EXPECTED_SAMPLES/16000/3600
        summary={'schema':'alimeeting-frozen-c-diagnostic-v1','evidence_class':'unadmitted-corpus-trigger-diagnostics',
                 'source_revision':SOURCE_COMMIT,'source_tree':SOURCE_TREE,'build_receipt_sha256':sha(receipt_path),
                 'audit_sha256':AUDIT_SHA,'all_sessions_complete':True,'session_count':8,
                 'input_samples':EXPECTED_SAMPLES,'input_seconds':EXPECTED_SAMPLES/16000,'input_hours':hours,
                 'trigger_event_count':len(all_events),'trigger_events_per_input_hour':len(all_events)/hours,
                 'per_keyword_counts':{str(k):sum(e['keyword_id']==k for e in all_events) for k in (1,2)},
                 'events_sha256':sha(out/'events.jsonl'),'meetings':records,'accepted_negative_duration_s':0,
                 'confirmed_far_available':False,'frr_available':False,'latency_available':False,
                 'training_runs':0,'parameter_changes':False,'shipping_approved':False,'recalibration_required':True,
                 'started_utc':started_utc,'wall_seconds':time.monotonic()-started,
                 'annotation_note':'Official transcript text scan only. Unknown/ambiguous target status is not relabelled negative.'}
        save(out/'summary.json',summary)
        return summary
    except BaseException as exc:
        save(out/'failure.json',{'schema':'alimeeting-diagnostic-failed-v1','error':str(exc),
              'completed_meetings':records,'all_sessions_complete':False,'accepted_negative_duration_s':0,
              'confirmed_far_available':False,'automatic_retry':False})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--build-receipt',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    run(a.audit,a.build_receipt,a.out,a.execute)


if __name__=='__main__':
    main()
