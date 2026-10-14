#!/usr/bin/env python3
"""Single frozen C development control on exactly 13 reviewed machine labels.

This is neither human-gold FRR nor a new AliMeeting exposure. No training,
parameter overrides, padding, trimming, automatic retries or sample selection.
"""
import argparse
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import time
import unicodedata
import wave
import zipfile

BASE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('frozen_diagnostic', BASE/'diagnostic_runner.py')
diagnostic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diagnostic)
check, sha, save = diagnostic.check, diagnostic.sha, diagnostic.save
EXPECTED = {
    'public.zip':'f8bf70a0539d6ca7bd11dfed9229a087c0c7d27b943550f73136cd765f54ef0c',
    'report.json':'5c4c4c909d9e3b8a02e5d39b0dfbafae1630d13028227d611aef8379a50538f4',
    'results.json':'3b53948ddb3a4160d159fdb64efe75579858e4909a390b0ad1a59eeb5e29ad05',
    'generation-receipts.json':'6662d0b13a9aaea64c70db98428b71b4f6c4d2d144d4c5fe1382b83a9159dac2',
}
ASR_AUDIT_SHA='a29ad37a9ce989602bb668bad0f7208ce4cd8f35724428ecfdcf4105698c2103'
ALI_SUMMARY_SHA='e34600b0cc8816e3e23dda888782c8eabb831fb9bea658518bb41a12f51f1c06'
TARGETS={'你好小窝':1,'小窝小窝':2}
FAMILIES={'qwen06','sensevoice','whisper'}


def normalize(text):
    check(type(text) is str, 'ASR text must be string')
    # Identical normalization to the reviewed pilot screen. No homophone rewrite.
    return ''.join(c for c in unicodedata.normalize('NFKC',text)
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def unique(rows, key):
    result={}
    for row in rows:
        identity=row[key]
        check(identity not in result, 'duplicate '+key)
        result[identity]=row
    return result


def select(report, generation, results):
    slots=unique(report['slots'],'sample_id')
    generated=unique(generation,'sample_id')
    asr={}
    for row in results:
        key=(row['sample_id'],row['family'])
        check(key not in asr, 'duplicate ASR sample/family')
        asr[key]=row
    selected=[]
    for sid, slot in slots.items():
        if slot.get('three_model_training_candidate') is not True:
            continue
        check(re.fullmatch(r'tqp-qwen(?:06|17)-\d{2}',sid) is not None, 'unsafe sample ID')
        check(slot.get('human_gold') is False and slot.get('training_admitted') is False, 'unexpected label admission')
        check(slot.get('target_vote_count')==3 and slot.get('audio_quality_pass') is True, 'candidate screening mismatch')
        gen=generated[sid]; target=gen['target_text']
        check(target in TARGETS and gen['status']=='success', 'unsupported target/generation')
        for family in FAMILIES:
            row=asr[(sid,family)]
            check(row['status']=='success' and row['completeness']=='complete' and row['quality_flags']==[] and row['attempt']==1,
                  'ASR evidence incomplete')
            check(normalize(row['raw_text'])==target, 'ASR target differs')
        selected.append({'sample_id':sid,'target_text':target,'keyword_id':TARGETS[target],
                         'wav_sha256':gen['wav_sha256'],'native_wav_sha256':gen['native_wav_sha256'],
                         'tts_engine':gen['engine'],'label_source':'machine_three_asr',
                         'development_only':True,'human_gold':False,'training_admitted':False})
    check(len(selected)==13,'exactly 13 reviewed candidates required')
    check({t:sum(x['target_text']==t for x in selected) for t in TARGETS}=={'你好小窝':2,'小窝小窝':11},
          'fixed per-keyword denominators changed')
    return selected


def inspect_wav(body):
    check(len(body)<=10_000_000,'WAV size exceeds control bound')
    with wave.open(io.BytesIO(body),'rb') as wav:
        check((wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getcomptype())==(1,2,16000,'NONE'),
              'control WAV must be existing mono PCM16 16 kHz')
        count=wav.getnframes(); pcm=wav.readframes(count)
        check(400<=count<=16000*60 and len(pcm)==count*2,'WAV sample bounds/truncation')
        return count,hashlib.sha256(pcm).hexdigest()


def coverage(rows):
    answer={}
    for target,kid in TARGETS.items():
        group=[r for r in rows if r['target_text']==target]
        answer[target]={'keyword_id':kid,'input_clips':len(group),
            'clips_with_matching_trigger':sum(r['matching_event_count']>0 for r in group),
            'clips_with_any_trigger':sum(r['event_count']>0 for r in group),
            'clips_with_wrong_keyword_trigger':sum(r['wrong_keyword_event_count']>0 for r in group),
            'matching_events':sum(r['matching_event_count'] for r in group),
            'wrong_keyword_events':sum(r['wrong_keyword_event_count'] for r in group),
            'trigger_coverage':sum(r['matching_event_count']>0 for r in group)/len(group) if group else None}
    return answer


def prepare(result_dir,audit_path):
    result_dir=Path(result_dir).resolve();audit_path=Path(audit_path).resolve()
    checks={str(result_dir/name):digest for name,digest in EXPECTED.items()}
    checks[str(audit_path)]=ASR_AUDIT_SHA
    checks[str(BASE/'diagnostic-build/build-receipt.json')]=diagnostic.BUILD_RECEIPT_SHA
    checks[str(BASE/'diagnostic-run-p4-11b93abe/summary.json')]=ALI_SUMMARY_SHA
    checks[str(Path(__file__).resolve())]=sha(__file__)
    checks[str(BASE/'diagnostic_runner.py')]=sha(BASE/'diagnostic_runner.py')
    diagnostic.verify(checks)
    audit=json.loads(audit_path.read_text())
    check(audit['status']=='pass' and audit['archive_sha256']==EXPECTED['public.zip'] and audit['three_asr_candidates']==13,
          'wrong independent review')
    report=json.loads((result_dir/'report.json').read_text())
    gen=json.loads((result_dir/'generation-receipts.json').read_text())
    results=json.loads((result_dir/'results.json').read_text())
    rows=select(report,gen,results)
    receipt=json.loads((BASE/'diagnostic-build/build-receipt.json').read_text())
    check(receipt['source_revision']==diagnostic.SOURCE_COMMIT and receipt['source_tree']==diagnostic.SOURCE_TREE,
          'wrong frozen source')
    for key,digest in (('runner',diagnostic.RUNNER_SHA),('model',diagnostic.MODEL_SHA),('keyword_pack',diagnostic.PACK_SHA)):
        check(receipt[key]['sha256']==digest,'receipt identity mismatch')
        checks[receipt[key]['path']]=digest
    for source in receipt['source_inputs']: checks[source['path']]=source['sha256']
    diagnostic.verify(checks)
    pack=Path(receipt['keyword_pack']['path']).read_bytes()
    thresholds={str(struct.unpack_from('<I',pack,24+48*k)[0]):struct.unpack_from('<f',pack,28+48*k)[0] for k in range(2)}
    bodies={}
    with zipfile.ZipFile(result_dir/'public.zip') as archive:
        members=archive.infolist(); names=[i.filename for i in members]
        check(len(names)==35 and len(set(names))==35,'archive member count/duplicates changed')
        check(sum(i.file_size for i in members)<4_000_000,'archive expanded-size bound')
        for item in members:
            p=PurePosixPath(item.filename)
            check(not p.is_absolute() and '..' not in p.parts and not item.is_dir() and ((item.external_attr>>16)&0o170000)!=0o120000,
                  'unsafe archive member')
        for name in ('report.json','results.json','generation-receipts.json'):
            check(hashlib.sha256(archive.read(name)).hexdigest()==EXPECTED[name],'archive receipt differs')
        for row in rows:
            body=archive.read('audio/'+row['sample_id']+'.wav')
            check(hashlib.sha256(body).hexdigest()==row['wav_sha256'],'selected WAV hash mismatch')
            count,pcm_sha=inspect_wav(body)
            row.update(sample_count=count,duration_s=count/16000,pcm_sha256=pcm_sha)
            bodies[row['sample_id']]=body
    check(len({r['pcm_sha256'] for r in rows})==13,'duplicate selected PCM')
    return rows,bodies,receipt,checks,thresholds


def run(result_dir,audit_path,out,execute=False):
    check(execute,'requires explicit --execute')
    out=Path(out).resolve()
    check(not out.exists(),'output exists; will not overwrite, resume, or replay')
    rows,bodies,receipt,checks,thresholds=prepare(result_dir,audit_path)
    out.mkdir(parents=False,exist_ok=False);(out/'audio').mkdir()
    for row in rows:
        wav=out/'audio'/(row['sample_id']+'.wav')
        with wav.open('xb') as stream: stream.write(bodies[row['sample_id']])
        checks[str(wav)]=row['wav_sha256']
    save(out/'inputs.json',{'schema':'machine-three-asr-positive-control-inputs-v1','selected':rows,
         'sha256_by_path':checks,'selection':'exact report slots where three_model_training_candidate is true',
         'thresholds_from_frozen_pack':thresholds,'block_samples':160,'padding_samples':0,
         'reset_policy':'once at each real generated-clip boundary; continuous within each complete clip',
         'label_source':'machine_three_asr','development_only':True,'human_gold':False,
         'human_frr_available':False,'far_available':False,'training_admitted':False})
    started=time.monotonic(); started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    completed=[];all_events=[]
    try:
        for row in rows:
            diagnostic.verify(checks)
            sid=row['sample_id'];wav=out/'audio'/(sid+'.wav')
            stats_path=out/(sid+'.stats.json');stdout=out/(sid+'.events.jsonl');stderr=out/(sid+'.stderr.log')
            argv=[receipt['runner']['path'],receipt['model']['path'],receipt['keyword_pack']['path'],str(wav),sid,
                  '--block-samples','160','--stats-json',str(stats_path)]
            save(out/(sid+'.invocation.json'),{'argv':argv,'input':row})
            with stdout.open('xb') as so,stderr.open('xb') as se:
                process=subprocess.run(argv,stdout=so,stderr=se,timeout=60,check=False)
            check(process.returncode==0,f'{sid}: nonzero exit; no retry')
            events=[json.loads(line) for line in stdout.read_text().splitlines() if line.strip()]
            stats=json.loads(stats_path.read_text())
            diagnostic.validate_outputs(sid,row['sample_count'],events,stats)
            completed.append({**row,'processed_frames':stats['processed_frames'],'speech_frames':stats['speech_frames'],
                 'blank_top1_frames':stats['blank_top1_frames'],'event_count':len(events),
                 'matching_event_count':sum(e['keyword_id']==row['keyword_id'] for e in events),
                 'wrong_keyword_event_count':sum(e['keyword_id']!=row['keyword_id'] for e in events),
                 'outputs':{p.name:sha(p) for p in (stats_path,stdout,stderr)},'exit_code':process.returncode})
            save(out/(sid+'.completion.json'),completed[-1]);all_events.extend(events)
            print(json.dumps({'completed':len(completed),'sample_id':sid,'events':len(events)},ensure_ascii=False),flush=True)
        diagnostic.verify(checks)
        with (out/'events.jsonl').open('x',encoding='utf-8') as stream:
            for event in all_events:stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        summary={'schema':'machine-three-asr-frozen-positive-control-v1','label_source':'machine_three_asr',
          'development_only':True,'human_gold':False,'human_frr_available':False,'far_available':False,
          'generalization_claim_available':False,'training_admitted':False,'training_runs':0,
          'parameter_changes':False,'shipping_approved':False,'recalibration_required':True,
          'source_revision':diagnostic.SOURCE_COMMIT,'source_tree':diagnostic.SOURCE_TREE,
          'runner_sha256':diagnostic.RUNNER_SHA,'model_sha256':diagnostic.MODEL_SHA,'keyword_pack_sha256':diagnostic.PACK_SHA,
          'build_receipt_sha256':diagnostic.BUILD_RECEIPT_SHA,'asr_report_sha256':EXPECTED['report.json'],
          'asr_artifact_sha256':EXPECTED['public.zip'],'independent_asr_audit_sha256':ASR_AUDIT_SHA,
          'original_alimeeting_summary_sha256_unchanged':ALI_SUMMARY_SHA,'all_clips_complete':True,
          'clip_count':len(completed),'input_samples':sum(r['sample_count'] for r in completed),
          'input_seconds':sum(r['sample_count'] for r in completed)/16000,
          'trigger_event_count':len(all_events),'per_target':coverage(completed),'clips':completed,
          'events_sha256':sha(out/'events.jsonl'),'started_utc':started_utc,'wall_seconds':time.monotonic()-started,
          'interpretation':'Clip-level response control on newly machine-labelled synthetic development speech. '
                           'Absence of events on the separate AliMeeting stream cannot establish low false accepts or useful sensitivity.'}
        save(out/'summary.json',summary)
        return summary
    except BaseException as error:
        save(out/'failure.json',{'error':str(error),'completed':completed,'automatic_retry':False,'all_clips_complete':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir',type=Path,required=True)
    parser.add_argument('--asr-audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();run(args.result_dir,args.asr_audit,args.out,args.execute)

if __name__=='__main__':main()
