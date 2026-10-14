"""Read-only machine-candidate group coverage check; never allocates a split."""
import argparse,hashlib,json,zipfile
from pathlib import Path

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()

def assess(report,generation,plan,native_receipts):
    gen={r['sample_id']:r for r in generation}
    require(len(gen)==len(generation),'duplicate generation identity')
    design={r['cell_id']:r for cells in plan.values() for r in cells}
    native={r['cell_id']:r for receipt in native_receipts for r in receipt['outcomes']}
    require(len(native)==sum(len(r['outcomes']) for r in native_receipts),'duplicate native receipt identity')
    selected=[r for r in report['slots'] if r.get('three_model_training_candidate') is True]
    require(len({r['sample_id'] for r in selected})==len(selected),'duplicate selected identity')
    candidates=[]
    for row in selected:
        require(row.get('human_gold') is False and row.get('training_admitted') is False,'machine-only unadmitted labels')
        sid=row['sample_id'];g=gen[sid];p=design[sid];n=native[sid]
        require(n['status']=='generated' and n['attempts']==1 and g['status']=='success','single successful generation')
        require(n['engine']==p['engine']==g['engine']==row['engine'],'engine binding')
        require(n['text']==p['text']==g['target_text'] and n['speaker']==p['speaker'],'native target/speaker binding')
        require(n['raw_wav_sha256']==g['native_wav_sha256'],'native hash bridge')
        candidates.append({'sample_id':sid,'target_text':n['text'],'engine':n['engine'],'speaker':n['speaker'],'group_id':'preset-voice:'+n['speaker'],'native_wav_sha256':n['raw_wav_sha256'],'derived_wav_sha256':g['wav_sha256'],'human_gold':False,'training_admitted':False})
    coverage={word:{'candidate_count':sum(r['target_text']==word for r in candidates),'independent_voice_groups':sorted({r['group_id'] for r in candidates if r['target_text']==word}),'minimum_required_groups':3} for word in ('你好小窝','小窝小窝')}
    insufficient=[word for word,row in coverage.items() if len(row['independent_voice_groups'])<3]
    return {'schema':'next-ab-machine-data-coverage-v1','status':'BLOCKED_INSUFFICIENT_KW1_GROUPS' if '你好小窝' in insufficient else ('BLOCKED_INSUFFICIENT_GROUPS' if insufficient else 'COVERAGE_ONLY_NOT_ADMITTED'),'roles':['train','calibration','selection'],'grouping':'Conservatively group the same named preset voice across both TTS engines; engines/seeds are not independent speaker groups.','coverage':coverage,'candidates':candidates,'split_assignment_performed':False,'heldout_reuse_allowed':False,'labels_changed':False,'training_admitted':False,'matched_quality_ab_status':'NOT_RUN','numeric_fixture_status':'PASS','human_gold':False,'limitations':['Necessary group coverage check only, not sufficient dataset admission or a full split/quality audit.','Machine consensus does not establish human pronunciation truth.','No sample reuse or heldout borrowing can repair missing independent voice groups.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--remaining-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    module=a.repo/'research/tts-pilot-screening-v1';e=module/'evidence/asr-run-38573117922'
    report_raw=(e/'report.json').read_bytes();gen_raw=(e/'generation-receipts.json').read_bytes();plan_raw=(module/'expected-plan.json').read_bytes()
    binding=json.loads((module/'activation-template/activation.json').read_bytes());native=[];sources=[]
    for engine in ('qwen06','qwen17'):
        archive=a.remaining_dir/(engine+'.zip');raw=archive.read_bytes();entry=next(r for r in binding['input_artifacts'] if r['engine']==engine and r['phase']=='remaining')
        require(sha(raw)==entry['zip_sha256'],'frozen remaining archive identity')
        member=engine+'-receipt.json'
        with zipfile.ZipFile(archive) as z:body=z.read(member)
        require(sha(body)==entry['members'][member]['sha256'],'frozen native receipt identity')
        native.append(json.loads(body));sources.append({'engine':engine,'run_id':entry['run_id'],'artifact_id':entry['artifact_id'],'zip_sha256':sha(raw),'member':member,'member_sha256':sha(body)})
    result=assess(json.loads(report_raw),json.loads(gen_raw),json.loads(plan_raw),native)
    numeric_path='research/effect-data-model-v2/evidence/numeric-run-38552529009/numeric-receipt.json'
    numeric_raw=(a.repo/numeric_path).read_bytes();numeric=json.loads(numeric_raw)
    require(numeric['status']=='PASS' and numeric['evidence_scope']=='artificial-pcm-numeric-contract-only' and numeric['training_updates_per_arm']==2 and numeric['quality_metrics_claimed'] is False,'actual numeric fixture scope')
    result['sources']={'numeric_fixture_receipt':{'path':numeric_path,'sha256':sha(numeric_raw)},'native_generation_receipts':sources,'asr_report_sha256':sha(report_raw),'generation_projection_sha256':sha(gen_raw),'frozen_plan_sha256':sha(plan_raw),'speaker_binding_note':'Public generation projection omits speaker; actual original native receipts supply it, joined by sample ID/native hash and crosschecked against the frozen plan.'}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
