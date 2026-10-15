#!/usr/bin/env python3
"""Read-only, offline text consistency verification. No model execution or network."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

SOURCE_SHA = '1ef2b7e9b9d236c13f33b1bdee78d8ae754dc8fa'
ZIP_SHA = 'e9cb187fbc0c0233f95e95efa7ca064359d7b00e3e6205041c7da06b3dedc5fe'
RAW_HASHES = {'report.json': '97f0e461d345a22fed6896b6f8d498f941101c21deccfa72f58957337054b263', 'ledger.json': '4f210056be3e5d928cd6dabaae1f0966c3cc579043d1f7008def9ee25ecde31f', 'manifest.json': '32aa7a8cff84bbb06d57d3e4c05acfab8885435861eadbd1614a81a71a14416e'}  # Filled with original artifact byte identities below.

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def load(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'duplicate JSON key: ' + k)
            result[k] = v
        return result
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)

def derive(report, ledger):
    slots = report['slots']
    require(len(slots) == 16 and len({s['slot_id'] for s in slots}) == 16, 'slot coverage')
    require({s['slot_id'] for s in slots} == {'kw1s-%02d' % i for i in range(16)}, 'slot IDs')
    calls = ledger['slots']
    require(len(calls) == 64 and ledger['denominator'] == 64 and ledger['retry'] is False, 'ledger denominator/retry')
    require(all(c['attempts'] == 1 and c['reserved_no_retry'] for c in calls), 'attempt count')
    expected = {(s['slot_id'], 'tts', s['engine']) for s in slots}
    expected |= {(s['slot_id'], 'asr', r) for s in slots for r in ['qwen_asr', 'sensevoice', 'whisper_small']}
    require({(c['slot_id'], c['kind'], c['stage']) for c in calls} == expected, 'ledger identity')
    require(all(c['status'] == ('completed' if c['kind'] == 'tts' else 'success') for c in calls), 'call status')
    require(all(len(s['recognitions']) == 3 and {r['recognizer'] for r in s['recognitions']} == {'qwen_asr','sensevoice','whisper_small'} for s in slots), 'recognizer coverage')
    require(all(r['status'] == 'success' and r['attempts'] == 1 for s in slots for r in s['recognitions']), 'ASR completion')
    candidates = [s['slot_id'] for s in slots if s['machine_train_only_candidate']]
    require(candidates == report['candidate_ids'] == ['kw1s-05','kw1s-06','kw1s-07'], 'candidate identity')
    def grouped(keys):
        result=[]
        for key in sorted({tuple(s[k] for k in keys) for s in slots}):
            rows=[s for s in slots if tuple(s[k] for k in keys)==key]
            ids=[s['slot_id'] for s in rows if s['machine_train_only_candidate']]
            result.append(dict(zip(keys,key),total=len(rows),candidates=len(ids),candidate_ids=ids,three_model_consistent=sum(s['text_three_model_consistent'] for s in rows)))
        return result
    return {'status':report['status'], 'calls':report['consumed_calls'], 'call_ledger':calls,
            'slots':slots, 'candidate_ids':candidates, 'candidate_rate':len(candidates)/len(slots),
            'text_three_model_consistent':sum(s['text_three_model_consistent'] for s in slots),
            'by_engine':grouped(['engine']), 'by_voice':grouped(['speaker']),
            'by_instruction':grouped(['instruction']), 'by_engine_voice_instruction':grouped(['engine','speaker','instruction']),
            'group_coverage':report['group_coverage'], 'resources':report['resources'],
            'training_admitted':report['training_admitted'], 'shipping':report['shipping'],
            'admission_status':report['admission_status']}

def verify(root, archive=None):
    root=Path(root)
    retention=load(root/'source-retention.json')
    expected_files={'README.md','observation.json','report.json','ledger.json','manifest.json','verify_observation.py','test_verify_observation.py'}
    require(set(retention['files']) == expected_files, 'retention inventory')
    workflow = root.parents[1]/'.github/workflows/kw1-supplement-observation.yml'
    data = workflow.read_bytes()
    require(retention['workflow'] == {'path': '.github/workflows/kw1-supplement-observation.yml', 'bytes': len(data), 'sha256': digest(data)}, 'workflow identity')
    for name, entry in retention['files'].items():
        data=(root/name).read_bytes()
        require(len(data)==entry['bytes'] and digest(data)==entry['sha256'], 'retention mismatch: '+name)
    for name, sha in RAW_HASHES.items():
        require(digest((root/name).read_bytes())==sha, 'original artifact identity: '+name)
    report,ledger,manifest=(load(root/n) for n in ['report.json','ledger.json','manifest.json'])
    for name in ['report.json','ledger.json']:
        data=(root/name).read_bytes()
        require(manifest['files'][name]=={'bytes':len(data),'sha256':digest(data)}, 'manifest binding: '+name)
    obs=load(root/'observation.json')
    require(obs['results']==derive(report,ledger), 'observation differs from raw report/ledger')
    require(obs['source']['commit']==SOURCE_SHA and obs['run']['id']==38664865348, 'run/source identity')
    require(obs['run']['status']=='completed' and obs['run']['conclusion']=='success' and obs['run']['run_attempt']==1 and obs['run']['head_sha']==SOURCE_SHA, 'terminal run identity')
    require(obs['artifact']['sha256']==ZIP_SHA and obs['artifact']['bytes']==2424250 and obs['artifact']['id']==11942325284, 'artifact identity')
    require(obs['source']['config_sha256']=='110494b07d3fb84c7b2537889e364ea9349b8f45e92774c3a033b398b6bbe916', 'config identity')
    require(obs['results']['training_admitted'] is False and obs['results']['shipping'] is False, 'admission limitation')
    if archive is not None:
        data=Path(archive).read_bytes()
        require(len(data)==2424250 and digest(data)==ZIP_SHA, 'ZIP identity')
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            require(len(names)==len(set(names))==235, 'ZIP file count/duplicates')
            require(set(names)==set(manifest['files'])|{'manifest.json'}, 'ZIP inventory')
            for name in ['report.json','ledger.json','manifest.json']:
                require(z.read(name)==(root/name).read_bytes(), 'ZIP text mismatch')
            for name,entry in manifest['files'].items():
                data=z.read(name)
                require(len(data)==entry['bytes'] and digest(data)==entry['sha256'], 'ZIP file mismatch: '+name)
    return {'verified':True,'scope':'offline_text_consistency_and_optional_zip_bytes','model_calls':0,'numeric_audio_recomputation':False,'zip_bytes_checked':archive is not None}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--zip',type=Path)
    a=p.parse_args()
    print(json.dumps(verify(a.root,a.zip),sort_keys=True))
