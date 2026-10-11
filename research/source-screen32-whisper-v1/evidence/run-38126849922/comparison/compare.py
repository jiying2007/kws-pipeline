"""Exploratory lexical comparison, not a native-evidence validator or training gate.
Optional terminal artifact must first pass the frozen host native validator.
The supplied checksum binds this report to those reviewed bytes; it is not proof
of validity by itself. Missing/failed outputs never become votes.
"""
import argparse, hashlib, json, pathlib, unicodedata
HERE=pathlib.Path(__file__).resolve().parent

def normalized(text):
    return ''.join(c for c in text if not unicodedata.category(c).startswith('P'))

def compare(data,terminal=None):
    rows=data['rows']; ids=[x['audio_id'] for x in rows]
    if len(ids)!=16 or len(set(ids))!=16 or ids!=[f'clip-{i:06d}' for i in range(1,17)]:
        raise ValueError('exact fixed 16-row roster required')
    third={}
    if terminal is not None:
        if not isinstance(terminal,list): raise ValueError('terminal must be a JSON array')
        for t in terminal:
            i=t['audio_id']
            if i not in ids or i in third: raise ValueError('unknown or duplicate audio ID')
            third[i]=t
    results=[]
    counts={'fixed_denominator':16,'qwen_sense_literal_agreement':0,'qwen_sense_normalized_agreement':0,'third_valid_outputs':0,'all_three_valid_outputs':0,'literal_three_way_agreement':0,'normalized_three_way_agreement':0,'normalized_three_way_agreement_matches_target':0,'normalized_three_way_agreement_differs_from_target':0,'normalized_three_way_disagreement':0,'incomplete_three_way':0}
    for row in rows:
        i=row['audio_id']; t=third.get(i,{'status':'NOT_RUN' if terminal is None else 'missing','raw_text':None})
        if i in third and t.get('wav_sha256')!=row['wav_sha256']: raise ValueError('terminal WAV mismatch: '+i)
        votes={}; statuses={}
        for model,r in [('qwen06',row['qwen06']),('sensevoice',row['sensevoice']),('whisper',t)]:
            statuses[model]=r.get('status','missing')
            valid=(r.get('status')=='completed_native_eot' and r.get('attempts')==1 and r.get('stop_reason')=='eot' and r.get('validation_errors')==[] and isinstance(r.get('raw_output_sha256'),str) and len(r['raw_output_sha256'])==64) if model=='whisper' else (r.get('status')=='success' and r.get('validation')=='validated_receipt' and r.get('completeness')=='complete' and not r.get('quality_flags'))
            # Empty/nonlexical outputs are not agreement votes, including punctuation-only.
            if valid and isinstance(r.get('raw_text'),str) and normalized(r['raw_text']).strip(): votes[model]=r['raw_text']
        allvalid=len(votes)==3
        lit=allvalid and len(set(votes.values()))==1
        norm=allvalid and len({normalized(v) for v in votes.values()})==1
        pair=all(k in votes for k in ('qwen06','sensevoice'))
        pairlit=pair and votes['qwen06']==votes['sensevoice']
        pairnorm=pair and normalized(votes['qwen06'])==normalized(votes['sensevoice'])
        target={m:{'literal_match':v==row['target_text'],'normalized_match':normalized(v)==normalized(row['target_text'])} for m,v in votes.items()}
        agrees_target=norm and all(x['normalized_match'] for x in target.values())
        for name,value in [('qwen_sense_literal_agreement',pairlit),('qwen_sense_normalized_agreement',pairnorm),('third_valid_outputs','whisper' in votes),('all_three_valid_outputs',allvalid),('literal_three_way_agreement',lit),('normalized_three_way_agreement',norm),('normalized_three_way_agreement_matches_target',agrees_target),('normalized_three_way_agreement_differs_from_target',norm and not agrees_target),('normalized_three_way_disagreement',allvalid and not norm),('incomplete_three_way',not allvalid)]: counts[name]+=int(value)
        results.append({'audio_id':i,'wav_sha256':row['wav_sha256'],'pcm_sha256':row['pcm_sha256'],'target_text':row['target_text'],'statuses':statuses,'valid_literal_votes':votes,'original_sensevoice_status':row['original_sensevoice']['status'],'qwen_sense_literal_agreement':pairlit,'qwen_sense_normalized_agreement':pairnorm,'literal_three_way_agreement':lit if allvalid else None,'normalized_three_way_agreement':norm if allvalid else None,'per_asr_vs_target':target,'role':'source_screen_only','human_gold':False,'training_admitted':False})
    return {'schema':'exploratory-three-asr-comparison-v1','status':'NOT_RUN' if terminal is None else 'EXPLORATORY_COMPARISON','normalization':data['normalization'],'unicode_database_version':unicodedata.unidata_version,'counts':counts,'rates_fixed_16':{k:v/16 for k,v in counts.items() if k!='fixed_denominator'},'preregistered_three_asr_roster':False,'native_artifact_validation':'NOT_PERFORMED_BY_THIS_COMPARATOR','role':'source_screen_only','human_gold':False,'training_admitted':False,'rows':results}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=pathlib.Path,default=HERE/'comparison-input.json');p.add_argument('--terminal',type=pathlib.Path);p.add_argument('--terminal-sha256');p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    raw=a.input.read_bytes();data=json.loads(raw);terminal=None
    if a.terminal:
        b=a.terminal.read_bytes()
        if not a.terminal_sha256 or hashlib.sha256(b).hexdigest()!=a.terminal_sha256: p.error('matching reviewed terminal SHA-256 required')
        terminal=json.loads(b)
    elif a.terminal_sha256: p.error('--terminal-sha256 requires --terminal')
    result=compare(data,terminal);result['input_sha256']=hashlib.sha256(raw).hexdigest();result['terminal_sha256']=a.terminal_sha256
    with a.output.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result['counts']))
if __name__=='__main__': main()
