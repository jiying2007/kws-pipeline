"""Public saved-scalar derivation; adapted from the original saved-only diagnostic.

Only JSON values, argmax order, collapse, subtraction, counts and sorting are
used. No model/audio/library reads, softmax, decoder replay or tuning.
Original diagnostic script SHA256 is recorded in PROVENANCE.json.
"""
import math

T=['blank','你','好','小','窝','屋']
FOCUS=['qwen3-kw1-serena','qwen3-kw2-eric','qwen3-repeat-nihao-eric']
selected_frames={
 FOCUS[0]:[75,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171],
 FOCUS[1]:[57,69,72,117,120,123,126,129,132,135,138],
 FOCUS[2]:[45,48,51,54,57,60,63,66],
}

def collapse(ids):
 return [t for i,t in enumerate(ids) if t and (i==0 or t!=ids[i-1])]
assert collapse([1,1,0,1,1,2,2,0])==[1,1,2]
assert collapse([0,0,0])==[]

def derive(raw, geometry, score):
    geo={r['recording']:r for r in geometry['rows']}
    meta={r['recording']:{'original_split':r['historical_split'],'voice':r['voice'],
          'label':r['declared_text'],'kind':r['kind'],'keyword_id':r['keyword_id'],
          'descriptive_outcome':r['descriptive_outcome']} for r in score['clips']}
    clips=[];focus=[];events=[];tie_count=0
    for start in [r for r in raw if r['kind']=='clip_start']:
     name=start['recording']; callbacks=[r for r in raw if r['kind']=='callback' and r['recording']==name]
     end=next(r for r in raw if r['kind']=='clip_end' and r['recording']==name)
     plans=geo[name]['callback_plan']; assert len(plans)==len(callbacks)
     rows=[];cbs=[];ev=[]
     for c,g in zip(callbacks,plans):
      for k in ['call_index','available_samples','call_samples','waveform_samples','fbank_rows','splice_rows','selected_rows','is_final_short','centers','decoder_total_frames']:
       assert c[k]==g[k],(name,k)
      assert len(c['logits'])==c['selected_rows']==len(c['centers'])
      ids=[]
      for i,(center,values) in enumerate(zip(c['centers'],c['logits'])):
       assert len(values)==6 and all(math.isfinite(v) for v in values)
       order=sorted(range(6),key=lambda k:(-values[k],k));top=order[0]
       tie_count+=sum(v==values[top] for v in values)>1
       ids.append(top)
       rows.append({'frame':center,'callback':c['call_index'],'row':i,'searched':i<c['decoder_rows_decoded'],
                    'ids_descending_logit':order,'logits':values,
                    'top_minus_runner_up':values[order[0]]-values[order[1]]})
      cbs.append({'callback':c['call_index'],'phase':c['phase'],'available_samples':c['available_samples'],
       'call_samples':c['call_samples'],'waveform_samples':c['waveform_samples'],'fbank_rows':c['fbank_rows'],
       'selected_rows':c['selected_rows'],'searched_rows':c['decoder_rows_decoded'],
       'frame_positions':c['centers'],'argmax_ids':ids,'final_short':bool(c['is_final_short'])})
      if c['state']:
       e={k:c[k] for k in ['keyword','start_frame','end_frame','score']}
       e.update(callback=c['call_index'],available_samples=c['available_samples'],available_audio_s=c['available_samples']/16000,
                searched_rows=c['decoder_rows_decoded'],selected_rows=c['selected_rows'],search_stop_frame=c['centers'][c['decoder_rows_decoded']-1])
       ev.append(e);events.append({'recording':name,**e})
     assert all(r['frame']==3*i for i,r in enumerate(rows)),name
     assert len(rows)==end['model_rows']
     runs=[]
     for r in rows:
      tid=r['ids_descending_logit'][0]
      if not runs or runs[-1]['token_id']!=tid:
       runs.append({'token_id':tid,'token':T[tid],'start_frame':r['frame'],'end_frame':r['frame'],
                    'start_callback':r['callback'],'start_row':r['row'],'end_callback':r['callback'],'end_row':r['row'],
                    'saved_rows':1,'searched_rows':int(r['searched'])})
      else:
       runs[-1].update(end_frame=r['frame'],end_callback=r['callback'],end_row=r['row'])
       runs[-1]['saved_rows']+=1;runs[-1]['searched_rows']+=int(r['searched'])
     nonblank=[r for r in runs if r['token_id']]
     ids=collapse([r['ids_descending_logit'][0] for r in rows]);assert ids==[r['token_id'] for r in nonblank]
     m=meta[name]
     clip={'recording':name,'historical_split':m['original_split'],'voice':m['voice'],'declared_text':m['label'],
      'kind':m['kind'],'keyword_id':m['keyword_id'],'observed_outcome':m['descriptive_outcome'],
      'audio_identity_from_saved_record':{k:start[k] for k in ['frames','wav_sha256','pcm_sha256']},
      'all_saved_rows':len(rows),'actually_searched_rows':sum(r['searched'] for r in rows),
      'derived_greedy_ctc':''.join(T[t] for t in ids),'nonblank_runs':nonblank,'callbacks':cbs,'observed_events':ev}
     clips.append(clip)
     if name in FOCUS:
      token_support=[]
      for tid in range(1,6):
       def margin(r):return r['logits'][tid]-max(v for j,v in enumerate(r['logits']) if j!=tid)
       best=max(rows,key=margin)
       token_support.append({'token_id':tid,'token':T[tid],'argmax_rows':sum(r['ids_descending_logit'][0]==tid for r in rows),
        'best_margin_over_other_tokens':margin(best),'best_margin_frame':best['frame'],'best_margin_callback':best['callback'],
        'best_margin_row':best['row'],'best_margin_logit':best['logits'][tid]})
      focus.append({'recording':name,'shorttail_gate':'NOT_FIRED_IN_ANY_CALLBACK',
       'minimum_callback_waveform_samples':min(c['waveform_samples'] for c in callbacks),
       'final_callback':cbs[-1],'all_saved_rows_searched':len(rows)==sum(r['searched'] for r in rows),
       'greedy_ctc':clip['derived_greedy_ctc'],'token_support_by_best_signed_margin':token_support,
       'selected_saved_row_competition':[r for r in rows if r['frame'] in selected_frames[name]],'observed_events':ev})
    assert len(clips)==20 and sum(c['all_saved_rows'] for c in clips)==1144
    assert sum(c['actually_searched_rows'] for c in clips)==1111 and len(events)==9
    assert sum(len(c['callbacks']) for c in clips)==127
    assert tie_count==0
    for f in focus:assert f['minimum_callback_waveform_samples']>=800
    shorttails=[{'recording':c['recording'],**cb} for c in clips for cb in c['callbacks'] if cb['final_short'] and cb['call_samples']<800]
    ranked=sorted(events,key=lambda e:e['score'])
    for i,e in enumerate(ranked,1):e['ascending_score_rank']=i
    return {'all20':clips,'focus3_numeric':focus,'raw_tail_under800_examples':shorttails,
            'observed_event_score_ranking_ascending':ranked,
            'totals':{'clips':20,'callbacks':127,'model_rows':1144,'searched_rows':1111,
                      'early_break_skipped_rows':33,'events':9,'max_logit_ties':tie_count}}
