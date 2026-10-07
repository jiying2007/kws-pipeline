"""Exact reviewed reference/native acquisition and scoring functions."""
from common import *
def reference_run(np,protocol,cases,inputs,out):
 from reference import freeze
 f=freeze(np,protocol,cases,inputs,out)
 return dict(passed=True,reference_freeze_sha256=sha(out/'reference-freeze.json'),reference_streams=8,reference_fbank_rows=351,reference_model_rows=115,reference_canonical_calls=17,reference_nonempty_model_sequences=7,candidate_model_steps=0,canonical_decoder_consumed_rows=f['canonical_decoder_consumed_rows'],workload=f['workload'])

def native_run(np,protocol,cases,inputs,out):
 prerequisites=reference_prerequisite();frozen=read(REFERENCE/'reference-freeze.json');sys.path.insert(0,str(CANONICAL))
 import authority,softmax,behavior,canonical as canonical_api,assets
 from native import Native
 from compare import exact,typed_accuracy
 from local_math import LocalAudit
 state,_=assets.load_state();local=LocalAudit(np,state);native=Native(np);checks=[];records=[]
 for case,record in zip(cases,frozen['records']):
  name=case['id'];folder=out/name;folder.mkdir(exist_ok=False);ref=REFERENCE/name;assert record['id']==name
  identities(protocol);assert sha(ref/'canonical.json')==record['canonical_sha256']and sha(ref/'reference.npz')==record['reference_sha256']
  canonical=read(ref/'canonical.json');authority.validate_metadata(canonical['reference_authorities'])
  with np.load(ref/'reference.npz',allow_pickle=False)as f:reference={k:f[k].copy()for k in f.files}
  pcm=inputs[name];whole=whole_events=whole_packed=None
  for route,parts in[('whole',case['whole_feeds']),('ragged',case['ragged_feeds'])]:
   assert sha_bytes(pcm.tobytes())==case['pcm_sha256']
   write(folder/(route+'-claim.json'),dict(id=name,route=route,reference_freeze_sha256=prerequisites['reference_freeze_sha256'],pcm_sha256=case['pcm_sha256'],retry_allowed=False))
   arrays,events,packed=native.evaluate(pcm,canonical['schedule'],parts);assert len(events)==len(canonical['calls']);local_arrays={};expected=[];observed=[]
   for index,(call,event)in enumerate(zip(canonical['calls'],events)):
    prefix=f'chunk{index:03d}';view=dict(call)
    for key in['fbank','spliced','cmvn','probabilities']:view[key]=reference[prefix+'_'+key]
    actual={key:arrays[prefix+'_'+key]for key in['fbank','spliced','cmvn','logits','probabilities']}
    exact(np,arrays[prefix+'_centers'],reference[prefix+'_centers'],name+'_'+route+'_'+prefix+'_centers')
    typed_accuracy(np,checks,name+'_'+route+'_'+prefix,actual,view,canonical['reference_authorities'],authority,canonical_api,softmax)
    expected.append(behavior.normalize_call(call));observed.append(behavior.normalize_call(event))
    before=len(checks);local_arrays.update(local.check(prefix,arrays,checks))
    for check in checks[before:]:check['name']=name+'_'+route+'_'+check['name']
   result=behavior.compare_records(expected,observed)
   checks.append(dict(name=name+'_'+route+'_canonical_behavior',gate='hard_event',passed=result['passed'],elements=len(events),failed_elements=len(result['issues']),behavior_comparison=result))
   save_arrays(np,folder/(route+'.npz'),arrays);save_arrays(np,folder/(route+'-local.npz'),local_arrays);write_compact(folder/(route+'-events.json'),events)
   with(folder/(route+'-events.bin')).open('xb')as f:f.write(packed);f.flush();os.fsync(f.fileno())
   if route=='whole':whole=arrays;whole_events=events;whole_packed=packed
   else:
    assert set(whole)==set(arrays)
    for key in arrays:exact(np,arrays[key],whole[key],name+'_partition_'+key)
    assert packed==whole_packed and json.dumps(events,sort_keys=True,allow_nan=False)==json.dumps(whole_events,sort_keys=True,allow_nan=False)
   records.append(dict(id=name,route=route,array_sha256=sha(folder/(route+'.npz')),local_sha256=sha(folder/(route+'-local.npz')),events_sha256=sha(folder/(route+'-events.json')),event_binary_sha256=sha(folder/(route+'-events.bin'))))
   assert sha_bytes(pcm.tobytes())==case['pcm_sha256'];allocation_check()
  del whole,whole_events,arrays,events,reference,canonical,local_arrays
 native.finish();local.finish();assert reference_prerequisite()==prerequisites
 return dict(passed=all(c['passed']for c in checks),**prerequisites,candidate_model_steps=native.counts['model_steps'],native_counts=native.counts,local_operation_counts=local.counts,checks=checks,hard_failed_checks=sum(not c['passed']for c in checks),route_records=records,all_saved_native_partition_outputs_bitexact=True,reference_types=authority.metadata(),original_official_tensor_comparisons='NOT_COMPUTED: unavailable historical tensors, no substitution',pointwise_canonical_intermediate_parity_claim=False,internal_decoder_count_basis='one internal softmax per consumed beam row, counted separately from full-row helper')
