"""One canonical mathematical stream per replacement identity, reference only."""
from common import *
def freeze(np,protocol,cases,inputs,out):
 sys.path.insert(0,str(CANONICAL));import canonical,schedule,authority
 schedules=[schedule.stream_schedule(c['samples'])for c in cases];verify_geometry(schedules);budget=schedule.operation_budget(schedules)
 pipeline=canonical.CanonicalPipeline();records=[];calls=frames=rows=sequences=decoder_rows=0
 for case,geometry in zip(cases,schedules):
  identities(protocol);folder=out/case['id'];folder.mkdir(exist_ok=False)
  write(folder/'canonical-claim.json',dict(id=case['id'],retry_allowed=False,pcm_sha256=case['pcm_sha256'],candidate_calls=0))
  pcm=inputs[case['id']];assert sha_bytes(pcm.tobytes())==case['pcm_sha256']
  result=pipeline.evaluate_pcm(pcm,expected_pcm_sha256=case['pcm_sha256'],ingress_chunks=case['whole_feeds'],with_events=True)
  assert result['contract_sha256']==CONTRACT_SHA
  assert result['schema']=='a20-canonical-pcm-pipeline.recovery-v1'and result['field_semantics']=='a20-canonical-pcm-pipeline.v3'
  assert result['schedule']==geometry and result['events_included']is True and result['hidden_native_calls']==result['hidden_torch_calls']==0
  assert result['sources_before']==result['sources_after'];authority.validate_metadata(result['reference_authorities'])
  assert result['workload']==schedule.operation_budget([geometry])
  nr=geometry['model_rows'];sequences+=int(nr>0)
  assert result['final_logits']['counters']==dict(reference_rows=nr,affine_calls=12*nr,memory_calls=4*nr,weighted_products=388984*nr)
  assert len(result['frame_records'])==geometry['fbank_rows'];arrays={};diagnostic=result['stage_diagnostics']
  assert diagnostic['diagnostic_only']is True and diagnostic['additional_model_evaluations']==0
  displays=diagnostic.pop('arrays');assert set(displays)=={f'stage{i}'for i in range(21)}
  for i,width in enumerate(DIMS):
   key=f'stage{i}';value=displays[key];assert value.dtype==np.float64 and value.shape==(nr,width)and np.isfinite(value).all();arrays['ideal_diagnostic_'+key]=value.copy()
   for field in['model_error_bounds','conversion_error_bounds','combined_error_bounds']:assert len(diagnostic[field][key])==nr
  diagnostic['array_storage']='reference.npz keys ideal_diagnostic_stage0..20; diagnostic only'
  del displays
  for i,call in enumerate(result['calls']):
   prefix=f'chunk{i:03d}';nf=geometry['calls'][i]['fbank_rows'];nr=geometry['calls'][i]['selected_rows'];assert call['geometry']==geometry['calls'][i]
   assert 0<=call['decoder']['rows_decoded']<=nr;decoder_rows+=call['decoder']['rows_decoded']
   for row in range(nf):
    frame=result['frame_records'][call['geometry']['fbank_begin']+row];assert frame['sample_start']==call['geometry']['window_sample_starts'][row]
    certs=frame['logfbank_round_certificates'];assert len(certs)==80 and[cert[2]for cert in certs]==[f'{int(v):08x}'for v in call['fbank'][row].view(np.uint32)]
   for row in range(nr):
    certs=call['ideal_probabilities'][row]['round_certificates'];assert len(certs)==6 and[cert[2]for cert in certs]==[f'{int(v):08x}'for v in call['probabilities'][row].view(np.uint32)]
   for key,shape in[('fbank',(nf,80)),('spliced',(nr,400)),('cmvn',(nr,400)),('probabilities',(nr,6))]:
    value=call.pop(key);assert value.dtype==np.float32 and value.shape==shape and np.isfinite(value).all();arrays[prefix+'_'+key]=value.copy()
   arrays[prefix+'_centers']=np.asarray(geometry['calls'][i]['centers'],dtype=np.uint64);call['array_prefix']=prefix;calls+=1;frames+=nf;rows+=nr
  assert sha_bytes(pcm.tobytes())==case['pcm_sha256'];save_arrays(np,folder/'reference.npz',arrays);write_compact(folder/'canonical.json',result)
  assert(folder/'canonical.json').stat().st_size<=20*1024**2
  records.append(dict(id=case['id'],reference_sha256=sha(folder/'reference.npz'),canonical_sha256=sha(folder/'canonical.json'),pcm_sha256=case['pcm_sha256']))
  del arrays,result;allocation_check()
 assert(calls,frames,rows,sequences)==(17,351,115,7)and 0<=decoder_rows<=115
 identities(protocol)
 frozen=dict(schema='a20-recovery-reference-freeze-v1',records=records,source_identity=pipeline.source_identity,weight_state_sha256=pipeline.weight_manifest['state_sha256'],reference_streams=8,reference_nonempty_model_sequences=sequences,canonical_decoder_consumed_rows=decoder_rows,reference_canonical_calls=calls,reference_fbank_rows=frames,reference_model_rows=rows,workload=budget,official_reference_forwards=0,torch_calls=0,native_model_steps_before_freeze=0,native_library_loads_before_freeze=0,files_sha256={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()})
 write(out/'reference-freeze.json',frozen);return frozen
