"""Reconstructed FFT64 trace capture. Library access only after all barriers."""
import ctypes as C,json,struct,traceback
from common import *
from compare import exact,cache_sequence
KEYWORDS={1:'你好小窝',2:'小窝小窝'}
class Hyp(C.Structure):
 _fields_=[('len',C.c_size_t),('pb',C.c_double),('pnb',C.c_double),('token',C.c_int32*128),('frame',C.c_int64*128),('prob',C.c_double*128)]
class Native:
 def __init__(self,np):
  self.np=np;p=paths();self.api=module('recovery_fft64_ctypes',p['candidate_ctypes']);self.lib,self.sizes=self.api.load(p['candidate_library'],p['candidate_resources'])
  self.bridge=C.CDLL(str(p['diagnostic_library']));fp=self.api.F32P
  self.lib.a20_accumulation_bits.argtypes=[];self.lib.a20_accumulation_bits.restype=C.c_int
  assert self.lib.a20_accumulation_bits()==64 and self.lib.donor_fft64_environment_ok()==1
  for name,args,result in [('u8_cmvn',[C.c_void_p,fp,fp],C.c_int),('u8_total_frames',[C.c_void_p],C.c_int64),('u8_last_active',[C.c_void_p],C.c_int64),('u8_hit_score',[C.c_void_p],C.c_double),('u8_decoder',[C.c_void_p],C.c_void_p),('u8_finished_valid',[C.c_void_p],C.c_int),('u8_reset_valid',[C.c_void_p],C.c_int),('u8_hyp_count',[C.c_void_p],C.c_size_t)]:
   fn=getattr(self.bridge,name);fn.argtypes=args;fn.restype=result
  for name,ctype in [('u8_front_bytes',self.api.PCM_Batch),('u8_result_bytes',self.api.Decoder_Result),('u8_batch_bytes',self.api.Stream_Batch)]:
   fn=getattr(self.bridge,name);fn.argtypes=[];fn.restype=C.c_size_t;assert fn()==C.sizeof(ctype)
  self.lib.a20d_softmax6.argtypes=[fp,fp];self.lib.a20d_softmax6.restype=C.c_int
  self.lib.a20d_get_hyp.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(Hyp)];self.lib.a20d_get_hyp.restype=C.c_int
  self.arena=self.api.Aligned_Arena(self.sizes['stream_bytes'],self.sizes['stream_alignment']);self.pointer=self.arena.pointer;self.ptr=lambda a:a.ctypes.data_as(fp)
  payload=PAYLOAD.read_bytes();assert sha_bytes(payload)==PAYLOAD_SHA
  self.weights=np.frombuffer(payload,dtype='<f4').copy();self.trace=np.zeros((21,250),np.float32)
  assert self.lib.a20fft64s_init(self.pointer,self.ptr(self.weights),len(self.weights))==0
  self.trace_callback=self.api.Trace_Callback(self.on_trace);self.callback=self.api.Stream_Callback(self.on_batch)
  assert self.lib.a20fft64s_set_trace(self.pointer,self.ptr(self.trace),self.trace_callback,None)==0
  self.counts=dict(streams=0,resets=0,application_feeds=0,finish_calls=0,canonical_calls=0,acoustic_calls=0,fbank_rows=0,model_steps=0,extra_probability_helpers=0,diagnostic_cmvn_helpers=0,internal_decoder_softmax=0,beam_rows=0)
 def on_trace(self,user,index,row,raw,trace,cache):
  if self.error:return
  try:
   np=self.np;assert index==len(self.records)and row==len(self.traces)and self.counts['model_steps']<230
   x=np.ctypeslib.as_array(raw,shape=(400,)).copy();t=np.ctypeslib.as_array(trace,shape=(5250,)).reshape(21,250)
   z=np.zeros(400,np.float32);assert self.bridge.u8_cmvn(self.pointer,self.ptr(x),self.ptr(z))==0
   self.counts['model_steps']+=1;self.counts['diagnostic_cmvn_helpers']+=1
   self.traces.append(([t[i,:width].copy()for i,width in enumerate(DIMS)],z,np.ctypeslib.as_array(cache,shape=(5632,)).reshape(128,11,4).copy(),x))
  except BaseException:self.error=traceback.format_exc()
 def on_batch(self,user,pointer):
  if self.error:return
  try:
   np=self.np;b=pointer.contents;f=b.frontend.contents;r=b.result.contents;index=len(self.records);count=int(f.selected_rows);prefix=f'chunk{index:03d}'
   assert self.counts['canonical_calls']<34 and index<len(self.expected)and f.call_index==index and count==len(self.traces)
   assert r.valid==int(count>0)and r.state in(0,1)and(r.keyword in(1,2)if r.state else r.keyword==0)
   geometry={key:int(getattr(f,key))for key in GEOMETRY_FIELDS};assert geometry=={k:self.expected[index][k]for k in GEOMETRY_FIELDS}
   self.counts['canonical_calls']+=1;self.counts['acoustic_calls']+=int(count>0);self.counts['fbank_rows']+=int(f.fbank_rows)
   a=self.arrays;a[prefix+'_fbank']=np.ctypeslib.as_array(f.fbank,shape=(f.fbank_rows*80,)).reshape(f.fbank_rows,80).copy()
   a[prefix+'_spliced']=np.ctypeslib.as_array(f.rows,shape=(count*400,)).reshape(count,400).copy();a[prefix+'_centers']=np.ctypeslib.as_array(f.centers,shape=(count,)).copy()
   a[prefix+'_logits']=np.ctypeslib.as_array(b.logits,shape=(count*6,)).reshape(count,6).copy();a[prefix+'_cache']=np.ctypeslib.as_array(b.cache,shape=(5632,)).reshape(128,11,4).copy()
   a[prefix+'_cmvn']=np.stack([v[1]for v in self.traces])if count else np.empty((0,400),np.float32)
   a[prefix+'_cache_rows']=np.stack([v[2]for v in self.traces])if count else np.empty((0,128,11,4),np.float32)
   raw=np.stack([v[3]for v in self.traces])if count else np.empty((0,400),np.float32);exact(np,raw,a[prefix+'_spliced'],prefix+'_trace_input')
   for i,width in enumerate(DIMS):a[prefix+f'_stage{i}']=np.stack([v[0][i]for v in self.traces])if count else np.empty((0,width),np.float32)
   stages=[a[prefix+f'_stage{i}']for i in range(21)];before=np.zeros((128,11,4),np.float32)if index==0 else a[f'chunk{index-1:03d}_cache']
   previous,after,final=cache_sequence(np,before,stages);a[prefix+'_cache_before']=before.copy();a[prefix+'_cache_previous_rows']=previous
   exact(np,after,a[prefix+'_cache_rows'],prefix+'_cache_taps');exact(np,final,a[prefix+'_cache'],prefix+'_cache_final');exact(np,stages[20],a[prefix+'_logits'],prefix+'_trace_logits')
   for i in[2,6,10,14,18]:exact(np,stages[i],np.maximum(stages[i-1],np.float32(0)),prefix+f'_relu{i}')
   probs=np.zeros((count,6),np.float32)
   for i in range(count):
    assert self.counts['extra_probability_helpers']<230 and self.lib.a20d_softmax6(self.ptr(a[prefix+'_logits'][i]),self.ptr(probs[i]))==0;self.counts['extra_probability_helpers']+=1
   a[prefix+'_probabilities']=probs;assert 0<=r.rows_decoded<=count
   self.counts['internal_decoder_softmax']+=int(r.rows_decoded);self.counts['beam_rows']+=int(r.rows_decoded)
   returned={}if not r.valid else dict(state=int(r.state),keyword=KEYWORDS[int(r.keyword)]if r.state else None,start=r.start_frame*.01 if r.state else None,end=r.end_frame*.01 if r.state else None,score=float(r.score)if r.state else None)
   frames=int(self.bridge.u8_total_frames(self.pointer));last=int(self.bridge.u8_last_active(self.pointer));hit=float(self.bridge.u8_hit_score(self.pointer));hyps=[]
   for i in range(self.bridge.u8_hyp_count(self.pointer)):
    h=Hyp();assert self.lib.a20d_get_hyp(self.bridge.u8_decoder(self.pointer),i,C.byref(h))==0 and h.len<=128
    hyps.append(dict(tokens=list(h.token[:h.len]),pb=float(h.pb),pnb=float(h.pnb),frames=list(h.frame[:h.len]),probabilities=list(h.prob[:h.len])))
   begin=int(f.available_samples-f.call_samples)*2;end=int(f.available_samples)*2
   self.packed.extend(struct.pack('<QQQiiiqqdQqqdB',begin,end,int(f.available_samples),int(r.valid),int(r.state),int(r.keyword),int(r.start_frame),int(r.end_frame),float(r.score),int(r.rows_decoded),frames,last,hit,0))
   self.records.append(dict(geometry=geometry,begin_byte=begin,end_byte=end,available_audio_samples=int(f.available_samples),decoder=dict(return_value=returned,rows_decoded=int(r.rows_decoded),decoder_total_frames=frames,last_active_pos=last,hit_score=hit,eof_flush=False),hypotheses=hyps));self.traces=[]
  except BaseException:self.error=traceback.format_exc()
 def evaluate(self,pcm,schedule,ingress):
  np=self.np;assert self.counts['streams']<16 and pcm.dtype==np.dtype('<i2')and pcm.shape==(schedule['samples'],)
  assert all(type(n)is int and n>=0 for n in ingress)and sum(ingress)==len(pcm)
  self.counts['streams']+=1;self.error=None;self.traces=[];self.arrays={};self.records=[];self.packed=bytearray();self.expected=schedule['calls']
  assert self.lib.a20fft64s_reset(self.pointer)==0 and self.bridge.u8_reset_valid(self.pointer)==1;self.counts['resets']+=1;offset=0
  for count in ingress:
   piece=np.ascontiguousarray(pcm[offset:offset+count]);offset+=count;self.counts['application_feeds']+=1;assert self.counts['application_feeds']<=55
   status=self.lib.a20fft64s_feed(self.pointer,piece.ctypes.data_as(self.api.I16P),count,self.callback,None);assert not self.error,self.error;assert status==0
  self.counts['finish_calls']+=1;assert self.counts['finish_calls']<=16
  status=self.lib.a20fft64s_finish(self.pointer,self.callback,None);assert not self.error,self.error
  assert status==0 and self.bridge.u8_finished_valid(self.pointer)==1 and len(self.records)==len(self.expected)and not self.traces and self.lib.donor_fft64_environment_ok()==1
  return self.arrays,self.records,bytes(self.packed)
 def finish(self):
  expected=dict(streams=16,resets=16,application_feeds=55,finish_calls=16,canonical_calls=34,acoustic_calls=26,fbank_rows=702,model_steps=230,extra_probability_helpers=230,diagnostic_cmvn_helpers=230)
  for k,v in expected.items():assert self.counts[k]==v,(k,self.counts[k])
  assert 0<=self.counts['beam_rows']==self.counts['internal_decoder_softmax']<=230
