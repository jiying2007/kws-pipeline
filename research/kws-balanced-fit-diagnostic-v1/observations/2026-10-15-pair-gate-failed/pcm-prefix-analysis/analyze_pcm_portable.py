#!/usr/bin/env python3
"""Portable fixed-input energy statistics. Standard library only; never extracts audio.

--output must not exist. If omitted, emit both reports as a JSON bundle to stdout.
Output hashes establish exact agreement with the frozen reports, not model quality.
"""
import argparse,array,hashlib,io,json,math,pathlib,sys,wave,zipfile
ARCHIVE_SIZE=1382028
ARCHIVE_SHA256='f8bf70a0539d6ca7bd11dfed9229a087c0c7d27b943550f73136cd765f54ef0c'
EXPECTED={'pcm-energy.json':'6cf75569e14fef2ec63ab557c750fcc1c1482b5325e8a52ab970cbb229934220','summary.json':'c165bcd51a8ee2b9dabd975ca396858a2b27b3d5e331caff82c423bb2436dbba'}
registry={'records': [{'slot_id': 'tqp-qwen06-05', 'record_id': 'tts-quality-remaining-20261014-v1:tqp-qwen06-05', 'wav_size': 46124, 'pcm_frames': 23040, 'derived_wav': {'sha256': '83fd70f99b3c98168aac162e59f95722fa7d97b8b4f696baff43e27c467efd4e'}, 'derived_pcm': {'sha256': '295333f506ec101e4544ec46dbe4d3a4af240345418f17ba8b165dc30b707b0c'}}, {'slot_id': 'tqp-qwen06-06', 'record_id': 'tts-quality-remaining-20261014-v1:tqp-qwen06-06', 'wav_size': 53804, 'pcm_frames': 26880, 'derived_wav': {'sha256': 'ec8211229fdc775925b8623f772e502f7978a05cb65132aeedbde81267fa1e89'}, 'derived_pcm': {'sha256': '34f326c5a8a3ee20a094806f35b5f2cb46a6c20f71cc4ea9bf7a4f72d2ebf9c9'}}]}
def sha(x):return hashlib.sha256(x).hexdigest()
def run(archive,output):
 with archive.open('rb') as stream:archive_bytes=stream.read(ARCHIVE_SIZE+1)
 if len(archive_bytes)!=ARCHIVE_SIZE or sha(archive_bytes)!=ARCHIVE_SHA256:raise ValueError('archive size or identity mismatch')
 def stats(x):
  return {'samples':len(x),'rms':math.sqrt(sum(v*v for v in x)/len(x))/32768,'peak':max(abs(v) for v in x)/32768,'exact_zero_fraction':sum(v==0 for v in x)/len(x),'integer_min':min(x),'integer_max':max(x)}
 results=[]
 with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
  for rid in ('tqp-qwen06-05','tqp-qwen06-06'):
   r=next(r for r in registry['records'] if r['slot_id']==rid)
   member='audio/'+rid+'.wav'
   entries=[v for v in z.infolist() if v.filename==member]
   if len(entries)!=1 or entries[0].file_size!=r['wav_size']:raise ValueError('WAV member bounds mismatch')
   with z.open(entries[0]) as stream: wav=stream.read(r['wav_size']+1)
   if len(wav)!=r['wav_size']:raise ValueError('WAV size mismatch')
   if sha(wav)!=r['derived_wav']['sha256']:raise ValueError('WAV identity mismatch')
   with wave.open(io.BytesIO(wav),'rb') as w:
    if (w.getnchannels(),w.getframerate(),w.getsampwidth(),w.getcomptype())!=(1,16000,2,'NONE'):raise ValueError('PCM geometry mismatch')
    if w.getnframes()!=r['pcm_frames']:raise ValueError('PCM frame count mismatch')
    pcm=w.readframes(r['pcm_frames'])
   if len(pcm)!=r['pcm_frames']*2 or sha(pcm)!=r['derived_pcm']['sha256']:raise ValueError('PCM identity mismatch')
   x=array.array('h');x.frombytes(pcm)
   if sys.byteorder!='little':x.byteswap()
   windows=[dict(index=i//320,start_ms=i/16,end_ms=(i+400)/16,**stats(x[i:i+400])) for i in range(0,len(x)-399,320)]
   maxr=max(v['rms'] for v in windows);relative=maxr*.01
   for v in windows:v.update(above_absolute_rms=v['rms']>=1e-4,above_relative_rms=v['rms']>=relative)
   def first(key):return next(({'index':v['index'],'start_ms':v['start_ms'],'end_ms':v['end_ms'],'rms':v['rms']} for v in windows if v[key]),None)
   results.append(dict(record_id=r['record_id'],pcm_sha256=sha(pcm),pcm_hash_verified=True,wav_hash_verified=True,whole=stats(x),prefix_105ms=stats(x[:1680]),prefix_200ms=stats(x[:3200]),max_window_rms=maxr,relative_rms_threshold=relative,first_absolute_crossing=first('above_absolute_rms'),first_relative_crossing=first('above_relative_rms'),windows=windows))
 out={'schema':'kws-pair-pcm-energy-description-v1','method':{'sample_rate':16000,'scale':32768,'window_samples':400,'hop_samples':320,'no_tail_padding':True,'absolute_rms_threshold':1e-4,'relative_threshold_fraction_of_own_max_window_rms':.01,'thresholds_fixed_before_reading_pcm':True,'crossing_comparison':'>=','prefix_lengths_samples':[1680,3200]},'limitations':['Energy only; no transcription or speech-onset ground truth.','Window crossing time is a window interval, not an exact onset.','This is not a model, training, ASR or TTS call.'],'new_model_calls':0,'new_optimizer_steps':0,'samples':results}
 
 full=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
 summary={k:v for k,v in out.items() if k!='samples'}
 summary['samples']=[{k:v for k,v in r.items() if k!='windows'} for r in out['samples']]
 summary['full_window_report_sha256']=sha(full)
 compact=(json.dumps(summary,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
 reports={'pcm-energy.json':full,'summary.json':compact}
 for name,raw in reports.items():
  if sha(raw)!=EXPECTED[name]:raise ValueError('recomputed report differs from frozen bytes')
 if output is None:
  print(json.dumps({name:json.loads(raw) for name,raw in reports.items()},ensure_ascii=False,indent=2))
 else:
  output.mkdir(parents=False,exist_ok=False)
  for name,raw in reports.items():
   with (output/name).open('xb') as stream:stream.write(raw)
  print(json.dumps({'status':'EXACT_FROZEN_REPORT_BYTES_MATCH','reports':EXPECTED,'new_model_calls':0,'new_optimizer_steps':0},sort_keys=True))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--archive',type=pathlib.Path,required=True)
 parser.add_argument('--output',type=pathlib.Path)
 args=parser.parse_args()
 try:run(args.archive,args.output)
 except Exception:
  print('Analysis failed: input identity, format, output availability, or frozen report mismatch.',file=sys.stderr)
  sys.exit(1)
