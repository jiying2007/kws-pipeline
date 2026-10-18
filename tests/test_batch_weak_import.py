"""Artificial byte fixtures only. Never claims real execution or Library receipt.

Envelope rebind helpers deliberately recompute all saved digests so negative
cases exercise semantic checks, not merely stale enclosing hashes.
"""
import copy
from contextlib import ExitStack
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'training')]
import batch_weak_import as b
from test_language_conditioned_queue import fixture, encoded, archive


def rehash_package(root):
    (root/'MANIFEST.json').write_bytes(encoded({p.relative_to(root).as_posix():b.digest(p.read_bytes())
        for p in root.rglob('*') if p.is_file() and p!=root/'MANIFEST.json'}))


def update(batch,name,fn):
    value=b.loads(batch['data'][name]);fn(value);batch['data'][name]=encoded(value)


def rebind(batch,ident='clip-000001'):
    """Rehash raw, boundary, parsed, terminal, native summary and archive."""
    d=batch['data'];p='output/'+ident+'/'
    raw=b.loads(d[p+'raw.json'])
    if batch['model_key']=='qwen06':raw['generation_evidence_sha256']=b.digest(d[p+'raw-generation.json'])
    d[p+'raw.json']=encoded(raw)
    parsed=b.loads(d[p+'parsed.json']);parsed['raw_evidence_sha256']=b.digest(d[p+'raw.json'])
    d[p+'parsed.json']=encoded(parsed)
    if batch['model_key']=='tts06':d[p+'derived-audio.json']=encoded(parsed)
    if p+'call-boundary.json' in d:
        update(batch,p+'call-boundary.json',lambda v:v.update(input_evidence_sha256=b.digest(d[p+'input-evidence.json'])))
    t=b.loads(d[p+'terminal.json']);t['result']=parsed
    t['retained_files']={n[len(p):]:b.digest(v) for n,v in d.items() if n.startswith(p) and n!=p+'terminal.json'}
    d[p+'terminal.json']=encoded(t)
    batch['summary']['items'][0]['terminal_sha256']=b.digest(d[p+'terminal.json'])
    rawzip=archive({'batch_id':batch['batch_id']},d)
    _,batch['data']=b.archive_bytes(rawzip,b.digest(rawzip))
    batch['data'].pop('PACKAGE-MANIFEST.json')
    return batch


def native_envelope(key,contents,parsed,receipt,items):
    ident='clip-000001';p='output/'+ident+'/'
    d={p+n[len('output/'):]:v for n,v in contents.items() if n.startswith('output/')
        and n not in ('output/native-declaration.json','output/worker-result.json','output/load-receipt.json')}
    consumed={'batch_id':'artificial','model_key':key,'item_id':ident,'plan_sha256':'a'*64,
              'consumed':True,'retry_authorized':False}
    d[p+'consume-receipt.json']=d[ident+'-consume.json']=encoded(consumed)
    d[p+'item-claim.json']=encoded({'item_id':ident,'model_key':key,'retry_authorized':False})
    t={'schema':b.NATIVE,'fixture':False,'model_key':key,'item_id':ident,'status':'completed_bound',
       'retry_authorized':False,'result':parsed,'retained_files':{},
       **{k:True for k in ('binding_verified','evidence_complete','call_started','callback_invoked','consumed',
                          'forward_completed','forward_completion_confirmed')}}
    d[p+'terminal.json']=encoded(t)
    result={'model_key':key,'batch_id':'artificial','plan_sha256':'a'*64,'data':d,'config':{'items':items},
       'load':{'receipt':{'load':receipt}},'summary':{'items':[{'item_id':ident,'status':'completed_bound',
       'binding_verified':True,'terminal_persistence_failed':False,'terminal_sha256':''}]},
       'controller':{'items':[{'item_id':ident,'status':'completed_bound','native_summary_confirmed':True,
        'forward_completed':True,'known_call_started':True}]}}
    return result


def asr_fixture(root,key='qwen06',emotion=None):
    _,content,_,audio,_=fixture(root,'qwen3-asr' if key=='qwen06' else 'sensevoice')
    receipt=b.loads(content['output/load-receipt.json'])['receipt']
    parsed=b.loads(content['output/parsed.json'])
    out=native_envelope(key,content,parsed,receipt,[{'opaque_id':'clip-000001','wav_sha256':audio['file_sha256'],
        'pcm_sha256':audio['pcm_sha256'],'frame_count':audio['frames']}])
    d=out['data'];p='output/clip-000001/'
    d['anonymous-input/'+audio['file_sha256']+'.wav']=content['input.wav']
    d['input-lineage.json']=encoded({'tts_readback_archive_sha256':'b'*64,'eligible_opaque_ids':['clip-000001']})
    d[p+'call-boundary.json']=encoded({'schema':b.NATIVE,'item_id':'clip-000001','input_evidence_sha256':'',
       'source_seed_provided':False,'asr_fixed_seed':0,'isolation_arguments':
       {'context':'','past_key_values':None,'use_cache':False} if key=='qwen06' else {'cache':{},'context':'','language':'zh'}})
    if emotion:
        ident=b.source_contract()['sense_metadata']['emotion_ids'][emotion]
        raw=b.loads(d[p+'raw.json']);par=b.loads(d[p+'parsed.json'])
        decoded=raw['raw_result'][0][0]['text'].replace('NEUTRAL',emotion)
        ids=par['token_binding']['raw_token_ids'];ids[1]=ident
        raw['raw_result'][0][0]['text']=decoded;raw['capture_calls']=[{'decoded':decoded,'token_ids':ids}]
        raw['raw_decode_attempts']=[{'decode_kwargs':{},'raw_decoded_result':decoded,'raw_token_ids':ids}]
        par['metadata'][1]=emotion;par['token_evidence']={'decoded':decoded,'token_ids':ids}
        par['token_binding'].update(raw_decoded_sha256=b.digest(decoded.encode()),metadata_token_ids=ids[:4])
        par['token_binding']['metadata_pieces'][1]='<|'+emotion+'|>'
        d[p+'raw.json']=encoded(raw);d[p+'parsed.json']=encoded(par)
    return rebind(out),audio,content['input.wav']


def tts_fixture():
    ident='clip-000001';p='output/'+ident+'/'
    request={'item_id':ident,'intended_text':'人工测试','speaker':'Dylan','seed':19}
    f=struct.pack('<f',0.125)*4800
    stream=io.BytesIO()
    with wave.open(stream,'wb') as w:w.setparams((1,2,16000,0,'NONE',''));w.writeframes(struct.pack('<h',4096)*3200)
    wav=stream.getvalue();audio,_=b.wav_observation(wav)
    before={'schema':b.NATIVE,'item_id':ident,'recipe':b.source_contract()['tts_recipe'],
      'cleared_per_generation_state':[{'attribute':'rope_deltas','module_index':1}],'request':request,'model_id':b.TTS_MODEL,'model_revision':b.TTS_REVISION,
      'cross_item_context':False,'tts_initial_past_key_values':None,'seed_reset':['random','numpy','torch']}
    termination={'ended_with_eos':True,'eos':2150,'hit_token_cap_without_eos':False,'iterations':2,'last_token_ids':[2150]}
    raw={'schema':b.NATIVE,'item_id':ident,'native_return_type':'builtins.tuple','forward_completion_confirmed':True,'termination':termination,'raw_evidence_complete':True,
      'token_capture':{'calls':1,'complete':True},'sample_rate_hz':24000,
      'waveforms':[{'path':'native-waveform-000.bin','dtype':'<f4','strides':[4],'bytes':len(f),'shape':[4800],'sha256':b.digest(f)}]}
    par={'model_id':b.TTS_MODEL,'model_revision':b.TTS_REVISION,'source_seed':19,'termination':termination,
      'source_float_values_sha256':b.digest(f),'source_signal_statistics':{'nonfinite_count':0,'peak':0.125,'abs_ge_one_count':0,'saturation_count':0},
      'asr_input_qualified':True,'signal_flags':[],'wav_sha256':audio['file_sha256'],'pcm_sha256':audio['pcm_sha256'],
      'frame_count':3200,'sample_rate_hz':16000,'channels':1,'sample_width_bytes':2,'raw_audio_sha256':b.digest(b'fixture raw float'),
      'normalized_peak':0.125,'normalized_signal_statistics':{'nonfinite_count':0,'saturation_count':0}}
    c={'output/raw.json':encoded(raw),'output/parsed.json':encoded(par),'output/input-evidence.json':encoded(before),
       'output/raw-generation.json':encoded({'schema':b.NATIVE,'item_id':ident,'forward_completion_confirmed':True,
       'returned_sequence_ids':[[17,2150]],'returned_sequence_shape':[1,2],'codec_eos_token_id':2150,
       'hidden_state_iteration_count':2,'returned_codebook_token_ids':[None,[[17]*16]]}),
       'output/native-waveform-000.bin':f,'output/'+ident+'.wav':wav,'output/'+ident+'.raw-float.wav':b'fixture raw float'}
    batch=native_envelope('tts06',c,par,{},[request]);rebind(batch)
    return batch,request


def cleanup_fixture():
    supervisor={'boot_id':'fixture','pid_namespace_inode':8,'pid_namespace_link':'pid:[8]'}
    child={'pid':41,'proc_start_ticks':25,'boot_id':'fixture'}
    data={'controller-source.py':b'artificial controller',
        'namespace-start.json':encoded({'batch_id':'artificial','plan_sha256':'a'*64,
            'controller_sha256':b.digest(b'artificial controller'),'supervisor_identity':supervisor}),
        'process-identity.json':encoded(child),
        'namespace-child.json':encoded({**supervisor,'captured':True,'pid':41,'child_start_ticks':25}),
        'terminal.json':encoded({'fixture':False,'returncode':0,'termination_reason':'normal_exit','model_leader_reaped':True,
            'postkill_group_state':{'live_processes':0}})}
    data['namespace-final.json']=encoded({'supervisor_identity':supervisor,'same_supervisor_process':True,
        'observation_scope':'original_supervisor_pid_namespace','cross_namespace_absence_is_proof':False,
        'observation_error':None,'child_identity':child,'group_after_terminal':{'live_processes':0},
        **{field:b.digest(data[name]) for field,name in [('terminal_sha256','terminal.json'),
            ('child_identity_sha256','process-identity.json'),('child_namespace_sha256','namespace-child.json')]}})
    return data


def screen_item(text='你好小窝',split='train'):
    return {'item_id':'artificial','opaque_id':'clip-000001','speaker':'Dylan','split':split,
       'intended_text':text,'intent_kind':'positive_candidate','source_row':{'item_id':'artificial','speaker':'Dylan','intended_text':text,'kind':'positive_candidate','lineage_group':'fixture-voice','split':split}}


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def check_asr(self,f):return b.verify_asr(f[0],'clip-000001',f[1],f[2],'b'*64)
    def test_native_qwen_complete(self):self.assertEqual(self.check_asr(asr_fixture(self.root)),'你好小窝')
    def test_native_sense_explicit_metadata(self):
        for emotion in ['NEUTRAL','ANGRY','SURPRISED','EMO_UNKNOWN']:
            with self.subTest(emotion=emotion):self.assertEqual(self.check_asr(asr_fixture(self.root,'sensevoice',emotion)),'你好小窝')
    def test_qwen_semantic_prefix_rebound(self):
        f=asr_fixture(self.root);p='output/clip-000001/'
        for n in ['input-evidence.json','raw-generation.json','raw.json']:
            def change(v):
                v['input_ids'][0][0]=15
                if 'returned_sequence_ids' in v:v['returned_sequence_ids'][0][0]=15
                if 'input_ids_after_generate' in v:v['input_ids_after_generate'][0][0]=15
            update(f[0],p+n,change)
        rebind(f[0])
        with self.assertRaisesRegex(ValueError,'language-only native token prefix'):self.check_asr(f)
    def test_qwen_input_mutated_rebound(self):
        f=asr_fixture(self.root);update(f[0],'output/clip-000001/raw.json',lambda d:d['input_ids_after_generate'][0].__setitem__(0,44));rebind(f[0])
        with self.assertRaisesRegex(ValueError,'input prefix changed'):self.check_asr(f)
    def test_sense_metadata_token_mismatch_rebound(self):
        f=asr_fixture(self.root,'sensevoice','ANGRY');p='output/clip-000001/'
        update(f[0],p+'parsed.json',lambda d:d['token_binding']['metadata_token_ids'].__setitem__(1,25007));rebind(f[0])
        with self.assertRaisesRegex(ValueError,'metadata token boundary'):self.check_asr(f)
    def test_condition_not_observed_rebound(self):
        f=asr_fixture(self.root);update(f[0],'output/clip-000001/parsed.json',lambda d:d.update(detected_language='zh'));rebind(f[0])
        with self.assertRaisesRegex(ValueError,'conditioned scope'):self.check_asr(f)
    def test_unknown_native_terminal_hash_rejected(self):
        f=asr_fixture(self.root);f[0]['summary']['items'][0]['terminal_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'exact item terminal'):self.check_asr(f)
    def test_partial_native_evidence_rebound(self):
        f=asr_fixture(self.root);update(f[0],'output/clip-000001/terminal.json',lambda d:d.update(evidence_complete=False));rebind(f[0])
        with self.assertRaisesRegex(ValueError,'terminal status'):self.check_asr(f)
    def test_actual_pcm_mismatch_rebound(self):
        f=asr_fixture(self.root);key='anonymous-input/'+f[1]['file_sha256']+'.wav'
        f[0]['data'][key]=f[0]['data'][key][:-2]+b'\x02\x00';rebind(f[0])
        with self.assertRaisesRegex(ValueError,'actual ASR input differs'):self.check_asr(f)
    def test_asr_tts_lineage_rebound(self):
        f=asr_fixture(self.root);update(f[0],'input-lineage.json',lambda d:d.update(tts_readback_archive_sha256='0'*64));rebind(f[0])
        with self.assertRaisesRegex(ValueError,'readback lineage'):self.check_asr(f)
    def test_tts_eos_and_pcm_good(self):
        batch,item=tts_fixture();self.assertEqual(b.verify_tts(batch,item)[1]['frames'],3200)
    def test_tts_missing_eos_rebound(self):
        batch,item=tts_fixture();update(batch,'output/clip-000001/raw-generation.json',lambda d:d['returned_sequence_ids'][0].__setitem__(-1,17));rebind(batch)
        with self.assertRaisesRegex(ValueError,'native token/EOS'):b.verify_tts(batch,item)
    def test_tts_codebook_token_rebound(self):
        batch,item=tts_fixture();update(batch,'output/clip-000001/raw-generation.json',lambda d:d['returned_codebook_token_ids'][1][0].__setitem__(0,18));rebind(batch)
        with self.assertRaisesRegex(ValueError,'codebook token'):b.verify_tts(batch,item)
    def test_tts_request_speaker_rebound(self):
        batch,item=tts_fixture();update(batch,'output/clip-000001/input-evidence.json',lambda d:d['request'].update(speaker='Sohee'));rebind(batch)
        with self.assertRaisesRegex(ValueError,'actual request'):b.verify_tts(batch,item)
    def test_no_lexical_repair_or_empty_negative(self):
        vocab={'啊':1281,'你':1027,'好':1401,'小':1462,'王':1973,'窝':2093,'屋':1479};audio={'frames':3200}
        item=screen_item('你好小王');item['intent_kind']='confusable_candidate'
        r=b.screen_text(item,'啊，你好，小王。','啊你好小王',audio,vocab)
        self.assertEqual(r['ctc_text'],'啊你好小王');self.assertEqual(r['ctc_ids'],[1281,1027,1401,1462,1973])
        self.assertFalse(r['four_token_consumer']['compatible']);self.assertEqual(r['status'],'machine_candidate')
        self.assertFalse(r['current_experiment_admission'])
        self.assertEqual(b.screen_text(item,'你好小窝','你好小窝',audio,vocab)['reason'],'negative_contains_keyword')
        self.assertEqual(b.screen_text(item,'你好小窝','你好小屋',audio,vocab)['reason'],'asr_disagreement')
        self.assertEqual(b.screen_text(item,'','',audio,vocab)['reason'],'empty_asr_not_proof_of_no_speech')
    def test_archive_traversal_and_duplicate(self):
        for names in [('../escape',),('same','same')]:
            out=io.BytesIO()
            with zipfile.ZipFile(out,'w') as z:
                for n in names:z.writestr(n,b'x')
            raw=out.getvalue()
            with self.assertRaises(ValueError):b.archive_bytes(raw,b.digest(raw))
    def test_no_model_runtime_import(self):
        self.check_asr(asr_fixture(self.root));b.verify_tts(*tts_fixture())
        for name in ['torch','transformers','funasr','sentencepiece','numpy','scipy']:
            self.assertNotIn(name,sys.modules)



    def test_same_namespace_cleanup_complete(self):b.verify_cleanup(cleanup_fixture(),'artificial','a'*64)
    def test_group_unknown_and_foreign_namespace_rebound(self):
        for field,value in [('group_after_terminal',{'live_processes':None}),('observation_error','unobservable'),
            ('same_supervisor_process',False),('cross_namespace_absence_is_proof',True)]:
            data=cleanup_fixture();d=b.loads(data['namespace-final.json']);d[field]=value;data['namespace-final.json']=encoded(d)
            raw=archive({'batch_id':'artificial'},data);_,data=b.archive_bytes(raw,b.digest(raw))
            with self.subTest(field=field),self.assertRaises(ValueError):b.verify_cleanup(data,'artificial','a'*64)
    def test_delivery_only_checks_supplied_bytes_and_identity(self):
        raw=b'artificial Library evidence, not a real receipt';n=len(raw)
        proof={'proof_scope':'parent_tool_receipt_and_local_readback_consistency','sha256':b.digest(raw),
            'bytes':n,'library_file_id':'fixture','file_id':'fixture-bytes','version':0,'readback_path':'fixture'}
        receipt={'upload':{'operation':'create_library_file','status':'succeeded','file_size_bytes':n,
            'library_file_id':'fixture','file_id':'fixture-bytes','current_version_number':0},
            'readback':{'transfers':[{'library_file_id':'fixture','file_id':'fixture-bytes','current_version_number':0,
                'size_bytes':n,'workspace_path':'fixture'}]}}
        b.verify_supplied_readback(proof,receipt,raw)
        with self.assertRaisesRegex(ValueError,'byte mismatch'):b.verify_supplied_readback(proof,receipt,raw+b'!')
        receipt['readback']['transfers'][0]['library_file_id']='different'
        with self.assertRaisesRegex(ValueError,'Library identity'):b.verify_supplied_readback(proof,receipt,raw)

    def rights_fixture(self):
        batch,item=tts_fixture();d=batch['data'];ident=item['item_id'];prefix='output/'+ident+'/'
        source={**item,'split':'train','kind':'positive_candidate','lineage_group':'fixture-voice'}
        item.update(split='train',intent_kind='positive_candidate')
        assets={'model_id':b.TTS_MODEL,'revision':b.TTS_REVISION,'files':[{'filename':'README.md','bytes':4,'sha256':b.digest(b'card')}]}
        batch['load']={'receipt':{'setup_verified':{'model_assets':{'qwen_tts':assets['files']},'source_members':[]}}}
        batch['archive_sha256']=b.digest(b'artificial archive')
        evidence={'model-card.md':b'card','model-locks.json':encoded({'qwen_tts':{'asset_lock':assets,'asset_lock_sha256':b.canonical(assets),'source_lock':[]}}),
            'source-manifest.jsonl':encoded(source)+b'\n','execution-plan.json':b'{}','unarmed-smoke-plan.json':b'{}',
            'tts-result.zip':b'artificial archive'}
        batch['plan_sha256']=b.digest(evidence['execution-plan.json'])
        review={'schema':'bounded-batch-source-rights-review-v1','status':'allowed_for_this_project_internal_noncommercial_research_only',
            'source_model_id':b.TTS_MODEL,'source_revision':b.TTS_REVISION,'voice_kind':'official_builtin_custom_voice',
            'voices':['Dylan','Serena'],'user_reference_audio_supplied':False,'public_audio_redistribution_rights':'not_established_by_this_record',
            'commercial_use_rights':'not_reviewed','tts_archive_sha256':batch['archive_sha256'],'old_source_rights_decision_reused':False,
            'permissions':{'claim_human_consent':False,'claim_human_gold':False,'claim_independent_test':False,
                'commercial_use_reviewed':False,'project_internal_noncommercial_research':True,'publish_raw_audio':False},
            'execution_plan_sha256':batch['plan_sha256'],'source_manifest_sha256':b.digest(evidence['source-manifest.jsonl']),
            'smoke_plan_sha256':b.digest(evidence['unarmed-smoke-plan.json']),'batch_id':batch['batch_id'],
            'model_asset_lock_sha256':b.canonical(assets)}
        for field,name,arch in [('tts_load_receipt_sha256','tts-load-receipt.json','output/load-receipt.json'),
            ('tts_session_summary_sha256','tts-session-summary.json','output/session-summary.json'),
            ('tts_batch_terminal_sha256','tts-batch-terminal.json','terminal.json')]:
            evidence[name]=d[arch]=b'{}';review[field]=b.digest(b'{}')
        parsed=b.loads(d[prefix+'parsed.json'])
        review['items']=[{**{k:item[k] for k in ('item_id','speaker','split','intended_text','seed')},
            'source_row':source,'source_row_canonical_sha256':b.canonical(source),'request':b.loads(d[prefix+'input-evidence.json'])['request'],
            'tts_terminal_sha256':b.digest(d[prefix+'terminal.json']),**{k:parsed[k] for k in ('wav_sha256','pcm_sha256','frame_count')}}]
        review['evidence_files']={n:{'bytes':len(v),'sha256':b.digest(v)} for n,v in evidence.items()}
        self.rights_contract=copy.deepcopy(b.source_contract())
        self.rights_contract['native_dependencies']['tts_setup_models']['sha256']=b.digest(evidence['model-locks.json'])
        return review,evidence,batch,[item]
    def fixture_rights_authority(self):
        stack=ExitStack();stack.enter_context(patch.object(b,'TTS_CARD_SHA256',b.digest(b'card')))
        stack.enter_context(patch.object(b,'source_contract',return_value=self.rights_contract))
        return stack
    def test_rights_fixture_bound(self):
        f=self.rights_fixture()
        with self.fixture_rights_authority():self.assertEqual(set(b.verify_rights(*f)),{'clip-000001'})
    def test_rights_license_string_not_authority(self):
        f=self.rights_fixture();f[0]['declared_model_license']='Apache-2.0';f[0]['source_revision']='unreviewed'
        with self.fixture_rights_authority():
            with self.assertRaisesRegex(ValueError,'source-rights scope'):b.verify_rights(*f)
    def test_rights_old_sohee_or_broadened_scope_rejected(self):
        for change in [lambda r:r.update(old_source_rights_decision_reused=True),lambda r:r['permissions'].update(publish_raw_audio=True)]:
            f=self.rights_fixture();change(f[0])
            with self.assertRaisesRegex(ValueError,'permissions outside'):b.verify_rights(*f)
    def test_rights_original_source_split_rebound(self):
        f=self.rights_fixture();r=f[0]['items'][0];r['source_row']['split']='development_a';r['source_row_canonical_sha256']=b.canonical(r['source_row'])
        with self.fixture_rights_authority():
            with self.assertRaisesRegex(ValueError,'source row not original'):b.verify_rights(*f)
    def test_rights_loaded_model_inventory_mismatch(self):
        f=self.rights_fixture();f[2]['load']['receipt']['setup_verified']['model_assets']['qwen_tts']=[]
        with self.fixture_rights_authority():
            with self.assertRaisesRegex(ValueError,'actual loaded assets'):b.verify_rights(*f)
    def test_rights_model_lock_self_rebound_rejected(self):
        f=self.rights_fixture();review,evidence,batch,items=f
        locks=b.loads(evidence['model-locks.json']);assets=locks['qwen_tts']['asset_lock']
        assets['files'][0]['sha256']='0'*64;locks['qwen_tts']['asset_lock_sha256']=b.canonical(assets)
        evidence['model-locks.json']=encoded(locks);review['model_asset_lock_sha256']=b.canonical(assets)
        batch['load']['receipt']['setup_verified']['model_assets']['qwen_tts']=assets['files']
        review['evidence_files']={n:{'bytes':len(v),'sha256':b.digest(v)} for n,v in evidence.items()}
        with self.fixture_rights_authority():
            with self.assertRaisesRegex(ValueError,'official model-locks source'):b.verify_rights(*f)
    def test_rights_actual_source_members_mismatch(self):
        f=self.rights_fixture();f[2]['load']['receipt']['setup_verified']['source_members']=[{'distribution':'qwen-tts','relative_path':'changed.py','sha256':'0'*64}]
        with self.fixture_rights_authority():
            with self.assertRaisesRegex(ValueError,'source members'):b.verify_rights(*f)
    def test_sense_decoded_metadata_cross_mismatch_rebound(self):
        batch,audio,wav=asr_fixture(self.root,'sensevoice','ANGRY');p='output/clip-000001/'
        raw=b.loads(batch['data'][p+'raw.json']);par=b.loads(batch['data'][p+'parsed.json'])
        new=raw['raw_result'][0][0]['text'].replace('ANGRY','SURPRISED')
        raw['raw_result'][0][0]['text']=new;raw['capture_calls'][0]['decoded']=new
        raw['raw_decode_attempts'][0]['raw_decoded_result']=new
        par['token_evidence']['decoded']=new;par['token_binding']['raw_decoded_sha256']=b.digest(new.encode())
        batch['data'][p+'raw.json']=encoded(raw);batch['data'][p+'parsed.json']=encoded(par);rebind(batch)
        with self.assertRaisesRegex(ValueError,'parsed outcome'):
            b.verify_asr(batch,'clip-000001',audio,wav,'b'*64)
    def loader_fixture(self):
        root=self.root/'package';root.mkdir();(root/'evidence').mkdir()
        batch,item=tts_fixture();wav=batch['data']['output/clip-000001/clip-000001.wav'];(root/'audio.wav').write_bytes(wav)
        audio,pcm=b.wav_observation(wav);vocab={'你':1027,'好':1401,'小':1462,'窝':2093}
        row=b.screen_text(screen_item(),'你好小窝','你好小窝',audio,vocab);row['audio_path']='audio.wav'
        record={'recording':'artificial','audio_path':'audio.wav','audio_sha256':audio['file_sha256'],
            'pcm_sha256':audio['pcm_sha256'],'samples':audio['frames'],'derived_pad300_pcm_sha256':b.digest(pcm+b'\0'*9600),
            'expected_output_frames':row['expected_output_frames'],'ctc_text':row['ctc_text'],'ctc_ids':row['ctc_ids'],
            'split':'train','pool':'positive','current_experiment_admission':False,'human_gold':False,'product_qualified':False,'independent_test':False,
            'source_metadata':row,'label_source':'weak-machine-agreement','purpose':'noncommercial-research-only'}
        b.write_json(root/'report.json',{'items':[row]});b.write_json(root/'fsmn-candidates.json',[record]);b.write_jsonl(root/'weak-fulltext.jsonl',[row])
        (root/'evidence/fsmn-tokens.txt').write_bytes(b'artificial vocabulary')
        b.manifest_directory(root);return root,vocab
    def test_fulltext_consumer_reads_candidate_without_admission(self):
        root,vocab=self.loader_fixture()
        with patch.object(b,'fsmn_vocabulary',lambda raw:vocab):self.assertEqual(len(list(b.load_fsmn_candidates(root))),1)
    def test_fulltext_consumer_fields_rebound(self):
        root,vocab=self.loader_fixture();saved=(root/'fsmn-candidates.json').read_bytes()
        for field,value in [('ctc_ids',[]),('split','development_a'),('derived_pad300_pcm_sha256','0'*64),
            ('current_experiment_admission',True),('human_gold',True),('product_qualified',True),('independent_test',True)]:
            rows=b.loads(saved);rows[0][field]=value;(root/'fsmn-candidates.json').write_bytes(encoded(rows));rehash_package(root)
            with self.subTest(field=field),patch.object(b,'fsmn_vocabulary',lambda raw:vocab):
                with self.assertRaises(ValueError):list(b.load_fsmn_candidates(root))
    def test_fulltext_conditioned_provenance_rebound(self):
        root,vocab=self.loader_fixture();saved=(root/'fsmn-candidates.json').read_bytes()
        for field,value in [('intended_text','changed'),('asr_protocols',['raw']),('source_language_conditions',['auto']),
            ('detected_language','Chinese'),('language_is_model_independent_observation',True)]:
            rows=b.loads(saved);rows[0]['source_metadata'][field]=value;(root/'fsmn-candidates.json').write_bytes(encoded(rows));rehash_package(root)
            with self.subTest(field=field),patch.object(b,'fsmn_vocabulary',lambda raw:vocab):
                with self.assertRaisesRegex(ValueError,'scope/provenance'):list(b.load_fsmn_candidates(root))
    def test_fulltext_manifest_report_rebound(self):
        root,vocab=self.loader_fixture();(root/'weak-fulltext.jsonl').write_bytes(b'');rehash_package(root)
        with patch.object(b,'fsmn_vocabulary',lambda raw:vocab):
            with self.assertRaisesRegex(ValueError,'manifest/report'):list(b.load_fsmn_candidates(root))
    def test_fulltext_semantic_source_rebound(self):
        for case in ('split','pool','ctc_text','source_pcm','source_frames','votes','lineage'):
            with self.subTest(case=case):
                root,vocab=self.loader_fixture();rows=b.loads((root/'fsmn-candidates.json').read_bytes());r=rows[0];s=r['source_metadata']
                if case=='split':r['split']=s['split']='development_a';r['pool']=None
                if case=='pool':r['pool']='other'
                if case=='ctc_text':r['ctc_text']=s['ctc_text']=s['machine_text']='小窝小窝';r['ctc_ids']=s['ctc_ids']=[vocab[c] for c in r['ctc_text']]
                if case=='source_pcm':s['pcm_sha256']='0'*64
                if case=='source_frames':s['frames']+=1
                if case=='votes':s['normalized_asr_votes']=['小窝小窝','小窝小窝']
                if case=='lineage':s['lineage_group']='changed'
                (root/'fsmn-candidates.json').write_bytes(encoded(rows));(root/'report.json').write_bytes(encoded({'items':[s]}));(root/'weak-fulltext.jsonl').write_bytes(encoded(s)+b'\n');rehash_package(root)
                with patch.object(b,'fsmn_vocabulary',lambda raw:vocab):
                    with self.assertRaisesRegex(ValueError,'semantics changed'):list(b.load_fsmn_candidates(root))
                import shutil;shutil.rmtree(root)
    def test_fulltext_missing_manifest_identity(self):
        root,vocab=self.loader_fixture();m=b.loads((root/'MANIFEST.json').read_bytes());del m['fsmn-candidates.json'];(root/'MANIFEST.json').write_bytes(encoded(m))
        with self.assertRaisesRegex(ValueError,'missing fulltext'):list(b.load_fsmn_candidates(root))


if __name__=='__main__':unittest.main()
