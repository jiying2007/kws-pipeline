"""Hand-built synthetic archives, no model files, runtime, or real audio inputs."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'training')]
from language_conditioned_vote import (SCHEMA, ADAPTER_SHA256, PROTOCOL_SHA256, MAINTAINED_SHA256,
    canonical, digest, loads, read_archive, verify_contents)
from import_language_conditioned_result import import_slot
from speech_data_queue import capture, evaluate, inspect_pcm16_wav
from weak_speech_dataset import load_examples
from sense_binding import CONTRACT_SHA256, TOKENIZER_SHA256


def encoded(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()


def fixture(root, family='qwen3-asr', text='你好小窝'):
    wavfile=root/'audio.wav'
    if not wavfile.exists():
        with wave.open(str(wavfile),'wb') as r:
            r.setparams((1,2,16000,0,'NONE','not compressed'));r.writeframes(b'\x01\x00'*3200)
    wav=wavfile.read_bytes();audio=inspect_pcm16_wav(wavfile);pcm=wav[44:]
    floatpcm=struct.pack('<'+'f'*3200,*([1/32768]*3200))
    binding={'schema':'cosyvoice12-pcm-binding-v1','opaque_id':'clip-000001',
             'operations':{'mix':False,'resample':False,'trim':False},
             'wav':{'bytes':len(wav),'format':'RIFF-WAVE-fmt16-PCM1-data-only','header_bytes':44,'sha256':digest(wav)}}
    for name,raw,dtype,enc in [('raw_pcm16',pcm,'<i2','signed-int16-le'),('float32_pcm',floatpcm,'<f4','ieee754-binary32-le')]:
        d={'bytes':len(raw),'channels':1,'dtype':dtype,'encoding':enc,'frame_count':3200,
           'sample_order':'unchanged','sample_rate_hz':16000,'sha256':digest(raw),'shape':[3200]}
        if name=='float32_pcm':d['normalization']='signed_int16 / 32768.0'
        binding[name]=d;binding[name+'_descriptor_sha256']=canonical(d)
    qwen=family=='qwen3-asr';model='Qwen/Qwen3-ASR-0.6B' if qwen else 'FunAudioLLM/SenseVoiceSmall'
    key='qwen06' if qwen else 'sensevoice';language='Chinese' if qwen else 'zh'
    slot={'slot_id':key+'-clip-000001','opaque_id':'clip-000001','model_key':key,'model_id':model,
          'language_argument':language,'wav_sha256':digest(wav),'pcm_sha256':digest(pcm),'frame_count':3200}
    conf={'schema':SCHEMA,'source_language_condition':language,'source_lock_sha256':'1'*64,
          'expected_schema_sha256':'2'*64,'decoder_manifest_sha256':'3'*64}
    recognizer={'id':key,'family':family,'model_id':model,'revision':'synthetic-fixture-only',
        'model_sha256':('4' if qwen else '5')*64,'config_sha256':canonical(conf),'blind':True,
        'protocol':SCHEMA,'protocol_config':conf,'archive_bindings':{}}
    dec={'schema':'source-language-single-slot-execution-v1','scope':'one_existing_pcm_one_source_language_conditioned_decode',
         'experiment_id':'synthetic-fixture-not-an-execution','model_id':model,'slot':slot,'source_language_condition':language,
         'context':'','hotwords':[],'target_text_provided_to_asr':False,'batch_size':1,'candidate_count':1,
         'max_inference_calls':1,'retries':0,'model_dtype':'float32','asset_lock_sha256':recognizer['model_sha256'],
         **{k:conf[k] for k in ('source_lock_sha256','expected_schema_sha256','decoder_manifest_sha256')}}
    receipt={'schema':SCHEMA,'model_id':model,'declaration':dec,'maintained_adapter_sha256':MAINTAINED_SHA256,
             'receipt':{'assets':{'model_id':model,'revision':recognizer['revision'],'asset_lock_sha256':recognizer['model_sha256']},
                        'device':'cpu','dtype':'float32','load':{'complete_checkpoint_verification_passed':True}}}
    before={'schema':SCHEMA,'model_id':model,'opaque_id':'clip-000001','input_binding':binding,'context':''}
    parsed={'schema':SCHEMA,'model_id':model,'opaque_id':'clip-000001','binding_verified':True,'detected_language':None,
            'language_is_model_independent_observation':False,'source_language_condition':language,'human_gold':False,
            'label_scope':'weak-research-only','transcript_repair_applied':False,'quality_flags':[],'completeness':'complete','raw_text':text}
    content={'input.wav':wav}
    if qwen:
        prefix=[151644,8948,198,151645,198,151644,872,198,151669,151676,151670,151645,198,151644,77091,198,11528,8453,151704]
        template='<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n'
        ids=[10,11,151645]
        before.update(input_ids=[prefix],empty_context_template=template,prompt_prefix=template+'language Chinese<asr_text>',
            force_language='Chinese',messages=[{'role':'system','content':''},{'role':'user','content':[{'type':'audio','audio':''}]}],
            generation_options={'do_sample':False,'eos_token_id':[151645,151643],'max_new_tokens':256},
            processor_tensor_evidence={'input_ids':{'dtype':'torch.int64','sha256':digest(struct.pack('<'+'q'*len(prefix),*prefix)),
                                                  'shape':[1,len(prefix)]}})
        generation={**before,'forward_completion_confirmed':True,'returned_sequence_ids':[prefix+ids],
                    'returned_sequence_shape':[1,len(prefix)+len(ids)]}
        content['output/raw-generation.json']=encoded(generation)
        raw={**generation,'generation_evidence_sha256':digest(content['output/raw-generation.json']),
             'generated_token_ids':ids,'input_ids_after_generate':[prefix],'input_ids_unchanged':True,'prefix_preserved':True,
             'prefix_token_count':len(prefix),'generation_tail_only':True,'decoded_tail':text,
             'batch_decode_skip_special_tokens':[text],'decoded_with_special_tokens':text+'<|im_end|>',
             'batch_decode_with_special_tokens':[text+'<|im_end|>']}
    else:
        tokenizer={'sha256':TOKENIZER_SHA256,'vocabulary_size':25055,'lexical_unknown_id':0,'lexical_unknown_piece':'<unk>',
                   'emo_unknown_id':25009,'emo_unknown_is_lexical_unknown':False}
        ids=[24884,25004,24993,25017,10153,11590,11885,15592]
        decoded='<|zh|><|NEUTRAL|><|Speech|><|woitn|>'+text
        before.update(source_language_condition='zh',hotwords=[],tokenizer_identity=tokenizer,
            inference_arguments={'language':'zh','use_itn':False,'text_norm':'woitn','fs':16000,'data_type':'sound',
                                 'output_timestamp':False,'ban_emo_unk':False,'device':'cpu'})
        receipt['receipt']['tokenizer_identity']=tokenizer
        raw={**before,'forward_completion_confirmed':True,'rich_postprocess_applied':False,'transcript_repair_applied':False,
             'raw_result_python_type':'tuple','raw_result':[[{'key':'clip-000001','text':decoded}],{}],
             'capture_calls':[{'decoded':decoded,'token_ids':ids}],
             'raw_decode_attempts':[{'decode_kwargs':{},'raw_decoded_result':decoded,'raw_token_ids':ids}]}
        parsed.update(metadata=['zh','NEUTRAL','Speech','woitn'],token_evidence={'decoded':decoded,'token_ids':ids},
            token_binding={'raw_token_ids':ids,'raw_decoded_sha256':digest(decoded.encode()),'raw_redecode_matches_capture':True,
                'metadata_boundary_verified':True,'lexical_unknown_present':False,'parser_contract_sha256':CONTRACT_SHA256,
                'tokenizer':tokenizer,'metadata_token_ids':ids[:4],'lexical_token_ids':ids[4:],
                'metadata_pieces':['<|zh|>','<|NEUTRAL|>','<|Speech|>','<|woitn|>']})
    content['output/raw.json']=encoded(raw);parsed['raw_evidence_sha256']=digest(content['output/raw.json'])
    content['output/parsed.json']=encoded(parsed)
    for n,v in [('output/input-evidence.json',before),('output/native-declaration.json',dec),('output/load-receipt.json',receipt),
        ('task.json',{'slot':slot,'config':{'protocol_sha256':PROTOCOL_SHA256,'maintained_adapter_sha256':MAINTAINED_SHA256,
            'manifest_sha256':conf['decoder_manifest_sha256']},'source_manifest':[
            {'path':'public/'+n,'sha256':h} for n,h in [('native_adapter.py',ADAPTER_SHA256),
             ('language_conditioned_protocol.py',PROTOCOL_SHA256),('adapters.py',MAINTAINED_SHA256)]]}),
        ('terminal.json',{'returncode':0,'termination_reason':'normal_exit','model_leader_reaped':True})]:content[n]=encoded(v)
    content['output/worker-result.json']=encoded({'status':'completed_bound','binding_verified':True,'slot_id':slot['slot_id'],
        'result':parsed,'raw_evidence_sha256':digest(content['output/raw.json']),'parsed_sha256':digest(content['output/parsed.json'])})
    manifest={'experiment_id':dec['experiment_id'],'slot_id':slot['slot_id']}
    lock={'item_id':'fixture','operation':key,'slot_id':slot['slot_id'],'experiment_id':dec['experiment_id']}
    return manifest,content,recognizer,audio,lock


def archive(manifest,content):
    manifest=copy.deepcopy(manifest)
    manifest['files']={n:{'bytes':len(b),'sha256':digest(b)} for n,b in content.items()}
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in content.items():z.writestr(n,b)
        z.writestr('PACKAGE-MANIFEST.json',encoded(manifest))
    return stream.getvalue()


def rebind_saved_hashes(f):
    """Recompute every envelope digest, leaving semantic disagreements intact."""
    manifest,content,recognizer,audio,lock=f
    raw=loads(content['output/raw.json'])
    if 'output/raw-generation.json' in content:
        raw['generation_evidence_sha256']=digest(content['output/raw-generation.json'])
    content['output/raw.json']=encoded(raw)
    parsed=loads(content['output/parsed.json'])
    parsed['raw_evidence_sha256']=digest(content['output/raw.json'])
    content['output/parsed.json']=encoded(parsed)
    worker=loads(content['output/worker-result.json'])
    worker.update(result=parsed,raw_evidence_sha256=digest(content['output/raw.json']),
                  parsed_sha256=digest(content['output/parsed.json']))
    content['output/worker-result.json']=encoded(worker)
    packed=archive(manifest,content);lock['archive_sha256']=digest(packed)
    manifest,content=read_archive(packed,lock['archive_sha256'])
    return manifest,content,recognizer,audio,lock


class ConditionedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.f=fixture(self.root)

    def change(self,name,action,f=None):
        f=self.f if f is None else f
        d=loads(f[1][name]);action(d);f[1][name]=encoded(d)
        return f

    def check(self,f=None):
        return verify_contents(*(self.f if f is None else f))

    def test_qwen_handmade(self):self.assertEqual(self.check(),'你好小窝')
    def test_sense_handmade(self):self.assertEqual(self.check(fixture(self.root,'sensevoice')),'你好小窝')
    def test_no_runtime_import(self):
        self.check();self.check(fixture(self.root,'sensevoice'))
        self.assertFalse(set(sys.modules)&{'torch','numpy','sentencepiece','qwen_asr','funasr','transformers'})
    def test_invalid_raw_variants(self):
        variants=[('decoded_tail','你好小屋'),('generated_token_ids',[1,2]),('prefix_preserved',False),
                  ('input_ids_after_generate',[[1]]),('prefix_token_count',1),('returned_sequence_shape',[2,2]),
                  ('returned_sequence_ids',[[1,2]]),('context','target'),('force_language','auto'),
                  ('generation_tail_only',False),('decoded_with_special_tokens','tamper'),('forward_completion_confirmed',False)]
        for field,value in variants:
            with self.subTest(field=field):
                f=copy.deepcopy(self.f);self.change('output/raw.json',lambda d:d.update({field:value}),f)
                with self.assertRaises(ValueError):self.check(f)
    def test_semantic_guards_after_all_hashes_rebound(self):
        cases=[
            ('qwen3-asr','output/raw.json',lambda d:d.update(prefix_preserved=False),'Qwen input prefix changed'),
            ('qwen3-asr','output/raw.json',lambda d:d.update(input_ids_after_generate=[[1]]),'Qwen input prefix changed'),
            ('sensevoice','output/raw.json',lambda d:d['capture_calls'][0].update(token_ids=[1]),'Sense native capture/token mismatch'),
            ('sensevoice','output/parsed.json',lambda d:d['token_binding'].update(metadata_token_ids=[1,2,3,4]),'Sense metadata token boundary'),
            ('qwen3-asr','output/parsed.json',lambda d:d.update(detected_language='Chinese'),'parsed scope/binding mismatch')]
        for family,name,mutate,reason in cases:
            with self.subTest(family=family,reason=reason):
                f=fixture(self.root,family)
                self.change(name,mutate,f)
                with self.assertRaisesRegex(ValueError,reason):self.check(rebind_saved_hashes(f))

    def test_bad_parsed_scope(self):
        for field,value in [('detected_language','zh'),('human_gold',True),('raw_text','小窝小窝'),
                            ('language_is_model_independent_observation',True),('quality_flags',['warning'])]:
            with self.subTest(field=field):
                f=copy.deepcopy(self.f);self.change('output/parsed.json',lambda d:d.update({field:value}),f)
                with self.assertRaises(ValueError):self.check(f)
    def test_bad_pcm(self):
        f=copy.deepcopy(self.f);f[1]['input.wav']=f[1]['input.wav'][:-2]+b'\x02\x00'
        with self.assertRaises(ValueError):self.check(f)
    def test_source_and_model_drift(self):
        for name,action in [('task.json',lambda d:d['source_manifest'][0].update(sha256='0'*64)),
            ('output/load-receipt.json',lambda d:d['receipt']['assets'].update(revision='other')),
            ('output/load-receipt.json',lambda d:d['receipt']['load'].update(complete_checkpoint_verification_passed=False)),
            ('output/native-declaration.json',lambda d:d.update(target_text_provided_to_asr=True))]:
            f=copy.deepcopy(self.f);self.change(name,action,f)
            with self.assertRaises(ValueError):self.check(f)
    def test_sense_token_tamper(self):
        f=fixture(self.root,'sensevoice');self.change('output/raw.json',lambda d:d['capture_calls'][0].update(token_ids=[1]),f)
        with self.assertRaises(ValueError):self.check(f)
    def test_sense_false_metadata(self):
        f=fixture(self.root,'sensevoice');self.change('output/parsed.json',lambda d:d['token_binding'].update(metadata_token_ids=[1,2,3,4]),f)
        with self.assertRaises(ValueError):self.check(f)
    def test_archive_identity(self):
        m,c,*_=self.f;b=archive(m,c)
        self.assertEqual(read_archive(b,digest(b))[1]['input.wav'],c['input.wav'])
        with self.assertRaises(ValueError):read_archive(b,'0'*64)
    def test_archive_traversal(self):
        m,c,*_=copy.deepcopy(self.f);c['../escape']=b'bad';b=archive(m,c)
        with self.assertRaises(ValueError):read_archive(b,digest(b))
    def test_archive_roster(self):
        m,c,*_=self.f;b=archive(m,c);stream=io.BytesIO(b)
        with zipfile.ZipFile(stream,'a') as z:z.writestr('unlisted',b'bad')
        b=stream.getvalue()
        with self.assertRaises(ValueError):read_archive(b,digest(b))
    def import_fixture(self,f=None):
        m,c,r,a,l=self.f if f is None else f;b=archive(m,c);l['archive_sha256']=digest(b)
        r['archive_bindings'][l['slot_id']]=l;p=self.root/(r['id']+'.zip');p.write_bytes(b)
        e=import_slot(p,self.root/r['id'],item_id='fixture',operation=r['id'],recognizer=r,lock=l)
        for ref in e['raw_artifacts']:ref['path']=r['id']+'/'+ref['path']
        e['archive_path']=r['id']+'/'+e['archive_path']
        e['archive_members']={k:r['id']+'/'+v for k,v in e['archive_members'].items()}
        return r,e
    def test_full_import_queue_loader(self):
        r1,e1=self.import_fixture();r2,e2=self.import_fixture(fixture(self.root,'sensevoice'))
        (self.root/'rights.txt').write_bytes(b'SYNTHETIC FIXTURE ONLY')
        (self.root/'tokens.txt').write_text('<blk> 0\nni3 1\nhao3 2\nxiao3 3\nwo1 4\n')
        item={'id':'fixture','source_kind':'real_speech','intent_kind':'positive','intended_text':'小窝小窝',
              'lineage_roots':['synthetic-fixture'],'split':'train','recipe_sha256':'0'*64,'resources':[],
              'rights':{'research_use':'allowed','path':'rights.txt','sha256':digest((self.root/'rights.txt').read_bytes())},
              'audio_path':'audio.wav',**self.f[3]}
        plan={'schema':'speech-data-queue-v1','call_budget':2,'keywords':['你好小窝','小窝小窝'],
              'recognizers':[r1,r2],'items':[item],'tokens_path':'tokens.txt','tokens_sha256':digest((self.root/'tokens.txt').read_bytes())}
        events=[e1,e2]
        for e in events:e['plan_sha256']=canonical(plan)
        (self.root/'plan.json').write_bytes(encoded(plan));(self.root/'events.json').write_bytes(encoded(events))
        report=capture(self.root/'plan.json',self.root/'events.json',self.root,self.root/'package')
        self.assertEqual(report['statistics']['known_calls'],2);self.assertEqual(report['items'][0]['machine_text'],'你好小窝')
        self.assertEqual(report['items'][0]['intended_text'],'小窝小窝')
        self.assertEqual(len(list(load_examples(self.root/'package'))),1)
        e1['detected_language']='zh'
        self.assertEqual(evaluate(plan,events,self.root)['items'][0]['status'],'quarantine')
    def test_loader_rejects_conditioned_metadata_tamper(self):
        self.test_full_import_queue_loader()
        package=self.root/'package'; path=package/'weak-train.jsonl'
        original=loads(path.read_bytes()); inventory=loads((package/'MANIFEST.json').read_bytes())
        changes={'asr_protocols':['anonymous-two-asr-raw-v1']*2,
                 'source_language_conditions':['auto','auto'], 'detected_language':'zh',
                 'language_is_model_independent_observation':True,'intended_text':'different intent'}
        for field,value in changes.items():
            for remove in (False,True):
                with self.subTest(field=field,missing=remove):
                    row=copy.deepcopy(original)
                    if remove:row.pop(field)
                    else:row[field]=value
                    path.write_bytes(encoded(row)+b'\n')
                    inventory['weak-train.jsonl']=digest(path.read_bytes())
                    (package/'MANIFEST.json').write_bytes(encoded(inventory))
                    with self.assertRaises(ValueError):list(load_examples(package))
        path.write_bytes(encoded(original)+b'\n')
        inventory['weak-train.jsonl']=digest(path.read_bytes())
        (package/'MANIFEST.json').write_bytes(encoded(inventory))
        self.assertEqual(len(list(load_examples(package))),1)

    def test_failed_import_retains_called_fact(self):
        self.change('output/raw.json',lambda d:d.update(prefix_preserved=False))
        with self.assertRaises(ValueError):self.import_fixture()
        failure=loads((self.root/'qwen06/import-failure.json').read_bytes())
        self.assertTrue((self.root/'qwen06/original.zip').exists())
        self.assertEqual(failure['retained_event']['call_state'],'called')
        self.assertEqual(failure['retained_event']['outcome'],'success')
        self.assertEqual(failure['retained_event']['import_validation'],'rejected')
    def test_existing_output_never_overwritten(self):
        self.import_fixture()
        with self.assertRaises(FileExistsError):self.import_fixture()


if __name__=='__main__':unittest.main()
