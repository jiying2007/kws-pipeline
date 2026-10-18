"""Bounded native batch archive -> research weak candidates, strictly offline.

No model/tokenizer imports. Complete private archives and original provenance
are retained. Source/receipt checks establish saved-byte consistency, not
cryptographic execution authenticity, acoustic gold, or training permission.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct
import unicodedata
import wave
import zipfile

from language_conditioned_vote import (SCHEMA as CONDITIONED, canonical, digest,
    loads, require, verify_pcm, verify_qwen, verify_sense)
from speech_data_queue import durable_write, inspect_pcm16_wav, actual_tokens, load_tokens
from speech_history_evidence import TTS_MODEL, TTS_REVISION, TTS_CARD_SHA256

SCHEMA='bounded-batch-weak-import-v1'
NATIVE='bounded-native-execution-v1'
SOURCE_SHA='30cd9db25ae5da28b3d812ee3f119e4b58df601930801f892a33fd5d4ef870d0'
FSMN_TOKENS_SHA='5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa'
KEYWORDS=('你好小窝','小窝小窝')


def normalize(text):
    require(type(text) is str and len(text)<=32768,'invalid transcript')
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def source_contract():
    raw=Path(__file__).with_name('batch-source-contract.json').read_bytes()
    require(digest(raw)==SOURCE_SHA,'unreviewed batch source contract')
    return loads(raw)


def archive_bytes(raw, expected):
    require(len(raw)<=64*1024**2 and digest(raw)==expected,'frozen archive mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        info=z.infolist();names=[r.filename for r in info]
        require(len(names)==len(set(names)) and 0<len(names)<=2048,'duplicate/oversized archive roster')
        require(sum(r.file_size for r in info)<=64*1024**2,'oversized uncompressed batch')
        for r in info:
            p=PurePosixPath(r.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in r.filename
                and not r.is_dir() and r.file_size<=8*1024**2
                and (r.external_attr>>16)&0o170000!=0o120000,'unsafe batch archive member')
        data={n:z.read(n) for n in names}
    manifest=loads(data['PACKAGE-MANIFEST.json'])
    require(set(data)=={'PACKAGE-MANIFEST.json',*manifest['files']},'batch archive roster mismatch')
    for n,v in manifest['files'].items():
        require(v=={'bytes':len(data[n]),'sha256':digest(data[n])},'batch archive member mismatch')
    return manifest,data


def verify_supplied_readback(proof, receipt, raw):
    """Only supplied receipt identity/byte consistency; no receipt authentication."""
    require(proof['proof_scope']=='parent_tool_receipt_and_local_readback_consistency'
        and proof['sha256']==digest(raw) and proof['bytes']==len(raw),'supplied readback byte mismatch')
    upload=receipt['upload'];transfers=receipt['readback']['transfers']
    require(upload['operation']=='create_library_file' and upload['status']=='succeeded'
        and upload['file_size_bytes']==len(raw) and len(transfers)==1,'supplied upload/readback incomplete')
    back=transfers[0]
    for name in ('library_file_id','file_id'):
        require(upload[name]==back[name]==proof[name],'supplied Library identity mismatch')
    require(upload['current_version_number']==back['current_version_number']==proof['version']
        and back['size_bytes']==len(raw) and back['workspace_path']==proof['readback_path'],
        'supplied readback version/path mismatch')


def verify_cleanup(data, batch_id, plan_sha):
    read=lambda n:loads(data[n])
    start,final,child,ns,terminal=(read(n) for n in ['namespace-start.json','namespace-final.json',
        'process-identity.json','namespace-child.json','terminal.json'])
    require(start['batch_id']==batch_id and start['plan_sha256']==plan_sha
        and start['controller_sha256']==digest(data['controller-source.py']),'namespace start binding mismatch')
    require(final['supervisor_identity']==start['supervisor_identity'] and final['same_supervisor_process'] is True
        and final['observation_scope']=='original_supervisor_pid_namespace'
        and final['cross_namespace_absence_is_proof'] is False and final['observation_error'] is None,
        'cleanup not observed by original supervisor')
    for field,name in [('terminal_sha256','terminal.json'),('child_identity_sha256','process-identity.json'),
                       ('child_namespace_sha256','namespace-child.json')]:
        require(final[field]==digest(data[name]),'cleanup sidecar hash mismatch')
    require(final['child_identity']==child and ns['captured'] is True and ns['pid']==child['pid']
        and ns['child_start_ticks']==child['proc_start_ticks'],'child process identity mismatch')
    for k in ('boot_id','pid_namespace_inode','pid_namespace_link'):
        require(ns[k]==start['supervisor_identity'][k],'child/supervisor namespace mismatch')
    require(child['boot_id']==ns['boot_id'] and terminal['fixture'] is False
        and terminal['returncode']==0 and terminal['termination_reason']=='normal_exit'
        and terminal['model_leader_reaped'] is True
        and final['group_after_terminal']['live_processes']==0
        and terminal['postkill_group_state']['live_processes']==0,'batch process cleanup incomplete')


def verify_batch(manifest,data,lock,execution_plan):
    batch_id=lock['batch_id'];key=lock['model_key'];read=lambda n:loads(data[n])
    require(manifest['batch_id']==batch_id,'batch identity mismatch')
    plan_sha=digest(execution_plan);plan=loads(execution_plan)
    task=read('task.json');config=read('session-config.json');control=read('output/manifest.json')
    declaration=read('output/session-claim.json');summary=read('output/session-summary.json')
    controller=read('summary.json');load=read('output/load-receipt.json')
    items=next(b['items'] for b in plan['batches'] if b['batch_id']==batch_id)
    require(1<=len(items)<=8 and len(items)==len(set(items)),'bounded unique batch required')
    idfield='item_id' if key=='tts06' else 'opaque_id'
    require([r[idfield] for r in config['items']]==items==lock['items'],'batch item roster changed')
    require(task['plan_sha256']==plan_sha and task['batch_id']==batch_id and task['config']==config
        and config['model_key']==key and config['schema']==NATIVE and config['armed'] is True
        and task['fixture'] is False,'batch task/source plan mismatch')
    require(control['items']==config['items'] and control['batch_id']==batch_id
        and declaration['manifest_sha256']==digest(data['output/manifest.json']),'native control manifest mismatch')
    require(declaration['schema']==NATIVE and declaration['batch_id']==batch_id and declaration['model_key']==key
        and declaration['armed'] is True and declaration['item_count']==len(items)
        and declaration['max_inference_calls']==len(items) and declaration['max_model_load_attempts']==1
        and declaration['inference_batch_size']==1 and declaration['retries']==0
        and declaration['cross_item_cache'] is False and declaration['cross_item_context'] is False
        and declaration['weight_reuse_only'] is True,'batch runtime declaration drift')
    contract=source_contract()
    require(declaration['source_identity']==contract['native_dependencies'],'batch native source identities changed')
    for name,sha in contract['execution_sources'].items():
        require(digest(data[name])==sha,'batch execution source changed')
    require(summary['schema']==NATIVE and summary['model_key']==key and summary['fixture'] is False
        and summary['closed'] is True and summary['load_count']==summary['load_attempt_count']==1
        and summary['not_started']==[] and summary['call_count']==summary['completed_calls']==len(items)
        and [r['item_id'] for r in summary['items']]==items,'native summary scope incomplete')
    require(controller['batch_id']==batch_id and controller['model_key']==key and controller['fixture'] is False
        and controller['known_model_loads']==1 and controller['known_call_started']==len(items)
        and controller['forward_completed']==len(items) and controller['unknown_consumption']==0
        and controller['unknown_completion']==0 and controller['retries']==0
        and [r['item_id'] for r in controller['items']]==items,'controller/native completion disagreement')
    require(load['schema']==NATIVE and load['fixture'] is False and load['model_key']==key
        and load['load_count']==load['load_attempt_count']==1,'unbound batch model load')
    if key!='tts06':
        expected=contract['asr_models'][key];assets=load['receipt']['load']['assets']
        require(all(assets[k]==v for k,v in expected.items()) and load['receipt']['model_id']==expected['model_id']
            and load['receipt']['load']['load']['complete_checkpoint_verification_passed'] is True,
            'ASR model/revision/checkpoint mismatch')
    else:
        require(load['receipt']['model_id']==TTS_MODEL and load['receipt']['revision']==TTS_REVISION,'TTS model/revision mismatch')
    require(load['receipt']['network_events']==[],'native network attempt present')
    verify_cleanup(data,batch_id,plan_sha)
    return {'model_key':key,'items':items,'config':config,'summary':summary,'controller':controller,
            'load':load,'declaration':declaration,'data':data,'archive_sha256':lock['archive_sha256'],
            'batch_id':batch_id,'plan_sha256':plan_sha}


def verify_item(batch,item_id):
    data=batch['data'];prefix='output/'+item_id+'/'
    read=lambda name:loads(data[prefix+name])
    terminal=read('terminal.json');parsed=read('parsed.json');raw=read('raw.json')
    row=next(r for r in batch['summary']['items'] if r['item_id']==item_id)
    parent=next(r for r in batch['controller']['items'] if r['item_id']==item_id)
    require(row['status']=='completed_bound' and row['binding_verified'] is True
        and row['terminal_persistence_failed'] is False and row['terminal_sha256']==digest(data[prefix+'terminal.json']),
        'native summary does not confirm exact item terminal')
    require(parent['native_summary_confirmed'] is True and parent['forward_completed'] is True
        and parent['status']=='completed_bound' and parent['known_call_started'] is True,'controller item not complete')
    require(terminal['schema']==NATIVE and terminal['item_id']==item_id and terminal['model_key']==batch['model_key']
        and terminal['fixture'] is False and terminal['status']=='completed_bound'
        and all(terminal[k] is True for k in ('binding_verified','evidence_complete','call_started','callback_invoked',
                                             'consumed','forward_completed','forward_completion_confirmed'))
        and terminal['retry_authorized'] is False and terminal['result']==parsed,'item terminal status/result mismatch')
    names={n[len(prefix):] for n in data if n.startswith(prefix)}
    require(set(terminal['retained_files'])==names-{'terminal.json'},'item retained roster incomplete')
    for name,sha in terminal['retained_files'].items():
        require(digest(data[prefix+name])==sha,'item retained artifact hash mismatch')
    consumed=read('consume-receipt.json');outer=loads(data[item_id+'-consume.json'])
    require(consumed==outer and consumed['consumed'] is True and consumed['retry_authorized'] is False
        and consumed['item_id']==item_id and consumed['model_key']==batch['model_key']
        and consumed['batch_id']==batch['batch_id'] and consumed['plan_sha256']==batch['plan_sha256'], 'item durable consumption binding mismatch')
    claim=read('item-claim.json')
    require(claim['item_id']==item_id and claim['model_key']==batch['model_key'] and claim['retry_authorized'] is False,'item claim mismatch')
    require(raw['forward_completion_confirmed'] is True and parsed['raw_evidence_sha256']==digest(data[prefix+'raw.json']),
        'item raw forward/result binding mismatch')
    return terminal,parsed,raw,read('input-evidence.json')


def wav_observation(raw):
    with wave.open(io.BytesIO(raw),'rb') as r:
        require((r.getframerate(),r.getnchannels(),r.getsampwidth())==(16000,1,2),'canonical PCM16 required')
        n=r.getnframes();pcm=r.readframes(n)
    require(0<n<=192000 and len(raw)==44+len(pcm) and len(pcm)==2*n and any(pcm),'invalid/silent PCM16')
    return {'file_sha256':digest(raw),'pcm_sha256':digest(pcm),'frames':n},pcm


def verify_tts(batch,item):
    ident=item['item_id'];prefix='output/'+ident+'/'
    term,parsed,raw,before=verify_item(batch,ident);data=batch['data']
    request=next(r for r in batch['config']['items'] if r['item_id']==ident)
    require(request==before['request'] and all(request[k]==item[k] for k in ('item_id','intended_text','speaker','seed')),
        'TTS actual request differs from frozen source item')
    require(before['schema']==raw['schema']==NATIVE and before['item_id']==raw['item_id']==ident
        and before['recipe']==source_contract()['tts_recipe']
        and before['cleared_per_generation_state']==[{'attribute':'rope_deltas','module_index':1}]
        and raw['native_return_type']=='builtins.tuple','TTS native recipe/input identity mismatch')
    require(before['model_id']==parsed['model_id']==TTS_MODEL and before['model_revision']==parsed['model_revision']==TTS_REVISION
        and before['cross_item_context'] is False and before['tts_initial_past_key_values'] is None
        and before['seed_reset']==['random','numpy','torch'] and parsed['source_seed']==request['seed'], 'TTS source/seed/context mismatch')
    generation=loads(data[prefix+'raw-generation.json']);ids=generation['returned_sequence_ids']
    require(generation['schema']==NATIVE and generation['item_id']==ident and generation['forward_completion_confirmed'] is True
        and type(ids) is list and len(ids)==1 and 1<len(ids[0])<=120
        and all(type(t) is int and 0<=t<=2150 for t in ids[0])
        and ids[0][-1]==generation['codec_eos_token_id']==2150 and 2150 not in ids[0][:-1]
        and generation['returned_sequence_shape']==[1,len(ids[0])], 'TTS native token/EOS mismatch')
    codebooks=generation['returned_codebook_token_ids']
    require(len(codebooks)==generation['hidden_state_iteration_count']==len(ids[0]) and codebooks[0] is None,
        'TTS codebook count mismatch')
    for index,entry in enumerate(codebooks[1:]):
        require(type(entry) is list and len(entry)==1 and len(entry[0])==16
            and all(type(x) is int and 0<=x<2048 for x in entry[0]) and entry[0][0]==ids[0][index], 'TTS codebook token mismatch')
    termination={'ended_with_eos':True,'eos':2150,'hit_token_cap_without_eos':False,'iterations':len(ids[0]),'last_token_ids':[2150]}
    require(raw['termination']==parsed['termination']==termination and raw['raw_evidence_complete'] is True
        and raw['token_capture']=={'calls':1,'complete':True} and raw['sample_rate_hz']==24000,'TTS termination/raw completeness mismatch')
    require(len(raw['waveforms'])==1,'TTS waveform count mismatch');w=raw['waveforms'][0]
    blob=data[prefix+w['path']]
    require(w['path']=='native-waveform-000.bin' and w['dtype']=='<f4' and w['strides']==[4]
        and w['bytes']==len(blob)==4*w['shape'][0] and 0<w['shape'][0]<=288000
        and w['sha256']==digest(blob)==parsed['source_float_values_sha256'],'TTS native waveform identity mismatch')
    floats=[v[0] for v in struct.iter_unpack('<f',blob)]
    require(all(math.isfinite(x) and abs(x)<1 for x in floats) and any(floats),'TTS native signal unhealthy')
    statistics={'nonfinite_count':0,'peak':max(map(abs,floats)),'abs_ge_one_count':0,'saturation_count':0}
    require(parsed['source_signal_statistics']==statistics,'TTS source signal statistics changed')
    wav=data[prefix+ident+'.wav'];audio,pcm=wav_observation(wav)
    require(parsed['asr_input_qualified'] is True and parsed['signal_flags']==[]
        and parsed['wav_sha256']==audio['file_sha256'] and parsed['pcm_sha256']==audio['pcm_sha256']
        and parsed['frame_count']==audio['frames']==math.ceil(len(floats)*2/3)
        and parsed['sample_rate_hz']==16000 and parsed['channels']==1 and parsed['sample_width_bytes']==2,
        'TTS derivative PCM qualification mismatch')
    require(parsed==loads(data[prefix+'derived-audio.json'])
        and parsed['raw_audio_sha256']==digest(data[prefix+ident+'.raw-float.wav']), 'TTS derivative/raw float artifact mismatch')
    require(0<parsed['normalized_peak']<1 and parsed['normalized_signal_statistics']['nonfinite_count']==0
        and parsed['normalized_signal_statistics']['saturation_count']==0,'TTS normalized signal unhealthy')
    return parsed,audio,wav


def verify_asr(batch,opaque,audio,wav,tts_archive_sha):
    key=batch['model_key'];prefix='output/'+opaque+'/';data=batch['data']
    terminal,parsed,raw,before=verify_item(batch,opaque)
    require(all(raw[k]==v for k,v in before.items()),'native raw/input evidence mismatch')
    model=source_contract()['asr_models'][key]['model_id'];language='Chinese' if key=='qwen06' else 'zh'
    require(raw['schema']==before['schema']==parsed['schema']==CONDITIONED
        and raw['model_id']==before['model_id']==parsed['model_id']==model
        and raw['opaque_id']==before['opaque_id']==parsed['opaque_id']==opaque, 'ASR model/schema/opaque mismatch')
    require(parsed['binding_verified'] is True and parsed['quality_flags']==[] and parsed['completeness']=='complete'
        and parsed['source_language_condition']==language and parsed['detected_language'] is None
        and parsed['language_is_model_independent_observation'] is False and parsed['human_gold'] is False
        and parsed['label_scope']=='weak-research-only' and parsed['transcript_repair_applied'] is False,'ASR conditioned scope mismatch')
    item=next(r for r in batch['config']['items'] if r['opaque_id']==opaque)
    require(set(item)=={'opaque_id','wav_sha256','pcm_sha256','frame_count'}
        and item['wav_sha256']==audio['file_sha256'] and item['pcm_sha256']==audio['pcm_sha256']
        and item['frame_count']==audio['frames'],'ASR config audio mismatch')
    require(data['anonymous-input/'+audio['file_sha256']+'.wav']==wav,'actual ASR input differs from generated PCM')
    lineage=loads(data['input-lineage.json'])
    require(lineage['tts_readback_archive_sha256']==tts_archive_sha and opaque in lineage['eligible_opaque_ids'], 'ASR TTS readback lineage mismatch')
    verify_pcm(wav,before['input_binding'],audio,opaque)
    boundary=loads(data[prefix+'call-boundary.json'])
    require(boundary['schema']==NATIVE and boundary['item_id']==opaque and boundary['input_evidence_sha256']==digest(data[prefix+'input-evidence.json'])
        and boundary['source_seed_provided'] is False and boundary['asr_fixed_seed']==0,'ASR call boundary mismatch')
    expected={'context':'','past_key_values':None,'use_cache':False} if key=='qwen06' else {'cache':{},'context':'','language':'zh'}
    require(boundary['isolation_arguments']==expected,'ASR cross-item context/cache drift')
    if key=='qwen06':
        generation=loads(data[prefix+'raw-generation.json'])
        require(raw['generation_evidence_sha256']==digest(data[prefix+'raw-generation.json']),'Qwen raw-generation hash mismatch')
        text=verify_qwen(raw,before,generation)
    else:
        metadata=source_contract()['sense_metadata']
        require(raw['tokenizer_identity']['sha256']==metadata['tokenizer_sha256']
            and raw['tokenizer_identity']['vocabulary_size']==metadata['vocabulary_size'],'batch Sense metadata tokenizer mismatch')
        text=verify_sense(raw,parsed,{'receipt':batch['load']['receipt']['load']},emotion_ids=metadata['emotion_ids'])
    require(text==parsed['raw_text'],'ASR native/parsed transcript mismatch')
    return text


def verify_rights(review, evidence, batch, items):
    require(review['schema']=='bounded-batch-source-rights-review-v1'
        and review['status']=='allowed_for_this_project_internal_noncommercial_research_only'
        and review['source_model_id']==TTS_MODEL and review['source_revision']==TTS_REVISION
        and review['voice_kind']=='official_builtin_custom_voice' and set(review['voices'])=={'Dylan','Serena'}
        and review['user_reference_audio_supplied'] is False
        and review['public_audio_redistribution_rights']=='not_established_by_this_record'
        and review['commercial_use_rights']=='not_reviewed'
        and review['tts_archive_sha256']==batch['archive_sha256'],'new batch source-rights scope mismatch')
    require(set(evidence)==set(review['evidence_files']),'rights evidence roster mismatch')
    require(review['permissions']=={'claim_human_consent':False,'claim_human_gold':False,'claim_independent_test':False,
        'commercial_use_reviewed':False,'project_internal_noncommercial_research':True,'publish_raw_audio':False}
        and review['old_source_rights_decision_reused'] is False,'rights permissions outside reviewed scope')
    for name,ref in review['evidence_files'].items():
        require(name in evidence and digest(evidence[name])==ref['sha256'] and len(evidence[name])==ref['bytes'], 'rights evidence changed')
    require(digest(evidence['model-card.md'])==TTS_CARD_SHA256,'official model card mismatch')
    require(digest(evidence['model-locks.json'])==source_contract()['native_dependencies']['tts_setup_models']['sha256'],
        'official model-locks source identity mismatch')
    locks=loads(evidence['model-locks.json'])['qwen_tts'];assets=locks['asset_lock']
    require(assets['model_id']==TTS_MODEL and assets['revision']==TTS_REVISION
        and canonical(assets)==locks['asset_lock_sha256'],'rights official model lock mismatch')
    inventory=[{k:a[k] for k in ('filename','bytes','sha256')} for a in assets['files']]
    require(batch['load']['receipt']['setup_verified']['model_assets']['qwen_tts']==inventory,'TTS actual loaded assets not covered by review')
    source_members=[{k:a[k] for k in ('distribution','relative_path','sha256')}
        for a in batch['load']['receipt']['setup_verified']['source_members']]
    require(source_members==locks['source_lock'],'TTS actual loaded source members not covered by lock')
    require(review['execution_plan_sha256']==batch['plan_sha256']==digest(evidence['execution-plan.json'])
        and review['source_manifest_sha256']==digest(evidence['source-manifest.jsonl'])
        and review['smoke_plan_sha256']==digest(evidence['unarmed-smoke-plan.json'])
        and review['batch_id']==batch['batch_id'],'rights plan/population binding mismatch')
    for field,name,archive_name in [('tts_load_receipt_sha256','tts-load-receipt.json','output/load-receipt.json'),
        ('tts_session_summary_sha256','tts-session-summary.json','output/session-summary.json'),
        ('tts_batch_terminal_sha256','tts-batch-terminal.json','terminal.json')]:
        require(review[field]==digest(evidence[name])==digest(batch['data'][archive_name]),'rights execution receipt mismatch')
    require(review['model_asset_lock_sha256']==canonical(assets)
        and digest(evidence['tts-result.zip'])==batch['archive_sha256'],'rights model/archive mismatch')
    sources=[loads(line) for line in evidence['source-manifest.jsonl'].splitlines() if line.strip()]
    indexed={r['item_id']:r for r in sources}
    reviewed={r['item_id']:r for r in review['items']}
    require(len(reviewed)==len(review['items'])==len(items) and len(indexed)==len(sources),'rights item roster mismatch')
    for item in items:
        ident=item['item_id'];r=reviewed[ident];source=indexed[ident]
        for k in ('item_id','speaker','split','intended_text','seed'):
            require(r[k]==item[k],'review item source/split mismatch: '+k)
        require(r['source_row']==source and r['source_row_canonical_sha256']==canonical(source)
            and source['kind']==item['intent_kind'],'rights source row not original manifest')
        for k in ('speaker','split','intended_text','seed'):
            require(source[k]==item[k],'original item source/split mismatch: '+k)
        require((item['speaker'],item['split']) in {('Dylan','train'),('Serena','development_a')},'voice/split lineage changed')
        prefix='output/'+ident+'/'
        require(r['tts_terminal_sha256']==digest(batch['data'][prefix+'terminal.json']),'rights terminal binding mismatch')
        require(r['request']==loads(batch['data'][prefix+'input-evidence.json'])['request'],'rights actual request mismatch')
        p=loads(batch['data'][prefix+'parsed.json'])
        require(all(r[k]==p[k] for k in ('wav_sha256','pcm_sha256','frame_count')),'rights PCM binding mismatch')
    return {r['item_id']:r['source_row'] for r in review['items']}


def fsmn_vocabulary(raw):
    require(digest(raw)==FSMN_TOKENS_SHA,'official FSMN vocabulary changed')
    result={word:int(number)-1 for word,number in (line.rsplit(maxsplit=1) for line in raw.decode().splitlines())
            if 0<=int(number)-1<2599}
    require(len(set(result.values()))==2599,'official FSMN class count mismatch')
    return result


def screen_text(item,qwen,sense,audio,vocab):
    a,b=normalize(qwen),normalize(sense)
    row={'item_id':item['item_id'],'opaque_id':item['opaque_id'],'speaker':item['speaker'],
         'lineage_group':item['source_row']['lineage_group'],
         'original_source_row':item['source_row'],
         'split':item['split'],'intended_text':item['intended_text'],'qwen_raw_text':qwen,'sense_raw_text':sense,
         'normalized_asr_votes':[a,b],'status':'quarantine','reason':'asr_disagreement','machine_text':None,
         'human_gold':False,'product_training_admitted':False,'research_weak_allowed':False,
         'current_experiment_admission':False,'independent_test':False,
         'asr_protocols':[CONDITIONED,CONDITIONED],'archive_protocol':NATIVE,
         'source_language_conditions':['Chinese','zh'],'detected_language':None,
         'language_is_model_independent_observation':False,**audio}
    if a!=b:return row
    row['machine_text']=a
    if not a:row['reason']='empty_asr_not_proof_of_no_speech';return row
    # Conservatively retain the old interrupted-wake subsequence protection.
    def contains(keyword):
        it=iter(a);return all(any(c==k for c in it) for k in keyword)
    is_intended_negative=item['intent_kind'] in ('negative','confusable','background','negative_candidate','confusable_candidate','background_candidate')
    if is_intended_negative and any(contains(k) for k in KEYWORDS):
        row['reason']='negative_contains_keyword';return row
    try:ids=[vocab[c] for c in a]
    except KeyError:row['reason']='fulltext_consumer_vocabulary_uncovered';return row
    frames=(1+(audio['frames']+4800-400)//160)//3
    require(ids and all(type(i)is int and 0<i<2599 for i in ids),'nonblank full transcript CTC targets required')
    if frames<len(ids)+sum(x==y for x,y in zip(ids,ids[1:])):
        row['reason']='insufficient_fsmn_ctc_frames';return row
    positive=any(k in a for k in KEYWORDS)
    row.update(status='machine_candidate',reason='machine_agreement_only',research_weak_allowed=True,
        actual_kind='positive' if positive else 'speech',ctc_text=a,ctc_ids=ids,
        fsmn_consumer_compatible=True,expected_output_frames=frames,
        pool_candidate='positive' if positive else 'other',
        label_scope='machine-weak-full-transcript-research-candidate-only')
    try:
        names=actual_tokens(a)
        row['four_token_consumer']={'compatible':True,'tokens':names}
    except ValueError:
        row['four_token_consumer']={'compatible':False,'reason':'consumer_specific_transcript_OOV','full_text_preserved':True}
    return row


def write_json(path,value):
    durable_write(path,json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2).encode())


def write_jsonl(path,rows):
    durable_write(path,b''.join(json.dumps(r,ensure_ascii=False,sort_keys=True).encode()+b'\n' for r in rows))


def manifest_directory(root):
    result={p.relative_to(root).as_posix():digest(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'MANIFEST.json'}
    write_json(root/'MANIFEST.json',result)


def import_package(config_path, output):
    """Fresh private package only. The config explicitly freezes supplied evidence."""
    config_path=Path(config_path);root=config_path.parent.resolve();output=Path(output)
    config_raw=config_path.read_bytes();config=loads(config_raw)
    output.mkdir(parents=True,exist_ok=False);durable_write(output/'raw-import-config.json',config_raw)
    def read_ref(ref):
        p=PurePosixPath(ref['path']);require(not p.is_absolute() and '..' not in p.parts,'contained input reference required')
        source=(root/str(p)).resolve(strict=True);require(source.is_relative_to(root) and source.is_file(),'input reference escapes root')
        raw=source.read_bytes();require(digest(raw)==ref['sha256'],'input frozen identity changed');return raw
    require(config['schema']==SCHEMA and config['execution_authorized'] is False,'offline import config required')
    data_dir=output/'evidence';data_dir.mkdir()
    try:
        execution_plan=read_ref(config['execution_plan']);durable_write(data_dir/'execution-plan.json',execution_plan)
        batches={}
        for lock in config['batches']:
            key=lock['model_key'];require(key not in batches,'duplicate model batch')
            directory=data_dir/key;directory.mkdir();raw=read_ref(lock['archive'])
            durable_write(directory/'result.zip',raw)
            require(lock['archive_sha256']==lock['archive']['sha256'],'archive lock mismatch')
            m,contents=archive_bytes(raw,lock['archive_sha256'])
            for name,data in contents.items():
                path=directory/'members'/name;path.parent.mkdir(parents=True,exist_ok=True);durable_write(path,data)
            expected_sidecars={'delivered.json','result-proof.json','reserve-proof.json','result-library-receipts.json','reserve-library-receipts.json'}
            require(set(lock['sidecars'])==expected_sidecars,'incomplete delivery sidecars')
            for name,ref in lock['sidecars'].items():
                durable_write(directory/name,read_ref(ref))
            proof=loads((directory/'result-proof.json').read_bytes());delivered=loads((directory/'delivered.json').read_bytes())
            require(proof['proof_scope']=='parent_tool_receipt_and_local_readback_consistency'
                and proof['sha256']==lock['archive_sha256'] and proof['bytes']==len(raw)
                and delivered['archive_sha256']==lock['archive_sha256'] and delivered['batch_id']==lock['batch_id'], 'supplied result readback binding mismatch')
            require(delivered['authority']=='parent_tool_attestation_and_local_bytes_xattrs_only' and delivered['no_retry'] is True,
                'delivery attestation scope mismatch')
            result_back=read_ref(lock['result_readback']);reserve_back=read_ref(lock['reserve_readback'])
            require(result_back==raw,'retained result readback differs from original archive')
            durable_write(directory/'result-readback.zip',result_back);durable_write(directory/'reserve-readback.zip',reserve_back)
            verify_supplied_readback(proof,loads((directory/'result-library-receipts.json').read_bytes()),result_back)
            reserve_proof=loads((directory/'reserve-proof.json').read_bytes())
            require(reserve_back==contents['reserve.zip'] and reserve_proof==loads(contents['reserve-proof.json']),
                'reserve archive/proof not original executed claim')
            verify_supplied_readback(reserve_proof,loads((directory/'reserve-library-receipts.json').read_bytes()),reserve_back)
            batches[key]=verify_batch(m,contents,lock,execution_plan)
        require(set(batches)=={'tts06','qwen06','sensevoice'},'three stage results required')
        rights_raw=read_ref(config['rights_review']);review=loads(rights_raw)
        rights_data={n:read_ref(r) for n,r in config['rights_evidence'].items()}
        rights_dir=data_dir/'rights';rights_dir.mkdir();durable_write(rights_dir/'SOURCE-RIGHTS-REVIEW.json',rights_raw)
        for n,raw in rights_data.items():
            p=PurePosixPath(n);require(not p.is_absolute() and '..' not in p.parts,'unsafe rights member')
            target=rights_dir/str(p);target.parent.mkdir(parents=True,exist_ok=True);durable_write(target,raw)
        sources=verify_rights(review,rights_data,batches['tts06'],config['items'])
        require(len({r['item_id'] for r in config['items']})==len(config['items'])
            and [r['item_id'] for r in config['items']]==batches['tts06']['items']
            and [r['opaque_id'] for r in config['items']]==batches['qwen06']['items']==batches['sensevoice']['items'],
            'cross-stage source/opaque roster mismatch')
        vocab_raw=read_ref(config['fsmn_tokens']);durable_write(data_dir/'fsmn-tokens.txt',vocab_raw);vocab=fsmn_vocabulary(vocab_raw)
        old_tokens=read_ref(config['four_tokens']);durable_write(data_dir/'four-tokens.txt',old_tokens)
        rows=[];fsmn=[];old_rows=[];weak=[]
        legacy=output/'four-token-consumer';legacy.mkdir();(legacy/'audio').mkdir()
        token_map=load_tokens(data_dir/'four-tokens.txt')
        for frozen_item in config['items']:
            item={**frozen_item,'source_row':sources[frozen_item['item_id']]}
            ident=item['item_id'];opaque=item['opaque_id']
            try:
                generated,audio,wav=verify_tts(batches['tts06'],item)
                qwen=verify_asr(batches['qwen06'],opaque,audio,wav,batches['tts06']['archive_sha256'])
                sense=verify_asr(batches['sensevoice'],opaque,audio,wav,batches['tts06']['archive_sha256'])
                row=screen_text(item,qwen,sense,audio,vocab)
                audio_rel='evidence/tts06/members/output/'+ident+'/'+ident+'.wav'
                row['audio_path']=audio_rel
                if row['status']=='machine_candidate':
                    _,pcm=wav_observation(wav)
                    fsmn.append({'recording':ident,'group':'bounded-batch-machine-weak','audio_path':audio_rel,
                        'audio_sha256':audio['file_sha256'],'pcm_sha256':audio['pcm_sha256'],'samples':audio['frames'],
                        'derived_pad300_pcm_sha256':digest(pcm+b'\0'*9600),'expected_output_frames':row['expected_output_frames'],
                        'ctc_text':row['ctc_text'],'ctc_ids':row['ctc_ids'],'source_metadata':row,
                        'split':item['split'],'pool':row['pool_candidate'] if item['split']=='train' else None,
                        'current_experiment_admission':False,'label_basis':'two-source-language-conditioned-ASR-full-text-agreement',
                        'human_gold':False,'product_qualified':False,'independent_test':False,'label_source':'weak-machine-agreement',
                        'purpose':'noncommercial-research-only'})
                old=copy_json(row)
                legacy_geometry_ok=(row['status']=='machine_candidate' and row['four_token_consumer']['compatible']
                    and max(0,1+(audio['frames']-400)//320)>=len(row['four_token_consumer']['tokens'])
                        +sum(a==b for a,b in zip(row['four_token_consumer']['tokens'],row['four_token_consumer']['tokens'][1:])))
                if row['status']=='machine_candidate' and legacy_geometry_ok:
                    names=row['four_token_consumer']['tokens'];ids=[token_map[n] for n in names]
                    require(all(i>0 for i in ids) and max(0,1+(audio['frames']-400)//320)>=len(ids)+sum(a==b for a,b in zip(ids,ids[1:])), 'legacy CTC geometry mismatch')
                    old.update(tokens=names,target_ids=ids)
                    durable_write(legacy/'audio'/(ident+'.wav'),wav)
                    weak.append({'audio':'audio/'+ident+'.wav','target_ids':ids,'tokens':names,'actual_text':row['machine_text'],
                        'kind':row['actual_kind'],'split':item['split'],'speaker_id':row['lineage_group'],'source_id':ident,
                        'file_sha256':audio['file_sha256'],'pcm_sha256':audio['pcm_sha256'],
                        'label_source':'weak-machine-agreement','purpose':'noncommercial-research-only','human_gold':False,'product_qualified':False,
                        **{k:row[k] for k in ('intended_text','asr_protocols','source_language_conditions','detected_language','language_is_model_independent_observation')}})
                elif row['status']=='machine_candidate':
                    old.update(status='quarantine',reason=('consumer_specific_insufficient_ctc_frames'
                        if row['four_token_consumer']['compatible'] else 'consumer_specific_transcript_OOV'),research_weak_allowed=False)
                old_rows.append(old)
            except (ValueError,KeyError,TypeError,IndexError,OSError) as exc:
                row={'item_id':ident,'status':'quarantine','reason':'native_batch_evidence_rejected','detail':str(exc),
                     'intended_text':item['intended_text'],'split':item['split'],'human_gold':False,'current_experiment_admission':False}
                old_rows.append(copy_json(row))
            rows.append(row)
        report={'schema':SCHEMA,'items':rows,'proposed_tasks':[],
            'statistics':{'source_items':len(rows),'machine_candidates':len(fsmn),'quarantined':len(rows)-len(fsmn),
                'candidate_yield':len(fsmn)/len(rows),'original_known_TTS_calls':len(batches['tts06']['items']),
                'original_known_ASR_calls':len(batches['qwen06']['items'])+len(batches['sensevoice']['items']),
                'new_model_calls':0,'training_runs':0},'authority':'saved-byte-consistency-machine-weak-only'}
        write_json(output/'report.json',report);write_json(output/'fsmn-candidates.json',fsmn)
        write_jsonl(output/'weak-fulltext.jsonl',[r for r in rows if r['status']=='machine_candidate'])
        write_jsonl(output/'quarantine.jsonl',[r for r in rows if r['status']=='quarantine'])
        write_json(legacy/'report.json',{'schema':'speech-data-queue-report-v1','items':old_rows})
        write_jsonl(legacy/'weak-train.jsonl',[r for r in weak if r['split']=='train'])
        write_jsonl(legacy/'weak-heldout.jsonl',[r for r in weak if r['split']!='train'])
        manifest_directory(legacy);manifest_directory(output)
        return report
    except Exception as exc:
        write_json(output/'IMPORT-FAILURE.json',{'status':'rejected-offline-import','error':str(exc),'new_model_calls':0})
        raise


def copy_json(value):
    return loads(json.dumps(value,ensure_ascii=False))


def load_fsmn_candidates(package):
    """Read-only metadata/PCM adapter, same official raw-ID-minus-one contract.

    Does not import the old top-level prepare/train scripts or construct features,
    models, optimizer, schedules, or grant current-experiment training admission.
    """
    package=Path(package).resolve(strict=True);manifest=loads((package/'MANIFEST.json').read_bytes())
    for name,sha in manifest.items():
        p=(package/name).resolve(strict=True)
        require(p.is_relative_to(package) and p.is_file() and digest(p.read_bytes())==sha,'candidate package manifest mismatch')
    require({'report.json','fsmn-candidates.json','weak-fulltext.jsonl','evidence/fsmn-tokens.txt'}<=set(manifest),
        'missing fulltext package manifest identity')
    vocabulary=fsmn_vocabulary((package/'evidence/fsmn-tokens.txt').read_bytes())
    report=loads((package/'report.json').read_bytes());byid={r['item_id']:r for r in report['items'] if r['status']=='machine_candidate'}
    require([loads(line) for line in (package/'weak-fulltext.jsonl').read_bytes().splitlines() if line.strip()]==list(byid.values()),
        'fulltext manifest/report disagreement')
    rows=loads((package/'fsmn-candidates.json').read_bytes());seen=set()
    for r in rows:
        require(r['recording'] not in seen,'duplicate fulltext candidate');seen.add(r['recording'])
        expected=byid[r['recording']]
        require(r['current_experiment_admission'] is False and r['human_gold'] is False and r['product_qualified'] is False
            and r['independent_test'] is False and r['source_metadata']==expected,'fulltext weak scope/provenance changed')
        require(r['ctc_text']==expected['machine_text'] and r['ctc_ids']==expected['ctc_ids']==[vocabulary[c] for c in r['ctc_text']]
            and r['split']==expected['split'] and r['audio_path']==expected['audio_path']
            and r['label_source']=='weak-machine-agreement' and r['purpose']=='noncommercial-research-only'
            and expected['asr_protocols']==[CONDITIONED,CONDITIONED]
            and expected['source_language_conditions']==['Chinese','zh'] and expected['detected_language'] is None
            and expected['language_is_model_independent_observation'] is False
            and expected['current_experiment_admission'] is False and expected['human_gold'] is False
            and expected['independent_test'] is False and expected['product_training_admitted'] is False
            and expected['research_weak_allowed'] is True,'fulltext IDs/split/provenance changed')
        p=(package/r['audio_path']).resolve(strict=True);require(p.is_relative_to(package) and r['audio_path'] in manifest,'audio outside fulltext package')
        audio,pcm=wav_observation(p.read_bytes())
        require(r['audio_sha256']==audio['file_sha256'] and r['pcm_sha256']==audio['pcm_sha256']
            and r['samples']==audio['frames'] and r['derived_pad300_pcm_sha256']==digest(pcm+b'\0'*9600)
            and r['expected_output_frames']==(1+(audio['frames']+4800-400)//160)//3,'fulltext PCM/derived geometry mismatch')
        original=expected['original_source_row']
        rebuilt=screen_text({'item_id':original['item_id'],'opaque_id':expected['opaque_id'],
            'speaker':original['speaker'],'split':original['split'],'intended_text':original['intended_text'],
            'intent_kind':original['kind'],'source_row':original},
            expected['qwen_raw_text'],expected['sense_raw_text'],audio,vocabulary)
        require(rebuilt['status']=='machine_candidate' and all(expected.get(k)==v for k,v in rebuilt.items()),
            'fulltext original source/votes/audio semantics changed')
        require(r['pool']==(rebuilt['pool_candidate'] if original['split']=='train' else None),
            'fulltext consumer pool/split semantics changed')
        ids=r['ctc_ids'];require(ids and all(type(i)is int and 0<i<2599 for i in ids)
            and r['expected_output_frames']>=len(ids)+sum(a==b for a,b in zip(ids,ids[1:])),'fulltext CTC geometry invalid')
        yield r
    require(seen==set(byid),'fulltext candidate roster incomplete')


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('import');run.add_argument('config',type=Path);run.add_argument('output',type=Path)
    check=sub.add_parser('inspect');check.add_argument('package',type=Path)
    args=parser.parse_args()
    if args.command=='import':print(json.dumps(import_package(args.config,args.output)['statistics'],sort_keys=True))
    else:
        rows=list(load_fsmn_candidates(args.package))
        print(json.dumps({'metadata_pcm_examples':len(rows),'train':sum(r['split']=='train' for r in rows),
            'development':sum(r['split']!='train' for r in rows),'current_experiment_admission':False,
            'new_model_calls':0,'training_runs':0},sort_keys=True))


if __name__=='__main__':main()
