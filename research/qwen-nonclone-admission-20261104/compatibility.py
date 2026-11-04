"""Read-only compatibility assessment, never a training admission issuer.

Missing historical observations remain missing. A source-screen result cannot
be promoted by filling runtime defaults or caller-authored permission flags.
"""
from pathlib import Path
import hashlib,json,sys,unicodedata
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from speech_label_admission import actual_tokens,PUNCTUATION
FAMILIES=('qwen06','sensevoice','whisper')
GENERATION_FIELDS=('num_beams','num_return_sequences','forced_bos_token_id','forced_eos_token_id','suppress_tokens','begin_suppress_tokens','bad_words_ids','force_words_ids','sequence_bias','stop_strings','do_sample','max_new_tokens','eos_token_id')

def require(ok,why):
    if not ok:raise ValueError(why)

def normalize(text):
    require(type(text)is str,'text required')
    return ''.join(c for c in text if c not in PUNCTUATION)

def assess(report,mapping,evidence):
    evidence=Path(evidence)
    require(report.get('planned_audio')==4 and report.get('planned_asr_slots')==12 and len(report['rows'])==4 and len(mapping['rows'])==4,'exact pilot denominator')
    source={r['sample_id']:r for r in mapping['rows']}
    require(len(source)==4 and len({r['pcm_sha256'] for r in source.values()})==4 and len({r['opaque_id'] for r in source.values()})==4,'unique source/PCM/opaque identities')
    require(set(source)=={r['sample_id'] for r in report['rows']},'complete source IDs')
    assessments=[]
    for row in report['rows']:
        m=source[row['sample_id']];obs=row['observations'];require(set(obs)==set(FAMILIES),'three families')
        require(all(o['opaque_id']==m['opaque_id'] and o['wav_sha256']==m['derived_sha256'] for o in obs.values()),'audio binding')
        successful=all(o['status']=='success' and o['completeness']=='complete' and not o['quality_flags'] and type(o.get('raw_text'))is str for o in obs.values())
        matches=successful and all(normalize(o['raw_text'])==normalize(m['requested_text']) for o in obs.values())
        reasons=[];tokens=None
        if not matches:reasons.append('machine_disagreement_or_target_mismatch')
        try:tokens=actual_tokens(m['requested_text'])
        except ValueError:reasons.append('requested_text_outside_reviewed_ctc_vocabulary')
        if tokens==[]:reasons.append('speech_cannot_be_empty_ctc_target')
        q=evidence/'primary/qwen06/qwen06'/(m['opaque_id']+'.decoder.json')
        s=evidence/'primary/sensevoice/sensevoice'/(m['opaque_id']+'.decoder.json')
        qd=json.loads(q.read_text());sd=json.loads(s.read_text())
        for d in (qd,sd):require(d['opaque_id']==m['opaque_id'] and d['wav_sha256']==m['derived_sha256'],'decoder/audio binding')
        qe,se=qd['evidence'],sd['evidence']
        # Record only observed compatibility. Do not manufacture native-v2 config.
        observed=dict(qwen_pcm_binding='input_binding' in qe,qwen_processor_tensor_hashes='processor_tensor_evidence' in qe,qwen_tokens_eos='token_ids' in qe and qe.get('metadata',{}).get('eos_terminated') is True,sense_pcm_binding='input_binding' in se,sense_raw_result='raw_result' in se,sense_tokenizer_binding='token_binding' in se)
        reasons += ['new_source_not_registered_as_reviewed_train_root','resolved_runtime_generation_config_not_captured']
        assessments.append(dict(sample_id=m['sample_id'],file_sha256=m['derived_sha256'],pcm_sha256=m['pcm_sha256'],source_family='qwen-customvoice-0.6b-vivian-20261104',split='train_only_if_later_admitted',machine_text_match=bool(matches),vocabulary_representable=tokens is not None and bool(tokens),declared_legacy_field_presence=observed,missing_runtime_snapshot_fields=list(GENERATION_FIELDS),blockers=reasons,label_authority='machine_screen_only',human_gold=False,training_admitted=False))
    return dict(schema='qwen-nonclone-training-compatibility-v1',scope='Read-only legacy evidence compatibility, not native-v2 certification or admission',artifact_authenticity_verified=False,decoder_content_semantics_verified=False,current_formal_admission_count=0,planned_audio=4,rows=assessments,reviewed_train_root_created=False,native_evidence_synthesized=False,training_run_started=False,human_gold=False,training_admitted=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True,type=Path);p.add_argument('--mapping',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=assess(json.loads((a.evidence/'report.json').read_text()),json.loads(a.mapping.read_text()),a.evidence)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
