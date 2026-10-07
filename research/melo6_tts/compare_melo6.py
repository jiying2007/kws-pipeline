"""Pure six-phrase join after independent saved-file/hash verification.

This helper does NOT itself establish filesystem freeze verification. Its caller
must verify both ASR raw freezes and the generation/blind identities first,
using the retained artifact verifier, then pass those bound saved records.
No audio/model I/O or inference occurs here. Acoustic/human truth stays UNKNOWN.
"""
import hashlib
from pathlib import Path
from types import ModuleType

TEXTS = ('你好小窝','小窝小窝','你好小屋','小屋小屋','你好你好','今天天气很好')
MODELS = ('sensevoice','qwen06')
HELPER_SHA = '9c5c1e4d31238f3e5e2daa6a5d102638f916183daee56d63735a11a37a60e891'


def helpers():
    path = Path(__file__).resolve().parent / 'comparison/audio_review.py'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA:
        raise ValueError('COMPARISON_HELPER_IDENTITY')
    module = ModuleType('melo_review_helpers')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def compare_verified_saved(intent_rows, blind_clips, asr_rows, binding):
    """Pure adapter; a binding hash is provenance, not self-proving verification."""
    a = helpers()
    required = {'plan_sha256','tts_generation_freeze_sha256','blind_job_sha256',
                'sensevoice_raw_freeze_sha256','qwen06_raw_freeze_sha256'}
    a.require(set(binding) == required, 'COMPARISON_BINDINGS')
    for value in binding.values():
        a.valid_hash(value)
    a.require(len(intent_rows) == 6 and [r['intended_text'] for r in intent_rows] == list(TEXTS),
              'FIXED_SIX_INTENTS')
    ids = [f'clip-{i:06d}' for i in range(1,7)]
    a.require([r['audio_id'] for r in blind_clips] == ids, 'FIXED_BLIND_ORDER')
    a.require(set(asr_rows) == set(MODELS), 'EXACT_TWO_RECOGNIZERS')
    indexed = {}
    for model in MODELS:
        rows = asr_rows[model]
        a.require(len(rows) == 6 and [r['opaque_id'] for r in rows] == ids, 'FULL_ASR_DENOMINATOR')
        indexed[model] = {r['opaque_id']: r for r in rows}
    results = []
    for intent, clip in zip(intent_rows, blind_clips):
        evidence = []
        intended = intent['intended_text']
        for model in MODELS:
            row = indexed[model][clip['audio_id']]
            a.require(row['wav_sha256'] == clip['wav_sha256'], 'ASR_AUDIO_BINDING')
            text = row.get('raw_text')
            successful = row['status'] == 'success' and row.get('completeness') == 'complete' and isinstance(text,str)
            evidence.append({'model':model,'status':row['status'],'raw_text':text,
                             'decoder_completeness':row.get('completeness','unknown'),
                             'normalized_text':a.normalize(text) if isinstance(text,str) else None,
                             'exact_intent_match':bool(successful and a.normalize(text)==a.normalize(intended)),
                             'reported_quality_flags':row.get('quality_flags',[])})
        lexical_support = all(r['exact_intent_match'] for r in evidence)
        results.append({'audio_id':clip['audio_id'],'wav_sha256':clip['wav_sha256'],
                        'intended_text':intended,'intent_repetition':a.repetition(intended),
                        'machine_text_agreement_only':lexical_support,
                        'assessment':'weak_lexical_support_only' if lexical_support else 'quarantine',
                        'quarantined_for_training':True,
                        'recognizer_evidence':evidence,'acoustic_tail_completeness':'UNKNOWN',
                        'human_transcript':'UNKNOWN','human_pronunciation':'UNKNOWN',
                        'gold_label':None,'ctc_target':None,'training_admission':False,
                        'kws_oov_policy':'evaluation only; never blank CTC'})
    all_supported = all(r['machine_text_agreement_only'] for r in results)
    return {'schema':'melo6-post-freeze-machine-comparison-v1','bindings':binding,'clips':results,
            'two_asr_agreement_means':'machine-level text support only',
            'required_control_scope':'all six: both wake phrases, both wo/wu controls, repetition and general-text controls',
            'all_six_weak_lexical_gate_passed':all_supported,
            'overall_result':'all_six_weak_lexical_support_only' if all_supported else 'failed_or_inconclusive_quarantine',
            'next_action':'stop; retain evidence for separate human review' if all_supported else 'stop; retain quarantine; no source advancement',
            'automatic_source_advancement':False,'automatic_retry':False,
            'speaker_isolated_split_possible':False,'kws_improvement_measured':False,
            'new_inference_calls':0,'human_review':'UNKNOWN'}
