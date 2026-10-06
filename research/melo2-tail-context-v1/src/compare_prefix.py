"""Saved-only original-prefix comparison for the single fixed 300 ms condition."""
from pathlib import Path
import hashlib,json,struct
ROOT=Path(__file__).resolve().parents[1]

def by_center(records,alias):
    rows={}
    for callback in records:
        if callback.get('kind')!='callback' or callback.get('recording')!=alias: continue
        for center,logits in zip(callback['centers'],callback['logits']):
            if center in rows: raise ValueError('duplicate center')
            rows[center]=(logits,callback)
    return rows

def compare_prefix(records,manifest):
    control_bytes=(ROOT/'controls/original_A20.raw.jsonl').read_bytes()
    receipt=json.loads((ROOT/'controls/saved-control-validation.json').read_text())
    if hashlib.sha256(control_bytes).hexdigest()!=receipt['source_raw_sha256']:
        raise ValueError('saved original control drift')
    controls=[json.loads(line) for line in control_bytes.splitlines()]
    streams=[]
    for row in manifest['rows']:
        alias=row['recording'];end=row['original_frames']
        if row.get('condition')!='appended_300ms_zero_context' or row.get('appended_samples')!=4800:
            raise ValueError('fixed appended-context condition required')
        old=by_center(controls,alias);new=by_center(records,alias)
        safe=[center for center in sorted(new) if (center+2)*160+400<=end]
        if safe!=sorted(old):
            raise ValueError('support-contained centers must exactly match saved original rows')
        comparisons=[]
        for center in safe:
            a,oc=old[center];b,nc=new[center]
            comparisons.append(dict(center=center,frontend_support_end_sample_exclusive=(center+2)*160+400,
                fp32_logits_equal=struct.pack('<6f',*a)==struct.pack('<6f',*b),
                original_callback_phase=oc['phase'],modified_callback_phase=nc['phase'],
                original_callback_available_samples=oc['available_samples'],modified_callback_available_samples=nc['available_samples']))
        events=[]
        for callback in records:
            if callback.get('kind')=='callback' and callback.get('recording')==alias and callback.get('state')==1:
                delta=callback['available_samples']-end
                events.append(dict(keyword=callback['keyword'],score=callback['score'],start_frame=callback['start_frame'],end_frame=callback['end_frame'],
                    available_samples=callback['available_samples'],original_file_end_sample=end,
                    availability_minus_original_file_end_samples=delta,availability_minus_original_file_end_seconds=delta/16000,
                    availability_after_original_file_end=delta>0,word_end_latency=None))
        equal=all(c['fp32_logits_equal'] for c in comparisons)
        streams.append(dict(recording=alias,original_frames=end,appended_samples=4800,
            original_wav_sha256=row['original_wav_sha256'],modified_wav_sha256=row['wav_sha256'],
            support_contained_rows=len(safe),all_support_contained_fp32_logits_equal=equal,
            differing_centers=[c['center'] for c in comparisons if not c['fp32_logits_equal']],comparisons=comparisons,
            new_context_centers=[c for c in sorted(new) if c not in safe],modified_events=events,
            original_event_count=receipt['selected_original_event_counts'][alias]))
    all_equal=all(s['all_support_contained_fp32_logits_equal'] for s in streams)
    return dict(condition='M1_M2_EXACT_4800_ZERO_SAMPLES_APPENDED',status='PREFIX_LOGITS_EQUAL' if all_equal else 'PREFIX_LOGIT_DIFFERENCE_OBSERVED',
        saved_original_raw_sha256=receipt['source_raw_sha256'],streams=streams,
        support_rule='Rightmost spliced fbank frame is center+2; its exclusive PCM end is (center+2)*160+400, required <= original EOF',
        expected_boundary_difference='The old third callback was finish (5/7 rows), while the modified third full feed has 10 rows and input availability 14400. Compare logits by absolute center; callback grouping and availability differences are expected.',
        event_time_scope='Available input relative to original file EOF, never acoustic word-end latency',
        qualification=False,original_human_and_machine_assessments_unchanged=True)
