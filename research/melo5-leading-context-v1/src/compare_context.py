"""Saved-only paired observations; no equality assertion across different precontexts."""
from pathlib import Path
import hashlib,json
from derived_binding import verify_derived_inputs
ROOT=Path(__file__).resolve().parents[1]
def load_controls(root=ROOT):
    receipt=json.loads((root/'controls/identities.json').read_text())
    groups={}
    for group,entry in receipt['saved_logs'].items():
        data=(root/entry['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('saved control raw drift')
        records=[json.loads(line) for line in data.splitlines()]
        if records[0]['model_sha256']!=receipt['model_sha256'] or records[0]['library_sha256']!=receipt['library_sha256']:raise ValueError('control model/library identity mismatch')
        if records[-1].get('kind')!='run_end' or records[-1].get('complete') is not True:raise ValueError('incomplete saved control')
        groups[group]=records
    return groups

def events(records,alias,prefix,original_frames):
    found=[]
    for c in records:
        if c.get('kind')!='callback' or c.get('recording')!=alias or c.get('state')!=1:continue
        source_available=c['available_samples']-prefix
        found.append(dict(keyword=c['keyword'],score=c['score'],start_frame=c['start_frame'],end_frame=c['end_frame'],centers=c['centers'],call_index=c['call_index'],phase=c['phase'],available_samples=c['available_samples'],available_seconds=c['available_samples']/16000,source_relative_available_samples=source_available,source_relative_available_seconds=source_available/16000,availability_minus_original_eof_samples=source_available-original_frames,event_available_during_leading_silence=bool(prefix and c['available_samples']<=prefix),word_end_latency=None))
    return found

def compare_context(records,manifest):
    binding=verify_derived_inputs(manifest)
    saved=load_controls();pairs=[]
    for row in manifest['rows']:
        alias=row['recording'];old=saved['positive300' if alias in ['M1','M2'] else 'negative300']
        pairs.append(dict(recording=alias,human_text=row['declared_text'],expected_keyword=row['label'],original_frames=row['original_frames'],original_eof_events=events(saved['original_eof'],alias,0,row['original_frames']),saved_tail300_events=events(old,alias,0,row['original_frames']),leading1500_tail300_events=events(records,alias,row['leading_samples'],row['original_frames'])))
    return dict(status='DESCRIPTIVE_PAIRED_CONTEXT_OBSERVATIONS',streams=pairs,derived_input_binding=binding,all_events_counted=True,event_time_scope='Subtract 24000 only from new input availability to express source-relative availability, preserving negative values; not acoustic word-end latency. Decoder frames and centers retain each stream original absolute coordinates.',logit_equality_expected=False,reason='Leading context changes frontend history, model caches and decoder history; center/absolute coordinates can shift.',qualification=False)
