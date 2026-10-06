"""Pure derived-audio checks supplement PR485, whose policy covers appended context only."""
from pathlib import Path
import hashlib,json
from prepare import inspect_wav,verify_context,LEADING_SAMPLES,APPENDED_SAMPLES,EXPECTED_IDS,EXPECTED_TEXTS
ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def verify_derived_inputs(manifest, root=ROOT):
    rows=manifest['rows']
    if [r['recording'] for r in rows]!=EXPECTED_IDS:raise ValueError('exact M1-M5 order required')
    checks=[]
    for row,text in zip(rows,EXPECTED_TEXTS):
        if (row.get('leading_samples'),row.get('appended_samples'),row.get('declared_text'),row.get('label'))!=(LEADING_SAMPLES,APPENDED_SAMPLES,text,{'M1':1,'M2':2}.get(row['recording'],0)):
            raise ValueError('fixed context and human label binding required')
        original=(root/'controls/data'/f"{row['recording']}.wav").read_bytes()
        changed=(root/row['local_wav_path']).read_bytes()
        verify_context(original,changed)
        oh=inspect_wav(original);dh=inspect_wav(changed)
        if row['original_wav_sha256']!=oh['wav_sha256'] or row['original_pcm_sha256']!=oh['pcm_sha256'] or row['original_frames']!=oh['frames']:
            raise ValueError('original audio identity drift')
        if any(row[k]!=v for k,v in dh.items()):raise ValueError('derived audio identity drift')
        left=44+LEADING_SAMPLES*2;right=left+oh['pcm_bytes']
        if row['middle_pcm_sha256']!=sha(changed[left:right]) or row['middle_pcm_sha256']!=oh['pcm_sha256']:
            raise ValueError('unchanged original middle PCM binding required')
        if row['zero_prefix_sha256']!=sha(changed[44:left]) or row['zero_tail_sha256']!=sha(changed[right:]):
            raise ValueError('zero context hash binding drift')
        old_group='positive300' if row['recording'] in ['M1','M2'] else 'negative300'
        old=(root/'controls'/old_group/'data'/f"{row['recording']}.wav").read_bytes()
        old_pcm=old[44:]
        if old_pcm!=original[44:]+b'\0'*(APPENDED_SAMPLES*2) or changed[left:]!=old_pcm:
            raise ValueError('same complete original PCM plus saved 300 ms postroll required')
        checks.append(dict(recording=row['recording'],leading_samples=LEADING_SAMPLES,appended_samples=APPENDED_SAMPLES,original_middle_preserved=True,saved_tail300_pcm_preserved=True,original_wav_sha256=oh['wav_sha256'],derived_wav_sha256=dh['wav_sha256']))
    return dict(status='VERIFIED_FIXED_LEADING_AND_POSTROLL_BYTES',streams=checks,scope='Actual derived WAV bytes, all five fixed leading contexts, entire original PCM and same postroll. Independent supplement to PR485; its eval-context-v1 policy does not define leading context.',frontend_calls=0,model_calls=0,decoder_calls=0)
