#!/usr/bin/env python3
"""Execute the actual A splitter -> B manifest projection contract (no Torch).

Fixtures contain generated PCM and invented identities, not user/data evidence.
A output is deliberately corrupted to span B splits; B must reject all retained
source linkage fields, rather than trusting upstream grouping blindly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import wave


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--comparison-dir',type=Path)
    args=p.parse_args()
    repo=args.repo.resolve()
    comparison=(args.comparison_dir or repo/'research/model_comparison').resolve()
    sys.path[:0]=[str(comparison),str(repo/'training'),str(repo/'tools')]
    import effect_dataset_split as upstream
    from corpus_identity import inspect_pcm16_wav
    from compare import LINEAGE_FIELDS, validate_split
    from runner import projected_identity
    assert set(upstream.GROUP_FIELDS + upstream.PCM_FIELDS) <= set(LINEAGE_FIELDS) | {'speaker_id','session_id'}
    checked=[]
    with tempfile.TemporaryDirectory(prefix='kws-split-bridge-') as directory:
        root=Path(directory)
        measured=[]
        for i in range(3):
            path=root/f'{i}.wav'
            with wave.open(str(path),'wb') as wf:
                wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(16000)
                wf.writeframes(struct.pack('<'+'h'*400,*[(i+1)*j for j in range(400)]))
            measured.append(inspect_pcm16_wav(path))
        # All of these are literal A fields, not a parallel B-only vocabulary.
        fields=('preset_voice_id','source_family_id','derivation_family_id','family_id',
                'original_pcm_sha256','source_pcm_sha256','reference_audio_sha256')
        for field in fields:
            rows=[]
            for i in range(3):
                name=f'fixture-{i}'
                row=dict(record_id=name,path=f'{i}.wav',audio_status='verified_pcm16',
                         speaker_id=name,session_id=name,source_id=name,source_family_id=name,
                         split='train',pcm_sha256=measured[i]['pcm_sha256'],
                         human_review=dict(status='complete',reviewer_id='fixture-reviewer',
                            listened_at_utc='2026-10-14T00:00:00Z',heard_text='artificial fixture only',
                            asr_or_target_seen_before_listening=False))
                if i < 2:
                    row[field]=('a'*64 if field.endswith('sha256') else 'shared-origin')
                rows.append(row)
            report=upstream.plan(rows)
            by_id={r['record_id']:r for r in report['rows']}
            assert by_id['fixture-0']['group_id']==by_id['fixture-1']['group_id'], field
            projected=[]
            for i,split in enumerate(('train','calibration','selection')):
                raw=by_id[f'fixture-{i}']
                # Simulate an unsafe downstream reassignment, keeping the actual
                # A-produced source metadata unchanged. B must independently fail.
                current={**measured[i],**{k:raw[k] for k in ('speaker_id','session_id','source_id')}}
                projected.append(projected_identity(current,raw,split))
            try:
                validate_split(projected)
            except ValueError as exc:
                assert 'leakage' in str(exc), (field,str(exc))
            else:
                raise AssertionError('A->B leakage accepted: '+field)
            checked.append(field)
        # Reference audio is a FILE digest, never a PCM digest. Check matching
        # against measured file bytes while retaining distinct PCM namespaces.
        base=[]
        for i,split in enumerate(('train','calibration','selection')):
            m={**measured[i], 'speaker_id':str(i),'session_id':str(i),'source_id':str(i)}
            raw={'reference_audio_sha256':measured[0]['file_sha256']} if i==1 else {}
            base.append(projected_identity(m,raw,split))
        try:validate_split(base)
        except ValueError as exc:assert 'leakage' in str(exc)
        else:raise AssertionError('reference audio -> measured file leakage accepted')
    print(json.dumps({'schema':'effect-split-model-bridge-v1','status':'PASS',
                      'upstream_fields_checked':checked,'reference_file_namespace_checked':True,
                      'torch_executed':False,'quality_claim':False,
                      'splitter_sha256':hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
