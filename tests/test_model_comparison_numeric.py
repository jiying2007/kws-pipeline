#!/usr/bin/env python3
"""Opt-in real Torch smoke; artificial PCM is NOT acoustic quality evidence.

Run as a script, not through unittest discovery. All optimization and numeric
inference occur in clean child processes under one <=120-second deadline.
No downloads, pretrained weights, product exporter, or protected evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import wave


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_worker(args):
    sys.path.insert(0, str(args.runner_file.parent))
    import runner
    # Exactly one runtime() call per clean process. It imports model before Torch.
    rt = runner.runtime(args.repo)
    torch, _, _, features, frontend, _, _ = rt
    import numpy
    assert str(torch.__version__).split('+')[0] == '2.13.0', str(torch.__version__)
    assert numpy.__version__ == '2.3.2', numpy.__version__
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    errors = {}
    for arm in ('control', 'treatment'):
        checkpoint = args.work / 'trained' / (arm + '.research.pt')
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert ckpt['synthetic_diagnostic_only'] is True
        assert ckpt['product_export_compatible'] is False and ckpt['frontend'] == 'logmel'
        torch.manual_seed(20261014)
        model = runner.build(rt, ckpt['spec'])
        initial = {k: v.detach().clone() for k,v in model.state_dict().items()}
        assert all(bool(torch.isfinite(v).all()) for v in ckpt['state_dict'].values())
        assert any(not torch.equal(initial[k], v) for k,v in ckpt['state_dict'].items())
        model.load_state_dict(ckpt['state_dict'], strict=True)
        model.eval()
        wav = args.work / 'selection.wav'
        with wave.open(str(wav), 'rb') as wf:
            n = wf.getnframes()
            pcm = torch.frombuffer(bytearray(wf.readframes(n)), dtype=torch.int16).float()/32768
        expected_frames = 1 + (n - 400)//320
        with torch.no_grad():
            x = features(pcm, 32, frontend=frontend)
            whole = model(x.unsqueeze(0))
            assert x.shape[0] == expected_frames
            for chunk in (1, 2, 7, 25):
                state = model.init_state(1, device='cpu', dtype=torch.float32)
                outputs, actual_features = [], []
                with wave.open(str(wav), 'rb') as wf:
                    for part in runner.iter_wav_features(rt, wf, chunk):
                        actual_features.append(part)
                        y, state = model.forward_chunk(part.unsqueeze(0), state)
                        outputs.append(y)
                combined = torch.cat(outputs)
                combined_features = torch.cat(actual_features)
                assert combined.shape == whole.shape == (expected_frames, 1, 5)
                torch.testing.assert_close(combined_features, x, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(combined, whole, atol=2e-6, rtol=2e-5)
                errors[f'{arm}/chunk{chunk}'] = float((combined-whole).abs().max())
            # EOF shorter than one full analysis window yields no features.
            with wave.open(str(args.work/'short.wav'), 'rb') as wf:
                assert list(runner.iter_wav_features(rt, wf, 1)) == []
        # Produce a deliberately invalid source binding for a fail-closed CLI check.
        if arm == 'control':
            ckpt['source_hashes']['training/model.py'] = '0'*64
            torch.save(ckpt, args.work/'tampered-source.pt')
            ckpt['source_hashes'] = {}
            torch.save(ckpt, args.work/'empty-source.pt')
    (args.work/'numeric.json').write_text(json.dumps({
        'torch_version': str(torch.__version__), 'numpy_version': numpy.__version__,
        'max_abs_logit_errors': errors, 'complete_frames': expected_frames,
        'chunk_sizes': [1,2,7,25], 'short_eof_frames': 0,
        'threads': [torch.get_num_threads(),torch.get_num_interop_threads()]}))


def smoke(args):
    start = time.monotonic()
    deadline = start + 120
    def run(cmd, *, expected_failure=None):
        left = deadline-time.monotonic()
        if left <= 0:
            raise TimeoutError('120-second total smoke deadline exhausted')
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=left)
        if expected_failure:
            assert result.returncode != 0 and expected_failure in result.stderr, (result.stdout,result.stderr)
        elif result.returncode:
            raise RuntimeError(f'command failed: {cmd}\n{result.stdout}\n{result.stderr}')
        return result
    with tempfile.TemporaryDirectory(prefix='kws-paired-numeric-') as directory:
        work = Path(directory)
        def wav(name, n, salt):
            # Literal arithmetic fixture, no TTS/voice sample or speech claim.
            values = [int(2400*math.sin((i+salt)*(0.037+salt*0.003))) for i in range(n)]
            path = work/(name+'.wav')
            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(16000)
                wf.writeframes(struct.pack('<'+'h'*n,*values))
            return path
        manifests={}
        samples=400+10*320+123
        for idx,split in enumerate(('train','calibration','selection')):
            audio=wav(split,samples,idx+1)
            row=dict(audio=str(audio), target_ids=[1,2,3,4], split=split,
                     speaker_id='artificial-'+split,session_id='fixture-'+split,
                     source_id='generated-'+split)
            rows=[row]
            if split=='train':
                # Also exercise a genuine empty CTC target without zero_infinity.
                negative=wav('train-empty',samples+37,9)
                rows.append(dict(row,audio=str(negative),target_ids=[],source_id='generated-empty'))
            manifest=work/(split+'.jsonl')
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifests[split]=str(manifest)
        wav('short',399,11)
        template=json.loads((args.runner_file.parent/'pilot.example.json').read_text())
        template.update(manifests=manifests,tokens=str(args.repo/'keywords/tokens.example.txt'),
                        epochs=1,seed=20261014,max_training_seconds=15)
        config=work/'config.json';config.write_text(json.dumps(template))
        prefix=[sys.executable,str(args.runner_file),'--repo',str(args.repo)]
        run(prefix+['train','--config',str(config),'--output',str(work/'trained')])
        training=json.loads((work/'trained/training-receipt.json').read_text())
        assert all(a['status']=='TRAINED' and a['sample_exposures']==2 for a in training['arms'])
        assert training['admission']['promotion_allowed'] is False
        assert training['admission']['purpose']=='synthetic-contract-test-only'
        assert training['shipping_approved'] is False
        run([sys.executable,str(Path(__file__).resolve()),'--worker','--repo',str(args.repo),
             '--runner-file',str(args.runner_file),'--work',str(work)])
        numeric=json.loads((work/'numeric.json').read_text())
        keywords=args.runner_file.parent/'keywords.example.json'
        events_by_arm={}
        for arm in ('control','treatment'):
            previous=None
            for chunk in (1,25):
                events=work/f'{arm}-chunk{chunk}.jsonl'
                run(prefix+['infer','--checkpoint',str(work/'trained'/f'{arm}.research.pt'),
                    '--wav',str(work/'selection.wav'),'--recording','artificial-selection',
                    '--keywords',str(keywords),'--output',str(events),'--chunk-frames',str(chunk)])
                receipt=json.loads(Path(str(events)+'.receipt.json').read_text())
                assert receipt['frames']==numeric['complete_frames']==11
                assert receipt['c_runtime_equivalent'] is False
                assert receipt['synthetic_diagnostic_only'] is True
                # Event timestamps/IDs exact; confidence floats allow numeric tolerance.
                current=[json.loads(line) for line in events.read_text().splitlines()]
                if previous is not None:
                    assert len(current)==len(previous)
                    for a,b in zip(current,previous):
                        assert a.keys()==b.keys()
                        for k in a:
                            if k=='confidence':assert abs(a[k]-b[k])<=2e-6
                            else:assert a[k]==b[k]
                previous=current
            events_by_arm[arm]=events
        # Exercise existing scorer with explicitly artificial empty reference.
        # No speech or wake accuracy is inferred from these invented PCM samples.
        refs=work/'references.jsonl'
        refs.write_text(json.dumps(dict(recording='artificial-selection',duration_s=samples/16000,expected=[]))+'\n')
        for arm,events in events_by_arm.items():
            summary=work/(arm+'-score.json')
            run([sys.executable,str(args.repo/'eval/score_events.py'),'--references',str(refs),
                 '--detections',str(events),'--summary',str(summary)])
            scored=json.loads(summary.read_text())
            assert {'false_rejects','false_accepts','far_per_hour','per_keyword'} <= scored.keys()
            assert scored['negative_recording_audio_hours'] > 0
        run(prefix+['infer','--checkpoint',str(work/'tampered-source.pt'),
            '--wav',str(work/'selection.wav'),'--recording','artificial-selection',
            '--keywords',str(keywords),'--output',str(work/'must-not-exist.jsonl')],
            expected_failure='checkpoint/source mismatch')
        assert not (work/'must-not-exist.jsonl').exists()
        run(prefix+['infer','--checkpoint',str(work/'empty-source.pt'),
            '--wav',str(work/'selection.wav'),'--recording','artificial-selection',
            '--keywords',str(keywords),'--output',str(work/'empty-must-not-exist.jsonl')],
            expected_failure='source_hashes keys mismatch')
        assert not (work/'empty-must-not-exist.jsonl').exists()
        result=dict(schema='paired-research-numeric-smoke-v1',status='PASS',
                    evidence_scope='artificial-pcm-numeric-contract-only',shipping_approved=False,
                    training_updates_per_arm=2,quality_metrics_claimed=False,
                    numerical_checks=numeric,source_binding_rejection=True,
                    score_interface_executed=True,runner_sha256=sha(args.runner_file),
                    test_sha256=sha(__file__),elapsed_seconds=time.monotonic()-start)
        if args.receipt:
            args.receipt.parent.mkdir(parents=True,exist_ok=True)
            with args.receipt.open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--runner-file',type=Path)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--work',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    args.repo=args.repo.resolve()
    args.runner_file=(args.runner_file or args.repo/'research/model_comparison/runner.py').resolve()
    (numeric_worker if args.worker else smoke)(args)

if __name__=='__main__':main()
