#!/usr/bin/env python3
"""Materialize the distinct R1 2601 port; optional exact private weight export.

Source preparation is standard-library only. Weight export explicitly requires
its owner-held R1 package and pinned core environment. No download or training.
The historical2599 module and its numerical FAIL are not changed or promoted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

CHECKPOINT = '99ac2239807ce5a93c8723a01df2e53c446f868245caccb2e7809deb6e943571'
CARD_SHA = '2079fce70bcd27d2991ce558807b3ef9707309051e6915946fc01821f8350936'
STATE = '71cf089fe9c163a0c76d394ed8a4e1937856dd831473ad3d2fb59a902be76b58'
MODEL_SHA = 'e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb'
TABLE_SHA = '49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49'
HERE = Path(__file__).resolve().parent

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def prepare_sources(repo, output):
    repo, output = Path(repo).resolve(), Path(output).absolute()
    require(not output.exists() and not any(p.is_symlink() for p in (output, *output.parents)), 'fresh output required')
    lock = json.loads((HERE / 'source-lock.json').read_text())
    for name, pin in lock['files'].items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe source path')
        p = repo / rel
        require(p.is_file() and not any(q.is_symlink() for q in (p, *p.parents)) and p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], 'source identity mismatch: ' + name)
    output.mkdir(parents=True)
    src = output / 'source'
    src.mkdir()
    (output / 'weights').mkdir()
    old = repo / 'research/fsmn2599-stream-c'
    for name in ('fsmn2599.c', 'fsmn2599.h'):
        text = (old / name).read_text().replace('FSMN2599', 'R1FSMN').replace('fsmn2599', 'r1fsmn')
        if name.endswith('.h'):
            for a, b in (('756933u','757215u'),('2599u','2601u'),('7599u','7601u'),('754334u','754614u')):
                require(text.count(a) == 1, 'ambiguous fixed geometry: ' + a)
                text = text.replace(a, b)
        text = text.replace('0x46323539', '0x52313236')
        (src / name.replace('fsmn2599','r1fsmn')).write_text('/* Modified2026-11-07: distinct2601 R1 port; old identity preserved. */\n' + text)
    for name in ('canonical_pcm.c', 'canonical_pcm.h'):
        text = (old / name).read_text().replace('kcf_pcm','r1_pcm').replace('canonical_pcm.h','r1_pcm.h')
        if name.endswith('.c'):
            begin = text.index('    for (unsigned i=0;i<4800;i++)')
            end = text.index('    if (!rc)', begin)
            text = text[:begin] + '    /* Exact R1 reference: no synthetic EOF input or outputs. */\n' + text[end:]
        else:
            text = text.replace('Append exactly 4800 samples, retain residual PCM for audit, never pad','Append zero samples, retain residual PCM for audit, never pad')
        (src / name.replace('canonical_pcm','r1_pcm')).write_text('/* Modified2026-11-07: R1 unpadded EOF, unchanged live transport. */\n' + text)
    for name in ('canonical_fbank.c','canonical_fbank.h','sha256.c','sha256.h','LICENSE','NOTICE.md','LICENSE.torchaudio'):
        shutil.copyfile(old / name, src / name)
    (src / 'vendor').mkdir()
    shutil.copyfile(repo / 'research/donor_fbank/fft_twiddles.h', src / 'vendor/fft_twiddles.h')
    for name in ('r1_stream.c','r1_stream.h','r1_pcm_cli.c'):
        shutil.copyfile(HERE / name, src / name)
    (src / 'model_identity.h').write_text('#ifndef R1_MODEL_IDENTITY_H\n#define R1_MODEL_IDENTITY_H\n#define R1_MODEL_SHA "' + MODEL_SHA + '"\n#define R1_FRONTEND_SHA "' + TABLE_SHA + '"\n#define R1_TABLE_FLOATS 20960u\n#endif\n')
    shutil.copyfile(HERE / 'build.sh', output / 'build.sh')
    (output / 'SOURCE-MANIFEST.json').write_text(json.dumps({'schema':'kws-r1-port-source-v1','checkpoint_sha256':CHECKPOINT,'files':{p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.rglob('*')) if p.is_file()},'model_forwards':0,'old_numerical_gate_unchanged':True}, sort_keys=True, indent=2)+'\n')
    return output

def export_weights(package, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    require(sys.version_info[:3] == (3,12,16), 'exact CPython3.12.16 required for weight/table export')
    require(sha(package / 'MODEL.json') == CARD_SHA, 'untrusted model-card bytes')
    card = json.loads((package / 'MODEL.json').read_text())
    require(card['checkpoint_sha256'] == CHECKPOINT and card['float_state_sha256'] == STATE, 'wrong model package')
    for name, pin in card['files'].items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe model path')
        p = package / rel
        require(p.is_file() and not any(q.is_symlink() for q in (p, *p.parents)) and p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], 'model package drift: ' + name)
    root = package / 'source/original'
    require(not any(name in sys.modules for name in ('research_runtime','training_state','research_fsmn_weak')), 'fresh export process required')
    sys.path[:0] = [str(root),str(root/'training'),str(root/'tools')]
    import research_runtime
    from training_state import load_checkpoint_snapshot,state_identity
    from research_fsmn_weak import load_upstream
    import research_weak_train
    runtime = research_runtime.load_manifest(root/'configs/runtime/research-runtime.json')
    research_runtime.check_interpreter(runtime)
    research_runtime.check_packages(runtime,'core')
    torch, _ = research_weak_train.torch_runtime()
    checkpoint, checkpoint_sha = load_checkpoint_snapshot(package/'model.pt')
    state = checkpoint['state_dict']
    require(checkpoint_sha == CHECKPOINT and state_identity(state)['sha256'] == STATE, 'actual state mismatch')
    names = ['global_cmvn.mean','global_cmvn.istd']
    for prefix in ('in_linear1','in_linear2'):
        names += ['backbone.'+prefix+'.linear.weight','backbone.'+prefix+'.linear.bias']
    for i in range(4):
        p = f'backbone.fsmn.{i}.'
        names += [p+'0.linear.weight',p+'1.conv_left.weight',p+'1.conv_right.weight',p+'2.linear.weight',p+'2.linear.bias']
    for prefix in ('out_linear1','out_linear2'):
        names += ['backbone.'+prefix+'.linear.weight','backbone.'+prefix+'.linear.bias']
    require(set(names)==set(state), 'tensor roster')
    payload = b''.join(state[n].detach().cpu().contiguous().numpy().astype('<f4').tobytes() for n in names)
    _, _, processor = load_upstream(package/'official',torch)
    k = processor['kaldi'].fbank.__globals__
    povey = k['_feature_window_function']('povey',400,0.42,torch.device('cpu'),torch.float32)
    mel, _ = k['get_mel_banks'](80,512,16000,20.0,0.0,100.0,-500.0,1.0)
    mel = torch.nn.functional.pad(mel,(0,1),mode='constant',value=0)
    tables = povey.numpy().astype('<f4').tobytes()+mel.numpy().astype('<f4').tobytes()
    require(len(payload)==3028860 and hashlib.sha256(payload).hexdigest()==MODEL_SHA, 'export payload mismatch')
    require(len(tables)==83840 and hashlib.sha256(tables).hexdigest()==TABLE_SHA, 'table export mismatch')
    for name, raw in (('model.f32le',payload),('frontend.f32le',tables)):
        with (output/'weights'/name).open('xb') as stream:
            stream.write(raw)
    shutil.copyfile(package/'official/upstream/LICENSE.torchaudio',output/'source/LICENSE.torchaudio')
    return {'model_sha256':MODEL_SHA,'frontend_sha256':TABLE_SHA,'model_forwards':0,'optimizer_updates':0}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=HERE.parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-package',type=Path)
    args=parser.parse_args()
    try:
        output=prepare_sources(args.repo,args.output)
        result=export_weights(args.model_package,output) if args.model_package else {'status':'source_only_no_weights','model_forwards':0}
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(2,str(exc)+'\n')
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':
    main()
