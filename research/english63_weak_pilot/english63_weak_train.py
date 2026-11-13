#!/usr/bin/env python3
"""Bounded, reproducible English-63 *weak-label* acoustic research pilot.

Two causal frame encoders (GRU and DS-TCN) with WHOLE-CLIP readout. No KWS
endpoint, source truth, training admission, product export or promotion.
Original synthetic-voice prompts are weak INTENT labels, not acoustic truth.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import wave
import zipfile

POLICY = 'english63-weak-clip-two-architecture-pilot-v1'
INPUT_SHA256 = 'c4c3cdfe363317f0b24fbb948183e5cf7ed433e7c2eef9bb46131651c9350007'
SOURCE_TTS_SHA256 = 'f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc'
SOURCE_REVIEW_SHA256 = '89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12'
SOURCE_RUN = 43242278962
CLASSES = {'background': 0, 'Hey Vosen': 1, 'Hey Mero': 2}
ROLES = {
    'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY': 'weak_diagnostic_only',
    'EXPOSED_DEVELOPMENT_NO_TUNE': 'development_no_tune',
    'QUARANTINE_AMBIGUOUS_NEARPHONE': 'nearphone_quarantine',
}
EXPECTED_COUNTS = {'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY': 30,
                   'EXPOSED_DEVELOPMENT_NO_TUNE': 12,
                   'QUARANTINE_AMBIGUOUS_NEARPHONE': 21}
TRAIN_VOICES = {'dylan', 'eric', 'ono_anna', 'sohee', 'vivian'}
DEV_VOICES = {'serena', 'uncle_fu'}
SEED = 20261113
STEPS = 48
HIDDEN = 32
FREQ_BANDS = 32
SHA64 = re.compile(r'[0-9a-f]{64}\Z')
CELL = re.compile(r'en-[0-9a-f]{24}\Z')


def require(flag: bool, why: str):
    if not flag:
        raise ValueError(why)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def wav_samples(raw: bytes):
    import numpy as np
    with wave.open(io.BytesIO(raw), 'rb') as w:
        require(w.getnchannels() == 1 and w.getsampwidth() == 2 and
                w.getframerate() == 16000 and w.getcomptype() == 'NONE',
                'EN_PILOT_WAV_GEOMETRY')
        nframes = w.getnframes()
        require(400 <= nframes <= 16000 * 12, 'EN_PILOT_WAV_DURATION')
        pcm = w.readframes(nframes)
        require(len(pcm) == nframes * 2 and not w.readframes(1),
                'EN_PILOT_WAV_PCM_TRUNCATED')
    return np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768, sha(pcm)


def load_strict_inputs(source: Path):
    """Never accept a reshuffled/faked source identity or a promoted label."""
    require(source.is_absolute() and source.is_file() and not source.is_symlink()
            and source.stat().st_size <= 8 * 1024 * 1024, 'EN_PILOT_SOURCE_INVALID')
    require(sha(source.read_bytes()) == INPUT_SHA256, 'EN_PILOT_ZIP_SHA_CHANGED')
    with zipfile.ZipFile(source) as z:
        infos = z.infolist()
        require(len(infos) == 64, 'EN_PILOT_ZIP_MEMBER_COUNT')
        paths = set()
        for info in infos:
            p = PurePosixPath(info.filename)
            require(not p.is_absolute() and '\\' not in info.filename and
                    '\x00' not in info.filename and
                    all(part not in ('.', '..', '') for part in p.parts) and
                    info.filename not in paths and
                    not stat.S_ISLNK(info.external_attr >> 16) and
                    info.file_size <= 4 * 1024 * 1024,
                    'EN_PILOT_ZIP_MEMBER_UNSAFE')
            paths.add(info.filename)
        require('MANIFEST.json' in paths, 'EN_PILOT_MANIFEST_MISSING')
        manifest = json.loads(z.read('MANIFEST.json'))
        require(type(manifest) is dict and
                manifest.get('schema') == 'english63-original-audio-weak-intake-v1' and
                manifest.get('status') == 'SOURCE_VERIFIED_OFFLINE_REPACK_NO_TRAINING' and
                manifest.get('source_tts_run_id') == SOURCE_RUN and
                manifest.get('source_tts_zip_sha256') == SOURCE_TTS_SHA256 and
                manifest.get('source_review_sha256') == SOURCE_REVIEW_SHA256 and
                manifest.get('english_class_roster') == CLASSES and
                manifest.get('counts') == EXPECTED_COUNTS and
                manifest.get('training_admitted') is False and
                manifest.get('formal_training_eligible') is False and
                manifest.get('model_promotion_allowed') is False and
                manifest.get('human_gold') is False,
                'EN_PILOT_MANIFEST_POLICY_CHANGED')
        rows = manifest.get('rows')
        require(type(rows) is list and len(rows) == 63,
                'EN_PILOT_WRONG_COUNT')
        expected_files = {'MANIFEST.json'}
        cell_ids, wav_hashes, pcm_hashes, speaker_roles = set(), set(), set(), {}
        audio = {}
        counts = dict.fromkeys(EXPECTED_COUNTS, 0)
        for r in rows:
            require(type(r) is dict, 'EN_PILOT_ROW_INVALID')
            action = r.get('action')
            family = r.get('source_family_id')
            cell = r.get('cell_id')
            role = r.get('role')
            path = r.get('local_relative_wav')
            need_label = action != 'QUARANTINE_AMBIGUOUS_NEARPHONE'
            expected_role = ('train-candidate' if family in
                             {'qwen3-preset-' + s for s in TRAIN_VOICES} else
                             'development-only' if family in
                             {'qwen3-preset-' + s for s in DEV_VOICES} else None)
            require(action in ROLES and expected_role is not None and
                    role == expected_role and
                    (action == 'EXPOSED_DEVELOPMENT_NO_TUNE') ==
                    (role == 'development-only' and need_label) and
                    type(cell) is str and CELL.fullmatch(cell) and
                    path == ROLES[action] + '/' + cell + '.wav' and
                    type(r.get('wav_sha256')) is str and
                    SHA64.fullmatch(r['wav_sha256']) and
                    type(r.get('pcm_sha256')) is str and
                    SHA64.fullmatch(r['pcm_sha256']) and
                    cell not in cell_ids and
                    r['wav_sha256'] not in wav_hashes and
                    r['pcm_sha256'] not in pcm_hashes and
                    r.get('training_admitted') is False and
                    r.get('formal_training_eligible') is False and
                    r.get('human_gold') is False and
                    r.get('trusted_negative') is False and
                    r.get('ctc_target_ids') is None and
                    ((r.get('weak_wholeclip_class') in (0,1,2)) if need_label else
                     r.get('weak_wholeclip_class') is None),
                    'EN_PILOT_ROW_ROLE_OR_ADMISSION_DRIFT')
            cell_ids.add(cell)
            wav_hashes.add(r['wav_sha256'])
            pcm_hashes.add(r['pcm_sha256'])
            speaker_roles.setdefault(family, set()).add(role)
            expected_files.add(path)
            raw = z.read(path)
            require(sha(raw) == r['wav_sha256'], 'EN_PILOT_WAV_CHANGED')
            samples, digest = wav_samples(raw)
            require(digest == r['pcm_sha256'], 'EN_PILOT_PCM_CHANGED')
            audio[cell] = samples
            counts[action] += 1
        require(set(z.namelist()) == expected_files and
                counts == EXPECTED_COUNTS and
                len(speaker_roles) == 7 and
                all(len(s) == 1 for s in speaker_roles.values()),
                'EN_PILOT_SOURCE_LEAK_OR_FILE_EXTRA')
    return rows, audio


def mel_bank():
    import numpy as np
    freq = np.linspace(0.0, 8000.0, 257, dtype=np.float64)
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    knots = mel_to_hz(np.linspace(hz_to_mel(70), hz_to_mel(7600), FREQ_BANDS+2))
    bank = np.zeros((FREQ_BANDS, 257), np.float64)
    for i in range(FREQ_BANDS):
        left, center, right = knots[i:i+3]
        bank[i, :] = np.maximum(0, np.minimum((freq-left)/(center-left),
                                              (right-freq)/(right-center)))
        bank[i] /= max(1.e-12, bank[i].sum())
    return bank.astype(np.float32)


def features(wav, bank):
    import numpy as np
    frames = np.lib.stride_tricks.sliding_window_view(wav, 400)[::160]
    require(50 <= len(frames) <= 1200, 'EN_PILOT_FRAMES_BUDGET')
    win = np.hanning(400).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * win, n=512, axis=1)) ** 2
    logmel = np.log(np.maximum(power @ bank.T, 1e-8)).astype(np.float32)
    require(logmel.shape == (len(frames), FREQ_BANDS) and
            bool(np.isfinite(logmel).all()), 'EN_PILOT_FEATURES_NONFINITE')
    return logmel


def train_only_norm(features_by_id, records):
    import numpy as np
    train = [features_by_id[r['cell_id']] for r in records
             if r['action'] == 'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY']
    joined = np.concatenate(train, axis=0).astype(np.float64)
    mean = joined.mean(axis=0).astype(np.float32)
    std = np.maximum(joined.std(axis=0), 0.5).astype(np.float32)
    require(np.isfinite(mean).all() and np.isfinite(std).all(),
            'EN_PILOT_FEATURE_NORM_INVALID')
    return mean, std


def create_model(kind):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class EncoderGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(FREQ_BANDS, HIDDEN, batch_first=True)
            self.classifier = nn.Linear(HIDDEN*2, 3)

        def forward(self, x, lengths):
            state, _ = self.rnn(x)
            mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            pooled = torch.cat((
                (state*mask.unsqueeze(-1)).sum(dim=1)/lengths.unsqueeze(-1),
                state.masked_fill(~mask.unsqueeze(-1), -1.e5).max(dim=1).values), dim=-1)
            return self.classifier(pooled)

    class CausalBlock(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.dilation = dilation
            self.dw = nn.Conv1d(HIDDEN,HIDDEN,3,dilation=dilation,groups=HIDDEN)
            self.pw = nn.Conv1d(HIDDEN,HIDDEN,1)
        def forward(self, x):
            return x + 0.5*self.pw(F.relu(self.dw(F.pad(x,(2*self.dilation,0)))))

    class EncoderTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv1d(FREQ_BANDS,HIDDEN,1)
            self.blocks = nn.Sequential(*(CausalBlock(d) for d in (1,2,4)))
            self.classifier = nn.Linear(HIDDEN*2,3)

        def forward(self,x,lengths):
            state = self.blocks(F.relu(self.proj(x.transpose(1,2)))).transpose(1,2)
            mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            pooled = torch.cat((
                (state*mask.unsqueeze(-1)).sum(dim=1)/lengths.unsqueeze(-1),
                state.masked_fill(~mask.unsqueeze(-1),-1.e5).max(dim=1).values),dim=-1)
            return self.classifier(pooled)

    if kind == 'gru':
        return EncoderGRU()
    if kind == 'dsc_tcn':
        return EncoderTCN()
    raise ValueError('EN_PILOT_UNKNOWN_MODEL_KIND')


def pad_tensor(records, by_id):
    import numpy as np
    import torch
    group = [by_id[r['cell_id']] for r in records]
    lens = torch.tensor([len(v) for v in group], dtype=torch.long)
    max_len = int(lens.max().item())
    stack = np.zeros((len(records), max_len, FREQ_BANDS), dtype=np.float32)
    for i, a in enumerate(group):
        stack[i,:len(a)] = a
    return torch.from_numpy(stack), lens


def score_model(model, records, by_id):
    import torch
    x,lens = pad_tensor(records,by_id)
    with torch.no_grad():
        logits = model(x,lens)
        pred = logits.argmax(dim=-1).cpu().tolist()
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
    out = []
    for row, cls, probs in zip(records,pred,prob):
        out.append({'cell_id':row['cell_id'],
                    'source_family_id':row['source_family_id'],
                    'role':row['role'], 'action':row['action'],
                    'requested_weak_intent_id':row['weak_wholeclip_class'],
                    'predicted_class':int(cls),
                    'raw_clip_probabilities':[round(float(p),6) for p in probs],
                    'prediction_is_acoustic_truth':False,
                    'window_or_streaming_event':False,
                    'human_gold':False,
                    'training_admitted':False,
                    'trusted_negative':False})
    return out


def save_safe_npz(path, model):
    import numpy as np
    arrays = {k: v.detach().cpu().contiguous().numpy() for k,v in model.state_dict().items()}
    with path.open('xb') as f:
        np.savez(f, **arrays)
    return sha(path.read_bytes()), {k:list(v.shape) for k,v in arrays.items()}


def train_one(kind, all_rows, normed, output):
    import torch
    from torch.nn import functional as F
    torch.manual_seed(SEED + (0 if kind=='gru' else 101))
    model=create_model(kind)
    model.train()
    train=sorted([r for r in all_rows if r['action']=='TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY'],key=lambda r:r['cell_id'])
    dev=sorted([r for r in all_rows if r['action']=='EXPOSED_DEVELOPMENT_NO_TUNE'],key=lambda r:r['cell_id'])
    quar=sorted([r for r in all_rows if r['action']=='QUARANTINE_AMBIGUOUS_NEARPHONE'],key=lambda r:r['cell_id'])
    train_x,lens=pad_tensor(train,normed)
    labels=torch.tensor([r['weak_wholeclip_class'] for r in train],dtype=torch.long)
    require(sorted(labels.unique().tolist()) == [0,1,2], 'EN_PILOT_CLASS_ROSTER')
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.01)
    start=time.monotonic()
    losses=[]
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits=model(train_x,lens)
        loss=F.cross_entropy(logits,labels)
        require(bool(torch.isfinite(loss).item()), 'EN_PILOT_NONFINITE_LOSS')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.item()))
    elapsed=time.monotonic()-start
    model.eval()
    train_pred=score_model(model,train,normed)
    dev_pred=score_model(model,dev,normed)
    quar_pred=score_model(model,quar,normed)
    weights_path=output/(kind+'-weights.npz')
    digest, shapes=save_safe_npz(weights_path,model)
    report={
        'architecture':kind,'class_roster':CLASSES,'feature_bands':FREQ_BANDS,
        'frame_length_ms':25,'frame_hop_ms':10,'model_hidden':HIDDEN,
        'readout':'whole-clip-mean-max-pool-not-streaming-event',
        'architecture_frame_encoder_causal':True,'seed':SEED+(0 if kind=='gru' else 101),
        'optimizer':'AdamW','lr':0.003,'updates':STEPS,'batch_size':30,
        'gradient_clip':1.0,'train_loss_start':round(losses[0],6),
        'train_loss_end':round(losses[-1],6),
        'training_wall_sec':round(elapsed,3),
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'weights_file':weights_path.name,'weights_sha256':digest,
        'tensor_shapes':shapes,
        'requested_weak_train_agreement':sum(r['requested_weak_intent_id']==r['predicted_class'] for r in train_pred),
        'requested_weak_train_total':len(train_pred),
        'exposed_development_weak_agreement_not_independent_accuracy':sum(r['requested_weak_intent_id']==r['predicted_class'] for r in dev_pred),
        'exposed_development_total':len(dev_pred),
        'quarantined_nearphone_wake_proposal_not_false_accept_count':sum(r['predicted_class'] in (1,2) for r in quar_pred),
        'quarantined_nearphone_total':len(quar_pred),
        'per_record':sorted(train_pred+dev_pred+quar_pred,key=lambda x:x['cell_id']),
        'human_gold':False,'training_admitted':False,'qualification_allowed':False,
        'independent_human_test':'NOT_RUN','continuous_FAR_per_hour':'NOT_RUN',
        'board_SSC305':'NOT_RUN','model_promotion_allowed':False,
    }
    return report


def run(input_path:Path,output:Path):
    import numpy as np
    import torch
    require(output.is_absolute() and output.parent.is_dir() and not output.exists() and
            not output.parent.is_symlink(), 'EN_PILOT_OUTPUT_CREATE_ONLY')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    rows,audio=load_strict_inputs(input_path)
    bank=mel_bank()
    by_id={cell: features(samples,bank) for cell,samples in audio.items()}
    mean,std=train_only_norm(by_id,rows)
    normed={cell:((feature-mean)/std).astype(np.float32) for cell,feature in by_id.items()}
    output.mkdir(mode=0o700)
    reports=[train_one(k,rows,normed,output) for k in ('gru','dsc_tcn')]
    result={
        'schema':POLICY,
        'status':'EXECUTED_SYNTHETIC_WEAK_LABEL_TRAINING_NOT_KWS_QUALIFICATION',
        'exact_input_sha256':INPUT_SHA256,'source_tts_sha256':SOURCE_TTS_SHA256,
        'training_script_sha256':sha(Path(__file__).read_bytes()),
        'runtime':{'python':sys.version.split()[0],'torch':torch.__version__,
                   'numpy':np.__version__,'backend':'cpu-deterministic-single-thread-mkldnn-off'},
        'source_three_evidence_join_sha256':SOURCE_REVIEW_SHA256,
        'train_source_families':sorted('qwen3-preset-'+s for s in TRAIN_VOICES),
        'exposed_dev_source_families':sorted('qwen3-preset-'+s for s in DEV_VOICES),
        'weak_train_count':30,'exposed_development_count':12,
        'nearphone_quarantine_count':21,
        'train_only_frontend_norm':{'mean32':mean.tolist(),'std32':std.tolist()},
        'pretrained_checkpoint_used':False,
        'source_intent_is_truth':False,
        'architecture_comparison_is_independent_evaluation':False,
        'training_admitted':False,'human_gold':False,
        'formal_promotion_allowed':False,'model_selection_made':False,
        'new_tts_calls':0,'new_asr_calls':0,'new_ipa_calls':0,
        'models':reports,
    }
    path=output/'RESEARCH_RESULT.json'
    with path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,ensure_ascii=False)
        stream.write('\n')
    path.chmod(0o600)
    print(json.dumps({'status':result['status'],
                      'model_summaries':[{k:m[k] for k in (
                          'architecture','parameter_count','train_loss_start','train_loss_end',
                          'requested_weak_train_agreement',
                          'exposed_development_weak_agreement_not_independent_accuracy',
                          'quarantined_nearphone_wake_proposal_not_false_accept_count',
                          'training_wall_sec','weights_sha256')} for m in reports]},sort_keys=True))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-zip',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        run(args.input_zip,args.output_dir)
    except (OSError,ValueError,EOFError,zipfile.BadZipFile) as exc:
        print('EN_PILOT_REJECT: '+str(exc), file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
