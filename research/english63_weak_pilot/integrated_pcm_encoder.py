#!/usr/bin/env python3
"""Exact-source English63 PCM -> log-Mel -> saved-weight encoder replay.

This is causal whole-clip end-logit equivalence for weak synthetic models.
It is NOT a continuous wake detector, speaker-independent accuracy, FAR/hour,
human FRR, formal training admission, model selection or SSC305 qualification.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import english63_weak_train as pilot
import pcm_frontend_parity as pcm

POLICY = 'english63-original-pcm-to-saved-encoder-final-logit-parity-v1'
STATUS = 'MEASURED_WEAK_ORIGINAL_PCM_END_LOGITS_NOT_CONTINUOUS_KWS'
CHUNKS = pcm.CHUNKS
MAX_LOGIT_DIFF = 2e-4
KINDS = ('gru', 'dsc_tcn')
SOURCE_RUN = 43242278962


def require(test, code):
    if not test:
        raise ValueError(code)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_result_metadata(report):
    """Dependency-free refusal of forged promotion and numeric receipts."""
    require(type(report) is dict and report.get('schema') == POLICY and
            report.get('status') == STATUS and
            report.get('source_tts_zip_sha256') == pcm.TTS_SHA256 and
            report.get('source_review_json_sha256') == pcm.REVIEW_SHA256 and
            report.get('runner_sha256') == digest(Path(__file__).read_bytes()) and
            report.get('saved_training_script_sha256') ==
                digest((ROOT/'english63_weak_train.py').read_bytes()) and
            report.get('saved_streaming_encoder_script_sha256') ==
                digest((ROOT/'streaming_parity.py').read_bytes()) and
            report.get('saved_frontend_script_sha256') ==
                digest((ROOT/'pcm_frontend_parity.py').read_bytes()) and
            report.get('source_audio_clips') == 63 and
            report.get('weak_train_role') == 30 and
            report.get('exposed_development_role') == 12 and
            report.get('nearphone_quarantined') == 21 and
            report.get('optimizer_updates') == 0 and
            report.get('new_audio_or_asr_generated') is False and
            report.get('human_gold') is False and
            report.get('formal_training_admitted') is False and
            report.get('threshold_selection') is False and
            report.get('continuous_wake_event_tested') is False and
            report.get('independent_english_frr') == 'NOT_RUN' and
            report.get('continuous_negative_far_per_hour') == 'NOT_RUN' and
            report.get('ssc305_board_qualification') == 'NOT_RUN' and
            report.get('model_selection_made') is False and
            report.get('model_promotion_allowed') is False,
            'EN_CHAIN_REPORT_UNSAFE')
    models = report.get('models')
    require(type(models) is list and len(models) == 2 and
            {m.get('kind') for m in models} == set(KINDS),
            'EN_CHAIN_REPORT_MODELS')
    for m in models:
        require(type(m.get('weights_sha256')) is str and
                len(m['weights_sha256']) == 64 and
                m.get('clip_count') == 63 and
                m.get('streamed_clip_comparisons') == 63*len(CHUNKS) and
                m.get('chunk_samples') == list(CHUNKS) and
                m.get('changed_final_classes') == 0 and
                m.get('final_clip_only') is True and
                m.get('continuous_event_or_far') is False and
                m.get('model_promotion_allowed') is False,
                'EN_CHAIN_REPORT_BOUNDARY')
        values = m.get('max_abs_logit_diff_by_chunk')
        require(type(values) is dict and set(values) == {str(n) for n in CHUNKS}
                and all(type(v) in (float, int) and math.isfinite(v) and
                        0 <= v <= MAX_LOGIT_DIFF for v in values.values()),
                'EN_CHAIN_REPORT_LOGITS')
    return True


class IncrementalEncoder:
    """Only state and current feature chunk are retained; no clip features."""
    def __init__(self, model, kind):
        import torch
        require(kind in KINDS, 'EN_CHAIN_MODEL_KIND')
        self.model, self.kind = model, kind
        self.observed = 0
        self.total = torch.zeros((1, pilot.HIDDEN), dtype=torch.float32)
        self.maximum = torch.full((1, pilot.HIDDEN), float('-inf'))
        self.gru_h = None
        self.conv_history = None
        if kind == 'dsc_tcn':
            self.conv_history = [
                torch.zeros((1, pilot.HIDDEN, 2*b.dilation), dtype=torch.float32)
                for b in model.blocks
            ]

    def consume(self, normalized):
        import numpy as np
        import torch
        from torch.nn import functional as F
        require(type(normalized) is np.ndarray and normalized.dtype == np.float32
                and normalized.ndim == 2 and normalized.shape[1] == pilot.FREQ_BANDS
                and 1 <= len(normalized) <= 24 and
                bool(np.isfinite(normalized).all()), 'EN_CHAIN_FRAME_CHUNK_INVALID')
        part = torch.from_numpy(normalized).unsqueeze(0)
        with torch.no_grad():
            if self.kind == 'gru':
                state, self.gru_h = self.model.rnn(part, self.gru_h)
            else:
                hidden = F.relu(self.model.proj(part.transpose(1, 2)))
                for layer, block in enumerate(self.model.blocks):
                    prior = torch.cat((self.conv_history[layer], hidden), dim=-1)
                    hidden = hidden + 0.5*block.pw(F.relu(block.dw(prior)))
                    self.conv_history[layer] = prior[:, :, -(2*block.dilation):]
                state = hidden.transpose(1, 2)
            self.total += state.sum(dim=1)
            self.maximum = torch.maximum(self.maximum, state.max(dim=1).values)
        self.observed += len(normalized)
        require(self.observed <= 1200, 'EN_CHAIN_FRAME_BUDGET')

    def final_logits(self):
        import torch
        require(50 <= self.observed <= 1200, 'EN_CHAIN_INCOMPLETE')
        with torch.no_grad():
            return self.model.classifier(
                torch.cat((self.total/self.observed, self.maximum), dim=-1)
            )


def load_models(saved):
    import numpy as np
    import torch
    require(saved.get('schema') == pilot.POLICY and
            saved.get('status') ==
                'EXECUTED_SYNTHETIC_WEAK_LABEL_TRAINING_NOT_KWS_QUALIFICATION' and
            saved.get('training_script_sha256') ==
                digest((ROOT/'english63_weak_train.py').read_bytes()) and
            saved.get('exact_input_sha256') == pilot.INPUT_SHA256 and
            saved.get('source_three_evidence_join_sha256') == pcm.REVIEW_SHA256 and
            saved.get('human_gold') is False and
            saved.get('training_admitted') is False and
            saved.get('formal_promotion_allowed') is False and
            saved.get('model_selection_made') is False,
            'EN_CHAIN_SAVED_REPORT_UNSAFE')
    records = saved.get('models')
    require(type(records) is list and len(records) == 2 and
            {r.get('architecture') for r in records} == set(KINDS),
            'EN_CHAIN_ARCHITECTURE_DRIFT')
    built = {}
    for item in records:
        kind = item['architecture']
        raw = base64.b64decode((ROOT/(kind+'-weights.npz.b64')).read_text().strip(),
                               validate=True)
        require(digest(raw) == item.get('weights_sha256') and
                len(raw) < 500000 and item.get('human_gold') is False and
                item.get('training_admitted') is False and
                item.get('model_promotion_allowed') is False,
                'EN_CHAIN_WEIGHTS_CHANGED')
        model = pilot.create_model(kind)
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            expected = model.state_dict()
            require(set(archive.files) == set(expected), 'EN_CHAIN_TENSOR_ROSTER')
            state = {}
            for key in archive.files:
                value = archive[key]
                require(value.dtype == np.float32 and
                        tuple(value.shape) == tuple(expected[key].shape) and
                        bool(np.isfinite(value).all()), 'EN_CHAIN_TENSOR_UNSAFE')
                state[key] = torch.from_numpy(value.copy())
        model.load_state_dict(state, strict=True)
        model.eval()
        original = item['per_record']
        require(type(original) is list and len(original) == 63 and
                len({r.get('cell_id') for r in original}) == 63,
                'EN_CHAIN_PREDICTION_ROSTER')
        built[kind] = (model, {r['cell_id']: r for r in original},
                       item['weights_sha256'])
    return built


def read_verified_audio(source_zip, review_json, saved):
    import numpy as np
    require(source_zip.is_absolute() and review_json.is_absolute() and
            source_zip.is_file() and review_json.is_file() and
            not source_zip.is_symlink() and not review_json.is_symlink() and
            source_zip.stat().st_size <= 16*1024*1024 and
            review_json.stat().st_size <= 1024*1024,
            'EN_CHAIN_INPUT_PATHS')
    require(digest(source_zip.read_bytes()) == pcm.TTS_SHA256 and
            digest(review_json.read_bytes()) == pcm.REVIEW_SHA256,
            'EN_CHAIN_SOURCE_CHANGED')
    pcm.verify_receipt(json.loads((ROOT/'PCM_FRONTEND_PARITY.json').read_text()))
    review = json.loads(review_json.read_text())
    rows = review.get('rows')
    require(review.get('schema') ==
                'english63-independent-evidence-joined-review-priority-v1' and
            review.get('status') == 'MACHINE_EVIDENCE_JOINED_REVIEW_NOT_GOLD' and
            review.get('source_tts_run_id') == SOURCE_RUN and
            review.get('audio_rows_reused') == 63 and
            review.get('human_gold') is False and
            review.get('training_admitted') is False and
            type(rows) is list and len(rows) == 63,
            'EN_CHAIN_REVIEW_NOT_BOUND')
    sources = {}
    roles = {'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY': 0,
             'EXPOSED_DEVELOPMENT_NO_TUNE': 0,
             'QUARANTINE_AMBIGUOUS_NEARPHONE': 0}
    families = {}
    with zipfile.ZipFile(source_zip) as z:
        paths = {p for p in z.namelist() if p.endswith('/audio-16k.wav')}
        require(len(paths) == 63, 'EN_CHAIN_AUDIO_FILES')
        for r in rows:
            cell = r.get('cell_id')
            family = r.get('source_family_id')
            role = r.get('role')
            intent = r.get('requested_intent')
            action = ('QUARANTINE_AMBIGUOUS_NEARPHONE' if intent == 'near_miss'
                      else 'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY'
                      if role == 'train-candidate'
                      else 'EXPOSED_DEVELOPMENT_NO_TUNE')
            require(type(cell) is str and pcm.CELL.fullmatch(cell) and
                    cell not in sources and
                    family in {'qwen3-preset-'+s for s in pcm.SPEAKERS} and
                    role in ('train-candidate', 'development-only') and
                    intent in ('wake','near_miss','ordinary_negative') and
                    ((role == 'train-candidate') ==
                     (family in {'qwen3-preset-'+s for s in pilot.TRAIN_VOICES})) and
                    r.get('training_admitted') is False and
                    r.get('human_gold') is False and
                    r.get('trusted_negative') is False and
                    r.get('phonetic_label_authenticated') is False,
                    'EN_CHAIN_REVIEW_ROW_UNSAFE')
            families.setdefault(family, set()).add(role)
            path = 'english-seven-voice-actual/capture/cells/'+cell+'/audio-16k.wav'
            require(path in paths, 'EN_CHAIN_SOURCE_WAV_MISSING')
            paths.remove(path)
            sources[cell] = (r, action, pcm.pcm_from_wav(z.read(path), r))
            roles[action] += 1
        require(not paths and len(sources) == 63 and
                roles == {'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY': 30,
                          'EXPOSED_DEVELOPMENT_NO_TUNE': 12,
                          'QUARANTINE_AMBIGUOUS_NEARPHONE': 21} and
                len(families) == 7 and all(len(x) == 1 for x in families.values()),
                'EN_CHAIN_SPLIT_ROSTER')
    bank = pilot.mel_bank()
    require(np.array_equal(bank, pcm.mel_bank()), 'EN_CHAIN_MEL_RECIPE_DRIFT')
    feats = {cell: pilot.features(samples, bank)
             for cell, (_, _, samples) in sources.items()}
    train_rows = [{'cell_id': cell, 'action': action}
                  for cell, (_, action, _) in sources.items()]
    mean, std = pilot.train_only_norm(feats, train_rows)
    norms = saved.get('train_only_frontend_norm')
    require(type(norms) is dict and
            np.array_equal(mean, np.asarray(norms['mean32'], dtype=np.float32)) and
            np.array_equal(std, np.asarray(norms['std32'], dtype=np.float32)),
            'EN_CHAIN_TRAIN_NORM_DRIFT')
    return sources, feats, bank, mean, std


def run(source_zip, review_json, output):
    import numpy as np
    import torch
    require(output.is_absolute() and output.parent.is_dir() and
            not output.exists() and not output.parent.is_symlink(),
            'EN_CHAIN_OUTPUT_CREATE_ONLY')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    saved = json.loads((ROOT/'RESULT.json').read_text())
    models = load_models(saved)
    sources, feats, bank, mean, std = read_verified_audio(source_zip, review_json,
                                                          saved)
    observations = []
    with torch.no_grad():
        for kind in KINDS:
            model, previous, weight_sha = models[kind]
            max_error = {str(n): 0.0 for n in CHUNKS}
            compared, class_drift = 0, 0
            for cell in sorted(sources):
                row, action, samples = sources[cell]
                prior = previous[cell]
                require(prior['action'] == action and
                        prior['role'] == row['role'] and
                        prior['source_family_id'] == row['source_family_id'] and
                        prior['human_gold'] is False and
                        prior['trusted_negative'] is False,
                        'EN_CHAIN_ORIGINAL_PREDICTION_POLICY')
                feat = ((feats[cell] - mean)/std).astype(np.float32)
                batch = model(torch.from_numpy(feat).unsqueeze(0),
                              torch.tensor([len(feat)]))
                cls = int(batch.argmax(dim=-1).item())
                prob = torch.softmax(batch, dim=-1).squeeze(0).numpy()
                require(cls == prior['predicted_class'] and
                        all(abs(float(a)-float(b)) < 2e-5
                            for a,b in zip(prob, prior['raw_clip_probabilities'])),
                        'EN_CHAIN_ORIGINAL_PREDICTION_DRIFT')
                for size in CHUNKS:
                    framer = pcm.FrameBuffer()
                    encoder = IncrementalEncoder(model, kind)
                    for pos in range(0, len(samples), size):
                        windows = framer.push(samples[pos:pos+size])
                        if windows:
                            mel = pcm.logmel(np.asarray(windows, dtype=np.float32),
                                             bank)
                            normalized = ((mel-mean)/std).astype(np.float32)
                            encoder.consume(normalized)
                    require(framer.finish() == len(feat) == encoder.observed,
                            'EN_CHAIN_FRAME_COUNT_DRIFT')
                    predicted = encoder.final_logits()
                    diff = float(torch.max(torch.abs(predicted-batch)).item())
                    require(math.isfinite(diff) and diff <= MAX_LOGIT_DIFF,
                            'EN_CHAIN_LOGIT_DIFFERENCE')
                    mismatch = int(predicted.argmax(dim=-1).item() != cls)
                    require(mismatch == 0, 'EN_CHAIN_FINAL_CLASS_DIFFERENCE')
                    compared += 1
                    class_drift += mismatch
                    max_error[str(size)] = max(max_error[str(size)], diff)
            require(compared == 315 and class_drift == 0, 'EN_CHAIN_COMPARISONS')
            observations.append({
                'kind':kind, 'weights_sha256':weight_sha, 'clip_count':63,
                'streamed_clip_comparisons':compared,
                'chunk_samples':list(CHUNKS),
                'max_abs_logit_diff_by_chunk':max_error,
                'changed_final_classes':class_drift,
                'final_clip_only':True, 'continuous_event_or_far':False,
                'model_promotion_allowed':False,
            })
    receipt = {
        'schema': POLICY, 'status': STATUS,
        'source_tts_zip_sha256': pcm.TTS_SHA256,
        'source_review_json_sha256': pcm.REVIEW_SHA256,
        'runner_sha256': digest(Path(__file__).read_bytes()),
        'saved_training_script_sha256': digest((ROOT/'english63_weak_train.py').read_bytes()),
        'saved_streaming_encoder_script_sha256': digest((ROOT/'streaming_parity.py').read_bytes()),
        'saved_frontend_script_sha256': digest((ROOT/'pcm_frontend_parity.py').read_bytes()),
        'source_audio_clips':63, 'weak_train_role':30,
        'exposed_development_role':12, 'nearphone_quarantined':21,
        'optimizer_updates':0, 'new_audio_or_asr_generated':False,
        'models':observations,
        'runtime':{'python':sys.version.split()[0],'torch':torch.__version__,
                   'numpy':np.__version__},
        'human_gold':False, 'formal_training_admitted':False,
        'threshold_selection':False, 'continuous_wake_event_tested':False,
        'independent_english_frr':'NOT_RUN',
        'continuous_negative_far_per_hour':'NOT_RUN',
        'ssc305_board_qualification':'NOT_RUN',
        'model_selection_made':False,'model_promotion_allowed':False,
    }
    verify_result_metadata(receipt)
    with output.open('x',encoding='utf-8') as f:
        json.dump(receipt,f,indent=2,sort_keys=True)
        f.write('\n')
    output.chmod(0o600)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tts-zip',type=Path,required=True)
    p.add_argument('--review-json',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        result=run(a.tts_zip,a.review_json,a.output)
    except (OSError,ValueError,KeyError,EOFError,RuntimeError,zipfile.BadZipFile) as e:
        print('EN_FULL_CHAIN_STOP: '+str(e),file=sys.stderr)
        return 2
    print(json.dumps({'status':result['status'],'models':result['models'],
                      'model_promotion_allowed':False},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
