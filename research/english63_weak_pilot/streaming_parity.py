"""Verify causal chunk equivalence of saved English-63 weak clip classifiers.

Research only. Retains the existing whole-clip pooled head; NO continuous wake
recognizer, FAR/FRR, acoustic gold, new training, source inference or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english63_weak_train as pilot

POLICY = 'english63-weak-pilot-encoder-chunk-parity-v1'
CHUNK_SIZES = (1, 5, 31, 'whole')
MAX_ABS_DIFF = 2e-5
KINDS = ('gru', 'dsc_tcn')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def chunks_for_frames(frames, size):
    require(type(frames) is int and 1 <= frames <= 1200,
            'EN_CHUNK_INVALID_FRAME_COUNT')
    if size == 'whole':
        return (frames,)
    require(type(size) is int and 1 <= size <= 1200,
            'EN_CHUNK_INVALID_SIZE')
    out = tuple(min(size, frames - pos) for pos in range(0, frames, size))
    require(sum(out) == frames and all(x > 0 for x in out),
            'EN_CHUNK_INVALID_PLAN')
    return out


def streamed_final_logits(model, kind, features, chunks):
    """Compute final weak clip logit using bounded encoder and pool states.

    The input is already precomputed per-clip Log-Mel; the audio frontend is NOT
    proven streaming here. No early wake event is inferred or scored.
    """
    import torch
    from torch.nn import functional as F
    require(kind in KINDS and isinstance(features, torch.Tensor) and
            features.dtype == torch.float32 and features.device.type == 'cpu' and
            features.ndim == 3 and tuple(features.shape[:1]) == (1,) and
            features.shape[-1] == pilot.FREQ_BANDS and
            type(chunks) in (tuple, list) and bool(chunks) and
            all(type(n) is int and n > 0 for n in chunks) and
            sum(chunks) == features.shape[1] and
            bool(torch.isfinite(features).all()),
            'EN_CHUNK_INVALID_INPUT')
    max_context = max(chunks)
    require(1 <= max_context <= 1200, 'EN_CHUNK_BUDGET')
    h = None
    history = None
    if kind == 'dsc_tcn':
        history = [features.new_zeros((1, pilot.HIDDEN, 2*block.dilation))
                   for block in model.blocks]
    total = features.new_zeros((1, pilot.HIDDEN))
    maximum = features.new_full((1, pilot.HIDDEN), float('-inf'))
    observed = 0
    with torch.no_grad():
        for n in chunks:
            part = features[:, observed:observed+n, :]
            if kind == 'gru':
                state, h = model.rnn(part, h)
            else:
                hidden = F.relu(model.proj(part.transpose(1, 2)))
                for layer, block in enumerate(model.blocks):
                    prior = torch.cat((history[layer], hidden), dim=-1)
                    hidden = hidden + 0.5 * block.pw(F.relu(block.dw(prior)))
                    history[layer] = prior[:, :, -(2*block.dilation):]
                state = hidden.transpose(1, 2)
            total += state.sum(dim=1)
            maximum = torch.maximum(maximum, state.max(dim=1).values)
            observed += n
        return model.classifier(torch.cat((total / observed, maximum), dim=-1))


def verify_saved_result(report, path):
    require(type(report) is dict and report.get('schema') == pilot.POLICY and
            report.get('status') ==
            'EXECUTED_SYNTHETIC_WEAK_LABEL_TRAINING_NOT_KWS_QUALIFICATION' and
            report.get('training_script_sha256') ==
            pilot.sha(Path(pilot.__file__).read_bytes()) and
            report.get('exact_input_sha256') == pilot.INPUT_SHA256 and
            report.get('source_tts_sha256') == pilot.SOURCE_TTS_SHA256 and
            report.get('source_three_evidence_join_sha256') ==
            pilot.SOURCE_REVIEW_SHA256 and
            report.get('human_gold') is False and
            report.get('training_admitted') is False and
            report.get('formal_promotion_allowed') is False and
            report.get('model_selection_made') is False and
            type(report.get('models')) is list and
            sorted(m.get('architecture') for m in report['models']) ==
            sorted(KINDS), 'EN_CHUNK_SAVED_RESEARCH_RECEIPT_INVALID')
    import numpy as np
    import torch
    states = {}
    for record in report['models']:
        kind = record['architecture']
        weight = path / (kind + '-weights.npz')
        require(weight.is_file() and not weight.is_symlink() and
                weight.stat().st_size <= 500000 and
                record.get('weights_file') == weight.name and
                pilot.sha(weight.read_bytes()) == record.get('weights_sha256') and
                record.get('human_gold') is False and
                record.get('training_admitted') is False and
                record.get('model_promotion_allowed') is False,
                'EN_CHUNK_SAVED_WEIGHT_INVALID')
        model = pilot.create_model(kind)
        with np.load(weight, allow_pickle=False) as arrays:
            require(set(arrays.files) == set(model.state_dict()),
                    'EN_CHUNK_WEIGHT_STATE_MISMATCH')
            state = {key: torch.from_numpy(arrays[key]) for key in arrays.files}
        model.load_state_dict(state, strict=True)
        model.eval()
        states[kind] = (model, record)
    return states



def verify_public_parity_receipt(value, saved):
    """Model-free audit of retained measurements, never acoustic qualification."""
    import math
    require(type(value) is dict and value.get('schema') == POLICY and
            value.get('status') ==
            'MEASURED_WEAK_CLIP_ENCODER_CHUNK_PARITY_NOT_CONTINUOUS_KWS' and
            value.get('source_input_zip_sha256') == pilot.INPUT_SHA256 and
            value.get('source_tts_zip_sha256') == pilot.SOURCE_TTS_SHA256 and
            value.get('source_review_sha256') == pilot.SOURCE_REVIEW_SHA256 and
            value.get('source_training_code_sha256') ==
            pilot.sha(Path(pilot.__file__).read_bytes()) and
            value.get('streaming_parity_code_sha256') ==
            pilot.sha(Path(__file__).read_bytes()) and
            value.get('split_policy') == {
                'weak_train': 30, 'exposed_development': 12,
                'quarantined_nearphones': 21} and
            value.get('new_audio_generated') is False and
            value.get('new_asr_calls') == value.get('new_ipa_calls') == 0 and
            value.get('optimizer_updates') == 0 and
            value.get('new_training_started') is False and
            value.get('threshold_selection') is False and
            value.get('human_gold') is False and
            value.get('formal_training_admitted') is False and
            value.get('model_promotion_allowed') is False and
            value.get('continuous_negative_far_per_hour') == 'NOT_RUN' and
            value.get('independent_human_frr') == 'NOT_RUN' and
            value.get('ssc305_board_profile') == 'NOT_RUN' and
            type(value.get('models')) is list and len(value['models']) == 2,
            'EN_CHUNK_RETAINED_RECEIPT_UNSAFE')
    require(type(saved) is dict and
            saved.get('training_script_sha256') ==
            value['source_training_code_sha256'] and
            type(saved.get('models')) is list and len(saved['models']) == 2,
            'EN_CHUNK_SAVED_BASELINE_INVALID')
    original = {m['architecture']: m for m in saved['models']}
    require(set(original) == set(KINDS), 'EN_CHUNK_ORIGINAL_MODEL_ROSTER')
    seen = set()
    for entry in value['models']:
        kind = entry.get('kind')
        require(kind in KINDS and kind not in seen and
                entry.get('weights_sha256') == original[kind]['weights_sha256'] and
                entry.get('input_clip_count') == 63 and
                entry.get('comparisons') == 252 and
                entry.get('chunk_plans') == [str(n) for n in CHUNK_SIZES] and
                entry.get('frame_frontend_streaming_tested') is False and
                entry.get('continuous_keyword_event_tested') is False and
                entry.get('streaming_farr_tested') is False and
                entry.get('model_promotion_allowed') is False,
                'EN_CHUNK_RETAINED_MODEL_UNSAFE')
        seen.add(kind)
        diffs = entry.get('max_abs_logit_diff_by_plan')
        errors = entry.get('final_clip_class_drift_by_plan')
        require(type(diffs) is dict and type(errors) is dict and
                set(diffs) == set(errors) == {str(n) for n in CHUNK_SIZES} and
                all(type(errors[k]) is int and errors[k] == 0 for k in errors) and
                all(type(diffs[k]) in (float, int) and
                    math.isfinite(diffs[k]) and 0 <= diffs[k] <= MAX_ABS_DIFF
                    for k in diffs), 'EN_CHUNK_RETAINED_NUMERIC_UNSAFE')
    return True


def run(input_zip, result_dir, output):
    import numpy as np
    import torch
    require(output.is_absolute() and output.parent.is_dir() and
            not output.parent.is_symlink() and not output.exists() and
            result_dir.is_absolute() and result_dir.is_dir() and
            not result_dir.is_symlink(), 'EN_CHUNK_CREATE_ONLY_PATHS')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    report = json.loads((result_dir / 'RESEARCH_RESULT.json').read_bytes())
    models = verify_saved_result(report, result_dir)
    rows, audio = pilot.load_strict_inputs(input_zip)
    bank = pilot.mel_bank()
    source_feats = {key: pilot.features(samples, bank)
                    for key, samples in audio.items()}
    mean = np.asarray(report['train_only_frontend_norm']['mean32'], dtype=np.float32)
    std = np.asarray(report['train_only_frontend_norm']['std32'], dtype=np.float32)
    measured_mean, measured_std = pilot.train_only_norm(source_feats, rows)
    require(np.array_equal(mean, measured_mean) and
            np.array_equal(std, measured_std), 'EN_CHUNK_TRAIN_ONLY_NORM_DRIFT')
    arrays = {cell: ((feat - mean) / std).astype(np.float32)
              for cell, feat in source_feats.items()}
    by_model = []
    for kind in KINDS:
        model, original = models[kind]
        diffs = {str(n): 0.0 for n in CHUNK_SIZES}
        classified = {str(n): 0 for n in CHUNK_SIZES}
        comparisons = 0
        with torch.no_grad():
            for cell, feature in sorted(arrays.items()):
                tensor = torch.from_numpy(feature).unsqueeze(0)
                batch_logits = model(tensor, torch.tensor([len(feature)]))
                for n in CHUNK_SIZES:
                    partition = chunks_for_frames(len(feature), n)
                    stream_logits = streamed_final_logits(model, kind, tensor, partition)
                    err = float((batch_logits - stream_logits).abs().max().item())
                    require(np.isfinite(err) and err <= MAX_ABS_DIFF,
                            'EN_CHUNK_NUMERIC_PARITY_FAILED')
                    label_diff = (batch_logits.argmax(dim=-1) !=
                                  stream_logits.argmax(dim=-1)).any().item()
                    require(not label_diff, 'EN_CHUNK_CLASS_DRIFT')
                    diffs[str(n)] = max(diffs[str(n)], err)
                    classified[str(n)] += int(label_diff)
                    comparisons += 1
        require(comparisons == 63 * len(CHUNK_SIZES),
                'EN_CHUNK_EXPECTED_RECORDS')
        by_model.append({
            'kind': kind,
            'weights_sha256': original['weights_sha256'],
            'input_clip_count': 63,
            'chunk_plans': [str(n) for n in CHUNK_SIZES],
            'comparisons': comparisons,
            'max_abs_logit_diff_by_plan': diffs,
            'final_clip_class_drift_by_plan': classified,
            'frame_frontend_streaming_tested': False,
            'continuous_keyword_event_tested': False,
            'streaming_farr_tested': False,
            'model_promotion_allowed': False,
        })
    result = {
        'schema': POLICY,
        'status': 'MEASURED_WEAK_CLIP_ENCODER_CHUNK_PARITY_NOT_CONTINUOUS_KWS',
        'source_input_zip_sha256': pilot.INPUT_SHA256,
        'source_tts_zip_sha256': pilot.SOURCE_TTS_SHA256,
        'source_review_sha256': pilot.SOURCE_REVIEW_SHA256,
        'source_training_code_sha256': pilot.sha(Path(pilot.__file__).read_bytes()),
        'streaming_parity_code_sha256': pilot.sha(Path(__file__).read_bytes()),
        'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__,
                    'numpy': np.__version__},
        'split_policy': {'weak_train': 30, 'exposed_development': 12,
                         'quarantined_nearphones': 21},
        'models': by_model,
        'new_audio_generated': False,
        'new_asr_calls': 0, 'new_ipa_calls': 0,
        'optimizer_updates': 0,
        'threshold_selection': False,
        'new_training_started': False,
        'human_gold': False,
        'formal_training_admitted': False,
        'independent_human_frr': 'NOT_RUN',
        'continuous_negative_far_per_hour': 'NOT_RUN',
        'ssc305_board_profile': 'NOT_RUN',
        'model_promotion_allowed': False,
    }
    verify_public_parity_receipt(result, report)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
    output.chmod(0o600)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-zip', type=Path, required=True)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        result = run(args.input_zip, args.result_dir, args.output)
        print(json.dumps({'status': result['status'],
                          'models': result['models'],
                          'model_promotion_allowed': False}, sort_keys=True))
    except (OSError, ValueError, RuntimeError, KeyError, EOFError) as exc:
        print('EN_CHUNK_PARITY_STOP: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
