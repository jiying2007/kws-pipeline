"""Stdlib archive integrity and scalar/token consistency only; no execution proof."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

FLAGS = ('human_gold', 'independent_evaluation', 'production_qualified',
         'reusable_weights_exported', 'shipping_qualified', 'training_admitted')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def numeric_types(value, key=None):
    """Reject bool-as-number and nonfinite JSON numerics in public statistics."""
    integer_fields = {'update', 'optimizer_call_reserved', 'optimizer_calls_reserved',
                      'input_frames', 'target_length', 'max_optimizer_calls',
                      'successful_optimizer_steps', 'unused_ab_optimizer_calls',
                      'new_tts_calls', 'new_asr_calls', 'new_model_calls', 'new_optimizer_steps', 'exit_code', 'child_max_rss_kib'}
    scalar_fields = {'finite_grad_norm', 'finite_loss', 'ctc_nll',
                     'ctc_nll_per_input_frame', 'blank_argmax_fraction',
                     'blank_mean_probability', 'macro_complete_rate',
                     'whole_process_wall_seconds', 'exact_zero_fraction', 'rms', 'peak'}
    if key in integer_fields:
        require(type(value) is int and value >= 0, 'integer field: ' + key)
    if key in scalar_fields:
        require(type(value) in (int, float) and math.isfinite(value), 'numeric field: ' + key)
    if key in ('per_class_complete_rate', 'final_class_complete_rates'):
        require(type(value) is dict and set(value) == {'KW1', 'KW2'}, 'class-rate fields')
        require(all(type(v) in (int, float) and math.isfinite(v) for v in value.values()), 'numeric class rates')
    if key in ('target_tokens', 'decode_tokens', 'raw_argmax_tokens', 'checkpoints'):
        require(type(value) is list and all(type(v) is int for v in value), 'integer array: ' + key)
    if isinstance(value, dict):
        for child_key, child in value.items():
            numeric_types(child, child_key)
    elif isinstance(value, list):
        for child in value:
            numeric_types(child)

def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    folder = root / 'observations/2026-10-15-pair-gate-failed'
    manifest = json.loads((folder / 'archive-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and '\\' not in name, 'unsafe manifest name')
        require(digest((folder / name).read_bytes()) == expected, 'archive file digest: ' + name)
    closure = json.loads((root / 'closure.json').read_text())
    numeric_types(closure)
    enabled = (folder / 'enabled-plan.json').read_bytes()
    current = (root / 'plan.json').read_bytes()
    require(enabled.replace(b'"execution_enabled": true', b'"execution_enabled": false') == current, 'plan controls changed')
    require(digest(enabled) == closure['enabled_plan_sha256'], 'enabled plan digest')
    require(digest(current) == closure['closed_plan_sha256'], 'closed plan digest')
    require(closure['consumed'] is True and closure['retry_authorized'] is False, 'campaign closure')
    require(closure['unused_ab_optimizer_calls'] == 1200 and closure['unused_budget_reusable'] is False, 'unused budget')
    hashes = json.loads((folder / 'result-hash-manifest.json').read_text())
    require(digest((folder / 'result-hash-manifest.json').read_bytes()) == closure['result_manifest_sha256'], 'result manifest digest')
    expected = {'claim.json', 'binding.json', 'terminal.json'}
    expected |= {f'{kind}-{i:04}.json' for kind in ('call', 'step') for i in range(1, 301)}
    expected |= {f'pair-{i:04}.json' for i in (0, 50, 100, 200, 300)}
    require(set(hashes['result_files']) == expected, 'result inventory')
    with zipfile.ZipFile(folder / 'raw-fit-result.zip') as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == expected, 'zip inventory')
        records = {}
        for name in sorted(expected):
            data = archive.read(name)
            require(digest(data) == hashes['result_files'][name], 'raw digest: ' + name)
            records[name] = json.loads(data)
    for record in records.values():
        numeric_types(record)
        for flag in FLAGS:
            if flag in record:
                require(record[flag] is False, 'qualification flag')
    require(records['binding.json']['plan_sha256'] == digest(enabled), 'binding plan')
    require(records['terminal.json']['status'] == 'pair_gate_failed', 'terminal status')
    require(records['terminal.json']['later_stages_skipped'] == ['A', 'B'], 'A/B skipped')
    require(records['terminal.json']['optimizer_calls_reserved'] == 300, 'reserved count')
    for i in range(1, 301):
        call, step = (records[f'{kind}-{i:04}.json'] for kind in ('call', 'step'))
        require(call == dict(optimizer_call_reserved=i, stage='pair', update=i), 'call sequence')
        require(step['stage'] == 'pair' and step['update'] == i, 'step sequence')
        for key in ('finite_grad_norm', 'finite_loss'):
            require(type(step[key]) in (int, float) and math.isfinite(step[key]) and step[key] >= 0, 'finite nonnegative step statistic')
        require(step['updated_weights_finite'] is True and step['weights_changed'] is True, 'step flags')
    curve = []
    for i in (0, 50, 100, 200, 300):
        record = records[f'pair-{i:04}.json']
        require(record['stage'] == 'pair' and record['update'] == record['optimizer_calls_reserved'] == i, 'checkpoint identity')
        require(len(record['samples']) == 2, 'pair size')
        rates = {}
        for sample in record['samples']:
            raw = sample['raw_argmax_tokens']
            require(len(raw) == sample['input_frames'] and len(raw) > 0, 'frame count')
            require(all(type(t) is int and 0 <= t < 5 for t in raw), 'token range')
            collapsed = [t for j, t in enumerate(raw) if t != 0 and (j == 0 or raw[j-1] != t)]
            require(collapsed == sample['decode_tokens'], 'CTC collapse')
            target = {'KW1': [1, 2, 3, 4], 'KW2': [3, 4, 3, 4]}[sample['target_class']]
            require(sample['target_tokens'] == target and sample['target_length'] == len(target), 'target')
            require(sample['complete_match'] is (collapsed == target), 'complete match')
            require(sample['blank_argmax_fraction'] == raw.count(0)/len(raw), 'blank argmax fraction')
            nll = sample['ctc_nll']
            require(math.isfinite(nll) and nll >= 0, 'finite NLL')
            require(math.isclose(sample['ctc_nll_per_input_frame'], nll/len(raw), rel_tol=2**-22, abs_tol=1e-12), 'float32 normalized NLL')
            require(math.isfinite(sample['blank_mean_probability']) and 0 <= sample['blank_mean_probability'] <= 1, 'blank probability range')
            require(sample['target_class'] not in rates, 'duplicate class')
            rates[sample['target_class']] = float(collapsed == target)
            curve.append(dict(update=i, **{k:v for k,v in sample.items() if k != 'raw_argmax_tokens'}))
        require(record['per_class_complete_rate'] == rates and record['macro_complete_rate'] == sum(rates.values())/2, 'class/macro rates')
    require(rates == {'KW1': 1.0, 'KW2': 0.0}, 'observed final outcome')
    saved_curve = json.loads((folder / 'learning-curve.json').read_text())
    numeric_types(saved_curve)
    require(curve == saved_curve, 'learning curve projection')
    require(closure['optimizer_calls_reserved'] == closure['successful_optimizer_steps'] == 300, 'closure counts')
    summary = json.loads((folder / 'public-summary.json').read_text())
    numeric_types(summary)
    require(summary['status'] == closure['status'] == 'pair_gate_failed', 'summary terminal')
    require(summary['optimizer_calls_reserved'] == summary['successful_optimizer_steps'] == 300, 'summary counts')
    require(summary['final_class_complete_rates'] == rates and summary['ab_executed'] is False, 'summary outcome')
    require(summary['new_tts_calls'] == summary['new_asr_calls'] == 0 and summary['model_state_exported'] is False, 'no new calls or state export')
    for flag in FLAGS:
        require(summary[flag] is False and closure[flag] is False, 'summary qualification')
    pcm = json.loads((folder / 'pcm-prefix-analysis/summary.json').read_text())
    numeric_types(pcm)
    require(pcm['full_window_report_sha256'] == digest((folder / 'pcm-prefix-analysis/pcm-energy.json').read_bytes()), 'PCM report digest')
    require(pcm['new_model_calls'] == pcm['new_optimizer_steps'] == 0, 'PCM analysis budget')
    require(pcm['samples'][0]['prefix_200ms']['exact_zero_fraction'] == 1.0, 'KW1 exact-zero prefix')
    for sample in records['pair-0300.json']['samples']:
        raw = sample['raw_argmax_tokens']
        transition_frames = [i for i in range(1, len(raw)) if raw[i] != raw[i-1]]
        require(max(transition_frames, default=0) <= 4, 'final greedy transitions end by 105 ms')
    return {'scope': 'archive integrity and reported scalar/token consistency; not process authenticity or NLL recomputation', 'result_files': len(expected), 'checkpoints': 5, 'status': 'pair_gate_failed'}

if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
