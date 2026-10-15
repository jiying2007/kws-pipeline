"""Public stdlib archive/receipt/token statistics; no process or model reconstruction."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

FLAGS = ('human_gold', 'independent_evaluation', 'production_qualified', 'reusable_weights_exported', 'shipping_qualified', 'training_admitted')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'finite numeric statistic')
    return value

def exact(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(exact(a[k], b[k]) for k in b)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b

def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    folder = root / 'observations/2026-10-15-completed-scientific-fail'
    archive_manifest = json.loads((folder / 'archive-manifest.json').read_text())
    for name, digest in archive_manifest['files'].items():
        require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and '\\' not in name, 'safe archive path')
        require(sha((folder / name).read_bytes()) == digest, 'archive digest: ' + name)
    closure = json.loads((root / 'closure.json').read_text())
    enabled = (folder / 'enabled-plan.json').read_bytes()
    require(enabled.replace(b'"execution_enabled": true', b'"execution_enabled": false') == (root / 'plan.json').read_bytes(), 'closed plan only flag change')
    require(closure['consumed'] is True and closure['retry_authorized'] is False and closure['remaining_budget'] == 0, 'consumed budget')
    hashes = json.loads((folder / 'result-hash-manifest.json').read_text())
    require(hashes['plan_sha256'] == sha(enabled) == closure['enabled_plan_sha256'], 'executed plan identity')
    require(sha((folder / 'result-hash-manifest.json').read_bytes()) == closure['result_manifest_sha256'], 'result manifest identity')
    expected = {'binding.json', 'intent.json', 'terminal.json', 'C-initialization.json', 'T-initialization.json'}
    expected |= {f'{kind}-{i:04}.json' for kind in ('call', 'step') for i in range(1, 601)}
    expected |= {f'{arm}-{i:04}.json' for arm in ('C', 'T') for i in (0, 50, 100, 200, 300)}
    require(set(hashes['result_files']) == expected, 'complete result inventory')
    with zipfile.ZipFile(folder / 'raw-anchor-result.zip') as z:
        require(len(z.namelist()) == len(expected) and set(z.namelist()) == expected, 'ZIP unique inventory')
        records = {}
        for name in sorted(expected):
            raw = z.read(name)
            require(sha(raw) == hashes['result_files'][name], 'raw file digest')
            records[name] = json.loads(raw)
    for row in records.values():
        for flag in FLAGS:
            if flag in row:
                require(row[flag] is False, 'qualification flags')
    baseline = json.loads((root / 'control-baseline.json').read_text())
    for arm in ('C', 'T'):
        require(records[arm + '-initialization.json']['initial_weight_sha256'] == baseline['initial_weight_sha256'], 'declared initialization')
    for i in range(1, 601):
        arm = 'C' if i <= 300 else 'T'
        update = i if i <= 300 else i - 300
        require(exact(records[f'call-{i:04}.json'], dict(optimizer_call_reserved=i, stage=arm, status='step_reserved', update=update)), 'call sequence')
        step = records[f'step-{i:04}.json']
        require(type(step['update']) is int and step['update'] == update and step['stage'] == arm and step['status'] == 'step_succeeded', 'successful step sequence')
        for key in ('finite_grad_norm', 'finite_loss', 'ctc_loss', 'anchor_loss'):
            require(number(step[key]) >= 0, 'nonnegative step statistic')
        require(step['updated_weights_finite'] is True and step['weights_changed'] is True, 'finite changed weights receipt')
        if arm == 'C':
            for key in ('finite_grad_norm', 'finite_loss'):
                require(exact(step[key], baseline['steps'][str(update)][key]), 'exact control step')
    curve = []
    for arm in ('C', 'T'):
        for update in (0, 50, 100, 200, 300):
            row = records[f'{arm}-{update:04}.json']
            require(exact(row['update'], update) and row['stage'] == arm, 'observation identity')
            require(exact(row['optimizer_calls_reserved'], update + (300 if arm == 'T' else 0)), 'observation call count')
            require(len(row['samples']) == 2, 'pair size')
            rates = {}
            for sample in row['samples']:
                cls = sample['target_class']; prefix = {'KW1': 23, 'KW2': 10}[cls]
                target = {'KW1': [1, 2, 3, 4], 'KW2': [3, 4, 3, 4]}[cls]
                raw = sample['raw_argmax_tokens']; lp = sample['log_probs']
                require(all(type(t) is int and 0 <= t < 5 for t in raw), 'integer raw tokens')
                require(exact(sample['input_frames'], len(raw)) and len(lp) == len(raw), 'frame count')
                for frame in lp:
                    require(len(frame) == 5, 'log probability classes')
                    for value in frame:
                        number(value)
                require(raw == [max(range(5), key=lambda k: f[k]) for f in lp], 'argmax from archived log probabilities')
                decoded = [t for j, t in enumerate(raw) if t and (not j or raw[j-1] != t)]
                require(exact(decoded, sample['decode_tokens']) and exact(target, sample['target_tokens']), 'collapse and target')
                require(sample['complete_match'] is (decoded == target), 'complete match')
                require(exact(sample['prefix_frames'], prefix), 'fixed prefix')
                require(exact(sample['prefix_nonblank_argmax_frames'], sum(t != 0 for t in raw[:prefix])), 'prefix nonblank count')
                require(exact(sample['first_nonblank_frame'], next((j for j, t in enumerate(raw) if t), None)), 'first nonblank')
                require(number(sample['blank_argmax_fraction']) == raw.count(0) / len(raw), 'blank fraction')
                rates[cls] = float(decoded == target)
                curve.append(dict(stage=arm, update=update, **{k:v for k,v in sample.items() if k not in ('log_probs', 'raw_argmax_tokens')}))
            require(exact(row['per_class_complete_rate'], rates) and number(row['macro_complete_rate']) == sum(rates.values())/2, 'class rates')
            if arm == 'C':
                for current, prior in zip(row['samples'], baseline['observations'][str(update)]):
                    require(exact({key: current[key] for key in prior}, prior), 'exact historical control observations')
    require(exact(curve, json.loads((folder / 'learning-curve.json').read_text())), 'learning curve projection')
    terminal = records['terminal.json']
    require(terminal['status'] == 'completed' and terminal['control_reproduced'] is True, 'completed control reproduction')
    require(terminal['treatment_prefix_success'] is True and terminal['treatment_fit_success'] is False and terminal['treatment_joint_success'] is False, 'scientific endpoint failure')
    final = records['T-0300.json']['samples']
    require([s['decode_tokens'] for s in final] == [[2, 3, 4], [3, 4, 3, 4]], 'frozen endpoint decodes')
    require(all(s['prefix_nonblank_argmax_frames'] == 0 for s in final), 'both prefixes blank')
    summary = json.loads((folder / 'public-summary.json').read_text())
    require(summary['scientific_validation'] == 'FAIL' and summary['evidence_validation'] == 'PASS', 'evidence versus science')
    for row in (closure, summary):
        require(exact(row['optimizer_calls_reserved'], 600) and exact(row['successful_optimizer_steps'], 600), 'closed count')
        for flag in FLAGS:
            require(row[flag] is False, 'closed qualification')
    return dict(result_files=1215, observations=10, control_exact=True, scientific_validation='FAIL', scope='archive integrity, receipts, exact control scalars and archived-log-probability argmax/token statistics; not process authenticity or independent model/CTC recomputation')

if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
