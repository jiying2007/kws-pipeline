"""Stdlib archive/receipt/token statistics, not process or model reconstruction."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

FLAGS = ('human_gold', 'independent_evaluation', 'production_qualified', 'reusable_weights_exported', 'shipping_qualified', 'training_admitted')
CHECKPOINTS = (0, 50, 100, 200, 300, 400, 500, 600)
STEP_FIELDS = ('samples', 'ctc_loss', 'anchor_loss', 'blank_lambda', 'update', 'finite_loss', 'finite_grad_norm')


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


PUBLIC_SCOPE = 'two exposed machine-labeled training samples only'
PUBLIC_VERIFICATION_SCOPE = 'archive integrity, receipt and saved-token statistics only; not process authenticity or independent model/CTC recomputation'
PUBLIC_LIMITATIONS = [
    'No human-gold or independent-evaluation claim.',
    'No true recall, false-activation-rate or generalization claim.',
    'No production, shipping or reusable-weights qualification.',
    'R600 is the sole primary endpoint; R400/R500 are descriptive only.',
    'No process/runtime authenticity or independent model/CTC recomputation claim from the public archive verifier.',
]


def verify_consumed(closure):
    require(closure['consumed'] is True and closure['execution_enabled'] is False and closure['retry_authorized'] is False and closure['budget_reusable'] is False, 'consumed budget flags')
    require(exact(closure['remaining_budget'], 0) and exact(closure['remaining_model_forward_budget'], 0), 'consumed typed budgets')
    require(closure['execution_status'] == 'completed', 'completed closure status')


def verify_public_summary(summary):
    require(exact(summary['scientific_scope'], PUBLIC_SCOPE), 'bounded scientific scope')
    require(exact(summary['public_verification_scope'], PUBLIC_VERIFICATION_SCOPE), 'bounded public verifier scope')
    require(exact(summary['limitations'], PUBLIC_LIMITATIONS), 'complete bounded limitations')
    require(summary['execution_status'] == 'completed' and summary['evidence_validation'] == 'PASS' and summary['scientific_validation'] == 'PASS', 'public status')
    for key, count in (('optimizer_calls_reserved', 600), ('successful_optimizer_steps', 600), ('model_forward_calls_reserved', 608), ('model_forward_calls_returned', 608), ('observations', 8), ('primary_endpoint_step', 600), ('new_tts_calls', 0), ('new_asr_calls', 0)):
        require(exact(summary[key], count), 'public typed count: ' + key)
    for flag in FLAGS:
        require(summary[flag] is False, 'public qualification: ' + flag)
    for key in ('historical_first300_exact_match', 'treatment_prefix_success', 'treatment_fit_success', 'treatment_joint_success'):
        require(summary[key] is True, 'public endpoint statement')


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    folder = root / 'observations/2026-10-15-completed-scientific-pass'
    manifest = strict_json((folder / 'archive-manifest.json').read_bytes())
    inventory = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name != 'archive-manifest.json'}
    require(set(manifest['files']) == inventory, 'complete archive inventory')
    for name, digest in manifest['files'].items():
        require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and '\\' not in name, 'safe archive path')
        require(sha((folder / name).read_bytes()) == digest, 'archive digest: ' + name)
    closure = strict_json((root / 'closure.json').read_bytes())
    enabled = (folder / 'enabled-plan.json').read_bytes()
    closed = (root / 'plan.json').read_bytes()
    plan = strict_json(enabled)
    require(plan['execution_enabled'] is True, 'executed enabled plan')
    require(enabled.count(b'"execution_enabled": true') == 1 and enabled.replace(b'"execution_enabled": true', b'"execution_enabled": false') == closed, 'closed plan only flag change')
    verify_consumed(closure)
    require(sha(closed) == closure['closed_plan_sha256'], 'closed plan identity')
    executed = strict_json((folder / 'executed-source-manifest.json').read_bytes())
    require(executed['files']['plan.json'] == sha(enabled), 'executed source plan')
    for name in ('historical-baseline.json', 'contract.py', 'run_horizon.py', 'numeric_preflight.py', 'requirements.research.lock', 'runtime-candidate.lock.json', 'prefix-energy-summary.json', 'prefix-energy-windows.json'):
        require(executed['files'][name] == sha((root / name).read_bytes()), 'unchanged executed source: ' + name)
    hashes = strict_json((folder / 'result-hash-manifest.json').read_bytes())
    require(hashes['plan_sha256'] == sha(enabled) == closure['enabled_plan_sha256'], 'executed plan identity')
    require(sha((folder / 'result-hash-manifest.json').read_bytes()) == closure['result_manifest_sha256'], 'result manifest identity')
    expected = {'binding.json', 'intent.json', 'terminal.json', 'R-initialization.json'}
    expected |= {f'{kind}-{i:04}.json' for kind in ('call', 'step') for i in range(1, 601)}
    expected |= {f'{kind}-{i:04}.json' for kind in ('forward', 'forward-return') for i in range(1, 609)}
    expected |= {f'R-{i:04}.json' for i in CHECKPOINTS}
    require(set(hashes['result_files']) == expected, 'complete result inventory')
    with zipfile.ZipFile(folder / 'raw-horizon-result.zip') as z:
        require(len(z.namelist()) == len(expected) and set(z.namelist()) == expected, 'ZIP unique inventory')
        records = {}
        for name in sorted(expected):
            raw = z.read(name)
            require(sha(raw) == hashes['result_files'][name], 'raw file digest: ' + name)
            records[name] = strict_json(raw)
    for row in records.values():
        for flag in FLAGS:
            if flag in row:
                require(row[flag] is False, 'qualification flags')
    baseline = strict_json((root / 'historical-baseline.json').read_bytes())
    require(sha((root / 'historical-baseline.json').read_bytes()) == plan['control_baseline_sha256'], 'historical baseline identity')
    require(records['R-initialization.json']['initial_weight_sha256'] == baseline['initial_weight_sha256'], 'declared initialization')
    require(records['binding.json']['plan_sha256'] == sha(enabled), 'binding plan identity')
    schedule = [('observation', 0)]
    for i in range(1, 601):
        schedule.append(('training', i))
        if i in CHECKPOINTS:
            schedule.append(('observation', i))
        require(exact(records[f'call-{i:04}.json'], dict(optimizer_call_reserved=i, stage='R', status='step_reserved', update=i)), 'optimizer reservation sequence')
        step = records[f'step-{i:04}.json']
        require(exact(step['update'], i) and step['stage'] == 'R' and step['status'] == 'step_succeeded', 'successful step sequence')
        for key in ('finite_grad_norm', 'finite_loss', 'ctc_loss', 'anchor_loss'):
            require(number(step[key]) >= 0, 'nonnegative step statistic')
        require(step['updated_weights_finite'] is True and step['weights_changed'] is True, 'finite changed weights receipt')
        if i <= 300:
            require(exact({k: step[k] for k in STEP_FIELDS}, {k: baseline['steps'][str(i)][k] for k in STEP_FIELDS}), 'exact historical T step components')
    require(len(schedule) == 608, 'forward budget')
    for i, (kind, update) in enumerate(schedule, 1):
        for prefix, status in (('forward', 'forward_reserved'), ('forward-return', 'forward_returned')):
            require(exact(records[f'{prefix}-{i:04}.json'], dict(kind=kind, model_forward_reserved=i, stage='R', status=status, update=update)), 'forward reservation/return sequence')
    curve = []
    stats = {}
    for update in CHECKPOINTS:
        name = f'R-{update:04}.json'
        row = records[name]
        require(exact(row['update'], update) and row['stage'] == 'R', 'observation identity')
        require(exact(row['optimizer_calls_reserved'], update), 'observation call count')
        require(len(row['samples']) == 2 and {s['target_class'] for s in row['samples']} == {'KW1', 'KW2'}, 'unique pair classes')
        rates = {}; stats[name] = {}
        for sample in row['samples']:
            cls = sample['target_class']; prefix = {'KW1': 23, 'KW2': 10}[cls]
            target = {'KW1': [1, 2, 3, 4], 'KW2': [3, 4, 3, 4]}[cls]
            raw = sample['raw_argmax_tokens']; lp = sample['log_probs']
            require(all(type(t) is int and 0 <= t < 5 for t in raw), 'integer raw tokens')
            require(exact(sample['input_frames'], len(raw)) and len(lp) == len(raw) == {'KW1': 71, 'KW2': 83}[cls], 'full frame count')
            for frame in lp:
                require(len(frame) == 5, 'log probability classes')
                for value in frame:
                    number(value)
            require(raw == [max(range(5), key=lambda k: f[k]) for f in lp], 'argmax from archived log probabilities')
            decoded = [t for j, t in enumerate(raw) if t and (not j or raw[j-1] != t)]
            require(exact(decoded, sample['decode_tokens']) and exact(target, sample['target_tokens']), 'collapse and target')
            match = decoded == target; count = sum(t != 0 for t in raw[:prefix])
            require(sample['complete_match'] is match, 'complete match')
            require(exact(sample['prefix_frames'], prefix) and exact(sample['prefix_nonblank_argmax_frames'], count), 'fixed prefix count')
            require(exact(sample['first_nonblank_frame'], next((j for j, t in enumerate(raw) if t), None)), 'first nonblank')
            require(number(sample['blank_argmax_fraction']) == raw.count(0) / len(raw), 'blank fraction')
            rates[cls] = float(match)
            stats[name][cls] = dict(complete_match=match, decode_tokens=decoded, joint_success=match and count == 0, prefix_nonblank_argmax_frames=count)
            curve.append(dict(stage='R', update=update, **{k:v for k,v in sample.items() if k not in ('log_probs', 'raw_argmax_tokens')}))
        require(exact(row['per_class_complete_rate'], rates) and number(row['macro_complete_rate']) == sum(rates.values())/2, 'class rates')
        if update <= 300:
            require(exact(row['samples'], baseline['observations'][str(update)]), 'exact complete historical T observations')
    require(exact(curve, strict_json((folder / 'learning-curve.json').read_bytes())), 'learning curve projection')
    terminal = records['terminal.json']
    require(terminal['status'] == 'completed' and terminal['historical_first300_reproduced'] is True, 'completed historical reproduction')
    require(all(terminal[k] is True for k in ('treatment_prefix_success', 'treatment_fit_success', 'treatment_joint_success')), 'primary endpoint success')
    for update in (400, 500, 600):
        require(all(s['joint_success'] is True for s in stats[f'R-{update:04}.json'].values()), 'descriptive checkpoint pair success')
    summary = strict_json((folder / 'public-summary.json').read_bytes())
    verify_public_summary(summary)
    acceptance = strict_json((folder / 'acceptance-summary.json').read_bytes())
    require(exact(acceptance['statistics'], stats), 'original acceptance token statistics')
    for row in (closure, summary, acceptance):
        require(row['scientific_validation'] == 'PASS' and row['evidence_validation'] == 'PASS', 'evidence versus science')
        require(exact(row['optimizer_calls_reserved'], 600) and exact(row['model_forward_calls_reserved'], 608), 'closed reserved counts')
        require(exact(row['primary_endpoint_step'], 600), 'only primary endpoint')
        for flag in FLAGS:
            require(row[flag] is False, 'closed qualification')
    for row in (closure, summary):
        require(exact(row['successful_optimizer_steps'], 600) and exact(row['model_forward_calls_returned'], 608) and exact(row['observations'], 8), 'closed successful counts')
    require(exact(acceptance['successful_step_receipts'], 600) and exact(acceptance['observations_verified'], 8) and exact(acceptance['model_forward_calls_returned'], 608), 'acceptance counts')
    require(exact(terminal['optimizer_calls_reserved'], 600) and exact(terminal['model_forward_calls_reserved'], 608) and exact(terminal['primary_endpoint_step'], 600), 'terminal endpoint counts')
    require(summary['historical_first300_exact_match'] is True and acceptance['historical_exact_match'] is True, 'history summary')
    final = records['R-0600.json']['samples']
    require(exact(summary['final_decodes'], {s['target_class']: s['decode_tokens'] for s in final}), 'summary decodes')
    require(exact(summary['final_prefix_nonblank_frames'], {s['target_class']: s['prefix_nonblank_argmax_frames'] for s in final}), 'summary prefixes')
    require(exact(summary['descriptive_pair_success_checkpoints'], [400, 500, 600]), 'descriptive checkpoint declaration')
    return dict(result_files=2428, observations=8, optimizer_calls=600, model_forward_calls=608, historical_first300_exact=True, scientific_validation='PASS', primary_endpoint_step=600, scope='archive integrity, ordered receipts, exact historical statistics and saved-log-probability argmax/token statistics; not process authenticity or independent model/CTC recomputation')


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
