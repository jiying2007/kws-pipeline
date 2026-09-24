#!/usr/bin/env python3
"""Select a retained epoch using train/calibration ONLY, fixed C decoder/thresholds.

The existing test split is development feedback, not a new independent holdout.
No test audio or test metrics are evaluated by this selector. Reused predecessor
exposure is reported as counts, never as independent examples or release FAR.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

from frozen_speech_ablation import verify_pool, resolve, sha, write
from preceding_context_evaluation import select_prior, speech_prefix
from startup_context_evaluation import pcm, wav, shifted, collect, measure
from startup_context_training import validate_recipe
from training_state import state_identity
from wake_pressure_balance import keyword_target_sequences
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = 'train-fit-calibration-context-epoch-selection-v1'
ARMS = ('silence-1s', 'prior-nonwake', 'continuous')
KEYWORDS = ('1', '2')


def confined(root: pathlib.Path, value: str) -> pathlib.Path:
    if not isinstance(value, str) or not value or pathlib.PurePosixPath(value).as_posix() != value:
        raise ValueError('invalid retained path')
    path = pathlib.Path(value)
    if path.is_absolute() or '..' in path.parts or any((root / pathlib.Path(*path.parts[:i])).is_symlink()
                                                        for i in range(1, len(path.parts) + 1)):
        raise ValueError('retained paths must be confined and not symlinks')
    result = root / path
    result.resolve().relative_to(root.resolve())
    if not result.is_file():
        raise ValueError('retained file missing')
    return result


def load_milestones(root: pathlib.Path, pool_sha: str, source_rows: list[dict]) -> tuple[dict, list[dict]]:
    index = json.loads((root / 'milestones.json').read_text())
    n = index.get('requested_epochs')
    if (index.get('policy') != 'context-milestone-retention-v1' or index.get('completed') is not True
            or index.get('release_authority') is not False or index.get('pool_sha256') != pool_sha
            or type(n) is not int or not 1 <= n <= 800):
        raise ValueError('incomplete or invalid milestone authority')
    expected = list(range(100, n + 1, 100)) + ([] if n % 100 == 0 else [n])
    records = index.get('records')
    if (index.get('expected_epochs') != expected or not isinstance(records, list)
            or [r.get('epoch') for r in records] != expected
            or any(type(r.get('epoch')) is not int for r in records)):
        raise ValueError('missing, duplicate or unexpected milestone epoch')
    common = None
    required_sources = [{'wav_sha256': r['wav_sha256'], 'target_ids': r['target_ids']}
                        for r in source_rows if r['split'] == 'train']
    for row in records:
        if row.get('directory') != f"milestones/epoch-{row['epoch']:04d}":
            raise ValueError('milestone directory mismatch')
        files = row.get('files')
        if not isinstance(files, dict) or set(files) != {'model.pt', 'model.kwm', 'model.kwm.provenance.json'}:
            raise ValueError('milestone file set mismatch')
        paths = {name: confined(root, row['directory'] + '/' + name) for name in files}
        if any(sha(paths[name]) != files[name] for name in files):
            raise ValueError('milestone bytes changed')
        cp = torch.load(paths['model.pt'], weights_only=True, map_location='cpu')
        recipe = cp['development_recipe']; validate_recipe(recipe)
        if (recipe['scope'] != 'all' or recipe['variant'] != 'grounded-ctc'
                or recipe['epochs'] != n or recipe['seed'] != index['seed']
                or recipe['pool_sha256'] != pool_sha or recipe['source_rows'] != required_sources
                or cp['epochs'] != row['epoch'] or len(cp['epoch_history']) != row['epoch']
                or [e['epoch'] for e in cp['epoch_history']] != list(range(1, row['epoch'] + 1))):
            raise ValueError('checkpoint recipe, source or history mismatch')
        identity = state_identity(cp['state_dict'])
        prov = json.loads(paths['model.kwm.provenance.json'].read_text())
        if (identity != cp['float_state_identity'] or identity['sha256'] != row['float_state_sha256']
                or prov['training']['float_state_identity'] != identity
                or prov['checkpoint']['sha256'] != files['model.pt']
                or prov['model']['sha256'] != files['model.kwm']
                or prov['training']['development_recipe'] != recipe):
            raise ValueError('checkpoint/export readback mismatch')
        binding = {'recipe': recipe, 'initial': cp['initial_float_state_identity'],
                   'corpus': cp['training_corpus_identity']}
        if common is not None and binding != common:
            raise ValueError('milestones do not belong to the same training trajectory')
        common = binding
    return index, records


def score_view(pool_root: pathlib.Path, rows: list[dict], model: pathlib.Path,
               runner: pathlib.Path, pack: pathlib.Path, output: pathlib.Path) -> dict:
    """Only receives a prefiltered train/calibration view; no test fallback."""
    if not rows or any(r['split'] not in ('train', 'calibration') for r in rows):
        raise ValueError('selection view must contain train/calibration only')
    words = keyword_target_sequences(pool_root/'tokens.example.txt', pool_root/'zh_cn_example.tsv')
    result = {}
    output.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'calibration'):
        selected = [r for r in rows if r['split'] == split]
        if not selected or not any(not r['expected'] for r in selected):
            raise ValueError('selection split lacks negative support')
        raw_rows = [pcm(resolve(pool_root, r['path'])) for r in selected]
        result[split] = {}
        for arm in ARMS:
            refs, dets = [], []
            if arm == 'continuous':
                stream = bytearray(b'\0\0' * 32000); events = []
                for r, raw in zip(selected, raw_rows):
                    events.extend(shifted(r['expected'], len(stream)//2)); stream.extend(raw)
                    stream.extend(b'\0\0' * ((-len(stream)//2) % 320 + 32000))
                ref = {'recording': split+'-continuous', 'duration_s': len(stream)/32000, 'expected': events}
                audio = output/'scratch.wav'; wav(audio, bytes(stream))
                refs.append(ref); dets.extend(collect(runner, model, pack, audio, ref['recording']))
            else:
                for i, (r, raw) in enumerate(zip(selected, raw_rows)):
                    before = b'\0\0' * 16000
                    if arm == 'prior-nonwake':
                        prior = select_prior(rows, r, i, words)
                        before = speech_prefix(pcm(resolve(pool_root, prior['path'])))
                    ref = {'recording': f'{split}-{i}', 'duration_s': (len(before)+len(raw))/32000,
                           'expected': shifted(r['expected'], len(before)//2)}
                    audio = output/'scratch.wav'; wav(audio, before+raw)
                    refs.append(ref); dets.extend(collect(runner, model, pack, audio, ref['recording']))
            result[split][arm] = measure(refs, dets)
            write(output/f'{split}-{arm}.json', {'references': refs, 'detections': dets,
                                               'metrics': result[split][arm]})
    (output/'scratch.wav').unlink()
    return result


def validate_metrics(row: dict) -> None:
    if not isinstance(row, dict):
        raise ValueError('missing event metrics')
    for k in ('expected', 'matched', 'false_rejects', 'false_accepts'):
        if type(row.get(k)) is not int or row[k] < 0:
            raise ValueError('event counts must be nonnegative integers')
    if row['expected'] == 0 or row['matched'] + row['false_rejects'] != row['expected']:
        raise ValueError('event positive support/count mismatch')
    per = row.get('per_keyword')
    if not isinstance(per, dict) or set(per) != set(KEYWORDS):
        raise ValueError('both keywords require explicit support')
    for k in KEYWORDS:
        r = per[k]
        if (any(type(r.get(c)) is not int or r[c] < 0 for c in ('expected','matched','false_rejects'))
                or not r['expected'] or r['matched'] + r['false_rejects'] != r['expected']):
            raise ValueError('invalid per-keyword counts')
    if any(sum(per[k][c] for k in KEYWORDS) != row[c] for c in ('expected','matched','false_rejects')):
        raise ValueError('aggregate/per-keyword counts disagree')


def eligibility(metrics: dict) -> tuple[bool, str]:
    if set(metrics) != {'train', 'calibration'}:
        raise ValueError('selection metrics must be train/calibration only')
    for split in metrics:
        if set(metrics[split]) != set(ARMS):
            raise ValueError('selection arms missing')
        for r in metrics[split].values(): validate_metrics(r)
    if any(r['matched'] != r['expected'] or r['false_accepts'] for r in metrics['train'].values()):
        return False, 'train-fit-or-incomplete-phrase-failed'
    if any(4*r['per_keyword'][k]['matched'] < 3*r['per_keyword'][k]['expected']
           for r in metrics['calibration'].values() for k in KEYWORDS):
        return False, 'calibration-per-keyword-recall-below-75-percent'
    return True, 'development-candidate-not-release'


def rank(row: dict) -> tuple:
    if type(row.get('epoch')) is not int or row['epoch'] <= 0:
        raise ValueError('invalid candidate epoch')
    ok, _ = eligibility(row['metrics'])
    if not ok: raise ValueError('ineligible candidate cannot be ranked')
    cal = list(row['metrics']['calibration'].values())
    return (max(r['false_accepts'] for r in cal), sum(r['false_accepts'] for r in cal),
            max(r['per_keyword'][k]['false_rejects']/r['per_keyword'][k]['expected'] for r in cal for k in KEYWORDS),
            sum(r['false_rejects'] for r in cal), row['epoch'])


def select(rows: list[dict]) -> dict | None:
    if not rows or len({r['epoch'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate candidate epochs')
    candidates = [r for r in rows if eligibility(r['metrics'])[0]]
    return min(candidates, key=rank) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pool','trial','runner','output'): parser.add_argument('--'+name, type=pathlib.Path, required=True)
    parser.add_argument('--pool-sha', required=True); args = parser.parse_args()
    pool_root, trial, runner, out = (p.resolve() for p in (args.pool,args.trial,args.runner,args.output))
    pool = verify_pool(pool_root, args.pool_sha)
    view = [r for r in pool['rows'] if r['split'] in ('train', 'calibration')]
    index_sha = sha(trial/'milestones.json'); index, retained = load_milestones(trial, args.pool_sha, view)
    out.mkdir(parents=True, exist_ok=False); pack = out/'keywords.kwk'
    subprocess.run([sys.executable,str(ROOT/'tools/compile_keywords.py'),'--tokens',str(pool_root/'tokens.example.txt'),
                    '--keywords',str(pool_root/'zh_cn_example.tsv'),'--out-pack',str(pack)],check=True,timeout=30,
                   stdout=subprocess.DEVNULL)
    runner_sha, pack_sha = sha(runner), sha(pack)
    report = {'policy': POLICY, 'completed': False, 'development_only': True, 'release_authority': False,
        'selected': None, 'pool_sha256': args.pool_sha, 'milestones_sha256': index_sha,
        'runner_sha256': runner_sha, 'pack_sha256': pack_sha, 'selection_splits': ['train','calibration'],
        'test_used_for_selection': False, 'minimum_calibration_recall_per_keyword': 0.75,
        'rank_order': ['max-calibration-FA','sum-calibration-FA','worst-keyword-FRR','sum-FR','earlier-epoch'],
        'implementation_sha256': {str(path.relative_to(ROOT)):sha(path) for path in (
            pathlib.Path(__file__), ROOT/'training/startup_context_evaluation.py',
            ROOT/'training/preceding_context_evaluation.py', ROOT/'eval/score_events.py')},
        'records': [], 'limitations': ['Reused synthetic calibration is development feedback, not release evidence.',
            'FA counts include repeated prior clips; not independent exposure or FAR/hour.',
            'Snapshot retention does not resume optimizer/RNG state.']}
    write(out/'selection.json', report)
    for record in retained:
        metrics = score_view(pool_root, view, trial/record['directory']/'model.kwm',runner,pack,
                             out/f"epoch-{record['epoch']:04d}")
        ok, reason = eligibility(metrics)
        report['records'].append({**record, 'metrics': metrics, 'eligible': ok, 'reason': reason})
        write(out/'selection.json', report)
        print(json.dumps({'epoch':record['epoch'],'eligible':ok,'reason':reason,
            'calibration':{a:[r['matched'],r['expected'],r['false_accepts']] for a,r in metrics['calibration'].items()}}),flush=True)
    chosen = select(report['records'])
    if sha(trial/'milestones.json') != index_sha or sha(runner) != runner_sha or sha(pack) != pack_sha:
        raise ValueError('selection inputs mutated')
    load_milestones(trial, args.pool_sha, view); verify_pool(pool_root,args.pool_sha)
    report['selected'] = {k:chosen[k] for k in ('epoch','directory','files','float_state_sha256')} if chosen else None
    report['selected_rank'] = list(rank(chosen)) if chosen else None
    report['completed'] = True; write(out/'selection.json',report)
    print(json.dumps({'selected_epoch':chosen['epoch'] if chosen else None,'release_authority':False}))


if __name__ == '__main__': main()
