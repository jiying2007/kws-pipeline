#!/usr/bin/env python3
"""Four predeclared ICL/x-vector calls, separate from the D1 candidate roster."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_tts_smoke as common
from research_job_lock import heavy_model_slot


def cases_for(plan, declaration, normalize):
    common.require(declaration['calls'] == 4 and declaration['seed'] == 20261105 and
                   declaration['prompt'] == '你好小窝' and
                   declaration['modes'] == ['ICL', 'x-vector-only'], 'FIXED_PAIR_DESIGN')
    eligible = []
    for ref in plan['references']:
        observations = (ref.get('reference_text_provenance') or {}).get('content_observations', [])
        if (ref['reference_text_status'] == 'dataset-provided-content-checked' and
                len(observations) == 3 and {x['engine'] for x in observations} ==
                {'qwen06', 'sensevoice', 'whisper'} and all(
                    x['input_mode'] == 'audio-only' and x['decoder_completed'] is True and
                    normalize(x['actual_text']) == normalize(ref['reference_text'])
                    for x in observations)):
            eligible.append(ref)
    eligible.sort(key=lambda row: row['reference_id'])
    common.require(len(eligible) >= 2 and [row['reference_id'] for row in eligible[:2]] ==
                   declaration['reference_ids'], 'PREDECLARED_TWO_THREE_ROUTE_MATCHES_REQUIRED')
    cells = []
    for ref in eligible[:2]:
        for mode in declaration['modes']:
            value = {'reference_id': ref['reference_id'], 'reference_mode': mode,
                     'reference_wav_sha256': ref['wav_sha256'],
                     'reference_pcm_sha256': ref['pcm_sha256'],
                     'prompt': declaration['prompt'], 'seed': declaration['seed'],
                     'model_revision': plan['model_revision']}
            cells.append({'case_id': 'pair-' + hashlib.sha256(common.canonical(value)).hexdigest()[:24],
                          **value, 'source_family_id': ref['source_family_id'],
                          'speaker_group_id': ref['speaker_group_id'], 'role': ref['role']})
    return cells


def prepare(args):
    parent_path = args.smoke_prepared.resolve() / 'recipe.json'
    parent = common.read(parent_path)
    package, plan = common.verify_recipe(parent)
    declaration = common.read(args.predeclaration)
    common.require(common.sha_file(parent_path) == declaration['parent_smoke_recipe_sha256'] and
                   common.sha_file(package / 'source-manifest.json') == declaration['source_manifest_sha256'],
                   'PAIR_PARENT_BINDING')
    bulk = common.load_bulk(package / 'producer-source')
    cases = cases_for(plan, declaration, bulk.p.reference_text.normalize)
    output = args.output.resolve()
    common.require(not output.exists(), 'PAIR_PACKAGE_ALREADY_EXISTS')
    output.mkdir(parents=True)
    common.once_bytes(output / 'predeclaration.json', args.predeclaration.read_bytes())
    common.once_bytes(output / 'source.py', Path(__file__).read_bytes())
    recipe = {'schema': 'active-tts-pair-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'prepared': str(output), 'parent_recipe': str(parent_path),
              'parent_recipe_sha256': common.sha_file(parent_path),
              'predeclaration': common.record(output / 'predeclaration.json'),
              'source': common.record(__file__), 'source_path': str(Path(__file__).resolve()),
              'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=common.ROOT, text=True).strip(),
              'cases': cases, 'generation': common.GENERATION, 'max_calls': 4,
              'max_retries': 0, 'timeout_seconds': 600, 'threads': 4,
              'seed_application': 'reset-after-prompt-immediately-before-generate',
              'case_order': 'reference-id-ascending-then-ICL-x-vector',
              'first_generation_before_selection': False, 'training_admitted': False,
              'qualification_allowed': False, 'quality_selection': False}
    common.once(output / 'recipe.json', recipe)
    return {'status': 'prepared', 'recipe': str(output / 'recipe.json'),
            'recipe_sha256': common.sha_file(output / 'recipe.json'), 'calls_allowed': 4}


def run(args):
    recipe = common.read(args.recipe)
    common.require(recipe['schema'] == 'active-tts-pair-v1' and recipe['max_calls'] == 4 and
                   recipe['max_retries'] == 0 and recipe['generation'] == common.GENERATION and
                   recipe['timeout_seconds'] == 600 and recipe['threads'] == 4, 'PAIR_RECIPE_CHANGED')
    common.require(common.record(__file__) == recipe['source'] and
                   common.record(Path(recipe['prepared']) / 'source.py') == recipe['source'] and
                   common.record(Path(recipe['prepared']) / 'predeclaration.json') == recipe['predeclaration'],
                   'PAIR_SOURCE_CHANGED')
    common.require(common.sha_file(recipe['parent_recipe']) == recipe['parent_recipe_sha256'],
                   'PARENT_RECIPE_CHANGED')
    if os.environ.get('KWS_RESEARCH_RECIPE_SHA256'):
        common.require(common.sha_file(args.recipe) == os.environ['KWS_RESEARCH_RECIPE_SHA256'], 'QUEUE_RECIPE_BINDING')
    parent = common.read(recipe['parent_recipe'])
    package, plan = common.verify_recipe(parent)
    bulk = common.load_bulk(package / 'producer-source')
    declaration = common.read(Path(recipe['prepared']) / 'predeclaration.json')
    common.require(cases_for(plan, declaration, bulk.p.reference_text.normalize) == recipe['cases'],
                   'PAIR_SELECTION_CHANGED')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    with heavy_model_slot(args.heavy_lock) as lease:
        common.require(lease.supervised, 'DURABLE_QUEUE_REQUIRED')
        common.once(output / 'invocation.json', {'recipe_sha256': common.sha_file(args.recipe),
            'source_revision': recipe['source_revision'], 'calls_allowed': 4,
            'heavy_lock': str(lease.path), 'inherited': lease.inherited, 'supervised': lease.supervised,
            'pid': os.getpid(), 'process_group': os.getpgrp(), 'started_at': datetime.now(timezone.utc).isoformat()})
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '4'
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(Path(parent['runtime']) / 'cache'))
        common.research_no_network.set_offline_environment()
        network = common.research_no_network.install()
        common.once(output / 'network-guard.json', network)
        import onnxruntime
        onnxruntime.disable_telemetry_events()
        common.verify_source_plan(bulk, package, plan)
        bulk.asset_lock(Path(parent['runtime']) / 'models/base', common.read(package / 'model-assets.json'),
                        plan['model_revision'])
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        delegate = bulk.QwenBulkClone(Path(parent['runtime']) / 'models/base', device='cpu')
        before = time.monotonic()
        model = delegate._model()
        common.once(output / 'model-loaded.json', {'seconds': time.monotonic() - before,
             'parameter_dtype': str(next(model.model.parameters()).dtype),
             'parameter_device': str(next(model.model.parameters()).device),
             'num_threads': torch.get_num_threads(), 'num_interop_threads': torch.get_num_interop_threads()})
        refs = {row['reference_id']: row for row in plan['references']}
        calls, outcomes = 0, []
        for case in recipe['cases']:
            folder = output / 'cells' / case['case_id']
            folder.mkdir(parents=True, exist_ok=False)
            common.once(folder / 'started.json', {**case, 'attempt': 1, 'recipe_sha256': common.sha_file(args.recipe)})
            ref = refs[case['reference_id']]
            before = time.monotonic()
            try:
                if case['reference_mode'] == 'ICL':
                    prompt = delegate._prompt(ref, package / 'reference-assets')
                else:
                    clip, rate = delegate._reference(ref, package / 'reference-assets')
                    prompt = model.create_voice_clone_prompt(ref_audio=(clip, rate), ref_text=None,
                                                            x_vector_only_mode=True)
                    delegate.prompt_creations += 1
                prompt_seconds = time.monotonic() - before
                torch.manual_seed(case['seed'])
                common.once(folder / 'call.json', {**case, 'generation': recipe['generation'],
                    'reference_wav_sha256': ref['wav_sha256'], 'reference_pcm_sha256': ref['pcm_sha256'],
                    'reference_text_status': ref['reference_text_status'],
                    'reference_text_passed': case['reference_mode'] == 'ICL',
                    'seed_application': recipe['seed_application'], 'prompt_seconds': prompt_seconds})
                calls += 1
                before = time.monotonic()
                wavs, rate = model.generate_voice_clone(text=case['prompt'], voice_clone_prompt=prompt,
                                                        **recipe['generation'])
                generation_seconds = time.monotonic() - before
                common.require(len(wavs) == 1, 'ONE_NATIVE_RETURN_PER_CALL')
                native = common.retain_native(folder, wavs[0], int(rate))
                common.once(folder / 'returned.json', {'generation_seconds': generation_seconds,
                                                       'native_stop_reason': 'not_exposed_by_voice_clone_API'})
                metrics = bulk.media.quality(wavs[0], int(rate))
                derived = bulk.media._write_waves(folder, wavs[0], int(rate), metrics)
                outcome = {'case_id': case['case_id'], 'status': 'GENERATED_PENDING_ASR',
                           'native': native, 'audio': derived, 'technical': metrics,
                           'training_admitted': False, 'qualification_allowed': False}
                common.once(folder / 'completed.json', outcome)
            except Exception as error:
                outcome = {'case_id': case['case_id'], 'status': 'FAILED_NO_RETRY',
                           'error_type': type(error).__name__, 'error': str(error),
                           'traceback': traceback.format_exc(), 'training_admitted': False}
                common.once(folder / 'failed.json', outcome)
            outcomes.append(outcome)
        common.require(calls <= 4 and len(outcomes) == 4, 'PAIR_CALL_LIMIT')
        end = resource.getrusage(resource.RUSAGE_SELF)
        common.once(output / 'inventory.json', {'schema': 'active-tts-output-inventory-v1',
            'files': {path.relative_to(output).as_posix(): common.record(path)
                      for path in sorted(output.rglob('*')) if path.is_file()}})
        summary = {'schema': 'active-tts-pair-result-v1', 'status': 'paired-smoke-unadmitted',
            'recipe_sha256': common.sha_file(args.recipe), 'source_revision': recipe['source_revision'],
            'generation_calls': calls, 'prompt_creations': delegate.prompt_creations,
            'cases': recipe['cases'], 'outcome_counts': dict(Counter(x['status'] for x in outcomes)),
            'wall_seconds': time.monotonic() - start, 'user_cpu_seconds': end.ru_utime - usage.ru_utime,
            'system_cpu_seconds': end.ru_stime - usage.ru_stime, 'peak_rss_kib': end.ru_maxrss,
            'inventory': common.record(output / 'inventory.json'), 'asr_calls': 0,
            'quality_selection': False, 'training_admitted': False, 'qualification_allowed': False}
        common.once(output / 'summary.json', summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('smoke-prepared', 'predeclaration', 'output'):
        prep.add_argument('--' + name, type=Path, required=True)
    execute = sub.add_parser('run')
    for name in ('recipe', 'output', 'heavy-lock'):
        execute.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args) if args.command == 'prepare' else run(args), sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
