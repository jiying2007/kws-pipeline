#!/usr/bin/env python3
"""Freeze and execute eight local Qwen Base candidates under the research queue.

The complete D1 plan is retained, but this entry point can consume only its first
eight cells. Queue claims and the create-only invocation record prevent a second
invocation from silently advancing to the next eight. Outputs are private,
unadmitted synthetic candidates; intended text is not an observed speech label.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_job_lock import heavy_model_slot
import research_runtime
import research_no_network

ROOT = Path(__file__).resolve().parents[1]
BULK = Path('research/auto-tts-voice-clone-20261102')
SOURCE_FILES = tuple(str(BULK / name) for name in
                     ('bulk_pool.py', 'bulk_runner.py', 'runner.py', 'campaign.py')) + tuple(
    'kws_data_contract/' + name for name in
    ('__init__.py', 'identity.py', 'source_registry.py', 'reference_text.py'))
GENERATION = dict(language='Chinese', non_streaming_mode=True, do_sample=True,
                  top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05,
                  subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0,
                  subtalker_temperature=0.9, max_new_tokens=256)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def once_bytes(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def once(path, value):
    once_bytes(path, canonical(value) + b'\n')


def relative(name):
    path = Path(name)
    require(type(name) is str and name and not path.is_absolute() and
            '..' not in path.parts and '\\' not in name, 'UNSAFE_RELATIVE_PATH')
    return path


def record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'EXPECTED_REGULAR_FILE')
    return {'bytes': path.stat().st_size, 'sha256': sha_file(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_import_preflight(raw, setup):
    """Validate a retained import-only receipt and setup binding, without rerunning it."""
    receipt = json.loads(raw)
    require(type(receipt) is dict and receipt.get('schema') == 'kws-runtime-import-preflight-v1'
            and receipt.get('role') == 'tts', 'TTS_IMPORT_PREFLIGHT_SCHEMA')
    require(all(type(receipt.get(key)) is int and receipt[key] == 0
                for key in ('returncode', 'model_objects_constructed', 'model_calls')),
            'TTS_IMPORT_PREFLIGHT_NOT_SUCCESSFUL_IMPORT_ONLY')
    require(type(receipt.get('stdout')) is str and type(receipt.get('stderr')) is str,
            'TTS_IMPORT_PREFLIGHT_STREAMS')
    argv = receipt.get('argv')
    require(type(argv) is list and len(argv) == 5 and all(type(value) is str and value for value in argv)
            and Path(argv[0]).is_absolute() and argv[1:4] == ['-B', '-I', '-c'],
            'TTS_IMPORT_PREFLIGHT_ARGV')
    elapsed = receipt.get('elapsed_seconds')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0
            and type(receipt.get('recorded_at')) is str
            and datetime.fromisoformat(receipt['recorded_at']).utcoffset() is not None,
            'TTS_IMPORT_PREFLIGHT_TIME')
    require(receipt.get('network_environment') == {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
            'TTS_IMPORT_PREFLIGHT_OFFLINE_ENVIRONMENT')
    # The real import writes a dependency warning before its one JSON result.
    results = [json.loads(line) for line in receipt['stdout'].splitlines() if line.lstrip().startswith('{')]
    require(len(results) == 1 and type(results[0]) is dict, 'TTS_IMPORT_PREFLIGHT_ONE_RESULT')
    result = results[0]
    require(set(result) == {'python', 'torch', 'numpy', 'scipy', 'soundfile', 'qwen_class',
                            'cuda_compiled', 'model_objects_constructed', 'model_calls'},
            'TTS_IMPORT_PREFLIGHT_RESULT_FIELDS')
    require(all(type(result.get(key)) is int and result[key] == 0
                for key in ('model_objects_constructed', 'model_calls'))
            and result.get('qwen_class') == 'Qwen3TTSModel' and result.get('cuda_compiled', 'missing') is None,
            'TTS_IMPORT_PREFLIGHT_RESULT')
    require(setup.get('status') == 'complete' and type(setup.get('active_python')) is str
            and result['python'] == setup['active_python'] and type(setup.get('installed_packages')) is dict
            and all(type(result[name]) is str and result[name] == setup['installed_packages'].get(name)
                    for name in ('torch', 'numpy', 'scipy', 'soundfile')),
            'TTS_IMPORT_PREFLIGHT_SETUP_VERSIONS')


def preflight_input(args):
    explicit = getattr(args, 'runtime_preflight', None)
    path = Path(explicit if explicit is not None else args.runtime / 'preflight.json').absolute()
    identity = {'path': str(path), **record(path)}
    raw = path.read_bytes()
    require(len(raw) == identity['bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256'],
            'TTS_IMPORT_PREFLIGHT_INPUT_CHANGED')
    if explicit is not None:
        validate_import_preflight(raw, read(args.runtime / 'setup.json'))
    return raw, identity if explicit is not None else None


def load_bulk(snapshot):
    # Each invocation is a fresh process: resolve the frozen data producer only.
    sys.path[:0] = [str(Path(snapshot) / BULK), str(snapshot)]
    return importlib.import_module('bulk_runner')


def selection(plan):
    require(plan.get('cell_layout') == 'd1-interleaved-32-v1' and
            len(plan['cells']) == 256 and len(plan['references']) == 8, 'D1_PLAN_SHAPE')
    cells = plan['cells'][:8]
    require(len({cell['cell_id'] for cell in cells}) == 8 and
            len({cell['reference_id'] for cell in cells}) == 8 and
            len({cell['source_family_id'] for cell in cells}) == 8 and
            Counter(cell['role'] for cell in cells) ==
            {'train-candidate': 6, 'development-only': 2}, 'SMOKE_SOURCE_COVERAGE')
    return [cell['cell_id'] for cell in cells]


def verify_source_plan(bulk, prepared, plan):
    """Rebuild source, rights, text and split bindings before any model load."""
    bulk.p.verify_plan(plan)
    expected = bulk.p.plan(read(prepared / 'source-manifest.json'), prepared / 'reference-assets',
                           repetitions=plan['repetitions'], batch_id=plan['batch_id'],
                           revision=plan['model_revision'],
                           max_calls=plan['budgets']['max_total_calls'], layout=plan.get('cell_layout'))
    require(bulk.p.canonical(expected) == bulk.p.canonical(plan), 'SOURCE_OR_PLAN_CHANGED')


def prepare(args):
    output = args.output.resolve()
    require(not output.exists(), 'PREPARED_PACKAGE_ALREADY_EXISTS')
    preflight_raw, preflight_source = preflight_input(args)
    output.mkdir(parents=True)
    snapshot = output / 'producer-source'
    source_records = {}
    for name in SOURCE_FILES:
        origin = args.data_root / relative(name)
        once_bytes(snapshot / name, origin.read_bytes())
        source_records[name] = record(snapshot / name)
    bulk = load_bulk(snapshot)
    manifest = bulk.p.read_json(args.sources)
    refs = bulk.p.validate_sources(manifest, args.audio_root)
    once_bytes(output / 'source-manifest.json', args.sources.read_bytes())
    audio_root = output / 'reference-assets'
    for ref in refs:
        for name in (ref['audio_path'], ref['rights_record_path']):
            target = audio_root / relative(name)
            if not target.exists():
                once_bytes(target, (args.audio_root / relative(name)).read_bytes())
    assets = read(ROOT / 'configs/runtime/qwen-base-tts-model-lock.json')
    model_lock = {'schema': bulk.LOCK_SCHEMA, 'model_id': assets['model_id'],
                  'revision': assets['revision'], 'files': [
        {'path': row['filename'], 'bytes': row['bytes'], 'sha256': row['sha256']}
        for row in assets['files']]}
    bulk.asset_lock(args.runtime / 'models/base', model_lock, assets['revision'])
    plan = bulk.p.plan(manifest, audio_root, repetitions=2, batch_id=args.batch_id,
                       revision=assets['revision'], max_calls=256, layout=bulk.p.D1_LAYOUT)
    cells = selection(plan)
    once(output / 'plan.json', plan)
    once(output / 'model-assets.json', model_lock)
    once_bytes(output / 'runtime-setup.json', (args.runtime / 'setup.json').read_bytes())
    once_bytes(output / 'runtime-preflight.json', preflight_raw)
    own_names = ('tools/active_tts_smoke.py', 'tools/research_job_lock.py',
                 'tools/research_no_network.py',
                 'tools/research_runtime.py', 'configs/runtime/research-runtime.json',
                 'research/qwen6_tts/runtime-lock.json', 'research/qwen6_tts/model-locks.json',
                 'configs/runtime/qwen-base-tts-model-lock.json')
    code = {}
    for name in own_names:
        once_bytes(output / 'execution-source' / name, (ROOT / name).read_bytes())
        code[str(ROOT / name)] = record(ROOT / name)
    recipe = {
        'schema': 'active-tts-smoke-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'kind': 'd1-first-eight', 'prepared': str(output), 'runtime': str(args.runtime.resolve()),
        'producer_source': source_records, 'execution_source': code,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'data_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.data_root, text=True).strip(),
        'source_manifest': record(output / 'source-manifest.json'),
        'plan': record(output / 'plan.json'), 'model_assets': record(output / 'model-assets.json'),
        'runtime_setup': record(output / 'runtime-setup.json'),
        'runtime_preflight': record(output / 'runtime-preflight.json'),
        **({'runtime_preflight_source': preflight_source} if preflight_source is not None else {}),
        'selected_cells': cells, 'generation': GENERATION, 'max_calls': 8, 'threads': 4,
        'timeout_seconds': 600, 'max_retries': 0, 'device': 'cpu', 'dtype': 'float32',
        'attention': 'sdpa', 'queue_supervision_required': True,
        'network_policy': 'native-seccomp-deny-before-ML-import-v1',
        'reference_mode': 'content-checked-or-verified-ICL-otherwise-x-vector',
        'raw_retention': 'exact-returned-array-and-float-WAV-before-technical-gates',
        'derived_pcm': 'scipy.resample_poly(2,3);rint(float64*32767);PCM16;no-gain-trim-clipping',
        'actual_spoken_labels_verified': False, 'training_admitted': False,
        'qualification_allowed': False}
    once(output / 'recipe.json', recipe)
    return {'status': 'prepared', 'recipe': str(output / 'recipe.json'),
            'recipe_sha256': sha_file(output / 'recipe.json'), 'calls_allowed': 8}


def verify_recipe(recipe):
    require(recipe['schema'] == 'active-tts-smoke-v1' and recipe['kind'] == 'd1-first-eight' and
            recipe['max_calls'] == 8 and recipe['max_retries'] == 0 and recipe['threads'] == 4 and
            recipe['timeout_seconds'] == 600 and recipe['queue_supervision_required'] is True and
            recipe.get('network_policy') == 'native-seccomp-deny-before-ML-import-v1' and
            recipe['generation'] == GENERATION, 'SMOKE_RECIPE_CHANGED')
    root = Path(recipe['prepared'])
    for field, name in (('source_manifest', 'source-manifest.json'), ('plan', 'plan.json'),
                        ('model_assets', 'model-assets.json'), ('runtime_setup', 'runtime-setup.json'),
                        ('runtime_preflight', 'runtime-preflight.json')):
        require(record(root / name) == recipe[field], 'PREPARED_INPUT_CHANGED:' + name)
    if 'runtime_preflight_source' in recipe:
        origin = recipe['runtime_preflight_source']
        require(type(origin) is dict and set(origin) == {'path', 'bytes', 'sha256'}
                and record(origin['path']) == {key: origin[key] for key in ('bytes', 'sha256')}
                and recipe['runtime_preflight'] == {key: origin[key] for key in ('bytes', 'sha256')},
                'TTS_IMPORT_PREFLIGHT_SOURCE_CHANGED')
        validate_import_preflight((root / 'runtime-preflight.json').read_bytes(), read(root / 'runtime-setup.json'))
    for name, expected in recipe['producer_source'].items():
        require(record(root / 'producer-source' / relative(name)) == expected, 'PRODUCER_SOURCE_CHANGED')
    require(set(recipe['producer_source']) == set(SOURCE_FILES), 'PRODUCER_SOURCE_ROSTER')
    for name, expected in recipe['execution_source'].items():
        require(record(name) == expected, 'EXECUTION_SOURCE_CHANGED')
    require(record(__file__) == recipe['execution_source'][str(Path(__file__).resolve())],
            'ENTRYPOINT_SOURCE_CHANGED')
    plan = read(root / 'plan.json')
    require(selection(plan) == recipe['selected_cells'], 'SMOKE_SELECTION_CHANGED')
    manifest = research_runtime.load_manifest()
    research_runtime.check_interpreter(manifest)
    research_runtime.check_packages(manifest, 'tts')
    expected = read(root / 'runtime-setup.json')
    require(expected['status'] == 'complete', 'TTS_SETUP_INCOMPLETE')
    for name, version in expected['installed_packages'].items():
        require(importlib.metadata.version(name) == version, 'RUNTIME_PACKAGE_CHANGED:' + name)
    for source in read(ROOT / 'research/qwen6_tts/model-locks.json')['qwen_tts']['source_lock']:
        path = importlib.metadata.distribution(source['distribution']).locate_file(source['relative_path'])
        require(sha_file(path) == source['sha256'], 'TTS_API_SOURCE_CHANGED')
    return root, plan


def retain_native(folder, audio, rate):
    """Preserve the actual model return even when the downstream gate rejects it."""
    import numpy as np
    import soundfile as sf
    value = np.asarray(audio)
    require(value.ndim == 1 and value.dtype.kind == 'f', 'NATIVE_ARRAY_SHAPE')
    target = folder / 'returned-array.npy'
    with target.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    wav = folder / 'returned-native.wav'
    with wav.open('xb') as stream:
        sf.write(stream, value, rate, format='WAV', subtype='FLOAT')
        stream.flush()
        os.fsync(stream.fileno())
    result = {'schema': 'active-tts-native-return-v1', 'array': record(target),
              'wav': record(wav), 'dtype': value.dtype.str, 'shape': list(value.shape),
              'sample_rate_hz': rate, 'samples_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
              'all_finite': bool(np.isfinite(value).all()), 'technical_gate_applied': False}
    once(folder / 'native.json', result)
    return result


class BoundedBackend:
    def __init__(self, delegate, output, recipe, native_writer=retain_native):
        self.delegate, self.output, self.recipe = delegate, Path(output), recipe
        self.native_writer = native_writer
        self.calls = 0
        self.seen = set()

    def synthesize(self, cell, ref, audio_root):
        require(cell['cell_id'] in self.recipe['selected_cells'] and
                cell['cell_id'] not in self.seen and self.calls < self.recipe['max_calls'],
                'SMOKE_CALL_BOUNDARY')
        self.seen.add(cell['cell_id'])
        folder = self.output / 'native' / cell['cell_id']
        folder.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        model = self.delegate._model()
        import torch
        torch.manual_seed(cell['seed'])
        try:
            prompt_start = time.monotonic()
            prompt = self.delegate._prompt(ref, audio_root)
            prompt_seconds = time.monotonic() - prompt_start
            checked = ref['reference_text_status'] in ('verified', 'dataset-provided-content-checked')
            invocation = {'schema': 'active-tts-generation-call-v1', 'cell_id': cell['cell_id'],
                          'reference_id': ref['reference_id'], 'source_family_id': ref['source_family_id'],
                          'reference_wav_sha256': ref['wav_sha256'], 'reference_pcm_sha256': ref['pcm_sha256'],
                          'reference_text_status': ref['reference_text_status'],
                          'reference_mode': 'ICL' if checked else 'x-vector-only',
                          'reference_text_passed': checked, 'seed': cell['seed'],
                          'intended_text': cell['prompt'], 'generation': self.recipe['generation'],
                          'reference_prompt_seconds': prompt_seconds, 'attempt': 1}
            once(folder / 'call.json', invocation)
            self.calls += 1
            before = time.monotonic()
            wavs, rate = model.generate_voice_clone(text=cell['prompt'], voice_clone_prompt=prompt,
                                                   **self.recipe['generation'])
            generation_seconds = time.monotonic() - before
            require(len(wavs) == 1, 'ONE_NATIVE_RETURN_PER_CALL')
            self.native_writer(folder, wavs[0], int(rate))
            once(folder / 'returned.json', {'generation_seconds': generation_seconds,
                                          'total_seconds': time.monotonic() - started,
                                          'native_stop_reason': 'not_exposed_by_voice_clone_API'})
            return wavs[0], int(rate)
        except Exception as error:
            once(folder / 'error.json', {'error_type': type(error).__name__, 'error': str(error),
                                         'traceback': traceback.format_exc(),
                                         'seconds': time.monotonic() - started})
            raise


def run(args):
    recipe = read(args.recipe)
    if os.environ.get('KWS_RESEARCH_RECIPE_SHA256'):
        require(sha_file(args.recipe) == os.environ['KWS_RESEARCH_RECIPE_SHA256'], 'QUEUE_RECIPE_BINDING')
    root, plan = verify_recipe(recipe)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    with heavy_model_slot(args.heavy_lock) as lease:
        require(lease.supervised, 'DURABLE_QUEUE_REQUIRED')
        once(output / 'invocation.json', {'recipe_sha256': sha_file(args.recipe),
                                         'source_revision': recipe['source_revision'],
                                         'calls_allowed': 8, 'selected_cells': recipe['selected_cells'],
                                         'heavy_lock': str(lease.path), 'inherited': lease.inherited,
                                         'supervised': lease.supervised, 'pid': os.getpid(),
                                         'process_group': os.getpgrp(), 'started_at': datetime.now(timezone.utc).isoformat()})
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '4'
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          HF_HOME=str(Path(recipe['runtime']) / 'cache'))
        research_no_network.set_offline_environment()
        network = research_no_network.install()
        once(output / 'network-guard.json', network)
        import onnxruntime
        onnxruntime.disable_telemetry_events()
        bulk = load_bulk(root / 'producer-source')
        verify_source_plan(bulk, root, plan)
        bulk.asset_lock(Path(recipe['runtime']) / 'models/base', read(root / 'model-assets.json'),
                        plan['model_revision'])
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        delegate = bulk.QwenBulkClone(Path(recipe['runtime']) / 'models/base', device='cpu')
        loading = time.monotonic()
        model = delegate._model()  # One load: a deterministic load error aborts the job.
        require(model.model.tts_model_type == 'base', 'BASE_MODEL_REQUIRED')
        once(output / 'model-loaded.json', {'seconds': time.monotonic() - loading,
             'model_type': model.model.tts_model_type, 'model_size': model.model.tts_model_size,
             'tokenizer_type': model.model.tokenizer_type,
             'num_threads': torch.get_num_threads(), 'num_interop_threads': torch.get_num_interop_threads(),
             'parameter_dtype': str(next(model.model.parameters()).dtype),
             'parameter_device': str(next(model.model.parameters()).device)})
        backend = BoundedBackend(delegate, output, recipe)
        result = bulk.run(plan, read(root / 'source-manifest.json'), root / 'reference-assets',
                          output / 'generation', backend, max_new_calls=8)
        require(result['total_started'] == len(backend.seen) == 8 and backend.calls <= 8,
                'EIGHT_SELECTED_CELLS_REQUIRED')
        end = resource.getrusage(resource.RUSAGE_SELF)
        files = {path.relative_to(output).as_posix(): record(path)
                 for path in sorted(output.rglob('*')) if path.is_file()}
        once(output / 'inventory.json', {'schema': 'active-tts-output-inventory-v1', 'files': files})
        summary = {'schema': 'active-tts-smoke-result-v1', 'status': 'completed', **result,
                   'recipe_sha256': sha_file(args.recipe), 'source_revision': recipe['source_revision'],
                   'generation_calls': backend.calls, 'prompt_creations': delegate.prompt_creations,
                   'selected_cells': recipe['selected_cells'], 'unstarted_plan_cells': 248,
                   'reference_modes': dict(Counter(read(output / 'native' / cell / 'call.json')['reference_mode']
                       for cell in recipe['selected_cells'] if (output / 'native' / cell / 'call.json').is_file())),
                   'wall_seconds': time.monotonic() - started, 'user_cpu_seconds': end.ru_utime - usage.ru_utime,
                   'system_cpu_seconds': end.ru_stime - usage.ru_stime, 'peak_rss_kib': end.ru_maxrss,
                   'inventory': record(output / 'inventory.json'), 'asr_calls': 0,
                   'human_gold': False, 'quality_selection': False}
        once(output / 'summary.json', summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for field in ('sources', 'audio-root', 'data-root', 'runtime', 'output'):
        prep.add_argument('--' + field, type=Path, required=True)
    prep.add_argument('--runtime-preflight', type=Path,
                      help='Retained successful TTS import-only receipt; defaults to legacy runtime/preflight.json')
    prep.add_argument('--batch-id', required=True)
    execute = sub.add_parser('run')
    for field in ('recipe', 'output', 'heavy-lock'):
        execute.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args) if args.command == 'prepare' else run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
