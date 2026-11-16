"""One public-main static export receipt. No model inference or local bypass.

The workflow writes STARTED before dependency/download steps, then invokes
export and an always-run finalizer. Tests exercise validate_context as a pure
function; they never impersonate Actions or run these execution stages locally.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = 'jiying2007/kws-pipeline'
BEFORE = '31238cafe3d6668874d446eae5dfa7c55028fb13'
SCHEMA = 'english-fixed-frontend-static-export-once-v1'
FILES = ('README.md', 'export_fixed_frontend.py', 'export_fixed_frontend_once.py',
         'fixed_frontend.c', 'fixed_frontend.h', 'frontend_replay.c',
         'test_fixed_frontend.py', 'test_fixed_frontend_independent.py')
WORKFLOW = '.github/workflows/english-embedding-backbone-assets-once.yml'
EXPECTED_ASSETS = {
    'melspectrogram.onnx': {'bytes': 1087958, 'sha256': 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f'},
    'embedding_model.onnx': {'bytes': 1326578, 'sha256': '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f'},
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def validate_context(env, event, head, porcelain):
    """Pure guard: fixture dictionaries do not initiate any execution."""
    need(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_EVENT_NAME') == 'push'
         and env.get('GITHUB_REPOSITORY') == REPOSITORY
         and env.get('GITHUB_REF') == 'refs/heads/main'
         and env.get('GITHUB_RUN_ATTEMPT') == '1', 'PUBLIC_MAIN_FIRST_ATTEMPT_REQUIRED')
    repo = event.get('repository', {})
    need(repo.get('full_name') == REPOSITORY and repo.get('private') is False
         and event.get('ref') == 'refs/heads/main' and event.get('before') == BEFORE,
         'EXACT_PUBLIC_MAIN_BEFORE_REQUIRED')
    sha = env.get('GITHUB_SHA', '')
    need(re.fullmatch('[0-9a-f]{40}', sha) is not None and sha != BEFORE
         and event.get('after') == sha and head == sha and not porcelain,
         'CLEAN_EXACT_ACTIONS_SOURCE_HEAD_REQUIRED')
    run_id = env.get('GITHUB_RUN_ID', '')
    need(re.fullmatch('[1-9][0-9]*', run_id) is not None, 'ACTUAL_RUN_ID_REQUIRED')
    return {'repository': REPOSITORY, 'before': BEFORE, 'source_head': sha,
            'run_id': run_id, 'run_attempt': 1,
            'run_url': f'https://github.com/{REPOSITORY}/actions/runs/{run_id}'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    need(path.is_file() and not path.is_symlink(), 'REGULAR_FILE_REQUIRED:' + path.name)
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': sha(raw)}


def write_once(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(raw)


def context(args):
    repo = HERE.parents[1]
    env = dict(os.environ)
    need(Path(env.get('GITHUB_WORKSPACE', '/missing')).resolve() == repo,
         'ACTUAL_ACTIONS_WORKSPACE_REQUIRED')
    event = json.loads(Path(env['GITHUB_EVENT_PATH']).read_text())
    git = lambda *a: subprocess.check_output(['git', '-C', str(repo), *a], text=True).strip()
    result = validate_context(env, event, git('rev-parse', 'HEAD'),
                              git('status', '--porcelain', '--untracked-files=all'))
    temp = Path(env['RUNNER_TEMP']).resolve()
    for path in (args.assets, args.output):
        need(path.is_absolute() and path.resolve().is_relative_to(temp)
             and path.resolve() != temp and not path.is_symlink(), 'RUNNER_TEMP_OUTPUT_REQUIRED')
    need(args.assets.resolve() != args.output.resolve()
         and not args.assets.resolve().is_relative_to(args.output.resolve())
         and not args.output.resolve().is_relative_to(args.assets.resolve()), 'SEPARATE_ASSET_AND_RECEIPT_DIRECTORIES')
    code = {str((HERE / name).relative_to(repo)): identity(HERE / name) for name in FILES}
    code[WORKFLOW] = identity(repo / WORKFLOW)
    return {'schema': SCHEMA, **result, 'code_files': code, 'expected_assets': EXPECTED_ASSETS,
            'model_calls': 0, 'onnx_inference_calls': 0, 'c_inference_calls': 0,
            'optimizer_updates': 0, 'tts_asr_phone_calls': 0,
            'numerical_parity_established': False, 'ssc305_qualification': False}


def execute_stage(args):
    receipt = context(args)
    output = args.output
    if args.stage == 'start':
        need(not output.exists(), 'FRESH_STATIC_EXPORT_RECEIPT_REQUIRED')
        output.mkdir(parents=True)
        write_once(output / 'STARTED.json', {**receipt, 'status': 'STARTED_STATIC_EXPORT_ONLY'})
        return 0
    started = json.loads((output / 'STARTED.json').read_text())
    need(all(started.get(k) == v for k, v in receipt.items()), 'UNCHANGED_STARTED_CONTEXT_AND_SOURCE')
    if args.stage == 'finalize':
        if not (output / 'RESULT.json').is_file() and not (output / 'FAILURE.json').is_file():
            write_once(output / 'FAILURE.json', {**receipt, 'status': 'FAILED_NO_RETRY',
                'stage': 'workflow-before-completed-export', 'error_type': 'IncompleteStaticExport',
                'reason': 'Download, dependency installation, or export did not produce a completed result.',
                'workflow_job_status': os.environ.get('STATIC_EXPORT_JOB_STATUS', 'unknown'),
                'export_attempted': (output / 'EXPORT_STARTED.json').exists()})
        return 0
    need(not any((output / name).exists() for name in ('EXPORT_STARTED.json', 'RESULT.json', 'FAILURE.json')),
         'STATIC_EXPORT_ALREADY_CONSUMED')
    write_once(output / 'EXPORT_STARTED.json', {**receipt, 'status': 'EXPORT_STARTED_NO_INFERENCE'})
    stage = 'pinned-parser-versions'
    assets = {}
    try:
        versions = {name: importlib.metadata.version(name) for name in ('numpy', 'onnx')}
        need(versions == {'numpy': '1.26.4', 'onnx': '1.17.0'}, 'PINNED_PARSER_VERSIONS')
        need('onnxruntime' not in sys.modules and 'torch' not in sys.modules, 'NO_INFERENCE_RUNTIME')
        stage = 'pure-local-file-export'
        spec = importlib.util.spec_from_file_location('_once_fixed_frontend_exporter', HERE / 'export_fixed_frontend.py')
        exporter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = exporter
        spec.loader.exec_module(exporter)
        for name, expected in exporter.ASSETS.items():
            row = identity(args.assets / name)
            need(row == {'bytes': expected[0], 'sha256': expected[1]}, 'EXACT_FROZEN_ASSET:' + name)
            assets[name] = row
        report = exporter.export_pair(args.assets, output / 'generated')
        need(report['real_inference_performed'] is False and report['numerical_parity_established'] is False,
             'PURE_EXPORT_REPORT_REQUIRED')
        need('onnxruntime' not in sys.modules and 'torch' not in sys.modules, 'NO_INFERENCE_RUNTIME')
        generated = {p.relative_to(output).as_posix(): identity(p)
                     for p in sorted((output / 'generated').iterdir())}
        write_once(output / 'RESULT.json', {**receipt, 'status': 'EXPORTED_STATIC_GRAPHS_ONLY',
            'export_attempted': True, 'parser_versions': versions, 'assets': assets,
            'generated': generated, 'graphs': {name: {k: graph[k] for k in
                ('asset_sha256', 'asset_bytes', 'input', 'output', 'constant_bytes', 'workspace_bytes')}
                for name, graph in report['graphs'].items()}})
        return 0
    except Exception as error:
        write_once(output / 'FAILURE.json', {**receipt, 'status': 'FAILED_NO_RETRY', 'stage': stage,
            'export_attempted': True, 'assets_verified_before_failure': assets,
            'error_type': type(error).__name__, 'reason': str(error)[:2000]})
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('start', 'export', 'finalize'))
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    return execute_stage(args)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
