#!/usr/bin/env python3
"""Current-deployment regression gate. No model execution or historical fallback.

The Actions API authenticates run/artifact origin; this verifier recomputes the
saved evidence. Receipts alone never authenticate a run or qualify a product.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT / 'tools')]
from aggregate_far import validate_shard
from statistical_bounds import poisson_rate_upper

SCHEMA = 'current-tuple-far-regression-v1'
WORKFLOW = '.github/workflows/far-nightly.yml'
SEEDS = (1103, 2207, 3301, 4409)
# Original rendered WAV bytes, not tokens or a guessed causal acoustic label.
KNOWN_FA_SOURCES = {
    '40a5ab93a7fad21dd3c234e659a092e890e8269eeeb0d69f04415124a53d6966',
    'ca75c2442d163e01fcedef656853116fd571a001241c7a6260bf38beb051f78c',
}
KNOWN_CONFIG = 'a515749b2987e1df23f588a1db8f1e9a7f710662ed6f8b4df92c09aacf40f5f6'
SOURCE_ROOTS = ('CMakeLists.txt', 'cmake', 'src', 'include', 'configs/parameter-contract.json',
                'tools/gen_parameter_limits.py', 'tools/kws_raw_stream.c', 'tools/tool_io.c',
                'tools/tool_io.h', 'eval/long_far_stream.py', 'training/synthetic_audio.py',
                'training/render_domains.py', 'training/acoustic_scene.py', 'training/frontend_spec.py')
CONFIGS = ('shipping.xiaowo.json', 'nightly.xiaowo-frozen-model.json', 'parameter-contract.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def evaluate(root):
    """Recompute the bounded four-shard contract from raw saved files."""
    root = Path(root)
    files = {}
    def bind(name):
        path = root / name
        files[name] = sha(path)
        return path
    source_binding = load(bind('far-regression/source-binding.json'))
    require(source_binding.get('schema') == 'far-runtime-source-binding-v1' and
            re.fullmatch('[0-9a-f]{40}', source_binding.get('source_sha', '')) and
            isinstance(source_binding.get('files'), dict) and source_binding['files'], 'missing runtime source binding')
    configs = {name: sha(bind('far-regression/' + name)) for name in CONFIGS}
    require(configs['nightly.xiaowo-frozen-model.json'] == KNOWN_CONFIG,
            'known failure generator config changed; applicability requires reviewed replacement')
    shipping = load(root / 'far-regression/shipping.xiaowo.json')
    require(configs['parameter-contract.json'] == shipping['parameter_contract']['sha256'], 'active parameter contract mismatch')
    runtime = sha(bind('far-regression/kws_raw_stream'))
    model, pack = shipping['model']['model_sha256'], shipping['model']['keyword_pack_sha256']
    manifest = sha(bind('nightly-negative.tsv'))
    domain = rows(bind('nightly-domain/domain-index.jsonl'))
    domain_hashes = {row['wav_sha256'] for row in domain if row.get('kind') != 'positive'}
    require(KNOWN_FA_SOURCES <= domain_hashes, 'known failure source audio missing; applicability unknown')
    aggregate = load(bind('far-aggregate.json'))
    counts, hours, injections, seconds = 0, 0.0, 0, 0.0
    summary_hashes = set()
    for seed in SEEDS:
        prefix = f'far-run-{seed}/'
        summary_path = bind(prefix + 'summary.json')
        summary_hashes.add(sha(summary_path))
        raw = load(summary_path)
        shard = validate_shard(raw, summary_path)
        require(raw.get('schema_version') == 2 and raw.get('seed') == seed, 'shard identity mismatch')
        require(raw.get('seconds') == 7200 and shard['audio_hours'] == 2.0, 'requires complete 7200-second shards')
        for field, value in [('runner_sha256', runtime), ('model_sha256', model),
                             ('keyword_pack_sha256', pack), ('negative_manifest_sha256', manifest)]:
            require(shard[field] == value, 'shard exact tuple mismatch: ' + field)
        require(raw.get('coverage_hard_negative_gain') == 0.9 and
                raw.get('min_hard_negative_injections_per_clip') == 1 and
                shard['hard_negative_rate_per_minute'] == 2.0, 'stream generation parameters changed')
        for name, field in [('detections.jsonl', 'detections_sha256'),
                            ('hard-negative-injections.jsonl', 'hard_negative_injections_sha256'),
                            ('captures.jsonl', 'captures_sha256')]:
            require(sha(bind(prefix + name)) == raw[field], 'raw shard evidence hash mismatch: ' + name)
        events = rows(root / prefix / 'detections.jsonl')
        schedule = rows(root / prefix / 'hard-negative-injections.jsonl')
        require(len(events) == shard['false_accepts'], 'detection count mismatch')
        require(len(schedule) == shard['hard_negative_injections'], 'injection count mismatch')
        covered = {r['source_sha256'] for r in schedule if r.get('coverage_required') is True and r.get('gain') == 0.9}
        require(domain_hashes <= covered, 'saved injections do not cover every negative source')
        counts += shard['false_accepts']; hours += shard['audio_hours']
        injections += shard['hard_negative_injections']; seconds += shard['hard_negative_audio_seconds']
    rate, upper = counts / hours, poisson_rate_upper(counts, hours, 0.95)
    expected = {'shards': 4, 'seeds': list(SEEDS), 'audio_hours': hours, 'false_accepts': counts,
                'far_per_hour': rate, 'far_upper_bound_per_hour': upper, 'confidence_level': 0.95,
                'max_far_per_hour': 0.0, 'max_upper_bound_per_hour': 0.4,
                'min_hard_negative_injections': 800, 'min_hard_negative_audio_seconds': 800.0,
                'hard_negative_injections': injections, 'hard_negative_audio_seconds': seconds,
                'runner_sha256': runtime, 'model_sha256': model, 'keyword_pack_sha256': pack,
                'negative_manifest_sha256': manifest, 'hard_negative_rate_per_minute': 2.0}
    for key, value in expected.items():
        require(aggregate.get(key) == value, 'aggregate does not match recomputed evidence: ' + key)
    require({r['sha256'] for r in aggregate['inputs']} == summary_hashes and len(aggregate['inputs']) == 4,
            'aggregate shard input set mismatch')
    passed = counts == 0 and upper <= 0.4 and injections >= 800 and seconds >= 800
    require(aggregate.get('qualified') is passed, 'aggregate verdict mismatch')
    return {'source_binding': source_binding, 'identity': {'runner_sha256': runtime, 'model_sha256': model, 'keyword_pack_sha256': pack,
                         'shipping_contract_sha256': configs['shipping.xiaowo.json'],
                         'nightly_contract_sha256': configs['nightly.xiaowo-frozen-model.json'],
                         'parameter_contract_sha256': configs['parameter-contract.json']},
            'status': 'passed' if passed else 'failed', 'qualified': passed,
            'metrics': expected, 'files': files, 'known_fa_sources_covered': sorted(KNOWN_FA_SOURCES),
            'product_qualification_allowed': False}


def produce(root, source, run_id, attempt):
    require(re.fullmatch('[0-9a-f]{40}', source or ''), 'exact source SHA required')
    require(type(run_id) is int and run_id > 0 and type(attempt) is int and attempt > 0, 'run/attempt required')
    record = {'schema': SCHEMA, 'source_sha': source, 'run_id': run_id, 'run_attempt': attempt}
    try:
        result = evaluate(root)
        require(result['source_binding']['source_sha'] == source, 'runtime source binding mismatch')
        record.update(result)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        record.update(status='unknown', qualified=False, blocker=str(exc), product_qualification_allowed=False)
    return record


def bind_source(root, source):
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == source,
            'producer checkout is not the executed source')
    paths = []
    for relative in SOURCE_ROOTS:
        path = ROOT / relative
        paths.extend(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else paths.append(path)
    records = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        expected = subprocess.check_output(['git', 'rev-parse', source + ':' + relative], cwd=ROOT, text=True).strip()
        require(blob == expected, 'runtime source modified after checkout: ' + relative)
        records[relative] = {'git_blob_sha1': blob, 'sha256': hashlib.sha256(data).hexdigest()}
    write(root / 'far-regression/source-binding.json', {'schema': 'far-runtime-source-binding-v1',
          'source_sha': source, 'files': records})


def verify_source_tree(binding, tree):
    require(tree.get('truncated') is False, 'runtime source tree is incomplete')
    blobs = {row['path']: row['sha'] for row in tree['tree'] if row['type'] == 'blob'}
    expected_paths = {path for path in blobs if any(path == prefix or path.startswith(prefix + '/') for prefix in SOURCE_ROOTS)}
    require(set(binding['files']) == expected_paths, 'runtime source file set mismatch')
    for path, row in binding['files'].items():
        require(row['git_blob_sha1'] == blobs[path], 'runner source differs from deployed source: ' + path)


def select_run(pages, workflow, repository, source):
    """Complete server-filtered history; never choose an older green run."""
    require(workflow.get('path') == WORKFLOW and type(workflow.get('id')) is int, 'official workflow identity mismatch')
    require(isinstance(pages, list) and pages, 'missing history pages')
    runs = [row for page in pages for row in page['workflow_runs']]
    totals = {page['total_count'] for page in pages}
    require(totals == {len(runs)} and len({r['id'] for r in runs}) == len(runs), 'incomplete or concurrent history pagination')
    require(runs, 'no current-source FAR run; historical qualification cannot substitute')
    for row in runs:
        require(row.get('head_sha') == source and row.get('workflow_id') == workflow['id'] and
                row.get('path') == WORKFLOW and row.get('repository', {}).get('full_name') == repository and
                row.get('head_repository', {}).get('full_name') == repository and
                row.get('head_branch') == 'main', 'history scope mismatch')
        require(type(row.get('run_number')) is int and type(row.get('run_attempt')) is int and row['run_attempt'] > 0,
                'run ordering/attempt missing')
    latest = max(runs, key=lambda row: (row['run_number'], row['run_attempt']))
    require(latest.get('event') in {'schedule', 'workflow_dispatch'}, 'unexpected official regression event')
    require(latest.get('status') == 'completed' and latest.get('conclusion') == 'success',
            'latest current-source FAR run is not successful; no older-green fallback')
    return {key: latest[key] for key in ('id', 'run_attempt', 'run_number', 'head_sha', 'workflow_id',
                                         'path', 'event', 'status', 'conclusion', 'updated_at')}


def gh_json(endpoint, paginate=False):
    command = ['gh', 'api', endpoint]
    if paginate:
        command += ['--paginate', '--slurp']
    return json.loads(subprocess.check_output(command))


def current_run(repository, source):
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository or ''), 'invalid repository')
    require(re.fullmatch('[0-9a-f]{40}', source or ''), 'invalid deployment source')
    workflow = gh_json(f'repos/{repository}/actions/workflows/far-nightly.yml')
    pages = gh_json(f"repos/{repository}/actions/workflows/{workflow['id']}/runs?head_sha={source}&per_page=100", True)
    return select_run(pages, workflow, repository, source)


def extract_verified(archive, digest, output):
    require(digest == 'sha256:' + sha(archive), 'complete artifact digest mismatch or missing')
    with zipfile.ZipFile(archive) as bundle:
        seen = set(); size = 0
        for item in bundle.infolist():
            name = PurePosixPath(item.filename)
            require(not name.is_absolute() and '..' not in name.parts and '\\' not in item.filename,
                    'unsafe artifact path')
            require(item.filename not in seen, 'duplicate artifact member')
            seen.add(item.filename); size += item.file_size
            require(not stat.S_ISLNK(item.external_attr >> 16) and size <= 1024 ** 3, 'unsupported or oversized artifact')
        require(bundle.testzip() is None, 'artifact CRC mismatch')
        bundle.extractall(output)


def verify_download(root, deployment, selected):
    receipt = load(root / 'far-regression/receipt.json')
    require(receipt.get('schema') == SCHEMA and receipt.get('source_sha') == deployment['source_sha'] and
            receipt.get('run_id') == selected['id'] and receipt.get('run_attempt') == selected['run_attempt'],
            'receipt source/run/attempt mismatch')
    recomputed = produce(root, deployment['source_sha'], selected['id'], selected['run_attempt'])
    require(receipt == recomputed, 'receipt does not match saved raw evidence')
    require(receipt['qualified'] is True and receipt['status'] == 'passed', 'current regression failed or unknown')
    identity = receipt['identity']
    for key in ('shipping_contract_sha256', 'nightly_contract_sha256'):
        require(identity[key] == deployment[key], 'deployment config mismatch: ' + key)
    for key in ('model_sha256', 'keyword_pack_sha256'):
        require(identity[key] == deployment['model_release'][key], 'deployment model/pack mismatch: ' + key)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('produce'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', required=True); p.add_argument('--run-id', type=int, required=True)
    p.add_argument('--attempt', type=int, required=True)
    p = subs.add_parser('fetch'); p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--repository', required=True); p.add_argument('--output', type=Path, required=True)
    p = subs.add_parser('recheck'); p.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'produce':
        bind_source(args.root, args.source)
        write(args.root / 'far-regression/receipt.json', produce(args.root, args.source, args.run_id, args.attempt))
    elif args.command == 'fetch':
        require(not args.output.exists(), 'fresh output directory required; stale evidence must not survive')
        deployment = load(args.deployment)
        require(deployment.get('repository') == args.repository, 'deployment repository mismatch')
        selected = current_run(args.repository, deployment['source_sha'])
        pages = gh_json(f"repos/{args.repository}/actions/runs/{selected['id']}/artifacts?per_page=100", True)
        artifacts = [a for page in pages for a in page['artifacts']]
        require({p['total_count'] for p in pages} == {len(artifacts)}, 'incomplete artifact pagination')
        matches = [a for a in artifacts if a['name'] == 'far-frozen-model-regression']
        require(len(matches) == 1 and matches[0].get('expired') is False, 'one live official regression artifact required')
        artifact = matches[0]
        require(type(artifact.get('size_in_bytes')) is int and 0 < artifact['size_in_bytes'] <= 1024 ** 3,
                'unsupported artifact size')
        args.output.mkdir(parents=True)
        archive = args.output / 'original-artifact.zip'
        with archive.open('wb') as out:
            subprocess.run(['gh', 'api', f"repos/{args.repository}/actions/artifacts/{artifact['id']}/zip"], stdout=out, check=True)
        root = args.output / 'evidence'
        extract_verified(archive, artifact.get('digest'), root)
        receipt = verify_download(root, deployment, selected)
        tree = gh_json(f"repos/{args.repository}/git/trees/{deployment['source_sha']}?recursive=1")
        verify_source_tree(receipt['source_binding'], tree)
        require(current_run(args.repository, deployment['source_sha']) == selected, 'regression selection changed during verification')
        write(args.output / 'verification.json', {'schema': SCHEMA, 'qualified': True, 'repository': args.repository,
              'selection': selected, 'artifact_id': artifact['id'], 'artifact_sha256': sha(archive),
              'receipt_sha256': sha(root / 'far-regression/receipt.json'), 'identity': receipt['identity'],
              'deployment_manifest_sha256': sha(args.deployment), 'product_qualification_allowed': False})
    else:
        record = load(args.record)
        require(record.get('schema') == SCHEMA and record.get('qualified') is True, 'missing verified regression')
        require(current_run(record['repository'], record['selection']['head_sha']) == record['selection'],
                'newer/changed regression run appeared before publication; approval is blocked')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print('regression blocked: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
