"""A fixed fresh campaign using an immutable, separately checked-out runtime."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
# The retained host imports its own module named contract. Do not register this
# campaign's contract under that shared name in the orchestration process.
_contract_spec = importlib.util.spec_from_file_location('fresh_campaign_contract', HERE / 'contract.py')
_contract = importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(_contract)
CAMPAIGN, RETAINED_COMMIT = _contract.CAMPAIGN, _contract.RETAINED_COMMIT
digest, file_hash, plan = _contract.digest, _contract.file_hash, _contract.plan
require, save, verify_environment = _contract.require, _contract.save, _contract.verify_environment
GIB = 1024 ** 3


def stage_sources(retained, destination):
    require(subprocess.check_output(['git', '-C', str(retained), 'rev-parse', 'HEAD'],
            text=True).strip() == RETAINED_COMMIT, 'retained commit mismatch')
    require(subprocess.check_output(['git', '-C', str(retained), 'status', '--porcelain'],
            text=True) == '', 'retained checkout must be clean')
    names = ('source-screen32-v1', 'tts-quality-pilot-v1', 'qwen6_tts', 'qwen6_asr')
    files = {}
    for name in names:
        source = retained / 'research' / name
        require(source.is_dir() and not source.is_symlink(), 'retained source missing')
        target = destination / 'research' / name
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(
            '__pycache__', 'evidence', 'history', 'runtime'))
        for path in target.rglob('*'):
            require(not path.is_symlink() and (path.is_dir() or path.is_file()), 'unexpected staged type')
            if path.is_file():
                relative = path.relative_to(destination).as_posix()
                require(file_hash(path) == file_hash(retained / relative), 'staged source changed')
                files[relative] = file_hash(path)
    quality = destination / 'research/tts-quality-pilot-v1'
    shutil.copyfile(quality / 'qwen06-model-lock.json', quality / 'qwen-model-lock.json')
    files['research/tts-quality-pilot-v1/qwen-model-lock.json'] = file_hash(quality / 'qwen-model-lock.json')
    current = destination / 'research/fresh-qwen16-20261030'
    current.mkdir()
    for name in ('worker.py', 'contract.py'):
        shutil.copyfile(HERE / name, current / name)
        files['research/fresh-qwen16-20261030/' + name] = file_hash(current / name)
    require(sum(path.stat().st_size for path in destination.rglob('*') if path.is_file()) <= 64 * 1024 ** 2,
            'staged source cap exceeded')
    return files


def publish(root, receipt):
    output = root / 'output'
    # A killed worker may leave a durable claim without a completed outcome.
    receipt['claimed_cells'] = sorted(path.stem for path in output.glob('*.claim'))
    receipt['claimed_count'] = len(receipt['claimed_cells'])
    receipt['unattempted_count'] = 16 - receipt['claimed_count']
    save(root / 'host-receipt.json', receipt)
    artifact = root / 'artifact'
    artifact.mkdir(exist_ok=False)
    paths = [root / name for name in ('host-receipt.json', 'source-freeze.json',
             'image-transfer.json', 'container-setup.json', 'container-tts.json')]
    allowed = {'generation-receipt.json', 'progress.json', 'runtime-observations.json'}
    for cell in plan():
        allowed.update(cell['cell_id'] + suffix for suffix in ('.claim', '.raw-float.wav', '.wav'))
    paths += [output / name for name in sorted(allowed)]
    total = 0
    files = {}
    for path in paths:
        if not path.exists():
            continue
        require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
                'regular unaliased evidence required')
        size = path.stat().st_size
        require(size <= 8 * 1024 ** 2, 'per-file evidence cap exceeded')
        total += size
        require(total <= 32 * 1024 ** 2, 'whole artifact cap exceeded')
        relative = path.relative_to(root)
        target = artifact / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[relative.as_posix()] = dict(bytes=size, sha256=file_hash(target))
    save(artifact / 'artifact-freeze.json', dict(schema='fresh-qwen16-artifact-v1',
        campaign=CAMPAIGN, source_head=receipt['source_head'], run_id=receipt['run_id'],
        files=files, ctc_training_allowed=False, qualification_allowed=False, human_gold=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retained-root', type=Path, required=True)
    args = parser.parse_args()
    event_path = Path(os.environ['GITHUB_EVENT_PATH'])
    require(event_path.stat().st_size <= 2 * 1024 ** 2, 'event size cap')
    verify_environment(os.environ, json.loads(event_path.read_bytes()))
    root = Path(os.environ['RUNNER_TEMP']) / (CAMPAIGN + '-' + os.environ['GITHUB_RUN_ID'])
    root.mkdir(exist_ok=False)
    for name in ('code', 'runtime', 'output'):
        (root / name).mkdir()
    with open(os.environ['GITHUB_OUTPUT'], 'a') as target:
        target.write('evidence_root=' + str(root / 'artifact') + '\n')
    started = time.monotonic()
    receipt = dict(schema='fresh-qwen16-host-v1', campaign=CAMPAIGN,
        source_head=os.environ['GITHUB_SHA'], retained_commit=RETAINED_COMMIT,
        run_id=os.environ['GITHUB_RUN_ID'], run_attempt=1, plan_sha256=digest(plan()),
        planned_calls=16, asr_calls=0, kws_training_calls=0, retry=False,
        status='started', stage='preflight', stages={}, ctc_training_allowed=False,
        qualification_allowed=False, human_gold=False)
    save(root / 'host-receipt.json', receipt)
    try:
        require(shutil.disk_usage(root).free >= 16 * GIB, '16 GiB current free disk required')
        files = stage_sources(args.retained_root.resolve(), root / 'code')
        save(root / 'source-freeze.json', dict(retained_commit=RETAINED_COMMIT,
             source_head=os.environ['GITHUB_SHA'], files=files))
        retained_source = args.retained_root.resolve() / 'research/source-screen32-v1'
        sys.path.insert(0, str(retained_source))
        spec = importlib.util.spec_from_file_location('fresh_retained_host', retained_source / 'hosted_run.py')
        host = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(host)
        receipt['capacity'] = host.host_capacity()
        lock = json.loads((retained_source / 'container-lock.json').read_bytes())
        receipt['stage'] = 'image'
        save(root / 'host-receipt.json', receipt)
        host.pull_image(lock, root / 'image-transfer.json')
        for stage in ('setup', 'tts'):
            host.host_capacity()
            receipt['stage'] = stage
            receipt['stages'][stage] = {'status': 'started'}
            save(root / 'host-receipt.json', receipt)
            name = 'fresh-qwen16-' + os.environ['GITHUB_RUN_ID'] + '-' + stage
            argv = host.container_args(name, lock['image'], root / 'code', root / 'runtime',
                                       root / 'output', 'tts', stage)
            split = argv.index(lock['image'])
            argv[split:split] = ['--env', 'TQP_ENGINE=qwen06']
            start = argv.index(lock['image']) + 1
            if stage == 'setup':
                argv[start:] = ['python', '-I', '-B', '/code/research/tts-quality-pilot-v1/setup_adapter.py',
                               '--profile', 'tts', '--stage', 'setup', '--runtime', '/runtime']
            else:
                argv[start:] = ['/runtime/venv/bin/python', '-I', '-B',
                               '/code/research/fresh-qwen16-20261030/worker.py']
            receipt['stages'][stage] = host.supervise_container(argv, name, root / 'output', stage,
                min(started + 7200, time.monotonic() + (1200 if stage == 'setup' else 4800)))
        result = json.loads((root / 'output/generation-receipt.json').read_bytes())
        require(result.get('status') == 'generated_16_pending_label_review' and
                result.get('generated') == 16 and result.get('attempted') == 16,
                'complete generation receipt required')
        receipt['status'] = 'generated_16_pending_label_review'
    except BaseException as error:
        receipt.update(status='stopped_no_retry', error_type=type(error).__name__)
        raise
    finally:
        receipt['wall_seconds'] = time.monotonic() - started
        publish(root, receipt)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print(json.dumps(dict(status='stopped_no_retry', error_type=type(error).__name__)), file=sys.stderr)
        raise SystemExit(1)
