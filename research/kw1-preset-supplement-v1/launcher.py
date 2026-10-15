#!/usr/bin/env python3
"""Reconstructed one-shot host entry; finalization never constructs a backend."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_common import require, document, read, sha, claim, exact_equal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GLOBAL_SECONDS = 18000

def boot_id():
    with open('/proc/sys/kernel/random/boot_id', 'r', encoding='ascii') as handle:
        value = handle.read(129).strip()
    require(len(value) == 36, 'host boot identity')
    return value

def checkout_identity(repo, expected):
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True,
        capture_output=True, text=True, timeout=15, env={'PATH': os.defpath, 'HOME': '/nonexistent'}).stdout.strip()
    require(head == expected, 'checkout must equal admitted Git source')

def validate_claim(repo, work, *, check_checkout=True):
    from admission import required_sources
    receipt = document(Path(work)/'campaign.claim.json')
    fields = {'schema','admitted','run_id','source_sha','workflow_id','config_sha256','source_files','max_tts_calls','max_asr_calls','retry'}
    require(type(receipt) is dict and set(receipt) == fields, 'exact admission receipt')
    require(receipt['schema'] == 'kw1-admission-receipt-v1' and receipt['admitted'] is True and receipt['retry'] is False, 'admitted once')
    require(type(receipt['run_id']) is int and receipt['run_id'] > 0 and type(receipt['workflow_id']) is int and receipt['workflow_id'] > 0, 'server identity integers')
    require(type(receipt['max_tts_calls']) is int and receipt['max_tts_calls'] == 16 and type(receipt['max_asr_calls']) is int and receipt['max_asr_calls'] == 48, 'fixed budget')
    config_path = Path(repo)/'research/kw1-preset-supplement-v1/execution-admission.json'
    config = document(config_path)
    require(config.get('enabled') is True and sha(read(config_path)) == receipt['config_sha256'], 'admission configuration unchanged')
    require(type(receipt['source_files']) is dict and set(receipt['source_files']) == required_sources() and exact_equal(receipt['source_files'], config.get('source_files')), 'complete unchanged source closure')
    for name, digest in receipt['source_files'].items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts, 'relative source')
        require(sha(read(Path(repo)/name)) == digest, 'admitted source changed')
    if check_checkout:
        checkout_identity(repo, receipt['source_sha'])
    return receipt

def remaining_deadline(work):
    stamp = document(Path(work)/'admission-clock.json')
    require(type(stamp) is dict and set(stamp) == {'schema','boot_id','monotonic','wall_time','budget_seconds'}, 'clock schema')
    require(stamp['schema'] == 'kw1-admission-clock-v1' and stamp['boot_id'] == boot_id(), 'same host boot required')
    require(type(stamp['budget_seconds']) is int and stamp['budget_seconds'] == GLOBAL_SECONDS, 'original global budget')
    for key in ('monotonic','wall_time'):
        require(type(stamp[key]) is float and math.isfinite(stamp[key]), 'finite original clock')
    elapsed_mono = time.monotonic() - stamp['monotonic']
    elapsed_wall = time.time() - stamp['wall_time']
    require(elapsed_mono >= 0 and elapsed_wall >= 0, 'clock rollback')
    remaining = GLOBAL_SECONDS - max(elapsed_mono, elapsed_wall)
    require(remaining > 0, 'campaign global budget expired')
    return time.monotonic() + remaining

def main(argv=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--admit', action='store_true')
    group.add_argument('--finalize', action='store_true')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--public', type=Path)
    args = parser.parse_args(argv)
    if args.admit:
        from admission import admit
        checkout_identity(REPO, os.environ.get('GITHUB_SHA'))
        start_mono, start_wall = time.monotonic(), time.time()
        admit(REPO, args.work)
        claim(args.work/'admission-clock.json', dict(schema='kw1-admission-clock-v1', boot_id=boot_id(), monotonic=start_mono, wall_time=start_wall, budget_seconds=GLOBAL_SECONDS))
        return 0
    require(args.public is not None, 'public destination required')
    receipt = validate_claim(REPO, args.work)
    from export_public import export
    from verify_public import verify
    if args.public.exists() or args.public.is_symlink():
        require(args.finalize, 'existing public output forbids execution')
        verify(args.public, expected_admission=receipt)
        return 0
    plan = document(HERE/'plan.json')
    if not args.finalize:
        # The durable execution claim is never renewed, even after interruption.
        deadline = remaining_deadline(args.work)
        claim(args.work/'execution.claim.json', dict(schema='kw1-execution-claim-v1', campaign_claim_sha256=sha(read(args.work/'campaign.claim.json')), attempt=1))
        from docker_backend import DockerBackend
        from campaign import execute
        backend = DockerBackend(REPO, args.work/'backend', deadline, source_lock={'files':receipt['source_files']})
        execute(plan, args.work, backend)
    # Export independently rebuilds only from authoritative durable evidence.
    export(plan, args.work, args.public, admission=receipt)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
