#!/usr/bin/env python3
"""DISABLED DRAFT: new independent 1200-update fixed-grouped-balance experiment; never retries or resumes.

Root must freeze actual source/plan/runner pins and confirm CLI before enabling.
No budget, output, or claim from the previous campaign may be reused.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import time

BASE = Path(__file__).resolve().parent
# Source/plan/runner pins and actual merged commit remain unknown; never infer or copy prior campaign pins.
LAUNCH_ENABLED = True
REVIEWED_SOURCE_COMMIT = '942b3c1495fa4e4007ca01604ca8a65ed2cd5bcf'
PLAN_SHA = '38cc02c58eb25b6da7af0e5592e5738d22beffed05beffa643643cd9cb902242'
RUNNER_SHA = 'f57e0fae61b314769cb1d3f3365b77f9ccb9400071be425b3ce7d04e66d53b6e'
RUNNER_NAME = 'run_grouped_balance.py'
CAMPAIGN_ID = 'kws-grouped-balance-20261015-v1'
CLAIM_RELATIVE = 'research-budget/kws-grouped-balance-20261015-v1.claim.json'
REVIEWED_SOURCE_PINS = {'research/consolidation/fit-history-recovery-2026-10-15.json': '6e0bf6141ddec49297ba00ed9a368fc66f323c06672d6906b5ab4fc482a041f7',
 'research/kw1-preset-supplement-observation-v1/observation.json': '95c67c26d04bfdbba41df11540c084ab99ad736e0b8147383d93808be40a919e',
 'research/kw1-preset-supplement-v1/blind_pcm.py': 'ab6164bb89c9b3b0e707375330b2e9c13b772a0fda1c252dfb2dda05f3539b64',
 'research/kw1-preset-supplement-v1/worker.py': '979e1709e24ff21e805abfd481994d4073c3220f217f20d6782d1307a77097b3',
 'research/kws-candidate-registry-v1/build_registry.py': '9d37fb852cdd5048a4d5c88f9e21efe2b3c1e18f25244babb2d811675ea9fbae',
 'research/kws-candidate-registry-v1/candidates.json': '28033edbbc1e444eae27c33af2e060d1654c4b7a072e65401b055fc27fc3b1b0',
 'research/kws-candidate-registry-v1/input-pins.json': '64c77a270c7f13d8252d10282191aaeb207f425c10fb89a2f01e390fa994e37c',
 'research/kws-grouped-balance-v1/README.md': '5238048c0fcbe440b3b66b27d72a41edac390dbf56b1c6c6e3489b3efe8db2d0',
 'research/kws-grouped-balance-v1/cohort-prefix-freeze.json': 'fe23e633925573e2b02ac8530e066214879a11cbccc2e4ef5875aa2c38006e87',
 'research/kws-grouped-balance-v1/contract.py': '6dca1cad9dba10f50e439f775b6d04d90f246544cbd5a634e30490589490bc74',
 'research/kws-grouped-balance-v1/numeric_preflight.py': '654b26668d9723d12e25a7b447df901cc2d5fae063bba27c53bd30c91efefefb',
 'research/kws-grouped-balance-v1/plan.json': '38cc02c58eb25b6da7af0e5592e5738d22beffed05beffa643643cd9cb902242',
 'research/kws-grouped-balance-v1/requirements.research.lock': '7aa51b6eebd2e2db290338345c30b8c7a8e197531746ee77efb7ef8678124dee',
 'research/kws-grouped-balance-v1/run_grouped_balance.py': 'f57e0fae61b314769cb1d3f3365b77f9ccb9400071be425b3ce7d04e66d53b6e',
 'research/kws-grouped-balance-v1/runtime-candidate.lock.json': 'b132c2c202e5ff00b6cbd1b2b630fa92dfe9ba46fcf0423dc0633381bbe80984',
 'research/kws-grouped-balance-v1/source-manifest.json': '745d7cce4d2863a7956315cff7a5a9b33ab36d4fd0f424091b6bb88f9c984f75',
 'research/kws-grouped-balance-v1/test_contract.py': 'd30eeda3fdc4fd3f29c69fdec84510a0c530431ba843f9a97b3cc651a6d655d8',
 'research/tts-pilot-screening-v1/evidence/asr-run-38573117922/thirteen-candidate-evidence.json': '9cb52b6f4a890a858f5fc96af89825ba63c5035fcaaefb0349a7806f3843745d',
 'research/tts-pilot-screening-v1/evidence/tts-input-20261014/artifact-inventory.json': 'cf20e6771eb0e0647dc890ab341a87367cbb77afb89c39b85377e8ace3c5ba4a',
 'research/tts-pilot-screening-v1/screen.py': '982582c0e4c7f957276520977ebfa3bad0cbb5753e16d50bbadef0835682523e',
 'training/frontend.py': 'd8ae687303d5c9c6bfdf6539c94f4ae463d910a4837ed77633d7656ff02a582e',
 'training/frontend_spec.py': 'fbd17236e1449101bed6df8f105baad476edb0b005e6e45cb83ebc7971d166e6',
 'training/model.py': '768800e25537367da770ef44178249208956d7c3c33f1eb7d60b1b9c98fe59c5'}  # Root freezes the complete merged source closure after independent review.
CLI_CONFIRMED = True
MODULE_RELATIVE = 'research/kws-grouped-balance-v1'


def require_frozen():
    if LAUNCH_ENABLED is not True or CLI_CONFIRMED is not True:
        raise ValueError('grouped-balance launch draft is disabled pending independent review')
    if not isinstance(REVIEWED_SOURCE_COMMIT, str) or not re.fullmatch('[0-9a-f]{40}', REVIEWED_SOURCE_COMMIT):
        raise ValueError('reviewed source commit is not frozen')
    for value in (PLAN_SHA, RUNNER_SHA):
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('reviewed source hash is not frozen')
    if not isinstance(RUNNER_NAME, str) or Path(RUNNER_NAME).name != RUNNER_NAME:
        raise ValueError('runner filename is not frozen')
    if not isinstance(CAMPAIGN_ID, str) or not CAMPAIGN_ID or CAMPAIGN_ID != 'kws-grouped-balance-20261015-v1':
        raise ValueError('new independent campaign ID required')
    if not isinstance(CLAIM_RELATIVE, str) or Path(CLAIM_RELATIVE).is_absolute() or '..' in Path(CLAIM_RELATIVE).parts or str(Path(CLAIM_RELATIVE)) != CLAIM_RELATIVE:
        raise ValueError('fixed new relative claim path required')
    if not isinstance(REVIEWED_SOURCE_PINS, dict) or not REVIEWED_SOURCE_PINS:
        raise ValueError('complete reviewed source closure is not frozen')
    if not isinstance(MODULE_RELATIVE, str) or Path(MODULE_RELATIVE).is_absolute() or '..' in Path(MODULE_RELATIVE).parts or str(Path(MODULE_RELATIVE)) != MODULE_RELATIVE:
        raise ValueError('new runner module path is not frozen')
    prefix = MODULE_RELATIVE + '/'
    if REVIEWED_SOURCE_PINS.get(prefix+'plan.json') != PLAN_SHA or REVIEWED_SOURCE_PINS.get(prefix+RUNNER_NAME) != RUNNER_SHA:
        raise ValueError('plan and runner must be in the reviewed source closure')
    for name, digest in REVIEWED_SOURCE_PINS.items():
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or str(Path(name)) != name:
            raise ValueError('unsafe source pin path')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid source pin hash')


def sha(path):
    """Bounded, stable no-alias read; access-time changes are intentionally ignored."""
    path = Path(path).absolute()
    if '..' in path.parts or path.resolve(strict=True) != path:
        raise ValueError('source aliases and symlinks are forbidden')
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    fd = None
    try:
        for part in path.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = nxt
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 32*1024**2:
                raise ValueError('unaliased bounded regular source required')
            raw = stream.read(32*1024**2 + 1)
            after = os.fstat(stream.fileno())
            stable = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            if len(raw) != before.st_size or stable(before) != stable(after):
                raise ValueError('source changed during bounded read')
        return hashlib.sha256(raw).hexdigest()
    finally:
        if fd is not None: os.close(fd)
        os.close(directory)


def write_new(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verified-source-commit', required=True)
    args = p.parse_args()
    require_frozen()  # Before filesystem mutations, subprocess creation, or imports.
    if args.verified_source_commit != REVIEWED_SOURCE_COMMIT:
        raise ValueError('externally verified source commit differs from frozen commit')
    repo = BASE / 'grouped-balance-integrated'
    module = repo / MODULE_RELATIVE
    for name, expected in REVIEWED_SOURCE_PINS.items():
        if sha(repo / name) != expected:
            raise ValueError('reviewed source closure changed before interpreter start')
    evidence = BASE / 'grouped-balance-execution-evidence'
    evidence.mkdir(exist_ok=False)
    claim_root = BASE / 'grouped-balance-campaign-ledger'
    claim_root.mkdir(exist_ok=False)
    (claim_root / Path(CLAIM_RELATIVE).parent).mkdir(parents=True, exist_ok=True)
    output = BASE / 'grouped-balance-result'
    if output.exists() or output.is_symlink():
        raise ValueError('existing output is never reused')
    home = evidence / 'home'; home.mkdir()
    tmp = evidence / 'tmp'; tmp.mkdir()
    command = [str(BASE / 'fit-venv/bin/python'), '-I', '-B', str(module / RUNNER_NAME),
        '--repo', str(repo), '--artifacts', str(BASE.parent / 'recovery-kws-20261015/artifacts'),
        '--wheelhouse', str(BASE / 'runtime-work/wheels'), '--output', str(output),
        '--claim-root', str(claim_root), '--plan-sha256', PLAN_SHA]
    command_sha = hashlib.sha256(json.dumps(command, separators=(',', ':')).encode()).hexdigest()
    cpu = min(os.sched_getaffinity(0))
    intent = dict(schema='research-grouped-balance-local-launch-intent-v1',
        campaign_id=CAMPAIGN_ID,
        parent_verified_source_commit=args.verified_source_commit,
        reviewed_source_pins=REVIEWED_SOURCE_PINS, independent_new_budget=True,
        previous_campaign_budget_reused=False, claim_relative=CLAIM_RELATIVE,
        plan_sha256=PLAN_SHA, runner_sha256=RUNNER_SHA, launcher_sha256=sha(Path(__file__)),
        command_sha256=command_sha, max_optimizer_updates=1200, max_model_forward_calls=1218,
        new_tts_calls=0, new_asr_calls=0, retry=False,
        actual_optimizer_updates='UNKNOWN_UNTIL_RESULT_VERIFICATION',
        wall_timeout_seconds=3600, cpu_limit_seconds=3600,
        virtual_address_space_limit_bytes=8*1024**3, per_file_limit_bytes=16*1024**2, runner_evidence_directory_limit_bytes=512*1024**2, core_dump_limit_bytes=0,
        cpu_affinity=[cpu], started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        production_qualified=False, reusable_weights_exported=False)
    write_new(evidence / 'launch-intent.json', intent)
    env = {'PATH': str(BASE / 'fit-venv/bin') + ':/usr/bin:/bin', 'HOME': str(home),
           'TMPDIR': str(tmp), 'LANG': 'C.UTF-8', 'OMP_NUM_THREADS': '1',
           'OMP_DYNAMIC': 'FALSE', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'NUMEXPR_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE', 'ATEN_CPU_CAPABILITY': 'default'}
    def limits():
        os.setsid()
        os.sched_setaffinity(0, {cpu})
        resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
        resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024**2, 16*1024**2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    start = time.monotonic(); timed_out = False
    with (evidence / 'stdout.log').open('xb') as out, (evidence / 'stderr.log').open('xb') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, env=env, cwd=repo, preexec_fn=limits)
        try:
            write_new(evidence / 'process-start.json', {'pid': child.pid, 'command_sha256': command_sha})
            try:
                code = child.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try: code = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); code = child.wait()
        except BaseException:
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            write_new(evidence / 'process-aborted.json', {
                'schema': 'research-grouped-balance-local-process-aborted-v1',
                'exit_code': child.returncode, 'command_sha256': command_sha,
                'actual_optimizer_updates': 'VERIFY_CALL_AND_STEP_LEDGER',
                'retry_authorized': False, 'production_qualified': False})
            raise
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(schema='research-grouped-balance-local-process-result-v1', exit_code=code,
        wall_timeout=timed_out, wall_seconds=time.monotonic()-start,
        user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
        child_max_rss_kib=usage.ru_maxrss, max_rss_scope='Linux RUSAGE_CHILDREN, not cgroup or SSC305',
        stdout_sha256=sha(evidence / 'stdout.log'), stderr_sha256=sha(evidence / 'stderr.log'),
        launcher_sha256=sha(Path(__file__)), command_sha256=command_sha,
        actual_optimizer_updates='VERIFY_CALL_AND_STEP_LEDGER',
        production_qualified=False)
    write_new(evidence / 'process-result.json', result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(code if code >= 0 else 1)

if __name__ == '__main__':
    main()
