"""Bounded stdlib capability inventory; never numerical execution admission."""
import argparse
import importlib.metadata
import itertools
import json
import platform
import re
import site
import subprocess
import sys

REQUIRED = {'torch': '2.11.0+cpu', 'numpy': '1.26.4'}
LIMITS = {'operator_rows': 798, 'wall_seconds': 120, 'rss_bytes': 536870912}
READ_PATHS = {'/proc/self/mountinfo', '/proc/self/cgroup', '/sys/fs/cgroup/memory.max'}
MAX_READ_BYTES = 65536
MAX_OUTPUT_BYTES = 16384


def version(value):
    """Reject arbitrary metadata, paths, controls and unexpectedly long text."""
    if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9._+!-]{1,64}', value):
        return value
    return None


def read(path):
    if path not in READ_PATHS:
        return None
    try:
        with open(path, 'rb') as f:
            data = f.read(MAX_READ_BYTES + 1)
        if len(data) > MAX_READ_BYTES:
            return None
        return data.decode('ascii')
    except (OSError, UnicodeError):
        return None


def memory_limit(value):
    if value is None:
        return None
    value = value.strip()
    if value == 'max':
        return 'unlimited'
    if re.fullmatch(r'[0-9]{1,19}', value) and int(value) <= 2**63-1:
        return int(value)
    return None


def cpu_probe():
    # Child returns only an exit code. No stderr, paths or environment are copied.
    code = ('import os,sys; a=min(os.sched_getaffinity(0)); '
            'os.sched_setaffinity(0,{a}); '
            'sys.exit(0 if len(os.sched_getaffinity(0))==1 and '
            'len(os.listdir("/proc/self/task"))==1 else 1)')
    try:
        p = subprocess.run([sys.executable, '-I', '-S', '-c', code],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return {'single_cpu_single_thread_verified': p.returncode == 0,
                'status': 'verified' if p.returncode == 0 else 'probe_failed'}
    except subprocess.TimeoutExpired:
        return {'single_cpu_single_thread_verified': False, 'status': 'probe_timeout'}
    except Exception:
        return {'single_cpu_single_thread_verified': False, 'status': 'probe_unavailable'}


def inspect():
    versions = {}
    for name in REQUIRED:
        try:
            matches = list(itertools.islice(importlib.metadata.Distribution.discover(
                name=name, path=site.getsitepackages()), 2))
            versions[name] = version(matches[0].version) if len(matches) == 1 else None
        except Exception:
            versions[name] = None
    mounts = read('/proc/self/mountinfo')
    membership = read('/proc/self/cgroup')
    cgroups = None if mounts is None else sum(' - cgroup' in line for line in mounts.splitlines())
    unified = None if membership is None else any(line.startswith('0::') for line in membership.splitlines())
    blockers = [name + '_metadata_missing_mismatched_or_unreadable'
                for name, required in REQUIRED.items() if versions[name] != required]
    blockers += ['rss_hard_enforcement_unverified', 'backend_thread_network_isolation_unverified',
                 'exact_backend_import_peak_unmeasured', 'historical_c_source_binding_unverified',
                 'single_op_independent_acceptance_missing', 'numerical_authorization_required']
    return {'schema': 'd20-no-model-admission-v1', 'python': version(platform.python_version()),
            'installed_distribution_metadata': versions,
            'distribution_metadata_scope': 'system_site_directories_no_pth_or_user_site',
            'required': REQUIRED, 'limits': LIMITS, 'cpu_stdlib_probe': cpu_probe(),
            'cgroup_capabilities': {'mount_count': cgroups, 'unified_membership_visible': unified,
                'visible_root_memory_max': memory_limit(read('/sys/fs/cgroup/memory.max')),
                'effective_process_limit_verified': False},
            'rlimit_as_is_rss': False, 'numerical_imports': 0, 'model_calls': 0,
            'operator_calls': 0, 'execution_ready': False, 'blockers': blockers}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    try:
        payload = json.dumps(inspect(), ensure_ascii=True, indent=2, allow_nan=False)+'\n'
        if len(payload.encode('ascii')) > MAX_OUTPUT_BYTES:
            raise ValueError('bounded output exceeded')
        with open(a.output, 'x') as f:
            f.write(payload)
    except Exception:
        print('ADMISSION_INVENTORY_FAILED', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
