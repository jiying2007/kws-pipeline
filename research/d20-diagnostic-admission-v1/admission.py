"""Bounded stdlib capability inventory; never numerical execution admission."""
import argparse
import email.parser
import email.policy
import importlib.metadata
import itertools
import json
import os
import pathlib
import platform
import re
import site
import stat
import subprocess
import sys

REQUIRED = {'torch': '2.11.0+cpu', 'numpy': '1.26.4'}
DISTRIBUTIONS = tuple(REQUIRED)
LIMITS = {'operator_rows': 798, 'wall_seconds': 120, 'rss_bytes': 536870912}
READ_PATHS = {'/proc/self/mountinfo', '/proc/self/cgroup', '/sys/fs/cgroup/memory.max'}
MAX_READ_BYTES = 65536
MAX_OUTPUT_BYTES = 16384


def version(value):
    """Reject arbitrary metadata, paths, controls and unexpectedly long text."""
    if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9._+!-]{1,64}', value):
        return value
    return None


def metadata_version(name):
    """Read one allowlisted wheel's metadata with a real byte cap; never import it.

    PathDistribution._path is used only to locate its directory. Public .version,
    .metadata and .read_text would load the whole file before we could bound it.
    Unsupported finders, zip/egg metadata, symlinks and ambiguous installs fail
    closed. Keep this function identical in the no-checkout hosted probe.
    """
    if name not in DISTRIBUTIONS:
        raise ValueError('distribution not allowlisted')
    roots = {pathlib.Path(p).resolve() for p in site.getsitepackages()}
    matches = list(itertools.islice(importlib.metadata.Distribution.discover(
        name=name, path=sorted(str(p) for p in roots)), 2))
    if not matches:
        raise importlib.metadata.PackageNotFoundError(name)
    if len(matches) != 1 or type(matches[0]) is not importlib.metadata.PathDistribution:
        raise ValueError('ambiguous or unsupported distribution')
    directory = matches[0]._path
    if (not isinstance(directory, pathlib.Path) or
            not directory.name.endswith('.dist-info') or
            directory.parent.resolve() not in roots):
        raise ValueError('unsupported metadata location')
    # Directory-relative, no-follow opens also reject symlink substitution races.
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    root_fd = os.open(directory.parent.resolve(), flags | os.O_DIRECTORY)
    try:
        directory_fd = os.open(directory.name, flags | os.O_DIRECTORY, dir_fd=root_fd)
        try:
            fd = os.open('METADATA', flags | os.O_NONBLOCK, dir_fd=directory_fd)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_READ_BYTES:
                    raise ValueError('metadata is not a bounded regular file')
                with os.fdopen(fd, 'rb', closefd=False) as source:
                    raw = source.read(MAX_READ_BYTES + 1)
            finally:
                os.close(fd)
        finally:
            os.close(directory_fd)
    finally:
        os.close(root_fd)
    if len(raw) > MAX_READ_BYTES:
        raise ValueError('metadata exceeds byte cap')
    decoded = raw.decode('utf-8', errors='strict')
    headers = email.parser.HeaderParser(policy=email.policy.strict).parsestr(decoded)
    values = {}
    for field in ('Metadata-Version', 'Name', 'Version'):
        entries = headers.get_all(field, [])
        if len(entries) != 1:
            raise ValueError('missing or duplicate identity field')
        values[field] = str(entries[0])
    normalize = lambda value: re.sub(r'[-_.]+', '-', value).lower()
    if (not re.fullmatch(r'[0-9]+\.[0-9]+', values['Metadata-Version']) or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', values['Name']) or
            normalize(values['Name']) != normalize(name) or
            not re.fullmatch(r'[A-Za-z0-9._+!-]{1,64}', values['Version'])):
        raise ValueError('invalid metadata identity')
    return values['Version']


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
            versions[name] = metadata_version(name)
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
