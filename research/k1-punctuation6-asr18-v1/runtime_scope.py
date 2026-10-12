"""Read-only proof of this Docker process's effective kernel limits, stdlib only.

This never establishes a scope or accepts a caller-supplied success declaration.
The launcher creates an unprivileged, private-cgroup-namespace container first.
Unknown or unsupported hosts fail before any third-party runtime import.
"""
from pathlib import Path
import os
import re

MEMORY = 12 * 1024 ** 3
PIDS = 256


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    with open(path, encoding="ascii") as source:
        value = source.read(65537)
    require(len(value) <= 65536, "oversized kernel scope record")
    return value


def verify_runtime_scope(*, network_required=True):
    """Return observed facts, not admission. Only setup may allow network access."""
    require(type(network_required) is bool, "invalid network requirement")
    require(os.geteuid() != 0, "container runtime must be non-root")
    require(read('/proc/self/cgroup').strip() == '0::/', "private cgroup-v2 root required")
    mounts = [line.split() for line in read('/proc/self/mountinfo').splitlines()]
    groups = [row for row in mounts if len(row) >= 10 and row[4] == '/sys/fs/cgroup'
              and ' - cgroup2 ' in ' '.join(row)]
    require(len(groups) == 1 and groups[0][3] == '/' and 'ro' in groups[0][5].split(','),
            "private read-only cgroup2 mount required")
    roots = [row for row in mounts if len(row) >= 10 and row[4] == '/']
    require(len(roots) == 1 and 'ro' in roots[0][5].split(','), "read-only container root required")
    expected = {'memory.max': str(MEMORY), 'memory.swap.max': '0',
                'cpu.max': '400000 100000', 'pids.max': str(PIDS)}
    observed = {key: ' '.join(read('/sys/fs/cgroup/' + key).split()) for key in expected}
    require(observed == expected, "actual kernel memory/swap/CPU/PID limits mismatch")
    status = {}
    for line in read('/proc/self/status').splitlines():
        key, separator, value = line.partition(':')
        if separator:
            status[key] = value.strip()
    require(status.get('NoNewPrivs') == '1' and status.get('Seccomp') == '2',
            "no-new-privileges and seccomp filter required")
    require(all(re.fullmatch('0+', status.get(key, '')) for key in
                ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')), "all capabilities must be dropped")
    require(len(status.get('NSpid', '').split()) == 1, "private PID namespace required")
    interfaces = sorted(path.name for path in Path('/sys/class/net').iterdir())
    if network_required:
        require(interfaces == ['lo'], "inference network must be disabled at container boundary")
    return {'schema': 'screen32-kernel-scope-v1', 'limits': observed,
            'uid': os.geteuid(), 'private_cgroup_root': True, 'read_only_root': True,
            'capabilities': 'none', 'no_new_privileges': True, 'seccomp_filter': True,
            'network_interfaces': interfaces, 'network_disabled': interfaces == ['lo']}
