"""Further restrict a local inference worker before importing any ML library.

Linux x86_64 only. This adds a seccomp filter to this process and descendants;
it does not alter host networking or relax an existing sandbox. Inherited socket
descriptors and pre-existing threads are rejected before installation.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import os
from pathlib import Path
import platform
import stat

ENVIRONMENT = {
    'ORT_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
    'DO_NOT_TRACK': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'GRADIO_ANALYTICS_ENABLED': 'False',
}


def set_offline_environment():
    os.environ.update(ENVIRONMENT)


def install():
    """Fail closed or install a kernel-level deny for new network communication."""
    if (platform.system(), platform.machine()) != ('Linux', 'x86_64'):
        raise RuntimeError('network guard requires Linux x86_64')
    tasks = list(Path('/proc/self/task').iterdir())
    if len(tasks) != 1:
        raise RuntimeError('network guard must precede native background threads')
    sockets = []
    for entry in Path('/proc/self/fd').iterdir():
        try:
            value = os.fstat(int(entry.name))
        except OSError as error:
            if error.errno == errno.EBADF:
                continue  # The enumeration's own already-closed directory FD.
            raise
        if stat.S_ISSOCK(value.st_mode):
            sockets.append(int(entry.name))
    if sockets:
        raise RuntimeError('inherited socket descriptors are forbidden')

    class Filter(ctypes.Structure):
        _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte),
                    ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ushort), ('filter', ctypes.POINTER(Filter))]

    # seccomp_data: syscall number at byte 0; audit architecture at byte 4.
    # A compat ABI cannot evade the native syscall-number deny list.
    kill = 0x80000000  # SECCOMP_RET_KILL_PROCESS
    denied = 0x00050000 | errno.EPERM
    instructions = [
        (0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E), (0x06, 0, 0, kill),
        (0x20, 0, 0, 0), (0x45, 0, 1, 0x40000000), (0x06, 0, 0, denied),
    ]
    # Linux x86_64 socket, connect, accept, sendto, sendmsg, bind, listen,
    # socketpair, accept4, sendmmsg. Denying creation plus rejecting inherited
    # sockets prevents even write()/io_uring from using a pre-existing socket.
    syscalls = (41, 42, 43, 44, 46, 49, 50, 53, 288, 307)
    # io_uring can create sockets without a socket syscall on newer kernels;
    # block its creation/entry/register paths as well, before any ring exists.
    syscalls += (425, 426, 427)
    for number in syscalls:
        instructions.extend(((0x15, 0, 1, number), (0x06, 0, 0, denied)))
    instructions.append((0x06, 0, 0, 0x7FFF0000))  # SECCOMP_RET_ALLOW
    filters = (Filter * len(instructions))(*(Filter(*row) for row in instructions))
    program = Program(len(filters), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), 'cannot set no_new_privs')
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:  # PR_SET_SECCOMP, FILTER
        raise OSError(ctypes.get_errno(), 'cannot install no-network filter')
    return {'schema': 'research-native-no-network-v1', 'platform': 'Linux-x86_64',
            'threads_at_install': 1, 'inherited_socket_descriptors': 0,
            'seccomp_installed': True, 'no_new_privileges': True,
            'denied_native_syscalls': list(syscalls),
            'filter_sha256': hashlib.sha256(bytes(filters)).hexdigest(),
            'network_attempt_count': None,
            'network_attempt_count_note': 'kernel denial is enforced; attempts are not counted',
            'environment': {name: os.environ.get(name) for name in ENVIRONMENT}}
