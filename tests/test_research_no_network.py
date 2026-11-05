#!/usr/bin/env python3
"""Prove native socket denial without sending any network request."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeNetworkGuard(unittest.TestCase):
    def child(self, body, **kwargs):
        code = 'import sys;sys.path.insert(0,' + repr(str(ROOT / 'tools')) + ');' + body
        return subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code],
                              capture_output=True, text=True, timeout=20, **kwargs)

    def test_python_and_libc_socket_creation_denied_file_io_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.child('''
import ctypes,errno,json,socket
from pathlib import Path
import research_no_network as guard
guard.set_offline_environment()
receipt=guard.install()
for family in (socket.AF_INET,socket.AF_INET6,socket.AF_UNIX):
 try:socket.socket(family,socket.SOCK_STREAM)
 except OSError as error:assert error.errno==errno.EPERM
 else:raise RuntimeError('Python socket was allowed')
libc=ctypes.CDLL(None,use_errno=True)
assert libc.syscall(41,2,1,0)==-1 and ctypes.get_errno()==errno.EPERM
assert libc.syscall(425,1,0)==-1 and ctypes.get_errno()==errno.EPERM
path=Path(''' + repr(str(Path(directory) / 'regular.txt')) + ''')
path.write_bytes(b'ordinary-file-still-works')
assert path.read_bytes()==b'ordinary-file-still-works'
print(json.dumps(receipt))
''')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)['seccomp_installed'])

    def test_inherited_socket_fd_is_rejected_before_filter(self):
        left, right = socket.socketpair()
        try:
            result = self.child('''
import research_no_network as guard
try:guard.install()
except RuntimeError as error:
 assert str(error)=='inherited socket descriptors are forbidden'
 print('rejected')
else:raise RuntimeError('inherited socket accepted')
''', pass_fds=(left.fileno(),))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'rejected')
        finally:
            left.close(); right.close()


if __name__ == '__main__':
    unittest.main()
